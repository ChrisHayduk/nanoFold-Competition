"""NanoFold-native reconstruction of the RFdiffusion architectural contract.

This is not the RosettaCommons training code (which is not public). It preserves
the claim-relevant ingredients: a RoseTTAFold-like 1D/2D/3D stack, diffusion of
residue frames, invariant graph messages, equivariant frame updates, sequence
and MSA conditioning, and all-atom reconstruction from the final frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


TRANSLATION_SIGMA_MAX = 5.0
IMPUTATION_AUGMENT_PROBABILITY = 0.15
IMPUTATION_TRANSLATION_SCALE = 0.75
IMPUTATION_ROTATION_FRACTION = 0.25
DIFFUSION_CONTRACT_ID = "cosine-translation-geodesic-haar-terminal-v2"
IMPUTATION_CONTRACT_ID = "stochastic-mask-blind-frame-completion-v2"
IDEAL_BACKBONE_COORDINATES = (
    (-0.53, 1.36, 0.0),
    (0.0, 0.0, 0.0),
    (1.52, 0.0, 0.0),
)
MODEL_FEATURE_KEYS = (
    "aatype",
    "restype",
    "msa",
    "msa_mask",
    "has_deletion",
    "deletion_value",
    "profile",
    "deletion_mean",
    "residue_index",
    "token_mask",
    "atom14_chemistry_mask",
)


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def diffusion_coefficients(
    timestep: torch.Tensor, *, total_steps: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the authoritative signal, translation-noise, and SO(3) schedule.

    The cosine translation path has an exact target-free terminal distribution:
    alpha(T) is exactly zero and sigma(T) is TRANSLATION_SIGMA_MAX. Rotations
    follow a geodesic path from identity to a Haar-uniform draw, with an exact
    Haar endpoint.
    """
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    fraction = (timestep.float() / float(total_steps)).clamp(0.0, 1.0)
    angle = 0.5 * torch.pi * fraction
    translation_alpha = torch.cos(angle)
    translation_sigma = TRANSLATION_SIGMA_MAX * torch.sin(angle)
    terminal = timestep >= total_steps
    translation_alpha = torch.where(
        terminal, torch.zeros_like(translation_alpha), translation_alpha
    )
    translation_sigma = torch.where(
        terminal,
        torch.full_like(translation_sigma, TRANSLATION_SIGMA_MAX),
        translation_sigma,
    )
    rotation_fraction = torch.where(terminal, torch.ones_like(fraction), fraction)
    return translation_alpha, translation_sigma, rotation_fraction


def terminal_translation_sigma(*, total_steps: int) -> float:
    """Derive the inference-prior scale from the training terminal timestep."""
    _, sigma, _ = diffusion_coefficients(
        torch.tensor([total_steps], dtype=torch.float32),
        total_steps=total_steps,
    )
    return float(sigma.item())


def ideal_backbone(
    *, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    return torch.tensor(
        IDEAL_BACKBONE_COORDINATES,
        device=device,
        dtype=dtype,
    )


def axis_angle_to_matrix(vector: torch.Tensor) -> torch.Tensor:
    """Rodrigues map with the correct first derivative at zero.

    The rotation-vector form is used directly instead of normalizing an axis.
    That distinction matters because every structure block starts with an
    exactly zero rotation update.  Normalizing the zero vector and then
    multiplying by its zero norm produces a zero autograd Jacobian, permanently
    freezing those zero-initialized rotation heads.
    """
    original_dtype = vector.dtype
    work = (
        vector.float()
        if vector.dtype in (torch.float16, torch.bfloat16)
        else vector
    )
    x, y, z = work.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(
        *work.shape[:-1], 3, 3
    )
    identity = torch.eye(3, device=work.device, dtype=work.dtype).expand(
        *work.shape[:-1], 3, 3
    )
    angle = work.norm(dim=-1)
    # torch.sinc(x) = sin(pi*x)/(pi*x).  These equivalent coefficients avoid
    # divisions by the angle and have their analytic limits at angle == 0.
    first_order = torch.sinc(angle / torch.pi)[..., None, None]
    second_order = (
        0.5 * torch.sinc(angle / (2.0 * torch.pi)).square()
    )[..., None, None]
    rotation = (
        identity + first_order * skew + second_order * (skew @ skew)
    )
    return rotation.to(original_dtype)


def uniform_rotation_from_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """Map normalized Gaussian quaternions to Haar-uniform SO(3) rotations."""
    quaternion = _normalize(quaternion)
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def scaled_uniform_rotation(
    quaternion: torch.Tensor, fraction: torch.Tensor
) -> torch.Tensor:
    """Geodesically scale a Haar rotation from identity to its full angle."""
    quaternion = _normalize(quaternion)
    quaternion = torch.where(
        (quaternion[..., :1] < 0),
        -quaternion,
        quaternion,
    )
    scalar = quaternion[..., :1].clamp(-1.0, 1.0)
    vector = quaternion[..., 1:]
    vector_norm = vector.norm(dim=-1, keepdim=True)
    axis = vector / vector_norm.clamp_min(1e-8)
    angle = 2.0 * torch.atan2(vector_norm, scalar)
    return axis_angle_to_matrix(axis * angle * fraction.unsqueeze(-1))


def frames_from_backbone(backbone: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return local-to-global rotations and CA translations from N/CA/C."""
    n, ca, c = backbone.unbind(dim=-2)
    x_axis = _normalize(c - ca)
    n_direction = n - ca
    y_axis = _normalize(
        n_direction - (n_direction * x_axis).sum(-1, keepdim=True) * x_axis
    )
    z_axis = _normalize(torch.cross(x_axis, y_axis, dim=-1))
    y_axis = _normalize(torch.cross(z_axis, x_axis, dim=-1))
    rotation = torch.stack((x_axis, y_axis, z_axis), dim=-1)
    return rotation, ca


def local_to_global(
    local: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor
) -> torch.Tensor:
    return torch.einsum("blij,blaj->blai", rotation, local) + translation.unsqueeze(-2)


def global_to_local(
    points: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor
) -> torch.Tensor:
    return torch.einsum("blij,blai->blaj", rotation, points - translation.unsqueeze(-2))


def _nearest_valid_value(
    value: torch.Tensor,
    valid: torch.Tensor,
    fallback: torch.Tensor,
) -> torch.Tensor:
    """Fill invalid sequence positions from the nearest valid sequence position."""
    batch, length = valid.shape
    indices = torch.arange(length, device=value.device)
    distance = (indices[:, None] - indices[None, :]).abs()
    distance = distance[None].expand(batch, -1, -1)
    distance = distance.masked_fill(~valid[:, None, :], length + 1)
    nearest = distance.argmin(dim=-1)
    batch_index = torch.arange(batch, device=value.device)[:, None]
    gathered = value[batch_index, nearest]
    has_valid = valid.any(dim=-1)
    has_valid = has_valid.reshape(batch, *([1] * (value.ndim - 1)))
    return torch.where(has_valid, gathered, fallback)


def _complete_translations(
    translation: torch.Tensor,
    valid: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    """Interpolate internal gaps and equivariantly extrapolate terminal gaps."""
    batch, length, _ = translation.shape
    indices = torch.arange(length, device=translation.device)
    index_grid = indices[None].expand(batch, -1)
    left_index = torch.cummax(
        torch.where(valid, index_grid, -torch.ones_like(index_grid)),
        dim=1,
    ).values
    right_index = torch.flip(
        torch.cummin(
            torch.flip(torch.where(valid, index_grid, length), dims=(1,)),
            dim=1,
        ).values,
        dims=(1,),
    )
    batch_index = torch.arange(batch, device=translation.device)[:, None]
    safe_left = left_index.clamp(0, length - 1)
    safe_right = right_index.clamp(0, length - 1)
    left_translation = translation[batch_index, safe_left]
    right_translation = translation[batch_index, safe_right]

    both = (left_index >= 0) & (right_index < length)
    denominator = (right_index - left_index).clamp_min(1).to(translation.dtype)
    interpolation_weight = (
        (index_grid - left_index).to(translation.dtype) / denominator
    ).unsqueeze(-1)
    interpolated = torch.lerp(left_translation, right_translation, interpolation_weight)

    left_direction = rotation[batch_index, safe_left, :, 0]
    right_direction = rotation[batch_index, safe_right, :, 0]
    left_offset = (
        (index_grid - left_index).to(translation.dtype).unsqueeze(-1)
        * 3.8
        * left_direction
    )
    right_offset = (
        (right_index - index_grid).to(translation.dtype).unsqueeze(-1)
        * 3.8
        * right_direction
    )
    completed = torch.where(
        both.unsqueeze(-1),
        interpolated,
        torch.where(
            (left_index >= 0).unsqueeze(-1),
            left_translation + left_offset,
            torch.where(
                (right_index < length).unsqueeze(-1),
                right_translation - right_offset,
                torch.zeros_like(translation),
            ),
        ),
    )
    return torch.where(valid.unsqueeze(-1), translation, completed)


def complete_corruption_frames(
    atom14: torch.Tensor,
    atom14_chemistry_mask: torch.Tensor,
    *,
    replacement_uniform: torch.Tensor | None = None,
    imputation_translation_noise: torch.Tensor | None = None,
    imputation_quaternion: torch.Tensor | None = None,
    augmentation_probability: float = IMPUTATION_AUGMENT_PROBABILITY,
    strict_complete_backbone: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build complete training frames without consulting target resolution masks.

    Deposited structures can omit N, CA, or C coordinates. Directly zeroing
    those residues makes the resolvedness pattern exactly readable by the
    denoiser and creates a train/inference support mismatch. Instead, this
    routine recognizes geometrically usable label frames and stochastically
    imputes unusable frames. A random subset of usable frames receives the same
    imputation augmentation, so imputation is not a deterministic missingness
    marker. Target resolution masks remain loss-only.
    """
    backbone = torch.nan_to_num(atom14[:, :, :3].float())
    support = atom14_chemistry_mask[:, :, :3].all(dim=-1)
    n, ca, c = backbone.unbind(dim=-2)
    n_vector = n - ca
    c_vector = c - ca
    n_length = n_vector.norm(dim=-1)
    c_length = c_vector.norm(dim=-1)
    frame_geometry_valid = (
        support
        & (n_length > 0.8)
        & (n_length < 2.2)
        & (c_length > 0.8)
        & (c_length < 2.2)
        & (torch.cross(c_vector, n_vector, dim=-1).norm(dim=-1) > 0.2)
    )
    if strict_complete_backbone and torch.any(support & ~frame_geometry_valid):
        incomplete = int((support & ~frame_geometry_valid).sum().item())
        raise ValueError(
            "strict complete-backbone mode rejected "
            f"{incomplete} chemistry-supported residues"
        )

    candidate_rotation, candidate_translation = frames_from_backbone(backbone)
    identity = torch.eye(3, device=backbone.device, dtype=backbone.dtype).expand(
        *candidate_rotation.shape
    )
    clean_rotation = _nearest_valid_value(
        candidate_rotation,
        frame_geometry_valid,
        identity,
    )

    # CA-only records are scientifically usable for translation corruption even
    # when no local N/CA/C frame can be formed. A plausible CA trace is detected
    # from sequence adjacency without consulting the resolved-atom mask.
    edge_distance = (ca[:, 1:] - ca[:, :-1]).norm(dim=-1)
    plausible_edge = (edge_distance > 2.5) & (edge_distance < 4.5)
    ca_trace_valid = frame_geometry_valid.clone()
    ca_trace_valid[:, 1:] |= plausible_edge
    ca_trace_valid[:, :-1] |= plausible_edge
    # Missing-coordinate slots in the NanoFold labels are exactly zero. Reject
    # that storage sentinel geometrically without reading its resolution mask.
    ca_trace_valid &= ca.abs().amax(dim=-1) > 1e-5
    ca_trace_valid &= support
    clean_translation = _complete_translations(
        candidate_translation,
        ca_trace_valid,
        clean_rotation,
    )
    support_weight = support.to(clean_translation.dtype).unsqueeze(-1)
    translation_centre = (clean_translation * support_weight).sum(
        dim=1, keepdim=True
    ) / support_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
    clean_translation = torch.where(
        support.unsqueeze(-1),
        clean_translation - translation_centre,
        torch.zeros_like(clean_translation),
    )

    if not 0.0 <= augmentation_probability <= 1.0:
        raise ValueError("augmentation_probability must be in [0, 1]")
    if replacement_uniform is None:
        replacement_uniform = torch.rand_like(candidate_translation[..., 0])
    if imputation_translation_noise is None:
        imputation_translation_noise = torch.randn_like(candidate_translation)
    if imputation_quaternion is None:
        imputation_quaternion = torch.randn(
            *candidate_translation.shape[:-1],
            4,
            device=candidate_translation.device,
            dtype=candidate_translation.dtype,
        )
    expected_scalar_shape = candidate_translation.shape[:-1]
    if replacement_uniform.shape != expected_scalar_shape:
        raise ValueError("replacement_uniform has an unexpected shape")
    if imputation_translation_noise.shape != candidate_translation.shape:
        raise ValueError("imputation_translation_noise has an unexpected shape")
    if imputation_quaternion.shape != (*expected_scalar_shape, 4):
        raise ValueError("imputation_quaternion has an unexpected shape")

    augmented = (
        (replacement_uniform < augmentation_probability)
        & frame_geometry_valid
        & support
    )
    rotation_replacement = support & (~frame_geometry_valid | augmented)
    translation_replacement = support & (~ca_trace_valid | augmented)
    rotation_anchor = frame_geometry_valid & ~augmented
    translation_anchor = ca_trace_valid & ~augmented

    corruption_rotation = _nearest_valid_value(
        candidate_rotation,
        rotation_anchor,
        identity,
    )
    rotation_perturbation = scaled_uniform_rotation(
        imputation_quaternion,
        torch.full_like(
            replacement_uniform,
            IMPUTATION_ROTATION_FRACTION,
        ),
    )
    corruption_rotation = torch.where(
        rotation_replacement[..., None, None],
        rotation_perturbation @ corruption_rotation,
        corruption_rotation,
    )
    corruption_translation = _complete_translations(
        candidate_translation,
        translation_anchor,
        corruption_rotation,
    )
    corruption_translation = torch.where(
        translation_replacement.unsqueeze(-1),
        corruption_translation
        + IMPUTATION_TRANSLATION_SCALE * imputation_translation_noise,
        corruption_translation,
    )
    corruption_centre = (corruption_translation * support_weight).sum(
        dim=1, keepdim=True
    ) / support_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
    corruption_translation = torch.where(
        support.unsqueeze(-1),
        corruption_translation - corruption_centre,
        torch.zeros_like(corruption_translation),
    )
    imputation_mask = rotation_replacement | translation_replacement
    return (
        clean_rotation,
        clean_translation,
        support,
        frame_geometry_valid,
        corruption_rotation,
        corruption_translation,
        imputation_mask,
    )


def corrupt_frames(
    base_rotation: torch.Tensor,
    base_translation: torch.Tensor,
    support: torch.Tensor,
    timestep: torch.Tensor,
    *,
    total_steps: int,
    translation_noise: torch.Tensor,
    rotation_quaternion: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Apply the shared forward frame corruption with explicit random draws."""
    if translation_noise.shape != base_translation.shape:
        raise ValueError("translation_noise has an unexpected shape")
    if rotation_quaternion.shape != (*base_translation.shape[:-1], 4):
        raise ValueError("rotation_quaternion has an unexpected shape")
    translation_alpha, translation_sigma, rotation_fraction = diffusion_coefficients(
        timestep, total_steps=total_steps
    )
    rotation_noise = scaled_uniform_rotation(
        rotation_quaternion,
        rotation_fraction[:, None].expand_as(base_translation[..., 0]),
    )
    noisy_rotation = rotation_noise @ base_rotation
    terminal_rotation = uniform_rotation_from_quaternion(rotation_quaternion)
    terminal = (timestep >= total_steps)[:, None, None, None]
    noisy_rotation = torch.where(terminal, terminal_rotation, noisy_rotation)
    noisy_translation = (
        translation_alpha[:, None, None] * base_translation
        + translation_sigma[:, None, None] * translation_noise
    )
    canonical_backbone = ideal_backbone(device=base_translation.device)
    local = canonical_backbone.expand(
        base_translation.shape[0],
        base_translation.shape[1],
        3,
        3,
    )
    noisy_backbone = local_to_global(local, noisy_rotation, noisy_translation)
    noisy_backbone = torch.where(
        support[:, :, None, None],
        noisy_backbone,
        torch.zeros_like(noisy_backbone),
    )
    return {
        "noisy_backbone": noisy_backbone,
        "noisy_rotation": noisy_rotation,
        "noisy_translation": noisy_translation,
        "translation_alpha": translation_alpha,
        "translation_sigma": translation_sigma,
        "rotation_fraction": rotation_fraction,
    }


def gather_nodes(nodes: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch, length, neighbors = indices.shape
    width = nodes.shape[-1]
    expanded = nodes[:, None].expand(batch, length, length, width)
    return torch.gather(
        expanded, 2, indices.unsqueeze(-1).expand(batch, length, neighbors, width)
    )


def gather_pairs(pairs: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch, length, neighbors = indices.shape
    width = pairs.shape[-1]
    return torch.gather(
        pairs,
        2,
        indices.unsqueeze(-1).expand(batch, length, neighbors, width),
    )


class Transition(nn.Module):
    def __init__(self, width: int, expansion: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.in_projection = nn.Linear(width, width * expansion * 2)
        self.out_projection = nn.Linear(width * expansion, width)
        nn.init.zeros_(self.out_projection.weight)
        nn.init.zeros_(self.out_projection.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        left, gate = self.in_projection(self.norm(value)).chunk(2, dim=-1)
        return value + self.out_projection(F.silu(left) * torch.sigmoid(gate))


class PairBiasedAttention(nn.Module):
    def __init__(self, state_width: int, pair_width: int, heads: int) -> None:
        super().__init__()
        if state_width % heads:
            raise ValueError("state width must be divisible by heads")
        self.heads = heads
        self.head_width = state_width // heads
        self.norm_state = nn.LayerNorm(state_width)
        self.norm_pair = nn.LayerNorm(pair_width)
        self.qkv = nn.Linear(state_width, state_width * 3, bias=False)
        self.pair_bias = nn.Linear(pair_width, heads, bias=False)
        self.output = nn.Linear(state_width, state_width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, state: torch.Tensor, pair: torch.Tensor, token_mask: torch.Tensor
    ) -> torch.Tensor:
        batch, length, width = state.shape
        q, k, v = self.qkv(self.norm_state(state)).chunk(3, dim=-1)
        q = q.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
        k = k.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
        v = v.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
        logits = torch.einsum("bhic,bhjc->bhij", q, k) * self.head_width**-0.5
        logits = logits + self.pair_bias(self.norm_pair(pair)).permute(0, 3, 1, 2)
        logits = logits.masked_fill(~token_mask[:, None, None, :], -1e4)
        attention = torch.softmax(logits.float(), dim=-1).to(v.dtype)
        output = torch.einsum("bhij,bhjc->bhic", attention, v)
        output = output.transpose(1, 2).reshape(batch, length, width)
        return state + self.output(output) * token_mask.unsqueeze(-1)


class ThreeTrackBlock(nn.Module):
    def __init__(
        self,
        *,
        state_width: int,
        pair_width: int,
        heads: int,
        edge_width: int,
        top_k: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.sequence_attention = PairBiasedAttention(state_width, pair_width, heads)
        self.sequence_transition = Transition(state_width)
        self.pair_norm = nn.LayerNorm(pair_width)
        self.state_norm = nn.LayerNorm(state_width)
        self.pair_update = nn.Sequential(
            nn.Linear(pair_width + 2 * state_width + 17, pair_width * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(pair_width * 2, pair_width),
        )
        nn.init.zeros_(self.pair_update[-1].weight)
        nn.init.zeros_(self.pair_update[-1].bias)
        self.pair_transition = Transition(pair_width, expansion=2)
        self.edge_mlp = nn.Sequential(
            nn.Linear(pair_width + 2 * state_width + 17, edge_width),
            nn.SiLU(),
            nn.Linear(edge_width, edge_width),
            nn.SiLU(),
        )
        self.state_update = nn.Linear(edge_width, state_width)
        self.frame_update = nn.Sequential(
            nn.LayerNorm(state_width),
            nn.Linear(state_width, state_width),
            nn.SiLU(),
            nn.Linear(state_width, 6),
        )
        nn.init.zeros_(self.state_update.weight)
        nn.init.zeros_(self.state_update.bias)
        nn.init.zeros_(self.frame_update[-1].weight)
        nn.init.zeros_(self.frame_update[-1].bias)

    @staticmethod
    def radial_basis(distance: torch.Tensor) -> torch.Tensor:
        centres = torch.linspace(
            0.0, 32.0, 16, device=distance.device, dtype=distance.dtype
        )
        return torch.exp(-((distance.unsqueeze(-1) - centres) / 2.0).square())

    def forward(
        self,
        state: torch.Tensor,
        pair: torch.Tensor,
        rotation: torch.Tensor,
        translation: torch.Tensor,
        token_mask: torch.Tensor,
        time_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, _ = state.shape
        state = self.sequence_attention(state, pair, token_mask)
        state = self.sequence_transition(state)

        state_normalized = self.state_norm(state)
        pair_normalized = self.pair_norm(pair)
        distance = torch.cdist(translation.float(), translation.float()).to(state.dtype)
        rbf = self.radial_basis(distance)
        time_pair = time_feature[:, None, None, :].expand(batch, length, length, 1)
        left = state_normalized[:, :, None, :].expand(-1, -1, length, -1)
        right = state_normalized[:, None, :, :].expand(-1, length, -1, -1)
        pair_delta = self.pair_update(
            torch.cat((pair_normalized, left, right, rbf, time_pair), dim=-1)
        )
        pair_mask = token_mask[:, :, None] & token_mask[:, None, :]
        pair = pair + pair_delta * pair_mask.unsqueeze(-1)
        pair = self.pair_transition(pair)

        valid_distance = distance.masked_fill(~pair_mask, 1e6)
        diagonal = torch.eye(length, device=distance.device, dtype=torch.bool)
        valid_distance = valid_distance.masked_fill(diagonal.unsqueeze(0), 1e6)
        neighbor_count = min(self.top_k, max(1, length - 1))
        neighbor_index = valid_distance.topk(
            neighbor_count, dim=-1, largest=False
        ).indices
        neighbor_state = gather_nodes(state_normalized, neighbor_index)
        centre_state = state_normalized[:, :, None].expand(-1, -1, neighbor_count, -1)
        neighbor_pair = gather_pairs(pair_normalized, neighbor_index)
        neighbor_distance = torch.gather(distance, 2, neighbor_index)
        neighbor_rbf = self.radial_basis(neighbor_distance)
        neighbor_time = time_feature[:, None, None, :].expand(
            batch, length, neighbor_count, 1
        )
        edge = self.edge_mlp(
            torch.cat(
                (
                    neighbor_pair,
                    centre_state,
                    neighbor_state,
                    neighbor_rbf,
                    neighbor_time,
                ),
                dim=-1,
            )
        )
        neighbor_valid = torch.gather(
            token_mask[:, None, :].expand(-1, length, -1), 2, neighbor_index
        )
        edge = edge * neighbor_valid.unsqueeze(-1)
        message = edge.sum(dim=2) / neighbor_valid.sum(dim=2, keepdim=True).clamp_min(1)
        state = state + self.state_update(message) * token_mask.unsqueeze(-1)

        local_translation, local_rotation = self.frame_update(state).chunk(2, -1)
        local_translation = 0.25 * torch.tanh(local_translation)
        local_rotation = 0.10 * torch.tanh(local_rotation)
        global_translation = torch.einsum("blij,blj->bli", rotation, local_translation)
        translation = translation + global_translation * token_mask.unsqueeze(-1)
        rotation = rotation @ axis_angle_to_matrix(local_rotation)
        return state, pair, rotation, translation


@dataclass(frozen=True)
class RFStyleConfig:
    state_width: int = 256
    pair_width: int = 128
    msa_width: int = 128
    blocks: int = 24
    heads: int = 8
    edge_width: int = 256
    top_k: int = 32
    dropout: float = 0.10
    checkpoint_blocks: bool = True


class RFdiffusionStyleModel(nn.Module):
    def __init__(self, config: RFStyleConfig) -> None:
        super().__init__()
        self.config = config
        self.aatype_embedding = nn.Embedding(22, config.state_width)
        self.profile_projection = nn.Linear(33, config.state_width)
        self.msa_embedding = nn.Embedding(32, config.msa_width)
        self.msa_projection = nn.Linear(config.msa_width + 2, config.state_width)
        self.left_pair = nn.Linear(config.state_width, config.pair_width)
        self.right_pair = nn.Linear(config.state_width, config.pair_width)
        self.relative_embedding = nn.Embedding(65, config.pair_width)
        self.blocks = nn.ModuleList(
            [
                ThreeTrackBlock(
                    state_width=config.state_width,
                    pair_width=config.pair_width,
                    heads=config.heads,
                    edge_width=config.edge_width,
                    top_k=config.top_k,
                    dropout=config.dropout,
                )
                for _ in range(config.blocks)
            ]
        )
        self.local_atom_head = nn.Sequential(
            nn.LayerNorm(config.state_width),
            nn.Linear(config.state_width, config.state_width * 2),
            nn.SiLU(),
            nn.Linear(config.state_width * 2, 14 * 3),
        )
        self.final_pair_to_state = nn.Linear(config.pair_width, config.state_width)
        canonical = torch.zeros(14, 3)
        canonical[:3] = ideal_backbone(device=canonical.device)
        canonical[3] = torch.tensor([2.15, 1.00, 0.0])  # O
        self.register_buffer("canonical_atom14", canonical, persistent=True)

    def initial_state(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_mask = batch["token_mask"]
        aatype = batch["aatype"].clamp(0, 21)
        profile_features = torch.cat(
            (batch["profile"], batch["deletion_mean"].unsqueeze(-1)), dim=-1
        )
        msa_mask = batch["msa_mask"].to(batch["profile"].dtype)
        msa_encoded = self.msa_embedding(batch["msa"])
        msa_encoded = (msa_encoded * msa_mask.unsqueeze(-1)).sum(dim=1) / msa_mask.sum(
            dim=1, keepdim=False
        ).unsqueeze(-1).clamp_min(1)
        deletion_features = torch.stack(
            (
                (batch["has_deletion"] * msa_mask).sum(dim=1)
                / msa_mask.sum(dim=1).clamp_min(1),
                (batch["deletion_value"] * msa_mask).sum(dim=1)
                / msa_mask.sum(dim=1).clamp_min(1),
            ),
            dim=-1,
        )
        state = (
            self.aatype_embedding(aatype)
            + self.profile_projection(profile_features)
            + self.msa_projection(torch.cat((msa_encoded, deletion_features), dim=-1))
        )
        state = state * token_mask.unsqueeze(-1)
        residue_index = batch["residue_index"]
        separation = (residue_index[:, :, None] - residue_index[:, None, :]).clamp(
            -32, 32
        ) + 32
        pair = (
            self.left_pair(state)[:, :, None]
            + self.right_pair(state)[:, None, :]
            + self.relative_embedding(separation)
        )
        pair_mask = token_mask[:, :, None] & token_mask[:, None, :]
        return state, pair * pair_mask.unsqueeze(-1)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        noisy_backbone: torch.Tensor,
        timestep: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        state, pair = self.initial_state(batch)
        rotation, translation = frames_from_backbone(noisy_backbone)
        time_feature = (timestep.float() / 50.0).unsqueeze(-1).to(state.dtype)
        token_mask = batch["token_mask"]

        for block in self.blocks:
            if self.config.checkpoint_blocks and self.training:
                state, pair, rotation, translation = checkpoint(
                    block,
                    state,
                    pair,
                    rotation,
                    translation,
                    token_mask,
                    time_feature,
                    use_reentrant=False,
                )
            else:
                state, pair, rotation, translation = block(
                    state,
                    pair,
                    rotation,
                    translation,
                    token_mask,
                    time_feature,
                )

        state = state + self.final_pair_to_state(pair.mean(dim=2))
        local_atoms = self.local_atom_head(state).reshape(
            state.shape[0], state.shape[1], 14, 3
        )
        local_atoms = local_atoms + self.canonical_atom14.to(local_atoms.dtype)
        atom14 = local_to_global(local_atoms, rotation, translation)
        # Output support is determined only by residue chemistry and padding.
        # The deposited-structure resolution mask is supervision and must not
        # enter the prediction path.
        atom14 = atom14 * batch["atom14_chemistry_mask"].unsqueeze(-1)
        return {
            "pred_atom14": atom14,
            "rotation": rotation,
            "translation": translation,
        }


def diffuse_backbone(
    atom14: torch.Tensor,
    atom14_chemistry_mask: torch.Tensor,
    timestep: torch.Tensor,
    *,
    total_steps: int = 50,
    strict_complete_backbone: bool = False,
) -> dict[str, torch.Tensor]:
    (
        clean_rotation,
        clean_translation,
        corruption_support,
        clean_frame_geometry_mask,
        corruption_rotation,
        corruption_translation,
        imputation_mask,
    ) = complete_corruption_frames(
        atom14,
        atom14_chemistry_mask,
        strict_complete_backbone=strict_complete_backbone,
    )
    translation_noise = torch.randn_like(clean_translation)
    rotation_quaternion = torch.randn(
        *clean_translation.shape[:-1],
        4,
        device=clean_translation.device,
        dtype=clean_translation.dtype,
    )
    corrupted = corrupt_frames(
        corruption_rotation,
        corruption_translation,
        corruption_support,
        timestep,
        total_steps=total_steps,
        translation_noise=translation_noise,
        rotation_quaternion=rotation_quaternion,
    )
    return {
        **corrupted,
        "clean_rotation": clean_rotation,
        "clean_translation": clean_translation,
        "corruption_support": corruption_support,
        "clean_frame_geometry_mask": clean_frame_geometry_mask,
        "imputation_mask": imputation_mask,
    }
