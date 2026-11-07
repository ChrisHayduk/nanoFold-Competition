"""Differentiable geometry utilities shared by both reconstruction arms."""

from __future__ import annotations

import math

import torch


def random_rotation(
    batch_size: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Uniform SO(3) matrices from normalized random quaternions."""
    quaternion = torch.randn(batch_size, 4, device=device, dtype=dtype)
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)
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
    ).reshape(batch_size, 3, 3)


def apply_rotation(points: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return torch.einsum("b...c,bdc->b...d", points, rotation)


def masked_centre(points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(points.dtype).unsqueeze(-1)
    centre = (points * weight).sum(dim=(-3, -2), keepdim=True) / weight.sum(
        dim=(-3, -2), keepdim=True
    ).clamp_min(1.0)
    return torch.where(weight.bool(), points - centre, torch.zeros_like(points))


def kabsch_align(
    mobile: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Rigid-align ``mobile`` to ``target`` using masked atom coordinates."""
    batch = mobile.shape[0]
    mobile_flat = mobile.reshape(batch, -1, 3)
    target_flat = target.reshape(batch, -1, 3)
    weight = mask.reshape(batch, -1, 1).to(mobile.dtype)
    denominator = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
    mobile_mean = (mobile_flat * weight).sum(dim=1, keepdim=True) / denominator
    target_mean = (target_flat * weight).sum(dim=1, keepdim=True) / denominator
    mobile_centered = (mobile_flat - mobile_mean) * weight
    target_centered = (target_flat - target_mean) * weight
    covariance = torch.einsum(
        "bni,bnj->bij", mobile_centered, target_centered
    )
    u, _, vh = torch.linalg.svd(covariance.float(), full_matrices=False)
    determinant = torch.det(u @ vh)
    correction = torch.eye(3, device=mobile.device, dtype=torch.float32).repeat(
        batch, 1, 1
    )
    correction[:, -1, -1] = torch.where(
        determinant < 0,
        torch.tensor(-1.0, device=mobile.device),
        torch.tensor(1.0, device=mobile.device),
    )
    rotation = (u @ correction @ vh).to(mobile.dtype)
    aligned = torch.einsum(
        "bni,bij->bnj", mobile_flat - mobile_mean, rotation
    ) + target_mean
    return aligned.reshape_as(mobile)


def aligned_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    aligned = kabsch_align(prediction, target, mask)
    squared = (aligned - target).square().sum(dim=-1)
    weight = mask.to(squared.dtype)
    return (squared * weight).sum() / weight.sum().clamp_min(1.0)


def smooth_lddt_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    atom_mask: torch.Tensor,
    *,
    cutoff: float = 15.0,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Smooth C-alpha lDDT loss, averaged over valid local atom pairs."""
    pred_ca = prediction[:, :, 1]
    true_ca = target[:, :, 1]
    ca_mask = atom_mask[:, :, 1]
    pred_distance = torch.cdist(pred_ca.float(), pred_ca.float())
    true_distance = torch.cdist(true_ca.float(), true_ca.float())
    pair_mask = ca_mask[:, :, None] & ca_mask[:, None, :]
    pair_mask &= true_distance < cutoff
    diagonal = torch.eye(
        true_distance.shape[-1], device=true_distance.device, dtype=torch.bool
    )
    pair_mask &= ~diagonal.unsqueeze(0)
    error = (pred_distance - true_distance).abs()
    score = torch.zeros_like(error)
    for threshold in (0.5, 1.0, 2.0, 4.0):
        score += torch.sigmoid(
            (torch.tensor(threshold, device=error.device) - error) / temperature
        )
    score = score / 4.0
    weight = pair_mask.to(score.dtype)
    return 1.0 - (score * weight).sum() / weight.sum().clamp_min(1.0)


def lognormal_sigma(
    batch_size: int,
    *,
    p_mean: float = -1.2,
    p_std: float = 1.5,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.exp(
        p_mean
        + p_std * torch.randn(batch_size, device=device, dtype=dtype)
    )


def sinusoidal_time_embedding(
    sigma: torch.Tensor, channels: int
) -> torch.Tensor:
    if channels % 2:
        raise ValueError("time-embedding width must be even")
    frequencies = torch.exp(
        torch.linspace(
            math.log(1.0),
            math.log(10_000.0),
            channels // 2,
            device=sigma.device,
            dtype=sigma.dtype,
        )
    )
    phase = torch.log(sigma.clamp_min(1e-8))[:, None] * frequencies[None]
    return torch.cat((phase.sin(), phase.cos()), dim=-1)
