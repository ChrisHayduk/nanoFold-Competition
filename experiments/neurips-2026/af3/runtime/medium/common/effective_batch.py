"""Effective-eight scheduling and accumulation shared by the scaled trainers.

Checkpointing is permitted only at optimizer-step boundaries. Physical rank
RNG states are saved by the unchanged runtime helper; resumes require the same
physical world size and this exact accumulation contract.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable


def accumulation_contract(physical_world_size: int) -> dict[str, Any]:
    if physical_world_size not in (1, 2, 4, 8):
        raise ValueError("physical world size must be 1, 2, 4 or 8")
    return {
        "schema_version": 1,
        "effective_batch_size": 8,
        "virtual_world_size": 8,
        "physical_world_size": physical_world_size,
        "accumulation_steps": 8 // physical_world_size,
        "logical_rank_mapping": "physical_rank + microbatch_index * physical_world_size",
        "loss_scaling": "each example loss divided by accumulation_steps; DDP averages physical ranks",
        "optimizer_scheduler_clip": "once per completed effective batch",
        "checkpoint_boundary": "optimizer-step boundaries only; no partial gradients are persisted",
        "rng_policy": "physical-rank RNG streams; exact same-physical-world resume only",
    }


def logical_ranks(physical_rank: int, physical_world_size: int) -> tuple[int, ...]:
    contract = accumulation_contract(physical_world_size)
    if not 0 <= physical_rank < physical_world_size:
        raise ValueError("physical rank is outside the process group")
    return tuple(physical_rank + micro * physical_world_size for micro in range(contract["accumulation_steps"]))


def accumulated_optimizer_step(
    *, model: Any, optimizer: Any, scheduler: Any,
    physical_rank: int, physical_world_size: int,
    forward_loss: Callable[[int], Any],
    require_finite: Callable[[Any], None], clip_grad: Callable[[], Any],
) -> tuple[Any, Any]:
    """Run one mean-of-eight update; forward_loss receives the logical rank."""
    ranks = logical_ranks(physical_rank, physical_world_size)
    optimizer.zero_grad(set_to_none=True)
    loss_sum = None
    for micro, logical_rank in enumerate(ranks):
        sync = model.no_sync() if physical_world_size > 1 and micro + 1 < len(ranks) else nullcontext()
        with sync:
            loss = forward_loss(logical_rank)
            require_finite(loss)
            (loss / len(ranks)).backward()
        loss_sum = loss.detach().clone() if loss_sum is None else loss_sum + loss.detach()
    gradient_norm = clip_grad()
    require_finite(gradient_norm)
    optimizer.step()
    scheduler.step()
    return loss_sum / len(ranks), gradient_norm
