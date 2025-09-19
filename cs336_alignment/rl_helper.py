from typing import Callable, Literal

import ray
import torch


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float = 1e-6,
    normalize_by_std: float = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    rollout_batch_size = len(rollout_responses)
    assert rollout_batch_size % group_size == 0, "Batch size must be divisible by group size"

    meta = dict()
    raw_rewards =list(reward_fn(response, truth).get("reward", 0) for response, truth in zip(rollout_responses, repeated_ground_truths))
    raw_rewards = torch.tensor(raw_rewards, dtype=torch.float32)
    group_means = raw_rewards.view(-1, group_size).mean(dim=1).repeat_interleave(group_size)
    meta["group_means"] = group_means.item()
    if normalize_by_std:
        group_stds = raw_rewards.view(-1, group_size).std(dim=1).repeat_interleave(group_size)
        meta["group_stds"] = group_stds.item()
        normalized_rewards = (raw_rewards - group_means) / (group_stds + advantage_eps)
    else:
        normalized_rewards = raw_rewards - group_means

    return normalized_rewards, raw_rewards, meta

def add_group_normalized_rewards_to_rollout(
    rollout: ray.data.Dataset,
    group_size: int,
    advantage_eps: float = 1e-6,
    normalize_by_std: float = False,
) -> ray.data.Dataset:
    rollout_batch_size = rollout.count()
    assert rollout_batch_size % group_size == 0, "Batch size must be divisible by group size"

    raw_rewards = list(row["reward"] for row in rollout.iter_rows())
    raw_rewards = torch.tensor(raw_rewards, dtype=torch.float32)
    group_means = raw_rewards.view(-1, group_size).mean(dim=1).repeat_interleave(group_size)
    if normalize_by_std:
        group_stds = raw_rewards.view(-1, group_size).std(dim=1).repeat_interleave(group_size)
        normalized_rewards = (raw_rewards - group_means) / (group_stds + advantage_eps)
        return rollout.zip(ray.data.from_items([{"advantage": adv.item(), "group_means": mean.item(), "group_stds": std.item()} for adv, mean, std in zip(normalized_rewards, group_means, group_stds)]))
    else:
        normalized_rewards = raw_rewards - group_means
        return rollout.zip(ray.data.from_items([{"advantage": adv.item(), "group_means": mean.item()} for adv, mean in zip(normalized_rewards, group_means)]))

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    return - raw_rewards_or_advantages.expand_as(policy_log_probs) * policy_log_probs

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    advantages = advantages.expand_as(policy_log_probs)
    g = torch.where(advantages < 0, (1-cliprange)*advantages, (1+cliprange)*advantages)
    ratio = torch.exp(policy_log_probs - old_log_probs)
    raw  = advantages * ratio
    return -torch.min(raw, g), {"clipped": raw > g}

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        assert raw_rewards is not None, "raw_rewards must be provided for no_baseline loss"
        return compute_naive_policy_gradient_loss(
            raw_rewards,
            policy_log_probs,
        ), {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "advantages must be provided for reinforce_with_baseline loss"
        return compute_naive_policy_gradient_loss(
            advantages,
            policy_log_probs,
        ), {}
    elif loss_type == "grpo_clip":
        assert advantages is not None, "advantages must be provided for grpo_clip loss"
        assert old_log_probs is not None, "old_log_probs must be provided for grpo_clip loss"
        assert cliprange is not None, "cliprange must be provided for grpo_clip loss"
        return compute_grpo_clip_loss(
            advantages,
            policy_log_probs,
            old_log_probs,
            cliprange,
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    masked_tensor = tensor * mask
    if dim is None:
        return masked_tensor.sum() / mask.sum()
    else:
        return masked_tensor.sum(dim=dim) / mask.sum(dim=dim)
    
def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    per_token_loss, meta = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange,
    )
    loss = masked_mean(per_token_loss, response_mask) / gradient_accumulation_steps
    loss.backward()
    return loss, meta