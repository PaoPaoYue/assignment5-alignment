import copy
from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import tempfile
from typing import Literal

import numpy as np
import ray
import ray.train
import ray.train.torch
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm.auto import tqdm
from vllm import SamplingParams
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_basics.optimizer import CosineAnnealingWithPrewarmRestarts
from cs336_basics.checkpoint import load_checkpoint
from cs336_alignment.common import *
from cs336_alignment.sft_helper import *
from cs336_alignment.rl_helper import *
from cs336_alignment.eval_model import Evaluator

import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainParams:
    run_name: str
    evaluator: Evaluator

    model_dir_path: str
    ckpt_path: str
    train_dir_path: str
    valid_dir_path: str
    valid_result_path: str

    train_cases: int = 512

    seed: int = 42

    lr: float = 1e-5
    rollout_batch_size: int = 256
    group_size: int = 8
    train_batch_size: int = 256
    val_batch_size: int = 12
    accumulate_steps: int = 128
    max_grad: float = 1
    loss_type: Literal[
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip",
    ] = "reinforce_with_baseline"
    use_std_normalization: bool = True
    grpo_cliprange: float = 0.2

    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.95
    optimizer_weight_decay: float = 0.0

    n_grpo_steps: int = 30
    val_step_freq: int = 10
    epochs_per_rollout_batch: int = 1

    def __post_init__(self):
        assert (
            self.train_batch_size % self.accumulate_steps == 0
        ), "train_batch_size must be divisible by gradient_accumulation_steps"
        self.micro_train_batch_size = self.train_batch_size // self.accumulate_steps
        assert (
            self.rollout_batch_size % self.group_size == 0
        ), "rollout_batch_size must be divisible by group_size"
        self.n_prompts_per_rollout_batch = self.rollout_batch_size // self.group_size
        assert (
            self.train_batch_size >= self.group_size
        ), "train_batch_size must be greater than or equal to group_size"
        self.n_microbatches_per_rollout_batch = (
            self.rollout_batch_size // self.micro_train_batch_size
        )
        self.off_policy = self.epochs_per_rollout_batch > 1 or self.train_batch_size < self.rollout_batch_size
        assert (
            self.off_policy and self.loss_type == "grpo_clip" or not self.off_policy
        ), "Off-policy training is only supported with grpo_clip loss."


def train_model(config: dict[any, any]):
    params = TrainParams(**config)
    init_random_seed(params.seed)
    mute_ray_data()

    train_dataset, valid_dataset = load_dataset(params.train_dir_path).limit(
        params.train_cases
    ), load_dataset(params.valid_dir_path)

    model_state_dict= torch.load(f"{params.ckpt_path}/checkpoint.pt",weights_only=False)

    model = AutoModelForCausalLM.from_pretrained(
        params.model_dir_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model.load_state_dict(model_state_dict)
    model = ray.train.torch.prepare_model(model)

    tokenizer = AutoTokenizer.from_pretrained(
        params.model_dir_path, trust_remote_code=True
    )
    optimizer = AdamW(
        model.parameters(),
        lr=params.lr,
        betas=(params.optimizer_beta1, params.optimizer_beta2),
        weight_decay=params.optimizer_weight_decay,
    )

    init_wandb(
        run_name=params.run_name,
        config={
            "batch_size": params.train_batch_size,
            "learning_rate": params.lr,
            "accumulate_steps": params.accumulate_steps,
            "max_grad": params.max_grad,
        },
    )
    
    # ========= 训练循环（最优模型保存）=========
    for grpo_step in range(1, params.n_grpo_steps + 1):
        rollout = sample_rollout(
            grpo_step,
            model_state_dict,
            train_dataset,
            params,
        )

        torch.cuda.empty_cache()

        model_state_dict = grpo_train(
            grpo_step,
            model,
            tokenizer,
            optimizer,
            rollout,
            params,
        )

        torch.cuda.empty_cache()

        if grpo_step % params.val_step_freq != 0 and grpo_step != params.n_grpo_steps:
            continue

        val_metrics = validate(
            grpo_step,
            model_state_dict,
            valid_dataset,
            params,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(model_state_dict, os.path.join(tmpdir, "checkpoint.pt"))
            checkpoint = ray.train.Checkpoint.from_directory(tmpdir)
            ray.train.report(metrics=val_metrics, checkpoint=checkpoint)

    ray.get(params.evaluator.close.remote())
    wandb.finish()


# ========== 训练与验证 ==========
@torch.no_grad()
def sample_rollout(
    grpo_step: int,
    model_state_dict: dict[str, torch.Tensor],
    dataset: ray.data.Dataset,
    params: TrainParams,
):
    evaluator = params.evaluator
    total = dataset.count()
    sampled = dataset.random_sample(
        params.n_prompts_per_rollout_batch / total, seed=params.seed + grpo_step * 1000
    ).limit(params.n_prompts_per_rollout_batch)
    assert sampled.count() == params.n_prompts_per_rollout_batch
    logger.info(
        f"GRPO step {grpo_step}: Sampled {sampled.count()} examples for training."
    )
    ray.get(evaluator.load_new_policy_weights.remote(model_state_dict))
    results, _ = ray.get(
        evaluator.evaluate.remote(
            "rollout",
            grpo_step,
            sampled,
            params.val_batch_size,
            sample_n=params.group_size,
        )
    )
    return add_group_normalized_rewards_to_rollout(
        results.select_columns(["problem", "response", "reward"]),
        params.group_size,
        normalize_by_std=params.use_std_normalization,
    )


def grpo_train(
    grpo_step: int,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    optimizer: torch.optim.Optimizer,
    dataset: ray.data.Dataset,
    params: TrainParams,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    running_entropy = 0.0

    old_log_prob_cache = dict()

    total = (
        dataset.count() + params.micro_train_batch_size - 1
    ) // params.micro_train_batch_size

    def batch_transform(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        prompt_strs = [R1_ZERO_PROMPT.format(question=prob) for prob in batch["problem"]]
        output_strs = [str(test) for test in batch["response"]]
        input_tensors = tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)

        batch["input_ids"] = input_tensors["input_ids"].numpy()
        batch["labels"] = input_tensors["labels"].numpy()
        batch["response_mask"] = input_tensors["response_mask"].numpy()

        del batch["problem"]
        del batch["response"]

        return batch

    dataset = dataset.map_batches(fn=batch_transform, batch_size=params.micro_train_batch_size)

    # pre-compute log probs of old policy when off-policy training
    if params.off_policy:
        for i, batch in enumerate(dataset.iter_batches(batch_size=params.micro_train_batch_size)):
            inputs = torch.from_numpy(batch["input_ids"]).to("cuda", non_blocking=True)
            labels = torch.from_numpy(batch["labels"]).to("cuda", non_blocking=True)

            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                token_probs = torch.softmax(model(inputs).logits, dim=-1)
                token_log_probs = token_probs.log()
                log_probs = torch.gather(
                    token_log_probs,
                    dim=-1,
                    index=labels.unsqueeze(-1),
                ).squeeze(-1)  # (B, seq_len)
                old_log_prob_cache[i] = log_probs

    for epoch in range(1, params.epochs_per_rollout_batch + 1):
        pbar = tqdm(
            dataset.iter_batches(batch_size=params.micro_train_batch_size),
            total=total,
            desc=f"Train | Epoch {epoch}",
            leave=False,
        )
        for i, batch in enumerate(pbar):
            inputs = torch.from_numpy(batch["input_ids"]).to("cuda", non_blocking=True)
            labels = torch.from_numpy(batch["labels"]).to("cuda", non_blocking=True)
            response_mask = torch.from_numpy(batch["response_mask"]).to("cuda", non_blocking=True)
            raw_rewards = torch.from_numpy(np.expand_dims(batch["rewards"], axis=-1)).to("cuda", non_blocking=True)
            advantages = torch.from_numpy(np.expand_dims(batch["advantages"], axis=-1)).to("cuda", non_blocking=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                token_probs = torch.softmax(model(inputs).logits, dim=-1) # (B, seq_len, vocab_size)
                token_log_probs = token_probs.log()
                log_probs = torch.gather(
                    token_log_probs,
                    dim=-1,
                    index=labels.unsqueeze(-1),
                ).squeeze(-1)  # (B, seq_len)
                old_log_probs = old_log_prob_cache.get(i)
                per_token_entropy = masked_mean(
                    -torch.sum(token_probs * token_log_probs, dim=-1), 
                    response_mask
                )
                loss, meta = grpo_microbatch_train_step(
                    log_probs,
                    response_mask,
                    params.accumulate_steps,
                    params.loss_type,
                    raw_rewards,
                    advantages,
                    old_log_probs,
                    params.grpo_cliprange,
                )

            if (i + 1) % params.accumulate_steps == 0:
                for p in model.parameters():
                    if p.grad is not None:
                        torch.nn.utils.clip_grad_value_(p, params.max_grad)
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * params.accumulate_steps
            running_entropy += per_token_entropy.item()

            pbar.set_postfix(
                loss=running_loss / (i + 1), entropy=running_entropy / (i + 1)
            )

            if (i + 1) % params.accumulate_steps == 0 and wandb.run is not None:
                report = {
                    "grpo_step": grpo_step,
                    "train_step": (epoch - 1) * total
                    + i // params.accumulate_steps
                    + 1,
                    "train/loss": loss.item() * params.accumulate_steps,
                    "train/entropy": per_token_entropy.item(),
                }
                if "group_means" in batch:
                    report["train/group_means"] = batch["group_means"].mean()
                if "group_stds" in batch:
                    report["train/group_stds"] = batch["group_stds"].mean()
                if "clipped" in meta:
                    report["train/clip_fraction"] = meta["clipped"].float().mean().item()
                wandb.log(report)

    return model.state_dict()


@torch.no_grad()
def validate(
    grpo_step: int,
    model_state_dict: dict[str, torch.Tensor],
    dataset: ray.data.Dataset,
    params: TrainParams,
) -> dict[str, float] | None:
    evaluator = params.evaluator
    ray.get(evaluator.load_new_policy_weights.remote(model_state_dict))
    _, analysis = ray.get(
        evaluator.evaluate.remote(
            "validation",
            grpo_step,
            dataset,
            params.val_batch_size,
            result_path=f"{params.valid_result_path}/grpo_step_{grpo_step}",
        )
    )
    return analysis


if __name__ == "__main__":
    run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    evaluator = Evaluator.options(num_gpus=0.1).remote(
        run_name=run_name,
        model_path=os.path.abspath("./models/qwen2.5-math-1.5b"),
        seed=42,
        sampling_params=SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=1024,
            min_tokens=4,
            include_stop_str_in_output=True,
            stop="</answer>",
            logprobs=10,
            seed=42,
        ),
        dtype=torch.bfloat16,
        # enable_prefix_caching=True,
        gpu_memory_utilization=0.1,
    )
    ray.get(evaluator.ready.remote())
    params = TrainParams(
        run_name=run_name,
        evaluator=evaluator,
        model_dir_path=os.path.abspath("./models/qwen2.5-math-1.5b"),
        ckpt_path=os.path.abspath("./artifacts/checkpoints/ei_ckpt"),
        train_dir_path=os.path.abspath("./datasets/train/math_12k/train"),
        valid_dir_path=os.path.abspath("./datasets/eval/math"),
        valid_result_path=os.path.abspath("./artifacts/results/rl-valid"),
    )
    trainer = ray.train.torch.TorchTrainer(
        train_model,
        train_loop_config=asdict(params),
        scaling_config=ray.train.ScalingConfig(
            num_workers=1, use_gpu=True, resources_per_worker={"GPU": 0.9}
        ),
        run_config=ray.train.RunConfig(
            name=run_name,
            checkpoint_config=ray.train.CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute="reward",
                checkpoint_score_order="max",
            ),
        ),
    )
    result = trainer.fit()
    best_ckpt, best_metrics = result.best_checkpoints[0]
    best_ckpt.to_directory(os.path.abspath("./artifacts/checkpoints/rl_ckpt"))
    logger.info(
        f"Train finished with metrics {best_metrics}, saved checkpoint to {os.path.abspath("./artifacts/checkpoints/rl_ckpt")}."
    )
