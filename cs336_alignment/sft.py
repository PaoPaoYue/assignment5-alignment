from dataclasses import asdict, dataclass, field
import os
from pathlib import Path

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

from cs336_basics.nn_utils import get_model_size
from cs336_basics.optimizer import CosineAnnealingWithPrewarmRestarts
from cs336_basics.checkpoint import save_checkpoint
from cs336_alignment.common import *
from cs336_alignment.sft_helper import *
from cs336_alignment.eval_model import Evaluator

import logging

logger = logging.getLogger(__name__)

__MAJOR_METRIC_NAME = "eval/reward"
__MAJOR_METRIC_GOAL = "maximize"


@dataclass
class TrainParams:
    model_dir_path: str
    train_dir_path: str
    valid_dir_path: str
    valid_result_path: str
    checkpoint_path: str

    valid_steps: list = field(default_factory=lambda: [8, 16, 32, 64]) # step = count / batch_size / accumulate_steps, default counts=[128, 256, 512, 1024]

    seed: int = 42

    lr: float = 1e-3
    batch_size: int = 1
    accumulate_steps: int = 1
    max_grad: float = 0
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.999
    optimizer_weight_decay: float = 0.01
    scheduler_t: int = 1
    scheduler_t_warmup: float = 0.1
    scheduler_t_mult: int = 1
    scheduler_min_lr: float = 0
    schduler_warmup_lr_factor: float = 0

    num_epochs: int = 1
    patience: int = 1
    min_delta: int = 1e-4


@dataclass
class TrainState:
    epoch: int
    best_metric: float
    epochs_no_improve: int = 0


def train_model(config: dict[any, any]):
    params = TrainParams(**config)
    init_random_seed(params.seed)
    mute_ray_data()
    state = TrainState(
        epoch=0,
        best_metric=(
            float("inf") if __MAJOR_METRIC_GOAL == "minimize" else -float("inf")
        ),
        epochs_no_improve=0,
    )

    train_dataset, valid_dataset =  load_dataset(params.train_dir_path).limit(512), load_dataset(params.valid_dir_path)
    model = AutoModelForCausalLM.from_pretrained(
        params.model_dir_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
      trust_remote_code=True,
    )
    model = ray.train.torch.prepare_model(model)
    tokenizer = AutoTokenizer.from_pretrained(params.model_dir_path, trust_remote_code=True)
    optimizer = AdamW(
        model.parameters(),
        lr=params.lr,
        betas=(params.optimizer_beta1, params.optimizer_beta2),
        weight_decay=params.optimizer_weight_decay,
    )
    scheduler = CosineAnnealingWithPrewarmRestarts(
        optimizer,
        T_0=params.scheduler_t,
        T_warmup=params.scheduler_t_warmup,
        T_mult=params.scheduler_t_mult,
        eta_min=params.scheduler_min_lr,
        eta_warmup_factor=params.schduler_warmup_lr_factor,
    )
    scaler = torch.amp.GradScaler()

    evaluator = Evaluator.options(num_gpus=0.5).remote(
        model_path=params.model_dir_path,
        seed=42,
        sampling_params=SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=1024,
            min_tokens=4,
            include_stop_str_in_output=True,
            stop="</answer>",
            logprobs=10,
        ),
        dtype=torch.bfloat16,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.95,
    )

    init_wandb(
        config={
            "batch_size": params.batch_size,
            "learning_rate": params.lr,
            "accumulate_steps": params.accumulate_steps,
            "max_grad": params.max_grad,
        }
    )
    wandb.watch(model, log="all", log_freq=200)

    total_params, trainable_params = get_model_size(model)
    logger.info(
        f"Starting training with parameters: total_params={total_params}, trainable_params={trainable_params}"
    )

    # ========= 训练循环（含早停与最优模型保存）=========
    for epoch in range(1, params.num_epochs + 1):
        train_metrics = train_one_epoch(
            epoch,
            model,
            tokenizer,
            train_dataset,
            optimizer,
            scheduler,
            scaler,
            evaluator,
            params,
        )
        
        val_metrics = validate(
            epoch,
            model,
            evaluator,
            valid_dataset,
            params,
            step="full",
        ) 

        # 选择指标做早停与保存
        current_metric = val_metrics[__MAJOR_METRIC_NAME]
        improved = (-1 if __MAJOR_METRIC_GOAL == "minimize" else 1) * (
            current_metric - state.best_metric
        ) > params.min_delta * abs(current_metric)

        state.epoch = epoch
        if improved:
            state.best_metric = current_metric
            state.epochs_no_improve = 0

            save_checkpoint_only_best(
                save_dir=Path(params.checkpoint_path),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                params=params,
                state=state,
            )

            wandb.summary["best_epoch"] = epoch
            wandb.summary["best_metric"] = state.best_metric
        else:
            state.epochs_no_improve += 1

        if state.epochs_no_improve >= params.patience:
            logger.info(
                f"Early stopping at epoch {epoch}: no improvement in {params.patience} epochs."
            )
            break

    logger.info(f"Training finished. Best {__MAJOR_METRIC_NAME}: {state.best_metric:.6f}")

    wandb.finish()


# ========== 训练与验证 ==========
def train_one_epoch(
    epoch: int,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    dataset: ray.data.Dataset,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    evaluator: Evaluator,
    params: TrainParams,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    running_entropy = 0.0

    total = (dataset.count() + params.batch_size - 1) // params.batch_size
    
    pbar = tqdm(
        dataset.iter_batches(batch_size=params.batch_size),
        total=total,
        desc=f"Train | Epoch {epoch}",
        leave=False,
    )
    for i, batch in enumerate(pbar):
        prompt_strs = [R1_ZERO_PROMPT.format(question=prob) for prob in batch["problem"]]
        output_strs = [R1_ZERO_OUTPUT.format(solution=sol, answer=ans) for sol, ans in zip(batch["solution"], batch["answer"])]
        input_tensors = tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)

        inputs = input_tensors["input_ids"].to("cuda", non_blocking=True)
        labels = input_tensors["labels"].to("cuda", non_blocking=True)
        response_mask = input_tensors["response_mask"].to("cuda", non_blocking=True)

        with torch.amp.autocast(device_type="cuda"):
            outputs = model(inputs)
            logits = outputs.logits  # (B, seq_len, vocab_size)
            probs = torch.softmax(logits, dim=-1)  # (B, seq_len, vocab_size)
            log_probs = probs.log()
            label_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # (B, seq_len)
            token_entropy = -torch.sum(probs * log_probs, dim=-1) # (B, seq_len)
            loss = masked_normalize(label_log_probs, response_mask, normalize_constant=label_log_probs.size(0) * params.accumulate_steps)
            per_token_entropy = masked_normalize(token_entropy, response_mask, normalize_constant=response_mask.sum())

        scaler.scale(loss).backward()
        if (i + 1) % params.accumulate_steps == 0:
            for p in model.parameters():
                if p.grad is not None:
                    torch.nn.utils.clip_grad_value_(p, params.max_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        scheduler.step((epoch - 1) + (i + 1) / total)

        running_loss += loss.item() * params.accumulate_steps
        running_entropy += per_token_entropy.item()

        pbar.set_postfix(
            loss=running_loss / (i + 1), entropy=running_entropy / (i + 1)
        )

        if (i + 1) % params.accumulate_steps == 0 and wandb.run is not None:
            wandb.log(
                {
                    "train_step": (epoch - 1) * total + i // params.accumulate_steps + 1,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/loss": running_loss / (i + 1),
                    "train/entropy": running_entropy / (i + 1),
                }
            )

        if (i + 1) // params.accumulate_steps in params.valid_steps:
            validate(
                epoch,
                model,
                evaluator,
                dataset,
                params,
                step=(i + 1) // params.accumulate_steps,
                async_no_return=True,
            )

    return {
        "train/loss": running_loss / (i + 1),
        "train/entropy": running_entropy / (i + 1),
    }


@torch.no_grad()
def validate(
    epoch: int,
    model: nn.Module,
    evaluator: Evaluator,
    dataset: ray.data.Dataset,
    params: TrainParams,
    step: any,
    async_no_return: bool = False,
) -> dict[str, float] | None:
    evaluator.load_new_policy_weights.remote(model.state_dict())
    if async_no_return:
        evaluator.evaluate.remote(
            dataset,
            params.batch_size,
            f"{params.valid_result_path}/epoch_{epoch}_step_{step}",
        )
        return
    else:
        _, analysis = ray.get(
            evaluator.evaluate.remote(
                dataset,
                params.batch_size,
                f"{params.valid_result_path}/epoch_{epoch}_step_{step}",
            )
        )
        return {__MAJOR_METRIC_NAME: 10}


def save_checkpoint_only_best(
    save_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    params: TrainParams,
    state: TrainState,
):
    """
    只保留当前最优模型：保存新 best，删除旧 best。
    返回新的 best 路径。
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = (
        f"best_epoch{state.epoch:03d}_{__MAJOR_METRIC_NAME}-{state.best_metric:.4f}.pt"
    )
    ckpt_path = save_dir / ckpt_name
    save_checkpoint(
        ckpt_path, model, optimizer, scheduler, train_param=params, train_state=state
    )
    logger.info(f"Saved best checkpoint to {ckpt_path}")

    # 删除旧同一目录下其他.pt
    for old_ckpt in save_dir.glob("best_epoch*.pt"):
        if old_ckpt != ckpt_path:
            try:
                old_ckpt.unlink()
            except Exception as e:
                logger.warning(f"[WARN] Failed to delete old checkpoint: {old_ckpt} ({e})")

if __name__ == "__main__":
    params = TrainParams(
        model_dir_path= os.path.abspath("./models/qwen2.5-math-1.5b"),
        train_dir_path= os.path.abspath("./datasets/train/math_12k/train"),
        valid_dir_path= os.path.abspath("./datasets/eval/math"),
        valid_result_path= os.path.abspath("./artifacts/results/sft-valid"),
        checkpoint_path= os.path.abspath("./artifacts/checkpoints/sft_ckpt"),
    )
    trainer = ray.train.torch.TorchTrainer(
        train_model,
        train_loop_config=asdict(params),
        scaling_config=ray.train.ScalingConfig(num_workers=1, use_gpu=True, resources_per_worker={"GPU": 0.5})
    )
    trainer.fit()