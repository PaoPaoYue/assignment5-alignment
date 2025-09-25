from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import tempfile

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


@dataclass
class TrainParams:
    run_name: str
    evaluator: Evaluator

    model_dir_path: str
    train_dir_path: str
    valid_dir_path: str
    valid_result_path: str

    train_cases: int = 512

    valid_steps: list = field(
        default_factory=lambda: [64, 128, 256, 512]
    )  # step = count / batch_size, default counts=[128, 256, 512, 1024]

    seed: int = 42

    lr: float = 5e-5
    batch_size: int = 2
    val_batch_size: int = 8
    accumulate_steps: int = 8
    max_grad: float = 1
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.999
    optimizer_weight_decay: float = 0.01
    scheduler_t: int = 1
    scheduler_t_warmup: float = 0
    scheduler_t_mult: int = 1
    scheduler_min_lr: float = 0
    schduler_warmup_lr_factor: float = 0

    num_epochs: int = 10
    val_epoch_freq: int = 1
    val_epoch_min: int = 8

def train_model(config: dict[any, any]):
    params = TrainParams(**config)
    init_random_seed(params.seed)
    mute_ray_data()

    train_dataset, valid_dataset = load_dataset(params.train_dir_path).limit(
        params.train_cases
    ), load_dataset(params.valid_dir_path)
    model = AutoModelForCausalLM.from_pretrained(
        params.model_dir_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
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
    scheduler = CosineAnnealingWithPrewarmRestarts(
        optimizer,
        T_0=params.scheduler_t,
        T_warmup=params.scheduler_t_warmup,
        T_mult=params.scheduler_t_mult,
        eta_min=params.scheduler_min_lr,
        eta_warmup_factor=params.schduler_warmup_lr_factor,
    )

    init_wandb(
        run_name=params.run_name,
        config={
            "batch_size": params.batch_size,
            "learning_rate": params.lr,
            "accumulate_steps": params.accumulate_steps,
            "max_grad": params.max_grad,
        },
    )

    # ========= 训练循环（最优模型保存）=========
    for epoch in range(1, params.num_epochs + 1):
        model_state_dict = train_one_epoch(
            epoch,
            model,
            tokenizer,
            train_dataset,
            valid_dataset,
            optimizer,
            scheduler,
            params,
        )

        if (epoch % params.val_epoch_freq != 0 and epoch != params.num_epochs) or epoch < params.val_epoch_min:
            continue

        torch.cuda.empty_cache()

        val_metrics = validate(
            epoch,
            model_state_dict,
            valid_dataset,
            params,
            step=epoch*((train_dataset.count() + params.batch_size - 1) // params.batch_size),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(model.state_dict(), os.path.join(tmpdir, "checkpoint.pt"))
            checkpoint = ray.train.Checkpoint.from_directory(tmpdir)
            ray.train.report(metrics=val_metrics, checkpoint=checkpoint)

    ray.get(params.evaluator.close.remote())
    wandb.finish()


# ========== 训练与验证 ==========
def train_one_epoch(
    epoch: int,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    dataset: ray.data.Dataset,
    valid_dataset: ray.data.Dataset,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
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
        prompt_strs = [
            R1_ZERO_PROMPT.format(question=prob) for prob in batch["problem"]
        ]
        output_strs = [
            R1_ZERO_OUTPUT.format(solution=sol, answer=ans)
            for sol, ans in zip(batch["solution"], batch["answer"])
        ]
        input_tensors = tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)

        inputs = input_tensors["input_ids"].to("cuda", non_blocking=True)
        labels = input_tensors["labels"].to("cuda", non_blocking=True)
        response_mask = input_tensors["response_mask"].to("cuda", non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(inputs)
            logits = outputs.logits  # (B, seq_len, vocab_size)
            probs = torch.softmax(logits, dim=-1)  # (B, seq_len, vocab_size)
            log_probs = probs.log()
            label_log_probs = torch.gather(
                log_probs, dim=-1, index=labels.unsqueeze(-1)
            ).squeeze(
                -1
            )  # (B, seq_len)
            token_entropy = -torch.sum(probs * log_probs, dim=-1)  # (B, seq_len)
            loss = masked_normalize(
                -label_log_probs,
                response_mask,
                normalize_constant=response_mask.sum() * params.accumulate_steps,
            )
            per_token_entropy = masked_normalize(
                token_entropy, response_mask, normalize_constant=response_mask.sum()
            )
        loss.backward()
        if (i + 1) % params.accumulate_steps == 0:
            for p in model.parameters():
                if p.grad is not None:
                    torch.nn.utils.clip_grad_value_(p, params.max_grad)
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step((epoch - 1) + i / total)

        running_loss += loss.item() * params.accumulate_steps
        running_entropy += per_token_entropy.item()

        pbar.set_postfix(loss=running_loss / (i + 1), entropy=running_entropy / (i + 1))

        if (i + 1) % params.accumulate_steps == 0:
            wandb.log(
                {
                    "train_step": (epoch - 1) * total
                    + i // params.accumulate_steps
                    + 1,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/loss": loss.item() * params.accumulate_steps,
                    "train/entropy": per_token_entropy.item(),
                }
            )

        # if (i + 1) in params.valid_steps:
        #     torch.cuda.empty_cache()
        #     validate(
        #         epoch,
        #         model,
        #         valid_dataset,
        #         params,
        #         step=total * epoch + (i + 1),
        #         async_no_return=True,
        #     )

    return model.state_dict()


@torch.no_grad()
def validate(
    epoch: int,
    state_dict: dict[str, torch.Tensor],
    dataset: ray.data.Dataset,
    params: TrainParams,
    step: int,
    async_no_return: bool = False,
) -> dict[str, float] | None:
    evaluator = params.evaluator
    ray.get(evaluator.load_new_policy_weights.remote(state_dict))
    if async_no_return:
        evaluator.evaluate.remote(
            "validation",
            step,
            dataset,
            params.val_batch_size,
            result_path=f"{params.valid_result_path}/epoch_{epoch}_step_{step}",
        )
        return
    else:
        _, analysis = ray.get(
            evaluator.evaluate.remote(
                "validation",
                step,
                dataset,
                params.val_batch_size,
                result_path=f"{params.valid_result_path}/epoch_{epoch}_step_{step}",
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
        train_dir_path=os.path.abspath("./datasets/train/math_12k/train"),
        valid_dir_path=os.path.abspath("./datasets/eval/math"),
        valid_result_path=os.path.abspath("./artifacts/results/sft-valid"),
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
            )
        ),
    )
    result = trainer.fit()
    best_ckpt, best_metrics = result.best_checkpoints[0]
    best_ckpt.to_directory(os.path.abspath("./artifacts/checkpoints/sft_ckpt"))
    logger.info(
        f"Train finished with metrics {best_metrics}, saved checkpoint to {os.path.abspath("./artifacts/checkpoints/sft_ckpt")}."
    )
