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

    valid_steps: list = field(default_factory=lambda: [64, 128, 256, 512]) # step = count / batch_size, default counts=[128, 256, 512, 1024]

    seed: int = 42

    lr: float = 1e-3
    batch_size: int = 2
    accumulate_steps: int = 4
    max_grad: float = 0
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.999
    optimizer_weight_decay: float = 0.01
    scheduler_t: int = 1
    scheduler_t_warmup: float = 0
    scheduler_t_mult: int = 1
    scheduler_min_lr: float = 1e-4
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

    evaluator = Evaluator.options(num_gpus=0.3).remote(
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
        gpu_memory_utilization=0.5,
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
        
        val_metrics = validate(
            epoch,
            model,
            evaluator,
            valid_dataset,
            params,
            step="full",
        ) 
        logger.info(f"Epoch {epoch} validation: {val_metrics}")


    logger.info(f"Training finished. Best {__MAJOR_METRIC_NAME}: {state.best_metric:.6f}")

    wandb.finish()


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
    logger.info(f"Validating setp={step}")
    ray.get(evaluator.load_new_policy_weights.remote(model.state_dict()))
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