import logging
import random
import time

import numpy as np
import ray
import torch
import wandb
from datasets import load_from_disk
from ray.data.context import DataContext
from vllm.model_executor import set_random_seed as vllm_set_random_seed

R1_ZERO_PROMPT = """
A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>
""".strip()

R1_ZERO_OUTPUT = """
{solution}</think> <answer>{answer}</answer>
""".strip()


def init_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    vllm_set_random_seed(seed)


def init_wandb(run_name, config: dict[str, any]):
    wandb.init(project="cs336-ass5", name=run_name, config=config)
    # Setup wandb metrics
    wandb.define_metric("train_step")  # the x‑axis for training
    wandb.define_metric("eval_step")  # the x‑axis for evaluation
    # everything that starts with train/ is tied to train_step
    wandb.define_metric("train/*", step_metric="train_step")
    # everything that starts with eval/ is tied to eval_step
    wandb.define_metric("eval/*", step_metric="eval_step")

    wandb.define_metric("train/loss", summary="min")
    wandb.define_metric("train/entropy", summary="min")
    wandb.define_metric("eval/reward", summary="max")
    wandb.define_metric("eval/format_reward", summary="max")


def mute_ray_data():
    logging.getLogger("ray.data").setLevel(logging.ERROR)
    DataContext.get_current().enable_progress_bars = False


def load_dataset(dataset_path: str) -> ray.data.Dataset:
    dataset = load_from_disk(dataset_path)
    ray_ds = ray.data.from_huggingface(dataset)
    columns = ["problem", "answer", "solution"] if "solution" in dataset.column_names else ["problem", "answer"]
    ray_ds = ray.data.range(ray_ds.count()).zip(ray_ds.select_columns(columns))
    return ray_ds
