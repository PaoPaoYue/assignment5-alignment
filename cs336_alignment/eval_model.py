import json
import logging
from itertools import islice
import os
from typing import Iterator

import numpy as np
import ray
import wandb
import torch
from tqdm import tqdm
from vllm import LLM, RequestOutput, SamplingParams

from cs336_alignment.common import R1_ZERO_PROMPT as PROMPT_TEMPLATE, init_random_seed
from cs336_alignment.common import mute_ray_data, load_dataset
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

logger = logging.getLogger(__name__)

@ray.remote(num_gpus=1, max_concurrency=1)
class Evaluator:
    def __init__(
        self,
        model_path: str,
        seed: int,
        sampling_params: SamplingParams,
        **kwargs,
    ):
        init_random_seed(seed)
        mute_ray_data()

        # from unittest.mock import patch
        # world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
        # profiling_patch = patch(
        # "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        # return_value=None
        # )
        self.llm = LLM(
            model=model_path,
            **kwargs,
        )
        logger.info(
            f"Evaluator initialized on device {ray.get_gpu_ids()} with model {model_path}"
        )
        self.sampling_params = sampling_params
        self.eval_step = -1

        self.__RESULT_FILE_MIN_ROWS = 100

    def evaluate(self, ds: ray.data.Dataset, batch_size: int = 4, result_path: str=None) -> dict[str, any]:
        logger.info("Evaluator starting evaluation##################")
        self.eval_step += 1
        result_buffer = []
        for batch in tqdm(
            ds.iter_batches(batch_size=batch_size),
            total=(ds.count() + batch_size - 1) // batch_size,
            desc=f"Eval | Step {self.eval_step}", leave=False
        ):
            prompts = [
                PROMPT_TEMPLATE.format(question=prob) for prob in batch["problem"]
            ]
            outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
            result_buffer.extend(log_generations(batch, outputs, self.sampling_params.logprobs or 0))

        if result_buffer:
            result = ray.data.from_items(result_buffer)
            result_buffer.clear()

        analysis = analyse_result(result)
        if wandb.run is not None:
            wandb.log(
                {
                    "eval_step": self.eval_step,
                    **{f"eval/{k}": v for k, v in analysis.items()},
                }
            )
        if result_path is not None:
            os.makedirs(result_path, exist_ok=True)
            result.write_csv(result_path, min_rows_per_file=self.__RESULT_FILE_MIN_ROWS)
            json.dump(analysis, open(f"{result_path}/analysis.json", "w"))
            logger.info(
                f"Evaluator finished eval step {self.eval_step}, results saved to {result_path}"
            )
        return result, analysis

    def load_new_policy_weights(self, state_dict: dict[str, any]):
        logger.info("Evaluator loading new policy weights#############")
        print("Evaluator loading new policy weights#############")
        # llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        # llm_model.load_weights(state_dict.items())


def log_generations(
    batch: dict[str, np.ndarray], outputs: list[RequestOutput], logprob_num: int
) -> Iterator[dict[str, any]]:
    for i, output in enumerate(outputs):
        for response in output.outputs:
            response_text = response.text.strip()
            rewards = r1_zero_reward_fn(response_text, batch["answer"][i])

            tokens = len(response.token_ids)

            if hasattr(response, "logprobs") and response.logprobs:
                # 把每个 token 的 top-k logprobs 转成二维数组 (num_tokens, k)
                logprob_lists = [
                    list(
                        islice((item.logprob for item in lp_dict.values()), logprob_num)
                    )
                    for lp_dict in response.logprobs
                    if lp_dict
                ]
                if logprob_lists:
                    logprob_arr = np.array(
                        logprob_lists, dtype=np.float32
                    )  # shape: (T, K)

                    probs = np.exp(logprob_arr)
                    probs /= probs.sum(axis=1, keepdims=True)
                    entropies = -np.sum(probs * np.log(probs + 1e-12), axis=1)
                    avg_entropy = float(entropies.mean())
                else:
                    avg_entropy = None
            else:
                avg_entropy = None

            yield {
                "id": int(batch["id"][i]),
                "problem": batch["problem"][i],
                "answer": batch["answer"][i],
                "response": response_text,
                "reward": rewards["reward"],
                "format_reward": rewards["format_reward"],
                "answer_reward": rewards["answer_reward"],
                "tokens": tokens,
                "entropy": avg_entropy,
            }


def analyse_result(ds: ray.data.Dataset) -> dict[str, any]:

    total_count = ds.count()

    # 条件过滤
    correct_answer = ds.filter(lambda r: r["answer_reward"] == 1)
    correct_format = ds.filter(lambda r: r["format_reward"] == 1)
    wrong_answer = ds.filter(lambda r: r["answer_reward"] != 1)
    wrong_format = ds.filter(lambda r: r["format_reward"] != 1)

    # 计算数量
    correct_answer_count = correct_answer.count()
    correct_format_count = correct_format.count()

    # 计算平均 token 长度
    avg_len = ds.mean("tokens") or 0
    correct_answer_avg_len = correct_answer.mean("tokens") or 0
    correct_format_avg_len = correct_format.mean("tokens") or 0
    wrong_answer_avg_len = wrong_answer.mean("tokens") or 0
    wrong_format_avg_len = wrong_format.mean("tokens") or 0

    # 计算平均 entropy
    avg_entropy = ds.mean("entropy") or 0
    correct_answer_avg_entropy = correct_answer.mean("entropy") or 0
    correct_format_avg_entropy = correct_format.mean("entropy") or 0
    wrong_answer_avg_entropy = wrong_answer.mean("entropy") or 0
    wrong_format_avg_entropy = wrong_format.mean("entropy") or 0

    return {
        "total_count": total_count,
        "correct_answer_count": correct_answer_count,
        "correct_format_count": correct_format_count,
        "correct_answer_rate": (
            correct_answer_count / total_count if total_count > 0 else 0
        ),
        "correct_format_rate": (
            correct_format_count / total_count if total_count > 0 else 0
        ),
        "avg_len": avg_len,
        "correct_answer_avg_len": correct_answer_avg_len,
        "correct_format_avg_len": correct_format_avg_len,
        "wrong_answer_avg_len": wrong_answer_avg_len,
        "wrong_format_avg_len": wrong_format_avg_len,
        "avg_entropy": avg_entropy,
        "correct_answer_avg_entropy": correct_answer_avg_entropy,
        "correct_format_avg_entropy": correct_format_avg_entropy,
        "wrong_answer_avg_entropy": wrong_answer_avg_entropy,
        "wrong_format_avg_entropy": wrong_format_avg_entropy,
    }


# ===== 调用示例 =====
if __name__ == "__main__":
    ray.init()
    evaluator = Evaluator.remote(
        model_path="./models/qwen2.5-math-1.5b",
        seed=42,
        sampling_params=SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=1024,
            min_tokens=4,
            include_stop_str_in_output=True,
            stop="</answer>",
            logprobs=10,  # 获取 top-10 logprobs
        ),
        # dtype="half",
        dtype=torch.bfloat16,
        # enable_prefix_caching=True,
        # gpu_memory_utilization=0.95
    )
    ds = load_dataset("./datasets/eval/math")
    _, analysis = ray.get(evaluator.evaluate.remote(ds, batch_size=16, result_path="./artifacts/results/eval"))
    logger.info(analysis)
