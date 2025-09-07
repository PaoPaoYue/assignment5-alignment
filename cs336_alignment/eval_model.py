import json
from datasets import load_from_disk
from vllm import LLM, SamplingParams
from tqdm import tqdm

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.common import R1_ZERO_PROMPT as PROMPT_TEMPLATE

def evaluate_model(model_path: str, save_path: str, dataset_path: str = "./datasets/eval/math"):
    """
    使用 vLLM 推理本地 Qwen2.5-Math 模型，对数学数据集进行评测，并保存结果为 JSON。

    Args:
        model_path (str): 本地模型路径
        save_path (str): 结果保存路径（JSON 文件）
        dataset_path (str): 数据集路径（默认 ./datasets/eval/math）
    """
    # 1. 加载数据集
    dataset = load_from_disk(dataset_path)

    # 2. 初始化 vLLM 模型
    llm = LLM(
        model=model_path,
        # trust_remote_code=True,
        # tensor_parallel_size=1,
        # gpu_memory_utilization=0.95,
        # dtype="half",
        # max_model_len=4096
    )

    sampling_params = SamplingParams(
        temperature=1.0,  # 确定性输出
        top_p=1.0,
        max_tokens=1024,
        include_stop_str_in_output=True,
        stop="</answer>"
    )


    # 3. 推理 + 校验
    results = []
    for i, item in enumerate(tqdm(dataset, desc="Evaluating", unit="problem")):
        problem = item["problem"]
        truth = item["answer"]

        # 应用模板
        prompt = PROMPT_TEMPLATE.format(question=problem)

        # 调用模型推理
        outputs = llm.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text.strip()

        # 校验
        reward_result = r1_zero_reward_fn(response, truth)

        # 保存结果
        results.append({
            "id": i,
            "prompt": prompt,
            "truth": truth,
            "response": response,
            "reward_result": reward_result
        })

    # 5. 保存到 JSON
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 推理与评测完成，结果已保存到 {save_path}")


# ===== 调用示例 =====
if __name__ == "__main__":
    evaluate_model(
        model_path="models/qwen2.5-math-1.5b",
        save_path="results/math_eval_results.json"
    )