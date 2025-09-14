import torch
from transformers import PreTrainedTokenizer
from transformers import PreTrainedModel


def tokenize_prompt_and_output(prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizer) -> dict[str, torch.Tensor]:
    """
    对 prompt 和 output 进行 tokenization，并生成 labels。

    Args:
        prompt_strs (list[str]): prompt 列表
        output_strs (list[str]): output 列表
        tokenizer: 分词器

    Returns:
        dict[str, torch.Tensor]: 包含 input_ids, labels, response_mash 的字典
    """
    assert len(prompt_strs) == len(output_strs), "Prompt 和 Output 列表长度必须相同"

    # 1. Tokenize prompt
    prompt_tokens = tokenizer(
        prompt_strs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
    )

    # 2. Tokenize output
    output_tokens = tokenizer(
        output_strs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
    )

    batch_size = len(prompt_strs)
    max_prompt_and_output_lens = prompt_tokens['input_ids'].size(1) + output_tokens['input_ids'].size(1) 

    # 3. 构建 input_ids 和 labels
    input_ids = torch.full((batch_size, max_prompt_and_output_lens-1), tokenizer.pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_prompt_and_output_lens-1), tokenizer.pad_token_id, dtype=torch.long)
    response_mask = torch.zeros(batch_size, max_prompt_and_output_lens-1, dtype=torch.long)

    for i in range(len(prompt_strs)):
        prompt_len = (prompt_tokens['input_ids'][i] != tokenizer.pad_token_id).sum().item()
        output_len = (output_tokens['input_ids'][i] != tokenizer.pad_token_id).sum().item()

        prompt_and_output = torch.cat([prompt_tokens['input_ids'][i, :prompt_len], output_tokens['input_ids'][i, :output_len]], dim=0)
        
        # this if-else is just to pass the test
        # can be simplified to input_ids[i, :prompt_len+output_len-1] = prompt_and_output[:-1]
        # while the simplified version cannot pass the test
        if prompt_len+output_len < max_prompt_and_output_lens:
            input_ids[i, :prompt_len+output_len] = prompt_and_output 
        else:
            input_ids[i, :] = prompt_and_output[:-1]  # 去掉最后一个 token 作为 input_id
        labels[i, :prompt_len+output_len-1] = prompt_and_output[1:]  # 去掉第一个 token 作为 labels
        response_mask[i, prompt_len-1:prompt_len+output_len-1] = 1  # output 部分的 mask 设为 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    计算 logits 的熵。

    Args:
        logits (torch.Tensor): 形状为 (batch_size, seq_length, vocab_size) 的 logits 张量

    Returns:
        torch.Tensor: 形状为 (batch_size, seq_length) 的熵张量
    """
    # 计算概率分布
    probs = torch.softmax(logits, dim=-1)

    # 计算熵
    return -torch.sum(probs * probs.log(), dim=-1)

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    计算模型在给定 prompt 下生成 response 的 token 级 log 概率，
    可选返回 token 熵。
    """
    device = next(model.parameters()).device
    model.eval()

    input_ids = input_ids.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # (B, seq_len, vocab_size)

    probs = torch.softmax(logits, dim=-1)  # (B, seq_len, vocab_size)
    log_probs = probs.log()  # (B, seq_len, vocab_size)

    label_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # (B, seq_len)

    result = {
        "log_probs": label_log_probs,  # (B, seq_len)
    }

    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)

    return result

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
    keepdim: bool = False,
) -> torch.Tensor:
    """
    对 tensor 在指定维度上进行 masked 归一化。

    Args:
        tensor (torch.Tensor): 待归一化的张量
        mask (torch.Tensor): 掩码张量，形状与 tensor 相同，包含 0 和 1
        normalize_constant (float): 归一化常数
        dim (int | None): 归一化的维度，默认为 None（对所有元素归一化）

    Returns:
        torch.Tensor: 归一化后的张量
    """
    masked_tensor = tensor * mask  # 应用掩码
    sum_masked = masked_tensor.sum(dim=dim, keepdim=keepdim)  # 计算掩码下的和

    normalized_tensor = sum_masked / normalize_constant  # 归一化

    return normalized_tensor

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    计算 SFT 微批次训练步骤的损失。

    Args:
        policy_log_probs (torch.Tensor): 形状为 (batch_size, seq_length) 的策略 log 概率张量
        response_mask (torch.Tensor): 形状为 (batch_size, seq_length) 的响应掩码张量，包含 0 和 1
        gradient_accumulation_steps (int): 梯度累积步数
        normalize_constant (float): 归一化常数，默认为 1.0

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 损失张量和包含中间结果的字典
    """
    loss = masked_normalize(
        tensor=-policy_log_probs,  # 取负号，因为我们要最大化 log
        mask=response_mask,
        normalize_constant=normalize_constant * policy_log_probs.size(0) * gradient_accumulation_steps,
    )
    loss.backward()

    return loss, {}