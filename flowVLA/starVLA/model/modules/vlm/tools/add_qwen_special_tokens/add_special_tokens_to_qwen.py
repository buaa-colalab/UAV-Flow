import argparse
import json
import os
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor

def add_new_tokens(
    model,
    tokenizer,
    new_tokens: List[str],
    init_strategy: str = "avg",
    as_special: bool = True,
) -> Tuple[Dict[str, int], int, int, int]:
    """
    向模型与 tokenizer 中添加新的 tokens（若不存在）。
    init_strategy: avg / normal / zero
    返回:
      - mapping: 所有目标 tokens 的 token_id 映射
      - added_now: 本次实际新增到 tokenizer 的数量
      - action_token_start_idx: 新增 embedding 起始下标（按模型旧 embedding 大小计算）
      - action_token_end_idx: 新增 embedding 结束下标（若无新增，则为 start_idx-1）
    说明:
      - tokenizer.vocab_size 为基础词表大小（不含已添加的 special/added tokens）
      - len(tokenizer) 为总词表大小（含 added/special tokens）
      - 模型的旧 embedding 大小以 model.get_input_embeddings().weight.shape[0] 为准
    """
    # 1) 计算需要新增的 tokens（相对 tokenizer 现有 vocab）
    vocab = tokenizer.get_vocab()  # 含已添加 tokens
    to_add = [t for t in new_tokens if t not in vocab]

    # 2) 记录模型当前的 embedding 尺寸（基础大小）
    old_embed = model.get_input_embeddings()
    old_embed_size = old_embed.weight.shape[0]

    # 3) 如有需要，先把 tokens 加到 tokenizer
    added_now = 0
    if to_add:
        if as_special:
            added_now = tokenizer.add_special_tokens({"additional_special_tokens": to_add})
        else:
            added_now = tokenizer.add_tokens(to_add)

    # 4) 目标总大小（tokenizer 总大小，基础 + 所有已添加）
    target_size = len(tokenizer)

    # 5) 若 tokenizer 总大小大于模型 embedding 大小，则需要 resize 并初始化新增行
    action_token_start_idx = old_embed_size
    action_token_end_idx = old_embed_size - 1  # 默认“无新增”
    if target_size > old_embed_size:
        model.resize_token_embeddings(target_size) # 一定会加到最后么？
        new_embed = model.get_input_embeddings()
        with torch.no_grad():
            if init_strategy == "avg":
                ref_vec = old_embed.weight.mean(dim=0, keepdim=True)
                for idx in range(old_embed_size, target_size):
                    new_embed.weight[idx].copy_(ref_vec[0])
            elif init_strategy == "zero":
                for idx in range(old_embed_size, target_size):
                    new_embed.weight[idx].zero_()
            elif init_strategy == "normal":
                for idx in range(old_embed_size, target_size):
                    nn.init.normal_(new_embed.weight[idx], mean=0.0, std=0.02)
            else:
                raise ValueError(f"未知 init_strategy: {init_strategy}")

        action_token_end_idx = target_size - 1

    # 6) 构造映射（返回请求关心的 tokens 的 id）
    mapping = {t: tokenizer.convert_tokens_to_ids(t) for t in new_tokens}
    return mapping, added_now, action_token_start_idx, action_token_end_idx

def save_bundle(model, tokenizer, mapping: Dict[str, int], save_dir: str, processor_src: str | None = None, padding_side: str | None = None):
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "added_token_id_map.json"), "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"[OK] 已保存到: {save_dir}")

    # 额外保存 AutoProcessor（生成 preprocessor_config.json），以便 AutoProcessor.from_pretrained(...) 加载
    try:
        src = processor_src or save_dir
        processor = AutoProcessor.from_pretrained(src, trust_remote_code=True)
        # 同步 processor.tokenizer 
        processor.tokenizer = tokenizer
        processor.save_pretrained(save_dir)
        print(f"[OK] AutoProcessor 已保存到: {save_dir}")
    except Exception as e:
        print(f"[WARN] 保存 AutoProcessor 失败: {e}")

def reload_and_check(save_dir: str, tokens: List[str]) -> bool:
    tok = AutoTokenizer.from_pretrained(save_dir, trust_remote_code=True)
    vocab = tok.get_vocab()
    missing = [t for t in tokens if t not in vocab]
    if missing:
        print(f"[WARN] 重新加载后仍缺失: {missing}")
        return False
    print("[OK] 重新加载检查通过，所有 token 均存在。")
    return True

def parse_tokens(args) -> List[str]:
    tokens: List[str] = []
    if args.tokens:
        tokens.extend([t.strip() for t in args.tokens.split(",") if t.strip()])
    if args.tokens_file:
        with open(args.tokens_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    tokens.append(line)
    # 去重保持顺序
    seen = set()
    ordered = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered

def main():
    parser = argparse.ArgumentParser(
        description="为 Qwen2.5-VL 模型添加特殊 tokens 并保存到本地。"
    )
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct", help="HF Hub 模型或本地路径")
    parser.add_argument("--save-dir", required=True, help="保存目录")
    parser.add_argument("--tokens", default="", help="逗号分隔 tokens，例如: <loc_x>,<loc_y>")
    parser.add_argument("--tokens-file", help="包含待添加 token 的文本文件（每行一个）")
    parser.add_argument("--init-strategy", default="avg", choices=["avg", "normal", "zero"], help="新增 embedding 初始化策略")
    parser.add_argument("--as-special", action="store_true", help="是否作为 special tokens 添加")
    parser.add_argument("--no-as-special", dest="as_special", action="store_false")
    parser.set_defaults(as_special=True)
    parser.add_argument("--padding-side", default="left", choices=["left", "right"])
    parser.add_argument("--device", default="cuda", help="cuda / cpu / mps / auto")
    args = parser.parse_args()

    tokens = parse_tokens(args)
    if not tokens:
        print("未提供任何 token，可使用 --tokens 或 --tokens-file")
        return

    print(f"[INFO] 待处理 tokens: {tokens}")

    print(f"[INFO] 加载模型: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.padding_side = args.padding_side
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map="auto" if args.device == "auto" else None,
        trust_remote_code=True,
    )

    # 额外打印三种大小，便于诊断
    base_tok_size = tokenizer.vocab_size                  # 基础词表大小
    total_tok_size = len(tokenizer)                       # 总词表大小
    model_embed_size = model.get_input_embeddings().weight.shape[0]  # 模型当前 embedding 大小
    print(f"[DEBUG] tokenizer.vocab_size(base) = {base_tok_size}")
    print(f"[DEBUG] len(tokenizer)(total)     = {total_tok_size}")
    print(f"[DEBUG] model.embed_size(before)  = {model_embed_size}")
    print(f"[DEBUG] added_in_tokenizer        = {total_tok_size - base_tok_size}")

    mapping, added, action_token_start_idx, action_token_end_idx = add_new_tokens(
        model=model,
        tokenizer=tokenizer,
        new_tokens=tokens,
        init_strategy=args.init_strategy,
        as_special=args.as_special,
    )
    new_model_embed_size = model.get_input_embeddings().weight.shape[0]
    print(f"[INFO] 本次新增到 tokenizer 的数量: {added}")
    # print(f"[INFO] Token 映射: {mapping}")
    print(f"[INFO] Action token idx 范围: [{action_token_start_idx}, {action_token_end_idx}]")
    print(f"[DEBUG] model.embed_size(after)   = {new_model_embed_size}")

    save_bundle(model, tokenizer, mapping, args.save_dir, processor_src=args.model_id, padding_side=args.padding_side)

    # 重新验证
    reload_and_check(args.save_dir, tokens)


def start_debugpy_once():
    """start debugpy once"""
    import debugpy
    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True

if __name__ == "__main__":
    start_debugpy_once()
    main()
