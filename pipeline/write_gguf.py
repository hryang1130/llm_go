#!/usr/bin/env python3
"""
自定义 GGUF 导出器 —— 把训练产物 (HF/llama 架构) 直接写成 GGUF

为什么不用 llama.cpp 自带的 convert_hf_to_gguf.py?
  它通过「编码固定文本再比对哈希」来识别 BPE pre-tokenizer, 只收录了
  HF 上知名模型的词表; 从零自训的词表永远无法命中, 直接报
  "BPE pre-tokenizer was not recognized"。
  本脚本用 gguf-py 手工写出 llama 架构 GGUF, 词表标记为 gpt2 预分词
  (与 train.py 中使用的 GPT-2 规则一致), 完全绕开该检查。

依赖: torch, safetensors (随 transformers), gguf (pip install gguf)
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent.parent

# HF 权重名 -> GGUF 张量名 (llama 架构)
TENSOR_MAP = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}


def map_layer(i: int, name: str) -> str:
    m = {
        "input_layernorm.weight": f"blk.{i}.attn_norm.weight",
        "self_attn.q_proj.weight": f"blk.{i}.attn_q.weight",
        "self_attn.k_proj.weight": f"blk.{i}.attn_k.weight",
        "self_attn.v_proj.weight": f"blk.{i}.attn_v.weight",
        "self_attn.o_proj.weight": f"blk.{i}.attn_output.weight",
        "post_attention_layernorm.weight": f"blk.{i}.ffn_norm.weight",
        "mlp.gate_proj.weight": f"blk.{i}.ffn_gate.weight",
        "mlp.up_proj.weight": f"blk.{i}.ffn_up.weight",
        "mlp.down_proj.weight": f"blk.{i}.ffn_down.weight",
    }
    return m.get(name)


def main():
    from gguf import GGUFWriter

    hf_dir = ROOT / "models" / "tinyllm-hf"
    out_path = ROOT / "models" / "tinyllm-f16.gguf"
    cfg = json.loads((hf_dir / "config.json").read_text(encoding="utf-8"))
    tj = json.loads((hf_dir / "tokenizer.json").read_text(encoding="utf-8"))

    n_layers = cfg["num_hidden_layers"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    hidden = cfg["hidden_size"]
    head_dim = hidden // n_heads

    print(f"[gguf] 加载权重 {hf_dir}")
    sd = load_file(str(hf_dir / "model.safetensors"))

    w = GGUFWriter(str(out_path), "llama")
    w.add_name("tinyllm")
    w.add_description("Tiny LLaMA trained from scratch (llm_go workflow)")
    w.add_uint32("llama.block_count", n_layers)
    w.add_uint32("llama.context_length", cfg["max_position_embeddings"])
    w.add_uint32("llama.embedding_length", hidden)
    w.add_uint32("llama.feed_forward_length", cfg["intermediate_size"])
    w.add_uint32("llama.attention.head_count", n_heads)
    w.add_uint32("llama.attention.head_count_kv", n_kv)
    w.add_uint32("llama.rope.dimension_count", head_dim)
    w.add_float32("llama.rope.freq_base", cfg.get("rope_theta", 10000.0))
    w.add_float32("llama.attention.layer_norm_rms_epsilon", cfg["rms_norm_eps"])

    # ---------- 词表 ----------
    vocab = tj["model"]["vocab"]                      # token -> id
    added = tj.get("added_tokens", [])                # 特殊 token 定义
    tokens = [""] * len(vocab)
    for tok, idx in vocab.items():
        tokens[idx] = tok
    n_vocab = len(tokens)
    special_ids = {a["id"] for a in added if a.get("special")}

    w.add_tokenizer_model("gpt2")                     # tokenizer.ggml.model
    w.add_tokenizer_pre("gpt-2")                       # 与 train.py 预分词规则一致
    w.add_array("tokenizer.ggml.tokens", tokens)
    w.add_array("tokenizer.ggml.scores", [0.0] * n_vocab)
    w.add_array("tokenizer.ggml.token_type",
                [2 if i in special_ids else 1 for i in range(n_vocab)])
    # gpt2 型 BPE 必需: 合并规则列表 ["a b", ...] 或 [[a, b], ...]
    merges = [" ".join(m) if isinstance(m, list) else m for m in tj["model"]["merges"]]
    w.add_array("tokenizer.ggml.merges", merges)

    # special token id 映射
    tok2id = {t: i for i, t in enumerate(tokens)}
    for name, key in [("bos", "tokenizer.ggml.bos_token_id"),
                      ("eos", "tokenizer.ggml.eos_token_id"),
                      ("pad", "tokenizer.ggml.padding_token_id")]:
        tok = f"<|{name}|>"
        if tok in tok2id:
            w.add_uint32(key, tok2id[tok])
    w.add_bool("tokenizer.ggml.add_bos_token", True)
    w.add_bool("tokenizer.ggml.add_eos_token", False)

    # ---------- 张量 (F32 -> F16, Norm 向量保留 F32) ----------
    n_tensor = 0
    for hf_name, t in sd.items():
        gguf_name = TENSOR_MAP.get(hf_name)
        if gguf_name is None:
            for i in range(n_layers):
                gguf_name = map_layer(i, hf_name.replace(f"model.layers.{i}.", ""))
                if gguf_name:
                    break
        if gguf_name is None:
            sys.exit(f"[gguf] 无法映射张量: {hf_name}")

        arr = t.detach().to(torch.float32).numpy()
        if arr.ndim == 2:                             # 权重矩阵转 F16
            arr = arr.astype(np.float16)
        w.add_tensor(gguf_name, arr)
        n_tensor += 1
    print(f"[gguf] 写入 {n_tensor} 个张量")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"[gguf] 完成: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
