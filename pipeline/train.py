#!/usr/bin/env python3
"""
阶段 1/4 —— 从零训练一个小型 LLaMA 架构语言模型

流程:
  1. 用 BPE 在本地语料上训练分词器 (不依赖任何预训练权重)
  2. 将语料切分为固定长度的训练样本
  3. 实例化 transformers 的 LlamaForCausalLM 并用 Trainer 训练
  4. 保存为 HuggingFace 格式 -> models/tinyllm-hf
     (该格式可直接被 llama.cpp 的 convert_hf_to_gguf.py 转换)

用法:
  python pipeline/train.py [--epochs 30] [--batch-size 8]
"""

import argparse
import math
import os
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "pipeline" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_tokenizer(cfg: dict):
    """在本地语料上训练一个 BPE 分词器。"""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast

    mcfg = cfg["model"]
    paths = cfg["paths"]
    corpus = ROOT / paths["corpus"]
    tokenizer_dir = ROOT / paths["tokenizer_dir"]
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    print(f"[tokenizer] 在 {corpus} 上训练 BPE 分词器, 词表 {mcfg['vocab_size']} ...")
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=mcfg["vocab_size"],
        special_tokens=["<|unk|>", "<|bos|>", "<|eos|>", "<|pad|>"],
        show_progress=True,
    )
    tok.train([str(corpus)], trainer)

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<|unk|>",
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        pad_token="<|pad|>",
        model_max_length=mcfg["max_position_embeddings"],
    )
    hf_tok.save_pretrained(str(tokenizer_dir))
    print(f"[tokenizer] 已保存到 {tokenizer_dir}, 实际词表大小 {hf_tok.vocab_size}")
    return hf_tok


def build_dataset(cfg: dict, tokenizer, block_size: int):
    """将语料编码并滑窗切分为等长样本。"""
    from torch.utils.data import Dataset

    paths = cfg["paths"]
    text = (ROOT / paths["corpus"]).read_text(encoding="utf-8")
    ids = tokenizer.encode(text)
    print(f"[data] 语料共 {len(text)} 字符, 编码为 {len(ids)} 个 token")

    step = block_size  # 不重叠滑窗 (语料少时重叠能增加样本量)
    chunks = [ids[i:i + block_size + 1] for i in range(0, len(ids) - 1, step)]
    chunks = [c for c in chunks if len(c) == block_size + 1]

    class LMDataset(Dataset):
        def __len__(self):
            return len(chunks)

        def __getitem__(self, idx):
            chunk = chunks[idx]
            return {"input_ids": chunk[:-1], "labels": chunk[1:]}

    print(f"[data] 切分为 {len(chunks)} 个样本, 每个长度 {block_size}")
    return LMDataset()


def main():
    parser = argparse.ArgumentParser(description="从零训练小型 LLaMA 模型")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置中的训练轮数")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true", help="忽略已有输出目录, 重新训练")
    args = parser.parse_args()

    cfg = load_config()
    tcfg = cfg["train"]
    mcfg = cfg["model"]
    output_dir = ROOT / cfg["paths"]["output_dir"]

    import torch
    from transformers import (
        LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments,
        default_data_collator,
    )

    random.seed(tcfg["seed"])
    torch.manual_seed(tcfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] 训练设备: {device}")

    tokenizer = train_tokenizer(cfg)
    dataset = build_dataset(cfg, tokenizer, tcfg["block_size"])

    model_cfg = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=mcfg["hidden_size"],
        intermediate_size=mcfg["intermediate_size"],
        num_hidden_layers=mcfg["num_hidden_layers"],
        num_attention_heads=mcfg["num_attention_heads"],
        num_key_value_heads=mcfg["num_key_value_heads"],
        max_position_embeddings=mcfg["max_position_embeddings"],
        rms_norm_eps=mcfg["rms_norm_eps"],
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = LlamaForCausalLM(model_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] 模型参数量: {n_params / 1e6:.2f}M")

    if args.no_resume and output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    training_args = TrainingArguments(
        output_dir=str(ROOT / "out" / "ckpt"),
        num_train_epochs=args.epochs or tcfg["epochs"],
        per_device_train_batch_size=args.batch_size or tcfg["batch_size"],
        learning_rate=tcfg["learning_rate"],
        warmup_steps=tcfg["warmup_steps"],
        logging_steps=tcfg["logging_steps"],
        save_strategy="no",
        report_to=[],
        seed=tcfg["seed"],
        use_cpu=(device == "cpu"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
    )
    trainer.train()

    final_loss = trainer.state.log_history[-1].get("train_loss")
    ppl = math.exp(final_loss) if final_loss and final_loss < 20 else float("inf")
    print(f"[train] 训练完成, 最终 loss={final_loss:.4f}, perplexity={ppl:.2f}")

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[train] 模型已保存到 {output_dir} (HF 格式, 可直接转 GGUF)")


if __name__ == "__main__":
    main()
