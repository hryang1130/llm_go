#!/usr/bin/env python3
"""
阶段 1.5/4 —— 指令微调 (LoRA SFT)

让「从零预训练」的模型学会听指令: 在 base 模型上挂 LoRA 适配器做监督微调,
训练完把适配器合并回权重, 产物仍是 HF 格式, 因此后续
write_gguf.py → quantize_sweep.py → eval.py → server 全链路直接复用。

数据格式 (JSONL, 一行一条):
  {"instruction": "什么是注意力机制？", "input": "", "output": "注意力机制……"}

用法:
  python pipeline/sft.py --check                      # 只看数据与配置
  python pipeline/sft.py                              # 默认读取 config sft.*
  python pipeline/sft.py --data data/sft_sample.jsonl --epochs 5 --lr 1e-3
  python pipeline/sft.py --steps 20                   # 只跑 20 步 (快速验证链路)

产物:
  models/tinyllm-hf-sft/      合并后的 HF 模型 (可直接导出 GGUF)
  out/lora-adapter/           仅适配器权重 (体积很小, 便于分享)
  out/sft.log                 训练日志
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, cfg_get, load_config, resolve_path, size_mb  # noqa: E402

PROMPT_TMPL = "### 指令：\n{instruction}\n### 回答：\n{output}"


def parse_args():
    p = argparse.ArgumentParser(description="LoRA 指令微调 (SFT)")
    p.add_argument("--data", default=None, help="指令数据 JSONL (默认 config sft.data)")
    p.add_argument("--base", default=None, help="base HF 模型目录 (默认 config paths.output_dir)")
    p.add_argument("--output", default=None, help="合并后模型输出目录 (默认 config sft.output_dir)")
    p.add_argument("--adapter-dir", default=None, help="适配器输出目录 (默认 out/lora-adapter)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--steps", type=int, default=None, help="限制训练步数 (快速验证链路)")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-length", type=int, default=None, help="单条样本最大 token 数")
    p.add_argument("--lora-r", type=int, default=None)
    p.add_argument("--no-merge", action="store_true", help="只保存适配器, 不合并进 base 权重")
    p.add_argument("--check", action="store_true", help="只做数据与配置自检")
    return p.parse_args()


def load_records(path: Path) -> list[dict]:
    """读 JSONL 指令数据, 兼容 {"instruction","input","output"} 与 {"prompt","response"}。"""
    records = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            sys.exit(f"[sft] 第 {i} 行不是合法 JSON: {exc}")
        instruction = obj.get("instruction") or obj.get("prompt") or ""
        extra_input = obj.get("input") or ""
        output = obj.get("output") or obj.get("response") or ""
        if extra_input:
            instruction = f"{instruction}\n{extra_input}"
        if not instruction or not output:
            sys.exit(f"[sft] 第 {i} 行缺少 instruction / output 字段")
        records.append({"instruction": instruction.strip(), "output": output.strip()})
    if not records:
        sys.exit(f"[sft] 数据为空: {path}")
    return records


def check_env(args, cfg) -> int:
    print("== SFT 环境自检 ==")
    data = resolve_path(args.data or cfg_get(cfg, "sft.data", "data/sft_sample.jsonl"))
    base = resolve_path(args.base or cfg_get(cfg, "paths.output_dir", "models/tinyllm-hf"))
    out = resolve_path(args.output or cfg_get(cfg, "sft.output_dir", "models/tinyllm-hf-sft"))
    print(f"  数据      : {'✓' if data.exists() else '✗'} {data}")
    if data.exists():
        recs = load_records(data)
        avg_in = sum(len(r["instruction"]) for r in recs) / len(recs)
        avg_out = sum(len(r["output"]) for r in recs) / len(recs)
        print(f"  样本数    : {len(recs)} 条 (平均指令 {avg_in:.0f} 字符 / 回答 {avg_out:.0f} 字符)")
    print(f"  base 模型 : {'✓' if (base / 'config.json').exists() else '✗ (先跑 train.py)'} {base}")
    print(f"  输出目录  : {out}")
    print(f"  LoRA      : r={cfg_get(cfg, 'sft.lora_r', 8)}, "
          f"alpha={cfg_get(cfg, 'sft.lora_alpha', 16)}, "
          f"dropout={cfg_get(cfg, 'sft.lora_dropout', 0.05)}")
    print(f"  训练      : epochs={cfg_get(cfg, 'sft.epochs', 3)}, "
          f"lr={cfg_get(cfg, 'sft.lr', 1e-3)}, batch={cfg_get(cfg, 'sft.batch_size', 4)}")
    ok = data.exists() and (base / "config.json").exists()
    print(f"== {'就绪' if ok else '未就绪'} ==")
    return 0 if ok else 1


def main():
    args = parse_args()
    cfg = load_config()

    if args.check:
        sys.exit(check_env(args, cfg))

    data_path = resolve_path(args.data or cfg_get(cfg, "sft.data", "data/sft_sample.jsonl"))
    base_dir = resolve_path(args.base or cfg_get(cfg, "paths.output_dir", "models/tinyllm-hf"))
    out_dir = resolve_path(args.output or cfg_get(cfg, "sft.output_dir", "models/tinyllm-hf-sft"))
    adapter_dir = resolve_path(args.adapter_dir or cfg_get(cfg, "sft.adapter_dir", "out/lora-adapter"))
    if not data_path.exists():
        sys.exit(f"[sft] 找不到数据 {data_path}")
    if not (base_dir / "config.json").exists():
        sys.exit(f"[sft] 找不到 base 模型 {base_dir}, 先运行 python pipeline/train.py")

    records = load_records(data_path)
    print(f"[sft] 载入 {len(records)} 条指令数据 ({data_path.name})")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)

    max_length = args.max_length or cfg_get(cfg, "sft.max_length", 256)
    tok = AutoTokenizer.from_pretrained(str(base_dir))
    model = AutoModelForCausalLM.from_pretrained(str(base_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    print(f"[sft] 设备 {device}, 单条最大长度 {max_length}, pad_id={pad_id}")

    # ---------- 构造样本: 指令部分不参与 loss ----------
    samples = []
    for r in records:
        prompt = PROMPT_TMPL.format(instruction=r["instruction"], output="")
        full = PROMPT_TMPL.format(instruction=r["instruction"], output=r["output"]) + \
            (tok.eos_token or "")
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tok(full, add_special_tokens=False)["input_ids"][:max_length]
        n_prompt = min(len(p_ids), len(f_ids))
        # 指令前缀 (含模板标记) 的 loss 屏蔽为 -100, 只学回答部分
        labels = [-100] * n_prompt + f_ids[n_prompt:]
        samples.append({"input_ids": f_ids, "labels": labels})

    class SFTDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, i):
            return samples[i]

    class PadCollator:
        """把同批样本右侧 padding 到批内最长长度。

        指令样本长短不一, 直接交给 default_data_collator 会因形状不一致报错;
        这里动态 padding 并同步生成 attention_mask, pad 位置的 loss 保持 -100。
        """

        def __call__(self, features):
            maxlen = max(len(f["input_ids"]) for f in features)
            input_ids, labels, attn = [], [], []
            for f in features:
                pad = maxlen - len(f["input_ids"])
                input_ids.append(f["input_ids"] + [pad_id] * pad)
                labels.append(f["labels"] + [-100] * pad)
                attn.append([1] * len(f["input_ids"]) + [0] * pad)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
            }

    lora = LoraConfig(
        r=args.lora_r or cfg_get(cfg, "sft.lora_r", 8),
        lora_alpha=cfg_get(cfg, "sft.lora_alpha", 16),
        lora_dropout=cfg_get(cfg, "sft.lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[sft] 可训练参数 {trainable / 1e6:.3f}M / 总参数 {total / 1e6:.3f}M "
          f"({trainable / total:.1%})")

    targs = TrainingArguments(
        output_dir=str(ROOT / "out" / "ckpt-sft"),
        num_train_epochs=args.epochs or cfg_get(cfg, "sft.epochs", 3),
        max_steps=args.steps or -1,
        per_device_train_batch_size=args.batch_size or cfg_get(cfg, "sft.batch_size", 4),
        learning_rate=args.lr or cfg_get(cfg, "sft.lr", 1e-3),
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=cfg_get(cfg, "train.seed", 42),
        use_cpu=(device == "cpu"),
    )
    trainer = Trainer(model=model, args=targs, train_dataset=SFTDataset(),
                      data_collator=PadCollator())
    trainer.train()
    # log_history 末项可能是汇总信息, 从后往前找第一条带 train_loss 的记录
    loss = next((h["train_loss"] for h in reversed(trainer.state.log_history)
                 if "train_loss" in h), None)
    print(f"[sft] 训练完成, 最终 loss={loss:.4f}" if loss is not None else "[sft] 训练完成")

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    print(f"[sft] 适配器已保存: {adapter_dir}")

    if args.no_merge:
        print("[sft] --no-merge: 跳过权重合并")
    else:
        merged = model.merge_and_unload()
        out_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(out_dir))
        tok.save_pretrained(str(out_dir))
        print(f"[sft] 合并后模型已保存: {out_dir}")
        print(f"[sft] 下一步: python pipeline/write_gguf.py --hf-dir {out_dir.as_posix()} "
              f"--out models/tinyllm-sft-f16.gguf")


if __name__ == "__main__":
    main()
