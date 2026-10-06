#!/usr/bin/env python3
"""
阶段 4+/4 —— 投机解码 (speculative decoding) 实验

思路: 用一个更小的 draft 模型快速猜 token, 再由 target 模型一次并行校验,
猜中的部分就省掉了 target 的自回归解码步数 —— 端侧场景里这是"不降精度换速度"
的常用手段。

前提: draft 与 target 必须共用同一套词表 (llama.cpp 的硬性要求)。本项目的
做法是让两者共用同一个 BPE 分词器:

  # 1) 训练 target (更大一档, 复用已有分词器)
  python pipeline/train.py --profile target --reuse-tokenizer
  # 2) 各自导出 + 量化
  python pipeline/write_gguf.py --hf-dir models/tinyllm-target-hf --out models/tinyllm-target-f16.gguf
  python pipeline/quantize_sweep.py --schemes Q4_K_M ...

用法:
  python pipeline/spec_decode.py --check
  python pipeline/spec_decode.py --llama-cpp D:/tools/llama.cpp \
      --draft models/tinyllm-q4_k_m.gguf --target models/tinyllm-target-q4_k_m.gguf

产物:
  out/spec_decode.md   基线 vs 投机解码的吞吐对比 (含加速比与接受率)
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ROOT, cfg_get, find_binary, http_json, http_text, load_config, markdown_table,
    resolve_path, size_mb, stage_report, supports_flag, try_find_binary, write_report,
)
from eval import PROMPT, Server  # noqa: E402  复用服务管理与提示词

DRAFT_PROMPTS = [
    "人工智能是计算机科学的一个分支，",
    "Transformer 架构是现代大语言模型",
    "深度学习使用多层神经网络",
]


def parse_args():
    p = argparse.ArgumentParser(description="投机解码实验 (draft + target)")
    p.add_argument("--llama-cpp", default=None, help="llama.cpp 目录 (或设 LLAMA_CPP_DIR)")
    p.add_argument("--server", default=None, help="llama-server 可执行文件路径")
    p.add_argument("--draft", default=None, help="draft 小模型 GGUF (默认 config spec.draft)")
    p.add_argument("--target", default=None, help="target 模型 GGUF (默认 config spec.target)")
    p.add_argument("--draft-max", type=int, default=None, help="单次猜测的最大 token 数")
    p.add_argument("--draft-min", type=int, default=None, help="低于该命中数时退回普通解码")
    p.add_argument("--draft-p-min", type=float, default=None, help="猜测的置信度下限")
    p.add_argument("--max-tokens", type=int, default=None, help="每次生成的 token 数")
    p.add_argument("--rounds", type=int, default=None, help="每组的测量轮数, 取中位数")
    p.add_argument("--ctx-size", type=int, default=None, help="上下文长度")
    p.add_argument("--out", default="out/spec_decode.md", help="报告输出路径")
    p.add_argument("--check", action="store_true", help="只做环境自检")
    return p.parse_args()


def gguf_meta(path: Path) -> dict:
    """读取 GGUF 元数据 (词表大小等), 用于校验 draft/target 是否共用词表。"""
    try:
        from gguf import GGUFReader                   # pip install gguf
    except Exception:
        return {}
    try:
        r = GGUFReader(str(path))
        tok_field = r.fields.get("tokenizer.ggml.tokens")
        if tok_field is None:
            return {}
        return {"vocab_size": len(tok_field.data),
                "arch": str(r.fields.get("general.architecture").contents()
                            if "general.architecture" in r.fields else "?")}
    except Exception:
        return {}


def story(args, cfg) -> tuple[Path, Path]:
    draft = resolve_path(args.draft or cfg_get(cfg, "spec.draft", "models/tinyllm-q4_k_m.gguf"))
    target = resolve_path(args.target or cfg_get(cfg, "spec.target",
                                                 "models/tinyllm-target-q4_k_m.gguf"))
    return draft, target


def check_env(args, cfg) -> int:
    print("== 投机解码环境自检 ==")
    server = try_find_binary("server", args.server, args.llama_cpp)
    draft, target = story(args, cfg)
    print(f"  llama-server : {server or '✗ 未找到 (必需)'}")
    for label, path in (("draft ", draft), ("target", target)):
        mark = "✓" if path.exists() else "✗ 不存在"
        print(f"  {label}       : {mark} {path}")
    vm_d, vm_t = gguf_meta(draft), gguf_meta(target)
    if vm_d and vm_t:
        same = vm_d.get("vocab_size") == vm_t.get("vocab_size")
        print(f"  词表校验     : draft {vm_d.get('vocab_size')} vs target "
              f"{vm_t.get('vocab_size')} -> {'一致 ✓' if same else '不一致 ✗ (投机解码无效)'}")
    elif draft.exists() and target.exists():
        print("  词表校验     : 未安装 gguf 包, 跳过 (pip install gguf 可启用)")
    ok = bool(server) and draft.exists() and target.exists()
    print(f"== {'就绪' if ok else '未就绪'} ==")
    return 0 if ok else 1


def measure(binary: Path, target: Path, ctx: int, max_tokens: int, rounds: int,
            draft: Path | None = None, extra: list[str] | None = None,
            tag: str = "") -> dict:
    """跑一组测量; draft 不为空时启用投机解码并从 /metrics 抓接受率。"""
    flags = list(extra or [])
    if draft:
        flags += ["--model-draft", str(draft)]
    with Server(binary, target, ctx, flags, tag=tag) as srv:
        dec, ttft, accept = [], [], None
        for i in range(rounds):
            res = None
            for prompt in DRAFT_PROMPTS:
                res = http_json(f"{srv.url}/completion", {
                    "prompt": prompt, "n_predict": max_tokens, "temperature": 0.0,
                    "cache_prompt": False, "top_k": 1,
                }, timeout=600)
                t = res.get("timings", {})
                if t.get("predicted_per_second"):
                    dec.append(t["predicted_per_second"])
                if t.get("prompt_ms"):
                    ttft.append(t["prompt_ms"])
            if draft:
                accept = accept_rate(srv) or accept
            print(f"    round {i + 1}: decode {dec[-1]:.1f} tok/s"
                  + (f", 接受率 {accept:.1%}" if accept else ""))
    return {
        "decode_tok_s": round(statistics.median(dec), 2) if dec else None,
        "ttft_ms": round(statistics.median(ttft), 1) if ttft else None,
        "accept_rate": accept,
    }


def _metric(text: str, name: str) -> float | None:
    """从 Prometheus 文本里取某个计数器的值 (跳过 HELP/TYPE 行)。"""
    m = re.search(rf"{re.escape(name)}\s+([0-9.eE+\-]+)", text)
    return float(m.group(1)) if m else None


def accept_rate(srv: Server) -> float | None:
    """从 llama-server 的 /metrics 里抓 draft 接受率。

    指标名在 llama.cpp 版本间变过, 这里两套都试:
      * 新版: llamacpp:spec_decode_num_accepted_tokens_total / ..._num_draft_tokens_total
      * 旧版: draft_n_accepted_total / draft_n_evaluated_total
    """
    text = http_text(f"{srv.url}/metrics")
    if not text:
        return None
    for acc_name, tot_name in (
        ("spec_decode_num_accepted_tokens_total", "spec_decode_num_draft_tokens_total"),
        ("draft_n_accepted_total", "draft_n_evaluated_total"),
    ):
        acc, tot = _metric(text, acc_name), _metric(text, tot_name)
        if acc is not None and tot:
            return acc / tot
    return None


def main():
    args = parse_args()
    cfg = load_config()

    if args.check:
        sys.exit(check_env(args, cfg))

    binary = find_binary("server", args.server, args.llama_cpp)
    draft, target = story(args, cfg)
    for label, path in (("draft", draft), ("target", target)):
        if not path.exists():
            sys.exit(f"[spec] {label} 模型不存在: {path}\n"
                     f"        请按 README「投机解码」一节训练并量化两个共用词表的模型")

    # 词表一致性校验: 不一致时投机解码不会有任何收益
    vm_d, vm_t = gguf_meta(draft), gguf_meta(target)
    if vm_d and vm_t and vm_d.get("vocab_size") != vm_t.get("vocab_size"):
        sys.exit(f"[spec] draft 词表 {vm_d['vocab_size']} 与 target "
                 f"{vm_t['vocab_size']} 不一致, 投机解码无效。\n"
                 f"        请用 --reuse-tokenizer 训练共用分词器的两个模型。")

    ctx = args.ctx_size or cfg_get(cfg, "eval.ctx_size", 512)
    max_tokens = args.max_tokens or cfg_get(cfg, "spec.max_tokens", 128)
    rounds = args.rounds or cfg_get(cfg, "spec.rounds", cfg_get(cfg, "eval.rounds", 3))
    draft_max = args.draft_max or cfg_get(cfg, "spec.draft_max", 8)
    draft_min = args.draft_min or cfg_get(cfg, "spec.draft_min", 2)
    p_min = args.draft_p_min if args.draft_p_min is not None else cfg_get(cfg, "spec.draft_p_min", 0.6)

    print(f"[spec] target: {target.name} ({size_mb(target):.1f} MB)")
    print(f"[spec] draft : {draft.name} ({size_mb(draft):.1f} MB)")
    print(f"[spec] draft-max={draft_max} draft-min={draft_min} p-min={p_min}, {rounds} 轮取中位数")

    # llama.cpp b6xxx 起把 --draft-max / --draft-min 改名为 --spec-draft-n-max / --spec-draft-n-min,
    # 这里按二进制实际支持的参数名组装, 兼容新旧版本 (探测失败时退回旧名)
    n_max_flag = ("--spec-draft-n-max" if supports_flag(binary, "--spec-draft-n-max")
                  else "--draft-max")
    n_min_flag = ("--spec-draft-n-min" if supports_flag(binary, "--spec-draft-n-min")
                  else "--draft-min")
    p_min_flag = ("--spec-draft-p-min" if supports_flag(binary, "--spec-draft-p-min")
                  else "--draft-p-min")
    if n_max_flag == "--draft-max":
        print("[spec] 该 llama-server 使用旧版投机解码参数名 (--draft-max/--draft-min)")

    print("\n[spec] 基线 (target 单独推理)")
    base = measure(binary, target, ctx, max_tokens, rounds, tag="base")
    print(f"    -> decode {base['decode_tok_s']} tok/s, TTFT {base['ttft_ms']} ms")

    print("\n[spec] 投机解码 (target + draft)")
    extra = [n_max_flag, str(draft_max), n_min_flag, str(draft_min)]
    if p_min is not None:
        extra += [p_min_flag, str(p_min)]
    spec = measure(binary, target, ctx, max_tokens, rounds,
                   draft=draft, extra=extra, tag="spec")
    print(f"    -> decode {spec['decode_tok_s']} tok/s, TTFT {spec['ttft_ms']} ms, "
          f"接受率 {spec['accept_rate']:.1%}" if spec["accept_rate"]
          else f"    -> decode {spec['decode_tok_s']} tok/s")

    speedup = (spec["decode_tok_s"] / base["decode_tok_s"]) if (
        base.get("decode_tok_s") and spec.get("decode_tok_s")) else None

    rows = [
        ["基线 (target)", f"{base['decode_tok_s']:.1f}" if base.get("decode_tok_s") else "—",
         f"{base['ttft_ms']:.0f}" if base.get("ttft_ms") else "—", "—", "—"],
        ["投机解码 (target+draft)",
         f"{spec['decode_tok_s']:.1f}" if spec.get("decode_tok_s") else "—",
         f"{spec['ttft_ms']:.0f}" if spec.get("ttft_ms") else "—",
         f"{speedup:.2f}×" if speedup else "—",
         f"{spec['accept_rate']:.1%}" if spec.get("accept_rate") else "—"],
    ]

    body = "\n".join([
        f"target: `{target.name}` ({size_mb(target):.1f} MB) · "
        f"draft: `{draft.name}` ({size_mb(draft):.1f} MB)",
        f"参数: {n_max_flag}={draft_max}, {n_min_flag}={draft_min}, "
        f"{p_min_flag}={p_min}, "
        f"上下文 {ctx}, 每组 {len(DRAFT_PROMPTS)} 个 prompt × {rounds} 轮取中位数, 温度 0",
        "",
        "## 结果",
        "",
        markdown_table(["配置", "解码(tok/s)", "TTFT(ms)", "加速比", "接受率"], rows),
        "",
        "## 说明",
        "",
        "- **接受率**是 draft 猜中的 token 占比。小 draft 模型的接受率通常不高, "
        "能拿到 1.2–1.6× 的解码加速已属正常; 接受率随 draft 与 target 的"
        "语料/参数规模差距变化。",
        "- draft 与 target **必须共用同一分词器**, 否则 llama.cpp 无法对齐候选 token, "
        "脚本会提前报错拦下。",
        "- 端侧部署时 draft 常驻内存也要计入预算 (本实验里 draft 仅 "
        f"{size_mb(draft):.1f} MB), 因此用极小 draft + 中等 target 是常见组合。",
        "- 若接受率过低 (<30%), 可适当调小 `--draft-max` 或提高 `--draft-p-min` 重跑对比。",
    ])
    write_report(resolve_path(args.out), "投机解码实验", body)
    print("[spec] 完成")

    # ---------- 阶段报告 ----------
    stage_report(
        "spec",
        summary=f"draft `{draft.name}` + target `{target.name}` 的投机解码对比"
                + (f"，解码从 {base['decode_tok_s']:.0f} 提到 **{spec['decode_tok_s']:.0f} tok/s**"
                   f"（{speedup:.2f}×）。" if speedup else "。"),
        metrics={
            "draft 模型": f"{draft.name} ({size_mb(draft):.1f} MB)",
            "target 模型": f"{target.name} ({size_mb(target):.1f} MB)",
            "最大猜测数": draft_max,
            "接受率": f"{spec['accept_rate']:.1%}" if spec.get("accept_rate") else "—",
            "加速比": f"{speedup:.2f}×" if speedup else "—",
        },
        tables=[("对比", markdown_table(
            ["配置", "解码(tok/s)", "TTFT(ms)", "加速比", "接受率"], rows))],
        artifacts=[draft, target],
        links=[("完整报告 (含调参建议)", "spec_decode.md")],
        notes="draft 与 target **必须共用分词器**（脚本会先校验词表大小）。"
              "接受率过低时可调小最大猜测数或提高 draft_p_min 重跑。",
    )


if __name__ == "__main__":
    main()
