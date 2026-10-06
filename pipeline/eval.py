#!/usr/bin/env python3
"""
阶段 4+/4 —— 端侧推理评测基准

把「量化到 1/4」这种说法变成可核对的数字。对每个 GGUF 产出:

  * 困惑度 PPL      —— 量化到底掉了多少精度 (优先 llama-perplexity, 缺失时可用 HF 模型兜底)
  * 首 token 延迟    —— 预填充耗时 (llama-server timings.prompt_ms)
  * 预填充吞吐       —— prompt tok/s, 反映长上下文的处理能力
  * 解码吞吐         —— 生成 tok/s, 端侧最关心的指标
  * 模型体积         —— 常驻内存的下限
  * KV cache 量化     —— --cache-type-k/v q8_0 后的速度与内存变化

用法:
  python pipeline/eval.py --check
  python pipeline/eval.py --llama-cpp D:/tools/llama.cpp
  python pipeline/eval.py --models models/tinyllm-f16.gguf models/tinyllm-q4_k_m.gguf
  python pipeline/eval.py --kv-quant                    # 额外测 KV cache 量化

产物:
  out/benchmark.md      人读的报告 (含与 F16 基线的对比)
  out/benchmark.json    机读结果, 便于画曲线或写进自动化
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ROOT, cfg_get, find_binary, free_port, http_json, load_config, markdown_table,
    process_memory_mb, read_log, resolve_path, run, size_mb, stage_report,
    try_find_binary, wait_for_http, write_report,
)

PROMPT = "人工智能是计算机科学的一个分支，"


def parse_args():
    p = argparse.ArgumentParser(description="GGUF 端侧推理评测")
    p.add_argument("--llama-cpp", default=None, help="llama.cpp 目录 (或设 LLAMA_CPP_DIR)")
    p.add_argument("--server", default=None, help="llama-server 可执行文件路径")
    p.add_argument("--models", nargs="*", default=None,
                   help="待评测 GGUF 列表 (默认 models/ 下的全部 *.gguf)")
    p.add_argument("--corpus", default=None, help="困惑度用的文本 (默认 config eval.corpus)")
    p.add_argument("--hf-dir", default=None,
                   help="llama-perplexity 缺失时, 用该 HF 目录配合 transformers 计算 PPL")
    p.add_argument("--ctx-size", type=int, default=None, help="上下文长度 (默认 config eval.ctx_size)")
    p.add_argument("--max-tokens", type=int, default=None, help="每次生成的 token 数")
    p.add_argument("--rounds", type=int, default=None, help="吞吐测量轮数, 取中位数")
    p.add_argument("--kv-quant", action="store_true", help="额外测 q8_0 KV cache 量化的影响")
    p.add_argument("--no-ppl", action="store_true", help="跳过困惑度, 只测速度")
    p.add_argument("--baseline", default=None, help="基线模型路径 (默认取 F16, 用于算相对变化)")
    p.add_argument("--out", default="out/benchmark.md", help="报告输出路径")
    p.add_argument("--json-out", default="out/benchmark.json", help="JSON 结果路径")
    p.add_argument("--check", action="store_true", help="只做环境自检")
    return p.parse_args()


def discover_models(args, cfg) -> list[Path]:
    if args.models:
        return [resolve_path(m) for m in args.models]
    models_dir = ROOT / "models"
    found = sorted(models_dir.glob("*.gguf"), key=lambda p: p.stat().st_size)
    if not found:
        sys.exit(f"[eval] {models_dir} 下没有 GGUF, 请先跑 train/export/quantize 流程")
    return found


def check_env(args, cfg) -> int:
    print("== 评测环境自检 ==")
    server = try_find_binary("server", args.server, args.llama_cpp)
    ppl_bin = try_find_binary("perplexity", None, args.llama_cpp)
    print(f"  llama-server     : {server or '✗ 未找到 (必需)'}")
    print(f"  llama-perplexity : {ppl_bin or '✗ 未找到 (PPL 将尝试 HF 兜底)'}")
    corpus = resolve_path(args.corpus or cfg_get(cfg, "eval.corpus", "data/corpus.txt"))
    print(f"  困惑度语料       : {'✓' if corpus.exists() else '✗'} {corpus}")
    try:
        models = discover_models(args, cfg)
        print(f"  待评测模型       : {len(models)} 个")
        for m in models:
            print(f"    - {m.name} ({size_mb(m):.1f} MB)")
    except SystemExit:
        print("  待评测模型       : ✗ 无")
    ok = bool(server)
    print(f"== {'就绪' if ok else '未就绪'} ==")
    return 0 if ok else 1


def hf_ppl(hf_dir: Path, corpus: Path, block_size: int) -> float | None:
    """用 transformers 在语料上算困惑度 (llama-perplexity 缺失时的兜底)。"""
    try:
        import math
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        return None
    if not (hf_dir / "config.json").exists():
        return None
    tok = AutoTokenizer.from_pretrained(str(hf_dir))
    # 我们自己对语料分块计算, 关闭分词器的整篇长度告警
    tok.model_max_length = int(1e30)
    model = AutoModelForCausalLM.from_pretrained(str(hf_dir))
    model.eval()
    ids = tok(corpus.read_text(encoding="utf-8"))["input_ids"]
    if len(ids) < block_size + 1:
        return None
    nll_total, n_tok = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(ids) - block_size, block_size):
            chunk = torch.tensor([ids[i:i + block_size + 1]])
            out = model(chunk[:, :-1], labels=chunk[:, 1:])
            nll_total += float(out.loss) * (block_size)
            n_tok += block_size
    return math.exp(nll_total / max(n_tok, 1))


def ppl_adapted_ctx(log_text: str) -> int | None:
    """llama-perplexity 报「语料只有 N 个 token」时, 给出它实际能跑的上下文。

    该工具要求 token 数 >= 2×ctx, 样例语料往往偏小, 因此取不超过 N/2 的最大 2 的幂。
    """
    m = re.search(r"tokenizes to only (\d+) tokens", log_text)
    if not m:
        return None
    n = int(m.group(1))
    return max(32, 1 << ((n // 2).bit_length() - 1))


def server_ppl(ppl_bin: Path, model: Path, corpus: Path, ctx: int) -> float | None:
    """用 llama-perplexity 计算困惑度; 语料偏小时自动下调上下文重试一次。"""
    from common import parse_ppl, read_log
    log = ROOT / "out" / f"ppl-{model.stem}.log"
    for attempt in range(2):
        try:
            # 不传 --chunks 0: 部分 llama.cpp 版本会把它当真只跑 0 个 chunk 而直接报错
            run([ppl_bin, "-m", model, "-f", corpus, "-c", str(ctx)],
                log_path=log, check=False)
        except Exception:
            return None
        ppl = parse_ppl(read_log(log))
        if ppl is not None:
            return ppl
        text = read_log(log)
        adapted = ppl_adapted_ctx(text)
        if attempt == 0 and adapted and adapted < ctx:
            print(f"    PPL: 语料过小, 上下文 {ctx} -> {adapted} 重试")
            ctx = adapted
            continue
        reason = next((ln.strip() for ln in reversed(text.splitlines())
                       if ln.strip().startswith("E ")), "未知原因")
        print(f"    PPL: 未能计算 —— {reason}")
        break
    return None


class Server:
    """llama-server 生命周期管理 (启动/探活/优雅退出)。"""

    def __init__(self, binary: Path, model: Path, ctx: int,
                 extra: list[str] | None = None, tag: str = ""):
        self.binary, self.model, self.ctx = binary, model, ctx
        self.extra = extra or []
        self.tag = tag
        self.port = free_port()
        self.proc: subprocess.Popen | None = None
        suffix = f"-{tag}" if tag else ""
        self.log = ROOT / "out" / f"server-{model.stem}{suffix}.log"

    def __enter__(self):
        self.log.parent.mkdir(parents=True, exist_ok=True)
        cmd = [str(self.binary), "-m", str(self.model), "--host", "127.0.0.1",
               "--port", str(self.port), "-c", str(self.ctx), "--metrics"] + self.extra
        print("+", " ".join(cmd), flush=True)
        self.fh = open(self.log, "w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(cmd, stdout=self.fh, stderr=subprocess.STDOUT)
        if not self._wait_ready():
            log_tail = read_log(self.log, tail=10)
            self.__exit__(None, None, None)
            raise RuntimeError(f"llama-server 启动失败, 详见 {self.log}\n{log_tail}")
        return self

    def _wait_ready(self, timeout: float = 120.0) -> bool:
        """等待 /health 就绪。

        进程若提前退出 (参数不被识别 / 模型加载失败) 立即返回失败并打印日志,
        避免为一个已经死掉的进程白等满整个超时。
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:        # 进程已退出
                print(f"[server] 进程提前退出 (exit={self.proc.returncode})", flush=True)
                return False
            if wait_for_http(f"{self.url}/health", timeout=1.0):
                return True
            time.sleep(0.5)
        return False

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def memory_mb(self) -> float | None:
        return process_memory_mb(self.proc.pid) if self.proc else None

    def complete(self, max_tokens: int) -> dict:
        t0 = time.time()
        data = http_json(f"{self.url}/completion", {
            "prompt": PROMPT, "n_predict": max_tokens, "temperature": 0.0,
            "cache_prompt": False, "top_k": 1,
        }, timeout=600)
        data["_wall_s"] = time.time() - t0
        return data

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        try:
            self.fh.close()
        except Exception:
            pass
        return False


def measure(args, server_bin: Path, model: Path, ctx: int, max_tokens: int,
            rounds: int, extra: list[str] | None = None, tag: str = "") -> dict:
    """跑一组吞吐测量, 返回中位数指标。"""
    with Server(server_bin, model, ctx, extra, tag=tag) as srv:
        dec, pre, ttft, wall = [], [], [], []
        for i in range(rounds):
            r = srv.complete(max_tokens)
            t = r.get("timings", {})
            if t.get("predicted_per_second"):
                dec.append(t["predicted_per_second"])
            if t.get("prompt_per_second"):
                pre.append(t["prompt_per_second"])
            if t.get("prompt_ms"):
                ttft.append(t["prompt_ms"])
            wall.append(r.get("_wall_s", 0.0))
            print(f"    round {i + 1}: decode {t.get('predicted_per_second', 0):.1f} tok/s, "
                  f"prefill {t.get('prompt_per_second', 0):.1f} tok/s")
        mem = srv.memory_mb()
    med = statistics.median
    return {
        "decode_tok_s": round(med(dec), 2) if dec else None,
        "prefill_tok_s": round(med(pre), 2) if pre else None,
        "ttft_ms": round(med(ttft), 1) if ttft else None,
        "wall_s": round(med(wall), 2) if wall else None,
        "server_rss_mb": round(mem, 1) if mem else None,
    }


def main():
    args = parse_args()
    cfg = load_config()

    if args.check:
        sys.exit(check_env(args, cfg))

    server_bin = find_binary("server", args.server, args.llama_cpp)
    models = discover_models(args, cfg)
    ctx = args.ctx_size or cfg_get(cfg, "eval.ctx_size", 512)
    max_tokens = args.max_tokens or cfg_get(cfg, "eval.max_tokens", 128)
    rounds = args.rounds or cfg_get(cfg, "eval.rounds", 3)
    corpus = resolve_path(args.corpus or cfg_get(cfg, "eval.corpus",
                                                 cfg_get(cfg, "paths.corpus", "data/corpus.txt")))
    ppl_bin = None if args.no_ppl else try_find_binary("perplexity", None, args.llama_cpp)
    print(f"[eval] llama-server: {server_bin}")
    print(f"[eval] 模型 {len(models)} 个, 上下文 {ctx}, 每轮生成 {max_tokens} tokens, {rounds} 轮取中位数")
    if not args.no_ppl:
        print(f"[eval] PPL: {'llama-perplexity' if ppl_bin else 'HF 兜底' if args.hf_dir else '跳过(缺少工具)'}")

    results = []
    for m in models:
        print(f"\n[eval] === {m.name} ({size_mb(m):.1f} MB) ===")
        row = {"model": m.name, "path": m.as_posix(), "size_mb": round(size_mb(m), 2)}

        if ppl_bin:
            row["ppl"] = server_ppl(ppl_bin, m, corpus, ctx)
        elif args.hf_dir:
            row["ppl"] = hf_ppl(resolve_path(args.hf_dir), corpus,
                                cfg_get(cfg, "train.block_size", 256))
        else:
            row["ppl"] = None

        row.update(measure(args, server_bin, m, ctx, max_tokens, rounds))
        print(f"    -> decode {row['decode_tok_s']} tok/s, TTFT {row['ttft_ms']} ms, PPL {row['ppl']}")

        if args.kv_quant:
            kv = ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
            kv_res = measure(args, server_bin, m, ctx, max_tokens, rounds,
                             extra=kv, tag="kvq")
            row["kv_q8"] = kv_res
            print(f"    -> KV q8_0: decode {kv_res['decode_tok_s']} tok/s, "
                  f"RSS {kv_res['server_rss_mb']} MB")

        results.append(row)

    # ---------- 相对基线的变化 ----------
    baseline = None
    if args.baseline:
        base_path = resolve_path(args.baseline)
    else:
        base_path = resolve_path(cfg_get(cfg, "paths.gguf_f16", "models/tinyllm-f16.gguf"))
    for r in results:
        if Path(r["path"]) == base_path:
            baseline = r
            break

    rows = []
    for r in results:
        def delta(key, lower_better=False):
            if not baseline or r is baseline or not baseline.get(key) or not r.get(key):
                return "—"
            change = (r[key] / baseline[key] - 1) * 100
            mark = "↓" if (change < 0) != lower_better else "↑"
            return f"{mark}{abs(change):.1f}%"
        kv = r.get("kv_q8") or {}
        rows.append([
            r["model"],
            f"{r['size_mb']:.1f} MB",
            f"{r['ppl']:.2f}" if r.get("ppl") else "—",
            delta("ppl", lower_better=True),
            f"{r['ttft_ms']:.0f}" if r.get("ttft_ms") else "—",
            f"{r['prefill_tok_s']:.1f}" if r.get("prefill_tok_s") else "—",
            f"{r['decode_tok_s']:.1f}" if r.get("decode_tok_s") else "—",
            f"{kv.get('decode_tok_s', 0):.1f}" if kv.get("decode_tok_s") else "—",
            f"{r['server_rss_mb']:.0f}" if r.get("server_rss_mb") else "—",
        ])

    # PPL 全缺时给出可操作的原因提示 (最常见是语料 token 数不足)
    ppl_note: list[str] = []
    if results and all(not r.get("ppl") for r in results):
        ppl_note = ["",
                    "> ⚠ 本机未取到困惑度。最常见原因是评测语料太短: "
                    "`llama-perplexity` 要求语料 token 数 ≥ 2×上下文长度 "
                    "(默认上下文 "
                    f"{ctx} → 至少 {2 * ctx} 个 token)。"
                    f"当前用例语料 `{corpus.name}` 偏小; 脚本已尝试自动下调上下文, "
                    "仍失败则请换用更长的留出语料 (把 `config.yaml` 的 `eval.corpus` 指向它)。"]

    body = "\n".join([
        f"测量条件: 上下文 {ctx}, 每轮生成 {max_tokens} tokens, {rounds} 轮取中位数; "
        f"PPL 语料 `{corpus.name}`; 解码温度 0。",
        f"基线: {Path(baseline['path']).name if baseline else '未匹配到 F16 (相对变化列留空)'}",
        *ppl_note,
        "",
        "## 总表",
        "",
        markdown_table(
            ["模型", "体积", "PPL", "PPL 变化", "TTFT(ms)", "预填充(tok/s)",
             "解码(tok/s)", "KV q8_0 解码", "服务端 RSS(MB)"], rows),
        "",
        "## 怎么读这张表",
        "",
        "- **PPL 变化**是量化损失的量化指标: 4bit 方案通常比 F16 高几个百分点, "
        "带 imatrix 校准的方案会明显更接近基线。",
        "- **解码吞吐**决定端侧体感速度; **预填充吞吐**决定长 prompt 的首字等待, "
        "也就是 TTFT。",
        "- **KV q8_0** 一列是开启 KV cache 量化后的解码速度: 内存更省, "
        "但在 CPU 上可能带来少量速度回退, 需要按设备权衡。",
        "- 体积 → 常驻内存的换算: GGUF 体积 + KV cache (与上下文长度、层数、KV 头数相关) "
        "+ 运行时开销。",
        "",
        "> 原始数据见 `out/benchmark.json`, 可用来画体积-精度-速度曲线。",
    ])
    write_report(resolve_path(args.out), "端侧推理评测基准", body)
    json_path = resolve_path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({"config": vars(args), "results": results},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] JSON 结果: {json_path}")

    # ---------- 阶段报告 ----------
    ok = [r for r in results if r.get("decode_tok_s")]
    fastest = max(ok, key=lambda r: r["decode_tok_s"]) if ok else None
    stage_report(
        "eval",
        summary=f"评测 {len(results)} 个模型：上下文 {ctx}、每轮 {max_tokens} tokens、"
                f"{rounds} 轮取中位数"
                + (f"，解码最快 **{Path(fastest['path']).name} "
                   f"{fastest['decode_tok_s']:.0f} tok/s**。" if fastest else "。"),
        metrics={
            "评测模型数": len(results),
            "上下文 / 生成长度": f"{ctx} / {max_tokens}",
            "测量轮数": rounds,
            "KV cache 量化": "开启 (q8_0)" if getattr(args, "kv_quant", False) else "关闭",
            "PPL 语料": corpus.name,
        },
        tables=[("总表", markdown_table(
            ["模型", "体积", "PPL", "PPL 变化", "TTFT(ms)", "预填充(tok/s)",
             "解码(tok/s)", "KV q8_0 解码", "服务端 RSS(MB)"], rows))],
        artifacts=[Path(r["path"]) for r in results if r.get("path")] + [json_path],
        links=[("完整报告 (含解读)", "benchmark.md")],
        notes="PPL 变化是量化损失的直接指标（4bit 通常高几个百分点）；解码吞吐决定端侧体感速度。"
              + ("\n\n> 本次有模型的 PPL 未取到，检查评测语料长度是否 ≥ 2×上下文。"
                 if len([r for r in results if not r.get("ppl")]) == len(results) else ""),
    )


if __name__ == "__main__":
    main()
