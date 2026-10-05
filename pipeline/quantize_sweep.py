#!/usr/bin/env python3
"""
阶段 3+/4 —— 量化精度工程: imatrix 校准 + 多方案量化对比

在原有「F16 -> 固定 Q4_K_M」的基础上做两件事:

  1. 支持重要度矩阵 (imatrix) 校准量化。
     用 llama-imatrix 在校准语料上统计每个张量对输出的重要度, 再交给
     llama-quantize --imatrix 使用。同样是 4bit, 带 imatrix 的方案能明显
     降低量化损失 (尤其小模型 + 低比特)。

  2. 批量跑多套量化方案并输出对比报告。
     Q8_0 / Q4_K_M / Q4_K_S / IQ4_XS 等逐一套跑, 记录产物体积、压缩率与耗时,
     生成 out/quantize_sweep.md。

用法:
  python pipeline/quantize_sweep.py --check                    # 只看环境是否就绪
  python pipeline/quantize_sweep.py --llama-cpp D:/tools/llama.cpp
  python pipeline/quantize_sweep.py --schemes Q4_K_M,Q4_K_S,IQ4_XS --imatrix auto
  python pipeline/quantize_sweep.py --input models/tinyllm-f16.gguf --no-imatrix

产物:
  models/<模型名>-<方案>.gguf      各方案的量化模型
  models/imatrix.dat               重要性矩阵 (供 llama-quantize --imatrix 使用)
  out/quantize_sweep.md            对比报告 (体积 / 压缩率 / 耗时)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ROOT, cfg_get, find_binary, load_config, markdown_table, read_log,
    resolve_path, run, size_mb, try_find_binary, write_report,
)

DEFAULT_SCHEMES = ["Q8_0", "Q4_K_M", "Q4_K_S", "IQ4_XS"]


def parse_args():
    p = argparse.ArgumentParser(description="imatrix 校准 + 多方案量化对比")
    p.add_argument("--llama-cpp", default=None, help="llama.cpp 目录 (或设 LLAMA_CPP_DIR)")
    p.add_argument("--quantize", default=None, help="llama-quantize 可执行文件路径")
    p.add_argument("--imatrix-bin", default=None, help="llama-imatrix 可执行文件路径")
    p.add_argument("--input", default=None, help="F16 GGUF 输入 (默认取 config.yaml)")
    p.add_argument("--schemes", default=None,
                   help=f"逗号分隔的量化方案, 默认 {','.join(DEFAULT_SCHEMES)}")
    p.add_argument("--imatrix", nargs="?", const="auto", default=None,
                   help="'auto' 生成校准矩阵, 或指定已有 .dat 文件; 不传则跳过")
    p.add_argument("--calibration", default=None, help="校准文本 (默认 config quantize.imatrix.calibration)")
    p.add_argument("--ctx-size", type=int, default=512, help="imatrix 统计上下文长度")
    p.add_argument("--imatrix-strict", action="store_true",
                   help="imatrix 失败时直接退出 (默认降级为普通量化)")
    p.add_argument("--out", default="out/quantize_sweep.md", help="报告输出路径")
    p.add_argument("--force", action="store_true", help="已存在的量化产物也重跑")
    p.add_argument("--check", action="store_true", help="只做环境自检")
    return p.parse_args()


def check_env(args, cfg) -> int:
    """环境自检: 二进制与输入文件是否就绪。"""
    print("== 量化环境自检 ==")
    quant = try_find_binary("quantize", args.quantize, args.llama_cpp)
    imat = try_find_binary("imatrix", args.imatrix_bin, args.llama_cpp)
    f16 = resolve_path(args.input or cfg_get(cfg, "paths.gguf_f16", "models/tinyllm-f16.gguf"))
    print(f"  llama-quantize : {quant or '✗ 未找到 (必需)'}")
    print(f"  llama-imatrix  : {imat or '✗ 未找到 (仅 imatrix 校准需要)'}")
    print(f"  F16 输入       : {f16 if f16.exists() else '✗ 不存在 (先跑 train.py + write_gguf.py)'}")
    calib = resolve_path(args.calibration or cfg_get(cfg, "quantize.imatrix.calibration", "data/corpus.txt"))
    print(f"  校准语料       : {calib if calib.exists() else '✗ 不存在'} "
          f"({size_mb(calib):.1f} MB)" if calib.exists() else f"  校准语料       : ✗ {calib}")
    schemes = (args.schemes or cfg_get(cfg, "quantize.schemes")) or DEFAULT_SCHEMES
    print(f"  量化方案       : {', '.join(schemes)}")
    ok = bool(quant) and f16.exists()
    print(f"== {'就绪' if ok else '未就绪'} ==")
    return 0 if ok else 1


def build_imatrix(args, cfg, imatrix_bin, f16_path: Path) -> Path | None:
    """生成重要性矩阵; 失败时视 --imatrix-strict 决定是否退出。"""
    if args.imatrix and args.imatrix != "auto":
        path = Path(args.imatrix)
        if not path.exists():
            sys.exit(f"[imatrix] 指定的文件不存在: {path}")
        print(f"[imatrix] 使用已有校准矩阵 {path}")
        return path

    calib = resolve_path(args.calibration or
                         cfg_get(cfg, "quantize.imatrix.calibration", "data/corpus.txt"))
    if not calib.exists():
        msg = f"[imatrix] 校准语料不存在: {calib}"
        if args.imatrix_strict:
            sys.exit(msg)
        print(msg + " -> 跳过 imatrix, 使用普通量化")
        return None

    out_path = resolve_path(cfg_get(cfg, "quantize.imatrix.output", "models/imatrix.dat"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log = ROOT / "out" / "imatrix.log"
    print(f"[imatrix] 校准语料 {calib} ({size_mb(calib):.1f} MB), 上下文 {args.ctx_size}")
    try:
        run([imatrix_bin, "-m", f16_path, "-f", calib,
             "-o", out_path, "-c", str(args.ctx_size),
             "--chunks", "0"], log_path=log)
    except Exception as exc:                     # 小语料/参数不兼容等
        msg = f"[imatrix] 生成失败: {exc}"
        print(read_log(log, tail=8))
        if args.imatrix_strict:
            sys.exit(msg)
        print(msg + " -> 降级为普通量化")
        return None
    if not out_path.exists():
        print("[imatrix] 未生成校准矩阵 -> 降级为普通量化")
        return None
    print(f"[imatrix] 已生成 {out_path} ({size_mb(out_path):.2f} MB)")
    return out_path


def quantize_one(quantizer: Path, f16: Path, out_path: Path, scheme: str,
                 imatrix: Path | None, force: bool) -> dict:
    """跑单个量化方案, 返回结果行 (失败时记录 error 字段)。"""
    log = ROOT / "out" / f"quantize-{scheme}.log"
    if out_path.exists() and not force:
        print(f"[{scheme}] 已存在 {out_path.name}, 跳过 (--force 可重跑)")
        return {"scheme": scheme, "path": out_path, "size": size_mb(out_path),
                "seconds": None, "imatrix": bool(imatrix), "skipped": True}

    cmd = [quantizer]
    if imatrix:
        cmd += ["--imatrix", imatrix]
    cmd += [f16, out_path, scheme]

    print(f"[{scheme}] 开始量化" + (" (imatrix)" if imatrix else ""))
    t0 = time.time()
    # 部分版本在输出重定向到文件时会以 iostream 错误返回非零, 但产物已生成,
    # 因此这里不 check, 统一以产物文件为准。
    run(cmd, log_path=log, check=False)
    elapsed = time.time() - t0

    if not out_path.exists() or out_path.stat().st_size == 0:
        print(read_log(log, tail=6))
        return {"scheme": scheme, "path": out_path, "size": float("nan"),
                "seconds": elapsed, "imatrix": bool(imatrix),
                "error": read_log(log, tail=1).strip()}
    print(f"[{scheme}] 完成: {size_mb(out_path):.1f} MB, 耗时 {elapsed:.1f}s")
    return {"scheme": scheme, "path": out_path, "size": size_mb(out_path),
            "seconds": elapsed, "imatrix": bool(imatrix)}


def main():
    args = parse_args()
    cfg = load_config()

    if args.check:
        sys.exit(check_env(args, cfg))

    quantizer = find_binary("quantize", args.quantize, args.llama_cpp)
    f16 = resolve_path(args.input or cfg_get(cfg, "paths.gguf_f16", "models/tinyllm-f16.gguf"))
    if not f16.exists():
        sys.exit(f"[sweep] 找不到 F16 GGUF: {f16}\n"
                 f"        请先运行: python pipeline/train.py 然后 python pipeline/write_gguf.py")

    schemes = [s.strip() for s in (args.schemes.split(",") if args.schemes
                                   else cfg_get(cfg, "quantize.schemes", DEFAULT_SCHEMES))
               if s.strip()]
    print(f"[sweep] llama-quantize: {quantizer}")
    print(f"[sweep] 输入 F16: {f16.name} ({size_mb(f16):.1f} MB)")
    print(f"[sweep] 方案: {', '.join(schemes)}")

    imatrix = None
    if args.imatrix:
        imatrix_bin = try_find_binary("imatrix", args.imatrix_bin, args.llama_cpp)
        if imatrix_bin:
            imatrix = build_imatrix(args, cfg, imatrix_bin, f16)
        else:
            print("[imatrix] 找不到 llama-imatrix -> 跳过校准")
            if args.imatrix_strict:
                sys.exit("[imatrix] --imatrix-strict 下缺少 llama-imatrix")

    results = []
    for scheme in schemes:
        out_path = f16.with_name(f"{f16.stem.replace('-f16', '')}-{scheme}.gguf")
        results.append(quantize_one(quantizer, f16, out_path, scheme, imatrix, args.force))

    f16_mb = size_mb(f16)
    rows = []
    for r in results:
        if r.get("error") or r["size"] != r["size"]:
            rows.append([r["scheme"], "失败", "—", "—", "—", r.get("error", "")[:60]])
            continue
        rows.append([
            r["scheme"],
            f"{r['size']:.1f} MB",
            f"{r['size'] / f16_mb:.1%}" if f16_mb == f16_mb else "—",
            f"{f16_mb / r['size']:.2f}×" if r["size"] else "—",
            "imatrix" if r["imatrix"] else "—",
            "已存在" if r.get("skipped") else (f"{r['seconds']:.1f}s" if r["seconds"] else "—"),
        ])

    body = "\n".join([
        f"输入模型: `{f16.name}` ({f16_mb:.1f} MB)",
        f"量化器: `{quantizer}`",
        f"重要性矩阵: {f'imatrix ({imatrix.name})' if imatrix else '未使用'}",
        "",
        "## 体积对比",
        "",
        markdown_table(["方案", "体积", "相对 F16", "压缩倍数", "校准", "耗时"], rows),
        "",
        "## 结论与用法",
        "",
        "- 体积/压缩倍数用于判断能否塞进目标设备的内存预算; 精度需配合 "
        "`python pipeline/eval.py` 的困惑度对比一起看。",
        "- 带 imatrix 的 4bit 量化通常能在同体积下取得更低的困惑度, 推荐"
        "先用 `--imatrix auto` 跑一遍 Q4_K_M 与 IQ4_XS 再决定上机方案。",
        "- 部署时把选中的 GGUF 交给 llama-server 即可, 例如:",
        "",
        "  ```bash",
        f"  llama-server -m {results[0]['path'].as_posix() if results else 'models/<方案>.gguf'} "
        "--host 127.0.0.1 --port 8081 --ctx-size 512",
        "  ```",
        "",
        "> 各方案的完整量化日志见 `out/quantize-<方案>.log`。",
    ])
    write_report(resolve_path(args.out), "量化方案对比 (imatrix 校准)", body)

    failed = [r["scheme"] for r in results if r.get("error")]
    if failed:
        sys.exit(f"[sweep] 以下方案失败: {', '.join(failed)}")
    print("[sweep] 全部完成")


if __name__ == "__main__":
    main()
