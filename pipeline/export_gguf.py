#!/usr/bin/env python3
"""
阶段 2+3/4 —— GGUF 导出与量化

  HF 模型 --(自写 write_gguf.py)--> F16 GGUF --(llama-quantize)--> Q4_K_M GGUF

前置条件: 只需要 llama.cpp 的 **可执行文件** (llama-quantize), 不需要 clone 源码仓库。
  去 https://github.com/ggml-org/llama.cpp/releases 下载预编译包, 解压到
  D:/tools/llama.cpp (含 bin/ 即可), 或用 --llama-cpp <目录> 指定。

为什么不用官方 convert_hf_to_gguf.py: 它用哈希白名单识别分词器, 从零自训的
BPE 词表不在名单里, 会报 "BPE pre-tokenizer was not recognized"。所以转换这一步
用项目自带的 pipeline/write_gguf.py。

用法:
  python pipeline/export_gguf.py --llama-cpp D:/tools/llama.cpp
  python pipeline/export_gguf.py --skip-convert          # 跳过转换, 只做量化
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "pipeline" / "config.yaml"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import find_binary, stage_report  # noqa: E402


def load_config() -> dict:
    import yaml
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def llama_cpp_dir(explicit: str | None) -> Path | None:
    """llama.cpp 运行时目录 (含 llama-quantize 的 release 解压目录或源码仓库)。"""
    for cand in (explicit, os.environ.get("LLAMA_CPP_DIR")):  # 显式参数 > 环境变量
        if cand:
            return Path(cand)
    return None


def run(cmd: list[str], **kw):
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def main():
    parser = argparse.ArgumentParser(description="导出 GGUF 并量化")
    parser.add_argument("--llama-cpp", default=None,
                        help="llama.cpp 目录 (含 bin/llama-quantize; 不需要源码仓库)")
    parser.add_argument("--quantize", default=None, help="llama-quantize 可执行文件路径")
    parser.add_argument("--skip-convert", action="store_true", help="跳过 HF->GGUF 转换")
    parser.add_argument("--quant-type", default=None, help="量化方案 (默认取 config.yaml)")
    args = parser.parse_args()

    cfg = load_config()
    paths = cfg["paths"]
    qtype = args.quant_type or cfg["quantize"]["type"]

    hf_dir = ROOT / paths["output_dir"]
    if not (hf_dir / "config.json").exists():
        sys.exit(f"[export] 未找到训练产物 {hf_dir}, 请先运行 train.py")
    if not (hf_dir / "tokenizer.model").exists() and not (hf_dir / "tokenizer.json").exists():
        sys.exit("[export] 训练产物缺少分词器文件")

    runtime = llama_cpp_dir(args.llama_cpp)
    if runtime:
        print(f"[export] llama.cpp 目录: {runtime}")

    f16_path = ROOT / paths["gguf_f16"]
    if not args.skip_convert:
        # 使用项目自带的 GGUF 导出器 (见文件头说明)
        run([sys.executable, ROOT / "pipeline" / "write_gguf.py"])
    else:
        print(f"[export] 跳过转换, 使用现有 {f16_path}")
    if not f16_path.exists():
        sys.exit(f"[export] 转换失败: {f16_path} 不存在")

    quantizer = find_binary("quantize", explicit=args.quantize, root=runtime)
    q4_path = ROOT / paths["gguf_q4"]
    print(f"[quantize] {f16_path.name} -> {q4_path.name} ({qtype})")
    # 注: 部分版本 llama-quantize 在输出重定向到管道时会以 iostream 错误
    # 返回非零, 但量化产物实际已生成。这里改为以产物文件为准。
    log_path = ROOT / "out" / "quantize.log"
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        subprocess.run([str(c) for c in (quantizer, f16_path, q4_path, qtype)],
                       stdout=log, stderr=subprocess.STDOUT)
    print(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-1])
    if not q4_path.exists() or q4_path.stat().st_size == 0:
        sys.exit(f"[quantize] 量化失败, 详见 {log_path}")

    f16_mb = f16_path.stat().st_size / 1e6
    q4_mb = q4_path.stat().st_size / 1e6
    print(f"[quantize] 完成! F16 {f16_mb:.1f} MB -> {qtype} {q4_mb:.1f} MB "
          f"(压缩到 {q4_mb / f16_mb:.0%})")
    print(f"[quantize] 量化产物: {q4_path}")

    # ---------- 阶段报告 (一次运行可能覆盖 export 与 quantize 两个节点) ----------
    if args.skip_convert:
        stage_report(
            "quantize",
            summary=f"用 `{qtype}` 把 F16 权重压到 **{q4_mb:.1f} MB**（原始 {f16_mb:.1f} MB，"
                    f"压缩到 {q4_mb / f16_mb:.0%}）。",
            metrics={
                "量化方案": qtype,
                "输入 (F16)": f"{f16_mb:.2f} MB",
                "输出": f"{q4_mb:.2f} MB",
                "压缩比": f"{q4_mb / f16_mb:.1%}",
                "量化器": str(quantizer),
            },
            artifacts=[q4_path, f16_path],
            logs=[log_path],
            notes="这是基础链路的单方案量化。要看多方案体积/精度对比并用 imatrix 校准，"
                  "运行「量化方案对比」节点（quantize_sweep.py）。",
        )
    else:
        stage_report(
            "export",
            summary=f"HF 权重导出为 GGUF（F16，**{f16_mb:.1f} MB**），"
                    f"随后量化为 {qtype}（{q4_mb:.1f} MB）。",
            metrics={
                "HF 目录": str(hf_dir.relative_to(ROOT).as_posix()),
                "F16 产物": f"{f16_mb:.2f} MB",
                "量化方案": qtype,
                "量化产物": f"{q4_mb:.2f} MB",
            },
            artifacts=[f16_path, q4_path],
            logs=[log_path],
            notes="导出用项目自带的 `pipeline/write_gguf.py`（不用官方 convert_hf_to_gguf.py，"
                  "因为它用哈希白名单识别分词器，认不出自训词表）。",
        )


if __name__ == "__main__":
    main()
