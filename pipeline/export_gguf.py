#!/usr/bin/env python3
"""
阶段 2+3/4 —— GGUF 导出与量化

  HF 模型 --(convert_hf_to_gguf.py)--> F16 GGUF --(llama-quantize)--> Q4_K_M GGUF

前置条件 (二选一):
  A. git clone https://github.com/ggml-org/llama.cpp 到本地 (推荐, 转换脚本在仓库里),
     量化器 llama-quantize 可以是源码编译的, 也可以是 release 下载的。
  B. 只下载 release 二进制: 无法转换格式, 但如果已有 GGUF 可直接量化。

用法:
  python pipeline/export_gguf.py --llama-cpp D:/tools/llama.cpp
  python pipeline/export_gguf.py --skip-convert          # 跳过转换, 只做量化
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "pipeline" / "config.yaml"


def load_config() -> dict:
    import yaml
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_llama_cpp(explicit: str | None) -> Path:
    """定位 llama.cpp 仓库目录。"""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("LLAMA_CPP_DIR")
    if env:
        candidates.append(Path(env))
    candidates += [
        Path("D:/tools/llama.cpp"),
        Path.home() / "llama.cpp",
        ROOT / "third_party" / "llama.cpp",
    ]
    for c in candidates:
        if (c / "convert_hf_to_gguf.py").exists():
            return c
    raise FileNotFoundError(
        "找不到 llama.cpp 仓库 (需要 convert_hf_to_gguf.py)。"
        "请先 git clone https://github.com/ggml-org/llama.cpp, "
        "然后用 --llama-cpp <路径> 指定, 或设置环境变量 LLAMA_CPP_DIR。"
    )


def find_quantize_binary(llama_cpp: Path, explicit: str | None) -> Path:
    """定位 llama-quantize 可执行文件。"""
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
    names = ["llama-quantize.exe", "llama-quantize"]
    search_dirs = [llama_cpp / "build" / "bin", llama_cpp / "bin", llama_cpp]
    # release 压缩包常见布局: llama.cpp/bin/llama-quantize.exe
    for d in search_dirs:
        for n in names:
            p = d / n
            if p.exists():
                return p
    w = shutil.which("llama-quantize")
    if w:
        return Path(w)
    raise FileNotFoundError(
        "找不到 llama-quantize。可从 https://github.com/ggml-org/llama.cpp/releases "
        "下载对应平台的预编译包, 把 llama-quantize(.exe) 放进 llama.cpp/bin/, "
        "或用 --quantize 指定完整路径。"
    )


def run(cmd: list[str], **kw):
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def main():
    parser = argparse.ArgumentParser(description="导出 GGUF 并量化")
    parser.add_argument("--llama-cpp", default=None, help="llama.cpp 仓库路径")
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

    llama_cpp = find_llama_cpp(args.llama_cpp)
    print(f"[export] llama.cpp 仓库: {llama_cpp}")

    f16_path = ROOT / paths["gguf_f16"]
    if not args.skip_convert:
        run([sys.executable, llama_cpp / "convert_hf_to_gguf.py",
             hf_dir, "--outfile", f16_path, "--outtype", "f16"])
    else:
        print(f"[export] 跳过转换, 使用现有 {f16_path}")
    if not f16_path.exists():
        sys.exit(f"[export] 转换失败: {f16_path} 不存在")

    quantizer = find_quantize_binary(llama_cpp, args.quantize)
    q4_path = ROOT / paths["gguf_q4"]
    print(f"[quantize] {f16_path.name} -> {q4_path.name} ({qtype})")
    run([quantizer, f16_path, q4_path, qtype])

    f16_mb = f16_path.stat().st_size / 1e6
    q4_mb = q4_path.stat().st_size / 1e6
    print(f"[quantize] 完成! F16 {f16_mb:.1f} MB -> {qtype} {q4_mb:.1f} MB "
          f"(压缩到 {q4_mb / f16_mb:.0%})")
    print(f"[quantize] 量化产物: {q4_path}")


if __name__ == "__main__":
    main()
