#!/usr/bin/env python3
"""
公共工具 —— 配置加载 / llama.cpp 二进制定位 / 命令执行 / 报告输出

新增的量化扫、评测、投机解码三个脚本都依赖这里的实现, 保证:
  * 二进制查找顺序与 export_gguf.py 一致 (显式参数 > 环境变量 > 常见目录 > PATH)
  * 所有子进程调用统一落日志, 便于在工作流界面里排查
  * 报告统一写成 Markdown, 可直接贴进 README 或简历

作者: llm_go 工作流
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "pipeline" / "config.yaml"

# llama.cpp 各阶段二进制的候选文件名 (release 包与源码编译产物命名略有差异)
BINARY_ALIASES = {
    "quantize": ("llama-quantize.exe", "llama-quantize"),
    "server": ("llama-server.exe", "llama-server"),
    "perplexity": ("llama-perplexity.exe", "llama-perplexity"),
    "imatrix": ("llama-imatrix.exe", "llama-imatrix"),
    "bench": ("llama-bench.exe", "llama-bench"),
    "cli": ("llama-cli.exe", "llama-cli"),
}

# 缺失时的提示语
BINARY_HINTS = {
    "quantize": "量化器 (llama-quantize), 用于 F16 -> Q4_K_M 等低比特量化",
    "server": "推理服务 (llama-server), 评测与投机解码实验都通过它的 HTTP 接口取 timings",
    "perplexity": "困惑度工具 (llama-perplexity), 用于量化前后的精度对比",
    "imatrix": "重要性矩阵工具 (llama-imatrix), 用于生成校准数据的重要性矩阵",
    "bench": "基准工具 (llama-bench), 可选, 提供更细的 pp/tg 曲线",
    "cli": "命令行推理 (llama-cli), 可选",
}

INSTALL_HINT = (
    "从 https://github.com/ggml-org/llama.cpp/releases 下载对应平台的预编译包 "
    "(如 llama-bXXXX-bin-win-cpu-x64.zip), 解压后把 bin 目录里的二进制放到 "
    "llama.cpp/bin/, 或用 --llama-cpp / LLAMA_CPP_DIR 指定目录。"
)


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
def load_config(path: str | Path | None = None) -> dict:
    """读取全局配置 (默认 pipeline/config.yaml)。"""
    with open(path or CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cfg_get(cfg: dict, dotted: str, default=None):
    """按 "a.b.c" 路径读配置, 缺失时返回 default。"""
    cur = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def resolve_path(value: str | Path) -> Path:
    """把配置里的相对路径解析成相对仓库根目录的绝对路径。"""
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


# --------------------------------------------------------------------------- #
# llama.cpp 二进制定位
# --------------------------------------------------------------------------- #
def candidate_roots(explicit: str | Path | None = None) -> list[Path]:
    roots: list[Path] = []
    if explicit:
        roots.append(Path(explicit))
    env = os.environ.get("LLAMA_CPP_DIR")
    if env:
        roots.append(Path(env))
    roots += [
        Path("D:/tools/llama.cpp"),
        Path("C:/tools/llama.cpp"),
        Path.home() / "llama.cpp",
        ROOT / "third_party" / "llama.cpp",
    ]
    return roots


def _search_dirs(root: Path) -> list[Path]:
    """release 包与源码编译的二进制常见位置。"""
    return [root / "build" / "bin", root / "bin", root, root / "build" / "bin" / "Release"]


def find_binary(kind: str, explicit: str | Path | None = None,
                root: str | Path | None = None) -> Path:
    """定位某个 llama.cpp 二进制, 找不到就抛出带安装提示的异常。"""
    found = try_find_binary(kind, explicit=explicit, root=root)
    if found:
        return found
    raise FileNotFoundError(
        f"找不到 {BINARY_HINTS.get(kind, kind)}。{INSTALL_HINT}"
    )


def try_find_binary(kind: str, explicit: str | Path | None = None,
                    root: str | Path | None = None) -> Path | None:
    """同 find_binary, 但找不到时返回 None (用于可选工具)。"""
    names = BINARY_ALIASES.get(kind, (kind,))
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        # 允许传入目录
        for n in names:
            if (p / n).is_file():
                return p / n
    roots = [Path(root)] if root else candidate_roots()
    for r in roots:
        for d in _search_dirs(r):
            for n in names:
                if (d / n).is_file():
                    return d / n
    for n in names:
        w = shutil.which(n)
        if w:
            return Path(w)
    return None


def binary_report(explicit: str | Path | None = None) -> dict[str, str | None]:
    """--check 用: 列出各二进制的定位结果。"""
    out: dict[str, str | None] = {}
    for kind in BINARY_ALIASES:
        p = try_find_binary(kind, explicit=explicit)
        out[kind] = str(p) if p else None
    return out


# --------------------------------------------------------------------------- #
# 子进程
# --------------------------------------------------------------------------- #
def run(cmd: list, log_path: Path | None = None, check: bool = True,
        env: dict | None = None, timeout: int | None = None) -> subprocess.CompletedProcess:
    """执行命令: 同时打到控制台与日志文件 (统一 UTF-8, 忽略坏字节)。"""
    cmd = [str(c) for c in cmd]
    print("+", " ".join(cmd), flush=True)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8", errors="replace") as log:
            log.write("+ " + " ".join(cmd) + "\n")
            log.flush()
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                  env=env, timeout=timeout)
    else:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace",
                              env=env, timeout=timeout)
        if proc.stdout:
            print(proc.stdout, end="", flush=True)
    if check and proc.returncode != 0:
        tip = f", 详见 {log_path}" if log_path else ""
        raise RuntimeError(f"命令失败 (exit {proc.returncode}){tip}: {' '.join(cmd)}")
    return proc


def read_log(path: Path, tail: int | None = None) -> str:
    if not Path(path).exists():
        return ""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if tail:
        return "\n".join(text.splitlines()[-tail:])
    return text


# --------------------------------------------------------------------------- #
# HTTP / 进程辅助
# --------------------------------------------------------------------------- #
def free_port(start: int = 18080) -> int:
    """找一个空闲端口, 避免与已有 llama-server 冲突。"""
    port = start
    while port < start + 200:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1
    raise RuntimeError("找不到空闲端口")


def wait_for_http(url: str, timeout: float = 90.0) -> bool:
    """轮询直到服务可用 (llama-server 启动需要加载权重的时间)。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3):
                return True
        except urllib.error.HTTPError:
            return True          # 有响应即视为可用
        except Exception:
            time.sleep(0.5)
    return False


def http_json(url: str, payload: dict | None = None, timeout: int = 180):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def http_text(url: str, timeout: int = 30) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8", "replace")
    except Exception:
        return ""


def process_memory_mb(pid: int) -> float | None:
    """进程驻留内存 (MB)。优先用 psutil, 没有就返回 None。"""
    try:
        import psutil  # type: ignore
        return psutil.Process(pid).memory_info().rss / 1e6
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# 报告输出
# --------------------------------------------------------------------------- #
def size_mb(path: str | Path) -> float:
    p = Path(path)
    return p.stat().st_size / 1e6 if p.exists() else float("nan")


def markdown_table(headers: list[str], rows: list[list]) -> str:
    """把二维数据渲染成 Markdown 表格。"""
    def cell(v) -> str:
        if v is None:
            return "—"
        if isinstance(v, float):
            return "—" if v != v else f"{v:,.2f}".rstrip("0").rstrip(".")
        return str(v)

    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(cell(v) for v in r) + " |")
    return "\n".join(out)


def write_report(path: str | Path, title: str, body: str) -> Path:
    """写一份带标题与生成时间的 Markdown 报告。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    p.write_text(f"# {title}\n\n_生成时间: {stamp}_\n\n{body}\n",
                 encoding="utf-8", newline="\n")
    print(f"[report] 已写入 {p}")
    return p


def parse_ppl(text: str) -> float | None:
    """从 llama-perplexity 输出里抓最终困惑度。"""
    m = re.findall(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)", text)
    if m:
        return float(m[-1])
    m = re.findall(r"\[\d+\]([0-9.]+)", text)      # 逐块输出取最后一个
    return float(m[-1]) if m else None
