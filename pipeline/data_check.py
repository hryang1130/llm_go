#!/usr/bin/env python3
"""数据准备节点: 检查训练语料的规模与质量 (仅标准库, 无依赖)"""

import sys
import re
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("用法: data_check.py <corpus_path>")
        sys.exit(1)
    corpus = Path(sys.argv[1])
    if not corpus.exists():
        print(f"[data] 错误: 语料文件不存在: {corpus}")
        sys.exit(1)

    text = corpus.read_text(encoding="utf-8")
    lines = [l for l in text.splitlines() if l.strip()]
    chars = len(text)
    # 简单质量检查: 空行比例、重复行
    dup_ratio = 1 - len(set(lines)) / max(len(lines), 1)

    print(f"[data] 语料文件: {corpus}")
    print(f"[data] 总字符数: {chars}")
    print(f"[data] 非空行数: {len(lines)}")
    print(f"[data] 重复行比例: {dup_ratio:.1%}")

    if chars < 1000:
        print("[data] 警告: 语料过少 (<1000 字符), 模型将学不到什么")
    if dup_ratio > 0.5:
        print("[data] 警告: 重复行超过一半, 建议去重")
    # 中英文字符分布
    zh = len(re.findall(r"[\u4e00-\u9fff]", text))
    en = len(re.findall(r"[a-zA-Z]", text))
    print(f"[data] 中文字符: {zh} / 英文字母: {en}")
    print("[data] 数据准备完成 ✓")


if __name__ == "__main__":
    main()
