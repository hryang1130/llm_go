#!/usr/bin/env python3
"""数据准备节点: 检查训练语料的规模与质量 (仅标准库, 无依赖)"""

import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import stage_report  # noqa: E402  (common 本身不依赖第三方库)


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

    warnings = []
    if chars < 1000:
        warnings.append("语料过少 (<1000 字符), 模型将学不到什么")
        print("[data] 警告: 语料过少 (<1000 字符), 模型将学不到什么")
    if dup_ratio > 0.5:
        warnings.append("重复行超过一半, 建议去重")
        print("[data] 警告: 重复行超过一半, 建议去重")
    # 中英文字符分布
    zh = len(re.findall(r"[\u4e00-\u9fff]", text))
    en = len(re.findall(r"[a-zA-Z]", text))
    print(f"[data] 中文字符: {zh} / 英文字母: {en}")
    print("[data] 数据准备完成 ✓")

    stage_report(
        "data",
        summary=f"语料 `{corpus.name}` 共 {chars:,} 字符 / {len(lines)} 非空行，"
                + ("质量检查通过。" if not warnings else "存在告警，见备注。"),
        metrics={
            "语料文件": str(corpus),
            "总字符数": f"{chars:,}",
            "非空行数": len(lines),
            "重复行比例": f"{dup_ratio:.1%}",
            "中文字符": zh,
            "英文字母": en,
        },
        tables=[("字符构成", common_pie(zh, en, chars))],
        artifacts=[corpus],
        notes=("**告警**\n\n" + "\n".join(f"- {w}" for w in warnings)) if warnings else
              "语料规模与重复度均在合理范围。继续训练即可；更换语料时建议保持 ≥ 几百 KB 纯文本。",
    )


def common_pie(zh: int, en: int, total: int) -> str:
    """字符构成的小表格 (纯标准库实现, 不依赖 common 的表格工具)。"""
    other = max(total - zh - en, 0)
    rows = [["中文", zh, f"{zh / max(total, 1):.1%}"],
            ["英文", en, f"{en / max(total, 1):.1%}"],
            ["其他(标点/空格)", other, f"{other / max(total, 1):.1%}"]]
    out = ["| 类型 | 字符数 | 占比 |", "|---|---|---|"]
    out += [f"| {a} | {b:,} | {c} |" for a, b, c in rows]
    return "\n".join(out)


if __name__ == "__main__":
    main()
