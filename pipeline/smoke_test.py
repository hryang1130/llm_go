#!/usr/bin/env python3
"""
冒烟测试:
  --hf       用 transformers 直接加载 HF 模型做生成 (量化前验证训练效果)
  --gguf     用 llama.cpp 的 llama-server 验证量化模型 (需服务已启动)

用法:
  python pipeline/smoke_test.py --hf
  python pipeline/smoke_test.py --gguf --url http://127.0.0.1:8081
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "pipeline" / "config.yaml"

PROMPTS = ["人工智", "深度学", "The transformer"]


def load_config() -> dict:
    import yaml
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_hf(cfg: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_dir = ROOT / cfg["paths"]["output_dir"]
    tok = AutoTokenizer.from_pretrained(str(hf_dir))
    model = AutoModelForCausalLM.from_pretrained(str(hf_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"[smoke] HF 模型加载成功, 设备 {device}")
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").to(device)
        out = model.generate(**ids, max_new_tokens=40, do_sample=True,
                             temperature=0.8, top_k=50)
        print(f"  prompt: {p!r} -> {tok.decode(out[0][ids['input_ids'].shape[1]:])}")


def test_gguf(cfg: dict, url: str):
    req = urllib.request.Request(
        url.rstrip("/") + "/completion",
        data=json.dumps({
            "prompt": PROMPTS[0],
            "n_predict": 40,
            "temperature": 0.8,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    print(f"[smoke] llama.cpp 服务 {url} 响应正常")
    print(f"  生成内容: {data.get('content', '')!r}")
    timings = data.get("timings", {})
    if timings:
        print(f"  生成速度: {timings.get('predicted_per_second', 0):.1f} tok/s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf", action="store_true", help="测试 HF 模型")
    parser.add_argument("--gguf", action="store_true", help="测试 llama.cpp 服务")
    parser.add_argument("--url", default=None, help="llama-server 地址")
    args = parser.parse_args()

    cfg = load_config()
    if args.hf:
        test_hf(cfg)
    if args.gguf:
        test_gguf(cfg, args.url or cfg["serve"]["llama_server_url"])
    if not args.hf and not args.gguf:
        parser.print_help()


if __name__ == "__main__":
    main()
