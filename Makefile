# ================= llm_go 全链路工作流 =================
# 用法: make <target>    (需要 make; Windows 可用 scoop/choco 安装或直接执行 make 中的命令)
#
#   make setup        安装 Python 依赖
#   make train        阶段1: 从零训练小型 LLaMA (draft)
#   make train-target 训练第二档模型 (target, 复用分词器; 投机解码用)
#   make export       阶段2: HF -> GGUF (F16)      需要 llama.cpp 仓库
#   make quantize     阶段3: F16 -> Q4_K_M          需要 llama-quantize
#   make sweep        量化精度工程: imatrix 校准 + 多方案对比
#   make sft          阶段1.5: LoRA 指令微调 (SFT)
#   make eval         评测: PPL + 延迟 + 吞吐 (+ KV cache 量化)
#   make spec         实验: 投机解码加速比
#   make check        环境自检 (量化 / 评测 / 投机解码 / SFT)
#   make smoke        冒烟测试 HF 模型
#   make serve        启动 llama-server + Go 网关
#   make gateway      只启动 Go 网关
#   make clean        清理训练与导出产物

PYTHON  ?= python
LLAMA_CPP ?= D:/tools/llama.cpp          # 修改为你的 llama.cpp 路径, 或用环境变量 LLAMA_CPP_DIR

.PHONY: setup train train-target export quantize sweep sft eval spec check smoke serve gateway clean

setup:
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) pipeline/train.py

train-target:
	$(PYTHON) pipeline/train.py --profile target --reuse-tokenizer

export:
	$(PYTHON) pipeline/export_gguf.py --llama-cpp $(LLAMA_CPP)

quantize:
	$(PYTHON) pipeline/export_gguf.py --llama-cpp $(LLAMA_CPP) --skip-convert

# imatrix 校准 + Q8_0/Q4_K_M/Q4_K_S/IQ4_XS 体积对比 -> out/quantize_sweep.md
sweep:
	$(PYTHON) pipeline/quantize_sweep.py --llama-cpp $(LLAMA_CPP) --imatrix auto

# LoRA 指令微调 -> models/tinyllm-hf-sft
sft:
	$(PYTHON) pipeline/sft.py

# PPL / TTFT / 解码吞吐 / KV cache 量化 -> out/benchmark.md
eval:
	$(PYTHON) pipeline/eval.py --llama-cpp $(LLAMA_CPP) --kv-quant

# 投机解码加速比 -> out/spec_decode.md
spec:
	$(PYTHON) pipeline/spec_decode.py --llama-cpp $(LLAMA_CPP)

check:
	$(PYTHON) pipeline/quantize_sweep.py --check
	$(PYTHON) pipeline/eval.py --check
	$(PYTHON) pipeline/spec_decode.py --check
	$(PYTHON) pipeline/sft.py --check

smoke:
	$(PYTHON) pipeline/smoke_test.py --hf

serve:
	llama-server -m models/tinyllm-q4_k_m.gguf --host 127.0.0.1 --port 8081 & \
	cd server && go run .

gateway:
	cd server && go run .

clean:
	rm -rf out models/tinyllm-hf models/tinyllm-target-hf models/tinyllm-hf-sft models/tokenizer *.gguf server/llm-gateway.exe
