# ================= llm_go 全链路工作流 =================
# 用法: make <target>    (需要 make; Windows 可用 scoop/choco 安装或直接执行 make 中的命令)
#
#   make setup      安装 Python 依赖
#   make train      阶段1: 从零训练小型 LLaMA
#   make export     阶段2: HF -> GGUF (F16)      需要 llama.cpp 仓库
#   make quantize   阶段3: F16 -> Q4_K_M          需要 llama-quantize
#   make smoke      冒烟测试 HF 模型
#   make serve      启动 llama-server + Go 网关
#   make gateway    只启动 Go 网关
#   make clean      清理训练与导出产物

PYTHON  ?= python
LLAMA_CPP ?= D:/tools/llama.cpp          # 修改为你的 llama.cpp 路径, 或用环境变量 LLAMA_CPP_DIR

.PHONY: setup train export quantize smoke serve gateway clean

setup:
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) pipeline/train.py

export:
	$(PYTHON) pipeline/export_gguf.py --llama-cpp $(LLAMA_CPP)

quantize:
	$(PYTHON) pipeline/export_gguf.py --llama-cpp $(LLAMA_CPP) --skip-convert

smoke:
	$(PYTHON) pipeline/smoke_test.py --hf

serve:
	llama-server -m models/tinyllm-q4_k_m.gguf --host 127.0.0.1 --port 8081 & \
	cd server && go run .

gateway:
	cd server && go run .

clean:
	rm -rf out models/tinyllm-hf models/tokenizer *.gguf server/llm-gateway.exe
