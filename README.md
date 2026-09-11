# llm_go —— LLM 全链路工作流: 训练 → 推理 → 量化 → 部署

从零开始训练一个小型 LLaMA 架构语言模型，导出为 GGUF，用 llama.cpp 量化压缩，最后通过 Go 网关对外提供 OpenAI 兼容的推理服务。

## 架构总览

```
┌─────────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────────┐
│ 阶段1 训练   │──▶│ 阶段2 导出    │──▶│ 阶段3 量化     │──▶│ 阶段4 部署    │
│ pipeline/   │   │ HF → GGUF    │   │ F16 → Q4_K_M  │   │ Go 网关 +    │
│ train.py    │   │ convert_hf_  │   │ llama-        │   │ llama-server │
│ (transformers)  │ to_gguf.py   │   │ quantize      │   │ (Gin, OpenAI │
│             │   │ (llama.cpp)  │   │               │   │  兼容 API)   │
└─────────────┘   └──────────────┘   └───────────────┘   └──────────────┘
```

模型规模约 **3~4M 参数**（hidden 256 / 4 层 / GQA），CPU 即可完成训练，用来走通全流程。

## 目录结构

```
llm_go/
├── data/corpus.txt          # 训练语料 (中英双语示例, 可替换)
├── pipeline/
│   ├── config.yaml          # 全局配置: 模型结构 / 训练超参 / 路径 / 量化方案
│   ├── train.py             # 阶段1: BPE 分词 + 从零训练 LLaMA
│   ├── export_gguf.py       # 阶段2+3: HF→GGUF(F16) + 量化(Q4_K_M)
│   └── smoke_test.py        # 冒烟测试 (--hf 测 HF 模型 / --gguf 测服务)
├── server/
│   ├── main.go              # 阶段4: Go 推理网关 (Gin)
│   └── go.mod
├── Dockerfile.pipeline      # 训练镜像
├── Dockerfile.server        # 网关镜像
├── docker-compose.yml       # llama-server + gateway 编排
└── Makefile
```

## 可视化工作流界面 (推荐入口)

不敲命令也能跑通全流程——内置节点编辑器，把流水线画在画布上：

```bash
cd server && go run .          # 或运行编译好的 llm-gateway.exe
# 浏览器打开 http://localhost:8080/web/
```

界面功能：

- **节点画布**：数据准备 → 模型训练 → 导出 GGUF → 量化 → 启动推理服务 → 冒烟测试，六个节点按拓扑序连线
- **自由编排**：拖拽节点调整位置，从右侧圆点拖到下一节点左侧圆点即可重新连线；滚轮缩放、空白处拖拽平移
- **参数面板**：点击节点编辑参数（训练轮数、批大小、llama.cpp 路径、量化方案、服务端口、测试 Prompt 等），也可「仅运行此节点」
- **实时日志**：点击「▶ 运行全部」后，后端按连线拓扑序逐节点执行，每个节点的 stdout 通过 SSE 实时显示在节点卡片内
- **运行控制**：随时「■ 停止」（会同时杀掉 llama-server 子进程），布局/连线/参数自动保存在浏览器本地

对应后端 API：`GET /api/workflow`（节点定义）、`POST /api/run`（SSE 执行日志）、`POST /api/cancel`

## 快速开始

### 0. 准备环境

```bash
pip install -r requirements.txt      # Python 3.10+, PyTorch CPU 即可
go version                           # Go 1.22+ (网关)

# 获取 llama.cpp (转换脚本在仓库里; 量化器可从 release 下载)
git clone https://github.com/ggml-org/llama.cpp D:/tools/llama.cpp
# 量化器: 从 https://github.com/ggml-org/llama.cpp/releases 下载对应平台包,
# 把 llama-quantize(.exe) 放入 D:/tools/llama.cpp/bin/
```

### 1. 训练

```bash
python pipeline/train.py                 # 默认 30 epochs, CPU 几分钟
python pipeline/train.py --epochs 60     # 更久 -> 更低 loss
python pipeline/smoke_test.py --hf       # 验证生成效果
```

产物: `models/tinyllm-hf/`（HuggingFace 格式）+ `models/tokenizer/`

### 2. 导出 GGUF (F16)

```bash
python pipeline/export_gguf.py --llama-cpp D:/tools/llama.cpp
```

产物: `models/tinyllm-f16.gguf`

> 导出使用项目自带的 `pipeline/write_gguf.py`（基于 gguf-py 手工写出 llama 架构 GGUF）。
> 不用官方 `convert_hf_to_gguf.py` 的原因：它通过哈希白名单识别 BPE 分词器，
> 从零自训的词表永远不在名单里，会报 "BPE pre-tokenizer was not recognized"。

### 3. 量化 (Q4_K_M)

```bash
python pipeline/export_gguf.py --llama-cpp D:/tools/llama.cpp --skip-convert
```

产物: `models/tinyllm-q4_k_m.gguf`（体积约为 F16 的 1/4）

### 4. 部署

```bash
# 终端 1: 启动 llama.cpp 推理服务
llama-server -m models/tinyllm-q4_k_m.gguf --host 127.0.0.1 --port 8081 --ctx-size 512

# 终端 2: 启动 Go 网关
cd server && go run .
```

测试:

```bash
curl http://localhost:8080/healthz

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"人工智能是什么"}],"max_tokens":64}'

# 流式输出 (SSE)
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"深度学习"}],"stream":true}'
```

Go 网关环境变量: `LLAMA_SERVER_URL`(默认 http://127.0.0.1:8081)、`GATEWAY_PORT`(默认 8080)、`DEFAULT_MAX_TOKENS`、`DEFAULT_TEMP`

### Docker 部署 (可选)

```bash
docker compose up -d          # 启动 llama-server + gateway
docker compose run pipeline   # 一键训练+导出 (需挂载 llama.cpp 仓库, 见 compose 注释)
```

## 换成自己的模型/数据

- **换语料**: 替换 `data/corpus.txt`，语料越多模型越"像话"（建议至少几百 KB 纯文本）
- **调模型**: 改 `pipeline/config.yaml` 的 `model` 段（层数/维度），注意 CPU 训练时间会随之增长
- **换量化方案**: 改 `config.yaml` 的 `quantize.type`（Q8_0 更准、Q4_K_S 更小）
- **常见坑**:
  - Trainer 需要 `accelerate`（已写入 requirements.txt）
  - llama-quantize 在输出重定向到管道时可能报 iostream 错误并返回非零，但产物已生成——`export_gguf.py` 已按产物判断成败
  - 新版 llama-server 对非流式 `/completion` 输出做严格 UTF-8 校验，小模型偶发的坏字节会 500；流式请求不受影响（工作流测试节点已用流式）
- **上 LoRA 微调**: 把 train.py 换成 peft 的 LoRA 训练 HF 现成模型（如 Qwen3-0.6B），后续导出/量化/部署流程完全复用
