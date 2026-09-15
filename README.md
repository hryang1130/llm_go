# llm_go —— LLM 全链路工作流: 训练 → 推理 → 量化 → 部署

从零开始训练一个小型 LLaMA 架构语言模型，导出为 GGUF，用 llama.cpp 量化压缩，最后通过 Go 网关对外提供 OpenAI 兼容的推理服务。

## 架构总览

```
┌─────────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────────┐
│ 阶段1 训练   │──▶│ 阶段2 导出    │──▶│ 阶段3 量化     │──▶│ 阶段4 部署    │
│ pipeline/   │   │ HF → GGUF    │   │ F16 → Q4_K_M  │   │ Go 网关 +    │
│ train.py    │   │ write_gguf.py│   │ llama-        │   │ llama-server │
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
│   ├── export_gguf.py       # 阶段2+3: 编排导出与量化
│   ├── write_gguf.py        # 阶段2: 自写 GGUF 导出器 (基于 gguf-py)
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

### 一键启动脚本 (Windows)

| 脚本 | 作用 |
|------|------|
| `start.bat` | 双击启动网关并自动打开工作流界面（自动探测系统 Python、自动清理端口残留） |
| `stop.bat`  | 一键停止网关与 llama-server |

`start.bat` 按 `PYTHON_CMD 环境变量 → 系统 PATH 中的 python` 顺序探测 Python。如果你的 Python 不在 PATH 里，先设置再运行：

```bat
set PYTHON_CMD=D:\envs\llm\Scripts\python.exe
start.bat
```

> 只装 Python 不影响界面启动；训练/导出节点执行时才真正调用它。

## 手动安装部署教程

从零在一台新机器上跑通全流程（以 Windows 为例，Linux/macOS 同理，路径换成对应格式）。

### 1. 安装 Python 环境并装依赖

要求 **Python 3.10+**（CPU 训练即可，无需 GPU）。

```bash
# 1) 建议创建独立虚拟环境
python -m venv D:\envs\llm_go
D:\envs\llm_go\Scripts\activate          # Linux/macOS: source D:/envs/llm_go/bin/activate

# 2) 安装依赖 (CPU 版 torch 体积小、足够本项目使用)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

> Windows 如果 `python` 不在 PATH，后续所有 `python` 命令都要换成完整路径，
> 或者用 `set PYTHON_CMD=D:\envs\llm_go\Scripts\python.exe` 让工作流引擎使用它。

### 2. 安装 Go (1.22+)

从 https://go.dev/dl/ 下载安装，确认 `go version` ≥ 1.22。然后编译网关：

```bash
cd server
go mod tidy
go build -o llm-gateway.exe .    # Linux/macOS: go build -o llm-gateway .
```

### 3. 获取 llama.cpp 运行时

本项目**只需要两个二进制**：`llama-server`（推理服务）和 `llama-quantize`（量化器）。
导出环节用的是项目自带的 `pipeline/write_gguf.py`，不需要 clone llama.cpp 源码。

- 去 https://github.com/ggml-org/llama.cpp/releases 下载对应平台的包（如 `llama-bXXXX-bin-win-cpu-x64.zip`）
- 解压到任意目录，例如 `D:\tools\llama.cpp\bin\`（Linux/macOS 也可以自己编译：`cmake -B build && cmake --build build`）

```bash
# 验证两个二进制可用
D:/tools/llama.cpp/bin/llama-server.exe --version
D:/tools/llama.cpp/bin/llama-quantize.exe --version
```

### 4. 跑通流水线（命令行方式）

```bash
# ① 数据准备 + 训练
python pipeline/train.py --epochs 150

# ② 导出 GGUF (F16) + 量化 (Q4_K_M)
python pipeline/export_gguf.py --llama-cpp D:/tools/llama.cpp
# 产物: models/tinyllm-f16.gguf -> models/tinyllm-q4_k_m.gguf
# 注: 导出用项目自带 write_gguf.py 而非官方 convert_hf_to_gguf.py,
#     因为后者用哈希白名单识别分词器, 从零自训的词表无法通过识别

# ③ 启动推理服务 (终端 1)
D:/tools/llama.cpp/bin/llama-server.exe -m models/tinyllm-q4_k_m.gguf --host 127.0.0.1 --port 8081 --ctx-size 512

# ④ 启动 Go 网关 (终端 2)
cd server && ./llm-gateway.exe
```

验证：

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

### 5. 跑通流水线（可视化界面方式）

启动网关后打开 http://localhost:8080/web/ ，点「▶ 运行全部」即可。
使用前在界面上检查两处参数：

- **导出 GGUF 节点** → `llamacpp_dir`：填第 3 步的 llama.cpp 目录（如 `D:/tools/llama.cpp`）
- **模型训练节点** → 训练轮数等参数按需调整（默认 150）

网关的环境变量（可选）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PYTHON_CMD` | `python` | 工作流节点调用的 Python 解释器（**建议指向你的虚拟环境**） |
| `LLAMA_CPP_DIR` | — | llama.cpp 目录默认值（也可在界面参数里填） |
| `GATEWAY_PORT` | `8080` | 网关监听端口 |
| `LLAMA_SERVER_URL` | `http://127.0.0.1:8081` | 上游 llama-server 地址 |

其他网关参数: `DEFAULT_MAX_TOKENS`、`DEFAULT_TEMP`

### 6. Docker 部署 (可选)

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
