# llm_go —— LLM 全链路工作流: 训练 → 推理 → 量化 → 部署

从零开始训练一个小型 LLaMA 架构语言模型，导出为 GGUF，用 llama.cpp 量化压缩，最后通过 Go 网关对外提供 OpenAI 兼容的推理服务。

## 架构总览

```
阶段1    训练      train.py            BPE 分词 + 从零训练 LLaMA (兼作投机解码的 draft)
阶段1.5  后训练    sft.py              LoRA 指令微调 (SFT), 合并回 HF 权重后复用后续链路
阶段2    导出      write_gguf.py       HF → GGUF (F16); 自写导出器绕过官方词表白名单
阶段3    量化      export_gguf.py      F16 → Q4_K_M (基础链路)
         量化工程  quantize_sweep.py   imatrix 校准 + Q8_0/Q4_K_M/Q4_K_S/IQ4_XS 体积对比
阶段4    部署      Go 网关 + llama-server  OpenAI 兼容 API, 流式输出, Docker 编排
         评测      eval.py             PPL / 首 token 延迟 / 解码吞吐 / KV cache 量化
         加速      spec_decode.py      draft + target 投机解码, 输出加速比与接受率
```

```
┌──────────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────────┐
│ 阶段1 训练    │──▶│ 阶段2 导出    │──▶│ 阶段3 量化     │──▶│ 阶段4 部署    │
│ train.py     │   │ write_gguf.py│   │ llama-quantize│   │ Go 网关 +    │
│ (transformers)│  │ HF → GGUF    │   │ (+ imatrix)   │   │ llama-server │
└──────────────┘   └──────────────┘   └───────────────┘   └──────────────┘
        │                                     ▲                   │
        │ 阶段1.5 sft.py                      │ quantize_sweep.py │ eval.py
        └──────────────▶ HF(微调后) ──────────┘                   ▼
                                                    spec_decode.py (draft+target 投机解码)
```

- **draft 模型**：默认的 3~4M 参数小模型（hidden 256 / 4 层 / GQA），CPU 即可训练。
- **target 模型**：`model_target` 段定义的更大一档（hidden 512 / 6 层），与 draft 共用同一分词器，
  供投机解码实验使用；同样可在 CPU 上训练。

## 目录结构

```
llm_go/
├── data/
│   ├── corpus.txt           # 训练语料 (中英双语示例, 可替换)
│   ├── eval.txt             # 评测文本 (困惑度用, 建议换成自己的留出语料)
│   └── sft_sample.jsonl     # 指令微调样例数据 (instruction/input/output)
├── pipeline/
│   ├── config.yaml          # 全局配置: 模型结构 / 训练超参 / 路径 / 量化 / 评测 / SFT
│   ├── common.py            # 公共工具: 二进制定位 / 命令执行 / Markdown 报告
│   ├── train.py             # 阶段1: BPE 分词 + 从零训练 (--profile tiny|target)
│   ├── sft.py               # 阶段1.5: LoRA 指令微调
│   ├── export_gguf.py       # 阶段2+3: 编排导出与量化
│   ├── write_gguf.py        # 阶段2: 自写 GGUF 导出器 (支持任意 HF 目录)
│   ├── quantize_sweep.py    # 阶段3+: imatrix 校准 + 多方案量化对比
│   ├── eval.py              # 阶段4+: PPL / 延迟 / 吞吐 / KV cache 量化评测
│   ├── spec_decode.py       # 阶段4+: 投机解码加速实验
│   └── smoke_test.py        # 冒烟测试 (--hf 测 HF 模型 / --gguf 测服务)
├── server/
│   ├── main.go              # 阶段4: Go 推理网关 (Gin)
│   ├── workflow.go          # 可视化工作流执行引擎 (含可选阶段)
│   └── web/index.html       # 节点编辑器前端
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

- **节点画布**：数据准备 → 模型训练 → 导出 GGUF → 量化 → 启动推理服务 → 冒烟测试，六个主流程节点按拓扑序连线
- **可选阶段**（虚线边框 + 「可选」标记）：训练 target 模型 / 指令微调 (LoRA) / 量化方案对比 / 评测基准 / 投机解码实验。
  这些阶段默认**不参与「运行全部」**，避免一键流程被额外依赖打断；点节点上的「仅运行此节点」单独执行
- **自由编排**：拖拽节点调整位置，从右侧圆点拖到下一节点左侧圆点即可重新连线；滚轮缩放、空白处拖拽平移
- **参数面板**：点击节点编辑参数（训练轮数、批大小、llama.cpp 路径、量化方案、服务端口、测试 Prompt、评测轮数等），也可「仅运行此节点」
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

## 进阶阶段：量化精度 / 评测 / 投机解码 / 后训练

主流程跑通之后，这四个阶段用来回答端侧部署真正关心的问题：**压到多少、掉多少精度、跑多快、怎么再快一点**。

先做一次环境自检（缺什么会直接告诉你）：

```bash
make check
# 或逐个: python pipeline/quantize_sweep.py --check / eval.py --check / spec_decode.py --check / sft.py --check
```

### 1. 量化精度工程：imatrix 校准 + 多方案对比

`llama-imatrix` 用校准语料统计每个权重张量对输出的重要度，`llama-quantize --imatrix` 再据此分配精度。
同样是 4bit，带校准的方案困惑度明显更接近 F16 —— 这正是"低比特量化下保住精度"的常用手段。

```bash
make sweep
# 等价于:
python pipeline/quantize_sweep.py --llama-cpp D:/tools/llama.cpp \
    --schemes Q8_0,Q4_K_M,Q4_K_S,IQ4_XS --imatrix auto --calibration data/corpus.txt
```

产物：

| 文件 | 内容 |
|------|------|
| `models/<模型>-<方案>.gguf` | 各方案的量化模型 |
| `models/imatrix.dat` | 校准得到的重要性矩阵（可复用） |
| `out/quantize_sweep.md` | 体积 / 压缩倍数 / 是否用校准 / 耗时 对比表 |
| `out/quantize-<方案>.log` | 每个方案的完整量化日志 |

要点：`--imatrix none` 可关掉校准做对照；`--force` 重跑已存在的产物；imatrix 失败会自动降级为普通量化（加 `--imatrix-strict` 可改为直接报错）。

### 2. 评测基准：PPL / 延迟 / 吞吐 / KV cache 量化

把"压缩到 1/4"变成一张可核对的表：

```bash
make eval
# 等价于:
python pipeline/eval.py --llama-cpp D:/tools/llama.cpp --kv-quant --rounds 3
```

| 指标 | 含义 | 来源 |
|------|------|------|
| PPL | 量化掉了多少精度 | `llama-perplexity -f data/eval.txt`（缺该工具时可用 `--hf-dir` 走 transformers 兜底） |
| TTFT | 首 token 延迟（预填充耗时） | `llama-server` 返回的 `timings.prompt_ms` |
| 预填充吞吐 | 长 prompt 的处理速度 | `timings.prompt_per_second` |
| 解码吞吐 | 生成速度（端侧最关心） | `timings.predicted_per_second`，多轮取中位数 |
| KV q8_0 解码 | 开启 KV cache 量化后的速度 | 以 `--cache-type-k/v q8_0` 再跑一组 |
| 服务端 RSS | 常驻内存 | 有 `psutil` 时自动读取 |

产物：`out/benchmark.md`（含与 F16 基线的相对变化）与 `out/benchmark.json`（可画曲线）。
用 `--models a.gguf b.gguf` 指定对比对象，默认扫 `models/` 下全部 GGUF。

### 3. 投机解码：draft + target 双模型

小模型快速猜测、大模型批量校验，猜中即省下大模型的自回归步数 —— 不牺牲精度的加速手段。

前提是 **draft 与 target 必须共用同一分词器**，所以先训练一个更大一档的 target：

```bash
# ① 训练 target（复用已有分词器，保证词表一致）
make train-target
# ② 导出 + 量化（沿用同一套脚本）
python pipeline/write_gguf.py --hf-dir models/tinyllm-target-hf \
    --out models/tinyllm-target-f16.gguf --name tinyllm-target
python pipeline/quantize_sweep.py --input models/tinyllm-target-f16.gguf --schemes Q4_K_M
# ③ 跑对比实验
make spec
```

产物 `out/spec_decode.md`：基线 vs 投机解码的解码吞吐、TTFT、加速比与 **接受率**（从 llama-server 的 `/metrics` 读取）。
脚本会先校验两个模型的词表大小，不一致直接拦下（否则投机解码不会有效果）。

### 4. 后训练：LoRA 指令微调

预训练模型只会续写，指令微调让它学会"回答问题"。LoRA 只训练低秩适配器，参数量占比通常不足 1%：

```bash
make sft
# 快速验证链路（只跑 20 步）:
python pipeline/sft.py --steps 20
# 只出适配器不合并:
python pipeline/sft.py --no-merge
```

数据格式（`data/sft_sample.jsonl`，一行一条）：

```json
{"instruction": "什么是模型量化？", "input": "", "output": "模型量化是用更低的比特宽度表示权重……"}
```

产物：`models/tinyllm-hf-sft/`（合并后的 HF 模型，可直接导出量化）与 `out/lora-adapter/`（体积很小的适配器）。
微调后的模型走同一条导出/量化/部署链路即可，例如：

```bash
python pipeline/write_gguf.py --hf-dir models/tinyllm-hf-sft --out models/tinyllm-sft-f16.gguf
python pipeline/quantize_sweep.py --input models/tinyllm-sft-f16.gguf --schemes Q4_K_M
```

### 推荐的实验顺序（把数字攒齐）

```text
make train → make export → make quantize        # 打底：能跑起来
make sweep                                      # 体积/压缩倍数对比（imatrix vs 不校准）
make eval                                       # 精度(PPL) - 延迟 - 吞吐 三维对比 + KV 量化
make train-target → 导出量化 target → make spec   # 拿到投机解码加速比与接受率
make sft → 导出量化 → 再 make eval                # 对比微调前后的困惑度与生成质量
```

每一步的 Markdown 报告都在 `out/` 下，可以直接作为实验记录或写进技术总结。

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
