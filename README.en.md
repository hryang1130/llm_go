> 🌐 **English** | [简体中文](README.md)

# llm_go — Full-Stack LLM Workflow: Train → Inference → Quantize → Deploy

Train a small LLaMA-architecture language model from scratch, export it to GGUF, compress it with llama.cpp quantization, and serve it through a Go gateway that exposes an OpenAI-compatible inference API.

## Architecture at a glance

```
Stage 1    Train       train.py            BPE tokenizer + from-scratch LLaMA (doubles as the draft model)
Stage 1.5  Post-train  sft.py              LoRA instruction tuning (SFT); merged back into HF weights
Stage 2    Export      write_gguf.py       HF → GGUF (F16); a custom exporter sidesteps the vocab whitelist
Stage 3    Quantize    export_gguf.py      F16 → Q4_K_M (baseline path)
           Quant eng.  quantize_sweep.py   imatrix calibration + Q8_0/Q4_K_M/Q4_K_S/IQ4_XS size comparison
Stage 4    Deploy      Go gateway + llama-server  OpenAI-compatible API, streaming, Docker orchestration
           Evaluate    eval.py              PPL / time-to-first-token / decode throughput / KV-cache quantization
           Speed up    spec_decode.py       draft + target speculative decoding; speedup and acceptance rate
```

```
┌──────────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────────┐
│ Stage 1      │──▶│ Stage 2      │──▶│ Stage 3       │──▶│ Stage 4      │
│ train.py     │   │ write_gguf.py│   │ llama-quantize│   │ Go gateway + │
│ (transformers)│  │ HF → GGUF    │   │ (+ imatrix)   │   │ llama-server │
└──────────────┘   └──────────────┘   └───────────────┘   └──────────────┘
        │                                     ▲                   │
        │ Stage 1.5 sft.py                    │ quantize_sweep.py │ eval.py
        └──────────────▶ HF (fine-tuned) ─────┘                   ▼
                                                 spec_decode.py (draft+target speculative decoding)
```

- **draft model**: the default 3–4M parameter model (hidden 256 / 4 layers / GQA), trainable on CPU.
- **target model**: a larger variant defined in the `model_target` section (hidden 512 / 6 layers).
  It shares the same tokenizer as the draft model and exists for speculative decoding experiments —
  also trainable on CPU.

## Repository layout

```
llm_go/
├── data/
│   ├── corpus.txt           # Training corpus (bilingual sample; replace it)
│   ├── eval.txt             # Evaluation text for perplexity (use your own held-out data)
│   └── sft_sample.jsonl     # Instruction-tuning samples (instruction/input/output)
├── pipeline/
│   ├── config.yaml          # Global config: model shape / hyperparameters / paths / quantize / eval / SFT
│   ├── common.py            # Shared helpers: binary lookup / command execution / Markdown reports
│   ├── train.py             # Stage 1: BPE tokenizer + from-scratch training (--profile tiny|target)
│   ├── sft.py               # Stage 1.5: LoRA instruction tuning
│   ├── export_gguf.py       # Stage 2+3: orchestrates export and quantization
│   ├── write_gguf.py        # Stage 2: custom GGUF exporter (works with any HF directory)
│   ├── quantize_sweep.py    # Stage 3+: imatrix calibration + multi-scheme quantization comparison
│   ├── eval.py              # Stage 4+: PPL / latency / throughput / KV-cache quantization benchmark
│   ├── spec_decode.py       # Stage 4+: speculative decoding speedup experiment
│   └── smoke_test.py        # Smoke test (--hf for the HF model / --gguf for the served endpoint)
├── server/
│   ├── main.go              # Stage 4: Go inference gateway (Gin)
│   ├── workflow.go          # Visual workflow execution engine (including optional stages)
│   └── web/index.html       # Node-editor front end
├── Dockerfile.pipeline      # Training image
├── Dockerfile.server        # Gateway image
├── docker-compose.yml       # llama-server + gateway orchestration
└── Makefile
```

## Visual workflow UI (recommended entry point)

You can run the whole pipeline without typing a single command — a built-in node editor lets you
lay the pipeline out on a canvas:

```bash
cd server && go run .          # or run the compiled llm-gateway.exe
# then open http://localhost:8080/web/
```

What the UI gives you:

- **Two ways to compose the pipeline**: the **Canvas** view is the original node editor (drag to connect);
  the **List** view lays the stages out in execution order — tick which ones to include, use ↑↓ to
  reorder (the main chain is rewired automatically), and reach *params / run / report* inline for each
  stage. Use it if you would rather not drag anything.
- **Preset templates**: one click switches between *Quick start / Full experiment / Accuracy first /
  Edge speed / Custom*, ticking the right stages and ordering them for you.
- **Per-stage reports**: every stage writes a Markdown report to `out/reports/<stage>.md` when it
  finishes (key metrics, artifact sizes, logs and notes), plus a `run.md` run summary. Open the
  drawer with the 📄 button in the header, or jump straight there from the *📄 report* button that
  appears on each node card.
- **Light/dark theme**: the ☀️/🌙 toggle in the header, remembered in `localStorage`.
- **Node canvas**: six main stages — data prep → train → export GGUF → quantize → start inference
  server → smoke test — connected in topological order.
- **Optional stages** (dashed border + "optional" tag): train the target model / LoRA instruction
  tuning / quantization scheme comparison / evaluation benchmark / speculative decoding experiment.
  These are **excluded from "Run all" by default** so that one click never stalls on an extra
  dependency; run them individually with "Run this node only".
- **Free-form editing**: drag nodes to reposition them, drag from the right-hand dot onto the next
  node's left-hand dot to reconnect; scroll to zoom, drag empty space to pan.
- **Parameter panel**: click a node to edit its parameters (epochs, batch size, llama.cpp path,
  quantization scheme, server port, test prompt, eval rounds, …) and to run just that node.
- **Live logs**: after you press "▶ Run all", the backend executes nodes in topological order and
  streams each node's stdout into its card over SSE.
- **Run control**: hit "■ Stop" at any time (this also kills the llama-server child process).
  Layout, edges and parameters are persisted in the browser.

Corresponding backend API: `GET /api/workflow` (node definitions), `POST /api/run` (SSE execution
logs), `POST /api/cancel`, `GET /api/reports` (report list), `GET /api/reports/content?name=<file>`
(raw report text).

### One-click launcher scripts (Windows)

| Script | Purpose |
|--------|---------|
| `start.bat` | Double-click to start the gateway and open the workflow UI (auto-detects Python, clears stale ports) |
| `stop.bat`  | Stops the gateway and llama-server in one go |

`start.bat` probes for Python in this order: the `PYTHON_CMD` environment variable, then `python` on
`PATH`. If your Python is not on `PATH`, set it first:

```bat
set PYTHON_CMD=D:\envs\llm\Scripts\python.exe
start.bat
```

> Python is not needed to open the UI; it is only invoked when a train/export node actually runs.

## Advanced stages: quantization quality / evaluation / speculative decoding / post-training

Once the main pipeline works, these four stages answer what edge deployment really cares about:
**how small can it get, how much accuracy does it lose, how fast does it run, and can it go faster.**

Start with an environment self-check (it tells you exactly what is missing):

```bash
make check
# or individually: python pipeline/quantize_sweep.py --check / eval.py --check / spec_decode.py --check / sft.py --check
```

### 1. Quantization quality engineering: imatrix calibration + scheme comparison

`llama-imatrix` uses a calibration corpus to score how much each weight tensor matters to the output,
and `llama-quantize --imatrix` then allocates precision accordingly. At the same 4-bit width, a
calibrated scheme lands noticeably closer to F16 in perplexity — the standard trick for keeping
accuracy at low bit widths.

```bash
make sweep
# equivalent to:
python pipeline/quantize_sweep.py --llama-cpp D:/tools/llama.cpp \
    --schemes Q8_0,Q4_K_M,Q4_K_S,IQ4_XS --imatrix auto --calibration data/corpus.txt
```

Artifacts:

| File | Contents |
|------|----------|
| `models/<model>-<scheme>.gguf` | The quantized model for each scheme |
| `models/imatrix.dat` | The importance matrix from calibration (reusable) |
| `out/quantize_sweep.md` | Size / compression ratio / calibrated-or-not / time comparison table |
| `out/quantize-<scheme>.log` | Full quantization log per scheme |

Notes: `--imatrix none` turns calibration off for an A/B baseline; `--force` re-runs existing
artifacts; if imatrix fails the script silently falls back to plain quantization (add
`--imatrix-strict` to make it a hard error instead).

### 2. Evaluation benchmark: PPL / latency / throughput / KV-cache quantization

Turn "shrunk to 1/4 the size" into a table you can check line by line:

```bash
make eval
# equivalent to:
python pipeline/eval.py --llama-cpp D:/tools/llama.cpp --kv-quant --rounds 3
```

| Metric | Meaning | Source |
|--------|---------|--------|
| PPL | How much accuracy quantization cost | `llama-perplexity -f data/eval.txt` (falls back to transformers via `--hf-dir` if the tool is missing) |
| TTFT | Time to first token (prefill) | `timings.prompt_ms` returned by `llama-server` |
| Prefill throughput | Prompt processing speed | `timings.prompt_per_second` |
| Decode throughput | Generation speed (what edge devices care about) | `timings.predicted_per_second`, median over rounds |
| KV q8_0 decode | Speed with KV-cache quantization on | A second pass with `--cache-type-k/v q8_0` |
| Server RSS | Resident memory | Read automatically when `psutil` is installed |

Artifacts: `out/benchmark.md` (including relative change versus the F16 baseline) and
`out/benchmark.json` (ready for plotting). Use `--models a.gguf b.gguf` to pick the comparison set;
by default it sweeps every GGUF under `models/`.

### 3. Speculative decoding: draft + target, two models

A small model guesses ahead quickly and the large model verifies in batches — whenever a guess is
right, an autoregressive step of the large model is saved. Speedup without sacrificing accuracy.

The prerequisite is that **draft and target must share one tokenizer**, so train the larger target
model first:

```bash
# 1) train the target model (reusing the existing tokenizer guarantees an identical vocab)
make train-target
# 2) export + quantize (same scripts as before)
python pipeline/write_gguf.py --hf-dir models/tinyllm-target-hf \
    --out models/tinyllm-target-f16.gguf --name tinyllm-target
python pipeline/quantize_sweep.py --input models/tinyllm-target-f16.gguf --schemes Q4_K_M
# 3) run the comparison
make spec
```

Artifact `out/spec_decode.md`: decode throughput, TTFT, speedup and **acceptance rate** for baseline
versus speculative decoding (read from llama-server's `/metrics`). The script checks the two models'
vocab sizes first and stops if they differ (speculative decoding simply would not work).

### 4. Post-training: LoRA instruction tuning

A pretrained model only continues text; instruction tuning teaches it to answer questions. LoRA
trains only a low-rank adapter, usually well under 1% of the parameters:

```bash
make sft
# quick end-to-end check (20 steps only):
python pipeline/sft.py --steps 20
# produce the adapter without merging:
python pipeline/sft.py --no-merge
```

Data format (`data/sft_sample.jsonl`, one record per line):

```json
{"instruction": "What is model quantization?", "input": "", "output": "Model quantization represents weights with fewer bits ..."}
```

Artifacts: `models/tinyllm-hf-sft/` (the merged HF model, ready to export and quantize) and
`out/lora-adapter/` (the much smaller adapter). A fine-tuned model goes through the exact same
export/quantize/deploy path:

```bash
python pipeline/write_gguf.py --hf-dir models/tinyllm-hf-sft --out models/tinyllm-sft-f16.gguf
python pipeline/quantize_sweep.py --input models/tinyllm-sft-f16.gguf --schemes Q4_K_M
```

### Recommended experiment order (to collect all the numbers)

```text
make train → make export → make quantize        # Baseline: get it running
make sweep                                      # Size / compression ratio (imatrix vs none)
make eval                                       # Accuracy (PPL) - latency - throughput, plus KV quantization
make train-target → export+quantize target → make spec   # Speculative decoding speedup and acceptance rate
make sft → export+quantize → make eval again     # Compare perplexity and output quality before/after tuning
```

Every step leaves a Markdown report under `out/`, ready to use as an experiment log or to fold into
a write-up.

## Manual installation and deployment

Getting the whole pipeline running on a fresh machine from zero (Windows as the example; Linux/macOS
are the same with the respective path format).

### 1. Set up Python and install dependencies

**Python 3.10+** is required (CPU training is enough — no GPU needed).

```bash
# 1) create a dedicated virtual environment
python -m venv D:\envs\llm_go
D:\envs\llm_go\Scripts\activate          # Linux/macOS: source D:/envs/llm_go/bin/activate

# 2) install dependencies (the CPU-only torch wheel is small and sufficient here)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

> On Windows, if `python` is not on `PATH`, substitute the full path for every `python` command
> below, or set `set PYTHON_CMD=D:\envs\llm_go\Scripts\python.exe` so the workflow engine uses it.

### 2. Install Go (1.22+)

Download and install from https://go.dev/dl/, verify `go version` ≥ 1.22, then build the gateway:

```bash
cd server
go mod tidy
go build -o llm-gateway.exe .    # Linux/macOS: go build -o llm-gateway .
```

### 3. Get the llama.cpp runtime

This project needs only **two binaries**: `llama-server` (inference) and `llama-quantize`
(quantizer). Export is handled by the bundled `pipeline/write_gguf.py`, so you do not need to clone
the llama.cpp source tree.

- Grab the archive for your platform from https://github.com/ggml-org/llama.cpp/releases
  (e.g. `llama-bXXXX-bin-win-cpu-x64.zip`)
- Unzip it anywhere, for example `D:\tools\llama.cpp\bin\` (on Linux/macOS you can also build it
  yourself: `cmake -B build && cmake --build build`)

```bash
# verify both binaries work
D:/tools/llama.cpp/bin/llama-server.exe --version
D:/tools/llama.cpp/bin/llama-quantize.exe --version
```

### 4. Run the pipeline (command line)

```bash
# 1) data prep + training
python pipeline/train.py --epochs 150

# 2) export GGUF (F16) + quantize (Q4_K_M)
python pipeline/export_gguf.py --llama-cpp D:/tools/llama.cpp
# artifacts: models/tinyllm-f16.gguf -> models/tinyllm-q4_k_m.gguf
# note: export uses the bundled write_gguf.py rather than the official convert_hf_to_gguf.py,
#       because the official one identifies tokenizers through a hash whitelist that a
#       from-scratch vocab can never match

# 3) start the inference server (terminal 1)
D:/tools/llama.cpp/bin/llama-server.exe -m models/tinyllm-q4_k_m.gguf --host 127.0.0.1 --port 8081 --ctx-size 512

# 4) start the Go gateway (terminal 2)
cd server && ./llm-gateway.exe
```

Verify:

```bash
curl http://localhost:8080/healthz

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is AI?"}],"max_tokens":64}'

# streaming (SSE)
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Deep learning"}],"stream":true}'
```

### 5. Run the pipeline (visual UI)

Start the gateway and open http://localhost:8080/web/, then hit "▶ Run all". Before running, check
two parameters in the UI:

- **Export GGUF node** → `llamacpp_dir`: the llama.cpp directory from step 3 (e.g. `D:/tools/llama.cpp`)
- **Model training node** → epochs and friends, as needed (150 by default)

Optional gateway environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHON_CMD` | `python` | Interpreter used by workflow nodes (**point it at your venv**) |
| `LLAMA_CPP_DIR` | — | Default llama.cpp directory (can also be set in the UI) |
| `GATEWAY_PORT` | `8080` | Gateway port |
| `LLAMA_SERVER_URL` | `http://127.0.0.1:8081` | Upstream llama-server address |

Other gateway parameters: `DEFAULT_MAX_TOKENS`, `DEFAULT_TEMP`

### 6. Docker deployment (optional)

```bash
docker compose up -d          # start llama-server + gateway
docker compose run pipeline   # one-shot train + export (needs the llama.cpp repo mounted; see compose comments)
```

## Swapping in your own model or data

- **Different corpus**: replace `data/corpus.txt`. The more text, the more coherent the model
  (a few hundred KB of plain text is the practical minimum).
- **Different model shape**: edit the `model` section of `pipeline/config.yaml` (layers / width).
  Remember CPU training time grows with it.
- **Different quantization scheme**: change `quantize.type` in `config.yaml` (Q8_0 is more accurate,
  Q4_K_S is smaller).
- **Common pitfalls**:
  - **Which Python does the gateway use**: workflow nodes run as child processes started by the
    gateway with `PYTHON_CMD` (or `python` on `PATH`). That `python` is often a stripped-down
    interpreter without the dependencies, which makes every node fail with
    `ModuleNotFoundError: No module named 'yaml'` and skips all downstream nodes. The gateway now
    probes `PYTHON_CMD → VIRTUAL_ENV → project .venv/venv/env → PATH` and actually tests
    `import yaml`, picking a working interpreter and logging it at startup; `start.bat` probes the
    same way. Still, setting it explicitly is best:
    `set PYTHON_CMD=D:\envs\llm_go\Scripts\python.exe`
  - **Export/quantize do not need the llama.cpp source repo**: the directory passed to
    `export_gguf.py --llama-cpp <dir>` only has to contain `bin/llama-quantize` (the release zip is
    enough); the conversion step is done by the bundled `write_gguf.py`
  - The Trainer needs `accelerate` (already in requirements.txt)
  - `llama-quantize` may report an iostream error and return non-zero when its output is piped, even
    though the artifact was produced — `export_gguf.py` judges success by the output file
  - Recent `llama-server` builds validate non-streaming `/completion` output strictly as UTF-8; a
    rare bad byte from a tiny model causes an HTTP 500. Streaming requests are unaffected (the
    workflow's test node already uses streaming)
  - **The `--chunks 0` trap**: some llama.cpp versions take it literally and run zero chunks —
    `llama-perplexity` errors out and `llama-imatrix` writes an empty 448-byte matrix. `eval.py` and
    `quantize_sweep.py` no longer pass it.
  - **Speculative decoding flags were renamed**: recent llama.cpp removed
    `--draft-max/--draft-min/--draft-p-min` in favor of
    `--spec-draft-n-max/--spec-draft-n-min/--spec-draft-p-min`. `spec_decode.py` probes `--help` and
    supports both generations.
  - **Speculative decoding metrics were renamed**: recent `/metrics` counters are
    `llamacpp:spec_decode_num_accepted_tokens_total` / `_num_draft_tokens_total` (older builds used
    `draft_n_accepted_total` / `draft_n_evaluated_total`); both are handled.
  - **Corpus length requirements**: PPL / imatrix need at least 2× the context length in tokens or
    they error out. The scripts automatically lower the context and retry when the corpus is short;
    if you replace `data/eval.txt`, keep it reasonably long (≥ 4 KB of text).
- **Going further with LoRA**: swap `train.py` for peft-based LoRA training of an off-the-shelf HF
  model (e.g. Qwen3-0.6B). The export/quantize/deploy pipeline is reused unchanged.
