# AGENT.md — Notes for AI agents working on `llm_go`

This file records conventions that are **not obvious from the code**. Follow them.

---

## 1. Commit messages MUST be in English

This is the one hard rule. All commits in this repository are written in **English**,
even though code comments, log strings and `README.md` are Chinese.

Format: [Conventional Commits](https://www.conventionalcommits.org/), imperative mood,
subject ≤ 72 chars, no trailing period. Body (optional) is a bullet list in English.

```text
feat: add imatrix-calibrated quantization sweep
fix: drop --chunks 0 which broke perplexity on newer llama.cpp
docs: document speculative decoding flag renames
ui: redesign workflow canvas with dark theme
chore: bump llama.cpp runtime to b11424
ci: cache python wheels in pipeline image
```

```
# BAD — never do this
fix: 修复端到端验证发现的兼容性问题
```

Language split by artifact:

| Artifact | Language |
|---|---|
| Commit subject / body | **English** (mandatory) |
| Code comments | Chinese (match surrounding code) |
| Runtime log lines / CLI help | Chinese |
| `README.md` | Chinese |
| `AGENT.md`, `LICENSE` | English |

---

## 2. Runtime requirements

- **Python 3.10+** with `pyyaml`, `torch`, `transformers`, `peft`, `datasets`, `accelerate`.
  On CPU only: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- **llama.cpp binaries** (build `b11424` or newer is known-good):
  `llama-quantize`, `llama-server`, `llama-perplexity`, `llama-imatrix`.
  Expected at `D:/tools/llama.cpp/bin/` in this workspace; configurable via
  `--llama-cpp` (pipeline scripts) or the `llamacpp_dir` node param (workflow UI).
- **Go 1.22+** to build the gateway (`cd server && go build -o llm-gateway.exe .`).
  If the local Go is older (e.g. 1.21.5) and the toolchain download is blocked,
  temporarily set `go 1.21` in `server/go.mod`, build, then restore the file.

### Python interpreter selection (common failure)

The gateway spawns pipeline scripts with the interpreter from `PYTHON_CMD`
(falling back to `python` on `PATH`). A bare `python` frequently resolves to an
interpreter **without** `yaml`/`torch`, which makes every node fail with
`ModuleNotFoundError`. Always point `PYTHON_CMD` at a venv that has the deps:

```bash
set PYTHON_CMD=D:\envs\llm_go\Scripts\python.exe
start.bat
```

`start.bat` probes `PYTHON_CMD` first, then `python` on `PATH`.

---

## 3. Layout conventions

- `pipeline/*.py` — one script per stage; each supports `--check` for an env self-test.
  `pipeline/common.py` holds shared helpers (`find_binary`, `run`, `supports_flag`, report writers).
- `server/` — Go gateway (Gin). `main.go` = HTTP/OpenAI-compatible API, `workflow.go` = node engine.
- `web/index.html` — single-file workflow UI (no build step, no framework). Served at `/web/`.
  Interaction state is persisted in `localStorage` under `wf_pos_v1` / `wf_edges_v1` /
  `wf_params_v1` / `wf_view_v1` — **do not rename these keys**, it wipes users' layouts.
- Artifacts: models in `models/`, reports and logs in `out/`. Never commit either
  (both are git-ignored) — commits carry code, config, docs and sample data only.

---

## 4. llama.cpp pitfalls (already worked around — keep the workarounds)

1. **Never pass `--chunks 0`.** Newer builds interpret it literally (0 chunks):
   `llama-perplexity` errors out and `llama-imatrix` writes an empty 448-byte matrix.
2. **Speculative decoding flags were renamed.** `--draft-max/-min/-p-min` →
   `--spec-draft-n-max/-n-min/-p-min`. Use `common.supports_flag()` to probe.
3. **Metrics were renamed.** `draft_n_accepted_total` → `llamacpp:spec_decode_num_accepted_tokens_total`.
   `spec_decode.py` matches both.
4. **imatrix / perplexity need corpus tokens ≥ 2×ctx.** Scripts auto-lower the context
   when the corpus is short; keep `data/eval.txt` reasonably long (~5 KB).
5. `llama-quantize` may exit non-zero while piping, but the artifact is fine —
   judge success by the output file, not the exit code.
6. Non-streaming `/completion` does strict UTF-8 validation; a tiny model's bad bytes
   cause HTTP 500. Use streaming (the `test` node already does).
7. `special_eos_id is not in special_eog_ids` is a **non-fatal** warning: our self-trained
   tokenizer names EOS `<|eos|>`, which is absent from llama.cpp's hardcoded EOG name list.

---

## 5. Verifying a change

```bash
python pipeline/<stage>.py --check                 # dependency self-test
python pipeline/eval.py --llama-cpp D:/tools/llama.cpp --rounds 2 --kv-quant
```

Artifacts to eyeball: `out/benchmark.md`, `out/quantize_sweep.md`, `out/spec_decode.md`.
For UI changes, load `http://localhost:8080/web/` and check the browser console for errors.
