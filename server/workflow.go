package main

// workflow.go —— 可视化工作流执行引擎
//
// 前端节点编辑器 (web/index.html) 通过以下 API 驱动流水线:
//
//	GET  /api/workflow  返回工作流定义 (节点/连线/参数)
//	POST /api/run       运行工作流, SSE 实时推送节点状态与日志
//	POST /api/cancel    取消当前运行 (并停止 llama-server 子进程)
//
// 执行顺序由前端返回的 edges 决定 (拓扑排序), 每个节点对应一个
// 本地命令/动作, stdout 逐行通过 SSE 推给前端节点卡片。

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
)

// ---------- 工作流定义 ----------

type WFParam struct {
	Key     string `json:"key"`
	Label   string `json:"label"`
	Default string `json:"default"`
}

type WFNode struct {
	ID     string    `json:"id"`
	Label  string    `json:"label"`
	Desc   string    `json:"desc"`
	Icon   string    `json:"icon"`
	X      float64   `json:"x"`
	Y      float64   `json:"y"`
	Params []WFParam `json:"params"`
	// Optional 节点默认不参与「运行全部」(一键全流程保持最短路径),
	// 只在「仅运行此节点」或显式指定时执行 —— 用于 SFT、评测、投机解码等
	// 需要额外模型的进阶阶段。
	Optional bool `json:"optional,omitempty"`
}

type WFEdge struct {
	From string `json:"from"`
	To   string `json:"to"`
}

type WFDef struct {
	Nodes []WFNode `json:"nodes"`
	Edges []WFEdge `json:"edges"`
}

func workflowDef() WFDef {
	return WFDef{
		Nodes: []WFNode{
			{
				ID: "data", Label: "数据准备", Icon: "📄", X: 60, Y: 200,
				Desc: "检查训练语料规模与质量",
				Params: []WFParam{
					{Key: "corpus", Label: "语料路径", Default: "data/corpus.txt"},
				},
			},
			{
				ID: "train", Label: "模型训练", Icon: "🏋️", X: 320, Y: 200,
				Desc: "BPE 分词 + 从零训练小型 LLaMA",
				Params: []WFParam{
					{Key: "epochs", Label: "训练轮数", Default: "150"},
					{Key: "batch_size", Label: "批大小", Default: "8"},
				},
			},
			{
				ID: "export", Label: "导出 GGUF (F16)", Icon: "📦", X: 580, Y: 200,
				Desc: "HF 格式 → GGUF (llama.cpp 转换脚本)",
				Params: []WFParam{
					{Key: "llamacpp_dir", Label: "llama.cpp 目录", Default: "D:/tools/llama.cpp"},
				},
			},
			{
				ID: "quantize", Label: "量化 (Q4_K_M)", Icon: "🗜️", X: 840, Y: 200,
				Desc: "F16 → Q4_K_M, 体积压缩约 4 倍",
				Params: []WFParam{
					{Key: "quant_type", Label: "量化方案", Default: "Q4_K_M"},
				},
			},
			{
				ID: "deploy", Label: "启动推理服务", Icon: "🚀", X: 1100, Y: 200,
				Desc: "llama-server 加载量化模型, 等待健康检查通过",
				Params: []WFParam{
					{Key: "llama_server_bin", Label: "llama-server 路径", Default: "llama-server"},
					{Key: "ctx_size", Label: "上下文长度", Default: "512"},
					{Key: "port", Label: "服务端口", Default: "8081"},
				},
			},
			{
				ID: "test", Label: "服务冒烟测试", Icon: "🔍", X: 1360, Y: 200,
				Desc: "向推理服务发送测试请求, 验证生成",
				Params: []WFParam{
					{Key: "prompt", Label: "测试 Prompt", Default: "人工智能"},
					{Key: "n_predict", Label: "生成 Token 数", Default: "48"},
				},
			},

			// ---------- 进阶阶段 (可选节点: 默认不参与「运行全部」, 单独点「运行此节点」) ----------
			{
				ID: "train_target", Label: "训练 target 模型", Icon: "🏗️", X: 60, Y: 440,
				Desc: "训练第二档模型 (复用分词器, 与 draft 词表一致, 供投机解码使用)",
				Optional: true,
				Params: []WFParam{
					{Key: "epochs", Label: "训练轮数", Default: "80"},
				},
			},
			{
				ID: "sft", Label: "指令微调 (LoRA)", Icon: "🎯", X: 320, Y: 440,
				Desc: "在 base 模型上做 LoRA SFT, 产物可直接导出 GGUF",
				Optional: true,
				Params: []WFParam{
					{Key: "epochs", Label: "训练轮数", Default: "3"},
					{Key: "steps", Label: "步数上限(0=不限)", Default: "0"},
				},
			},
			{
				ID: "sweep", Label: "量化方案对比", Icon: "🗜️", X: 580, Y: 440,
				Desc: "imatrix 校准 + Q8_0/Q4_K_M/Q4_K_S/IQ4_XS 体积对比",
				Optional: true,
				Params: []WFParam{
					{Key: "schemes", Label: "方案(逗号分隔)", Default: "Q8_0,Q4_K_M,Q4_K_S,IQ4_XS"},
					{Key: "imatrix", Label: "imatrix 校准(auto/none/路径)", Default: "auto"},
				},
			},
			{
				ID: "eval", Label: "评测基准", Icon: "📊", X: 840, Y: 440,
				Desc: "PPL / 首 token 延迟 / 解码吞吐 / KV cache 量化对比",
				Optional: true,
				Params: []WFParam{
					{Key: "rounds", Label: "测量轮数", Default: "3"},
					{Key: "max_tokens", Label: "每次生成 Token", Default: "128"},
					{Key: "kv_quant", Label: "KV cache 量化(1/0)", Default: "1"},
				},
			},
			{
				ID: "spec", Label: "投机解码实验", Icon: "⚡", X: 1100, Y: 440,
				Desc: "draft + target 双模型推测解码, 输出加速比与接受率",
				Optional: true,
				Params: []WFParam{
					{Key: "draft", Label: "draft 模型", Default: "models/tinyllm-q4_k_m.gguf"},
					{Key: "target", Label: "target 模型", Default: "models/tinyllm-target-q4_k_m.gguf"},
					{Key: "draft_max", Label: "单次最多猜测", Default: "8"},
				},
			},
		},
		Edges: []WFEdge{
			{From: "data", To: "train"},
			{From: "train", To: "export"},
			{From: "export", To: "quantize"},
			{From: "quantize", To: "deploy"},
			{From: "deploy", To: "test"},
		},
	}
}

// ---------- 运行器 ----------

type workflowRunner struct {
	mu          sync.Mutex
	running     bool
	cancel      context.CancelFunc
	llamaServer *exec.Cmd // deploy 节点启动的长驻进程, cancel 时一并停止
	root        string    // 项目根目录 (pipeline/ 所在处)
	pythonCmd   string
}

func newWorkflowRunner() *workflowRunner {
	root := findProjectRoot()
	py, note := detectPython(root)
	log.Printf("流水线 Python: %s", py)
	if note != "" {
		log.Printf("提示: %s", note)
	}
	return &workflowRunner{root: root, pythonCmd: py}
}

// detectPython 挑一个真正装了流水线依赖的 Python。
//
// 直接信任 PATH 里的 python 是常见翻车点: 它可能是个精简解释器 (缺 yaml/torch),
// 于是每个节点都以 ModuleNotFoundError 失败、后续节点全部跳过。
// 这里按 PYTHON_CMD → VIRTUAL_ENV → 项目内 venv → PATH 的顺序探测,
// 用 "import yaml" 实测, 再优先挑装了 torch 的那个 (训练/SFT 需要)。
func detectPython(root string) (string, string) {
	seen := map[string]bool{}
	var yamlOnly string
	for _, cand := range pythonCandidates(root) {
		if cand == "" || seen[cand] {
			continue
		}
		seen[cand] = true
		if !probePython(cand, "import yaml") {
			continue
		}
		if probePython(cand, "import torch") {
			return cand, "" // 依赖最全, 直接用
		}
		if yamlOnly == "" {
			yamlOnly = cand
		}
	}
	if yamlOnly != "" {
		return yamlOnly, "该解释器缺少 torch, 训练/指令微调节点会失败; 建议设置 PYTHON_CMD 指向装了依赖的虚拟环境"
	}
	return "python", "未找到含 pyyaml 的 Python 解释器, 流水线节点可能报 ModuleNotFoundError; 请设置 PYTHON_CMD"
}

// pythonCandidates 按优先级列出候选解释器。
func pythonCandidates(root string) []string {
	var c []string
	if v := os.Getenv("PYTHON_CMD"); v != "" {
		c = append(c, v)
	}
	if v := os.Getenv("VIRTUAL_ENV"); v != "" {
		c = append(c, filepath.Join(v, "Scripts", "python.exe"), filepath.Join(v, "bin", "python"))
	}
	for _, rel := range []string{".venv", "venv", "env"} {
		c = append(c,
			filepath.Join(root, rel, "Scripts", "python.exe"),
			filepath.Join(root, rel, "bin", "python"))
	}
	c = append(c, "python", "python3", "py")
	return c
}

// probePython 用一段极短的代码验证解释器可用且能 import 指定模块。
func probePython(python, code string) bool {
	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, python, "-c", code)
	cmd.Stdout = io.Discard
	cmd.Stderr = io.Discard
	return cmd.Run() == nil
}

// findProjectRoot 从当前目录向上查找包含 pipeline/ 的目录作为项目根。
func findProjectRoot() string {
	if p := os.Getenv("PROJECT_ROOT"); p != "" {
		return p
	}
	wd, err := os.Getwd()
	if err != nil {
		return "."
	}
	dir := wd
	for i := 0; i < 5; i++ {
		if info, err := os.Stat(filepath.Join(dir, "pipeline")); err == nil && info.IsDir() {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return wd // 兜底: 使用当前目录
}

// SSE 事件
type wfEvent struct {
	Event  string `json:"event"`            // status | log | done
	Node   string `json:"node,omitempty"`
	Status string `json:"status,omitempty"` // running | success | failed | skipped
	Line   string `json:"line,omitempty"`
	OK     bool   `json:"ok,omitempty"`
	Msg    string `json:"msg,omitempty"`
}

type runRequest struct {
	Params map[string]string `json:"params"` // 节点参数扁平化: "train.epochs" 等
	Only   []string          `json:"only"`   // 仅运行这些节点 (调试用); 空 = 全部
}

func (r *workflowRunner) param(req *runRequest, node, key, def string) string {
	if v, ok := req.Params[node+"."+key]; ok && v != "" {
		return v
	}
	for _, n := range workflowDef().Nodes {
		if n.ID == node {
			for _, p := range n.Params {
				if p.Key == key {
					return p.Default
				}
			}
		}
	}
	return def
}

func (r *workflowRunner) emit(ch chan wfEvent, e wfEvent) {
	ch <- e
}

// runNode 执行单个节点, 返回是否成功
func (r *workflowRunner) runNode(ctx context.Context, node string, req *runRequest, ch chan wfEvent) bool {
	r.emit(ch, wfEvent{Event: "status", Node: node, Status: "running"})

	var cmd *exec.Cmd
	var action func() error

	switch node {
	case "data":
		corpus := r.param(req, node, "corpus", "data/corpus.txt")
		p := corpus
		if !filepath.IsAbs(p) {
			p = filepath.Join(r.root, p)
		}
		cmd = exec.CommandContext(ctx, r.pythonCmd, "pipeline/data_check.py", p)
		cmd.Dir = r.root

	case "train":
		epochs := r.param(req, node, "epochs", "150")
		batch := r.param(req, node, "batch_size", "8")
		cmd = exec.CommandContext(ctx, r.pythonCmd, "pipeline/train.py",
			"--epochs", epochs, "--batch-size", batch)
		cmd.Dir = r.root

	case "export":
		dir := r.param(req, node, "llamacpp_dir", "D:/tools/llama.cpp")
		cmd = exec.CommandContext(ctx, r.pythonCmd, "pipeline/export_gguf.py", "--llama-cpp", dir)
		cmd.Dir = r.root

	case "quantize":
		dir := r.param(req, "export", "llamacpp_dir", "D:/tools/llama.cpp")
		qt := r.param(req, node, "quant_type", "Q4_K_M")
		cmd = exec.CommandContext(ctx, r.pythonCmd, "pipeline/export_gguf.py",
			"--llama-cpp", dir, "--skip-convert", "--quant-type", qt)
		cmd.Dir = r.root

	case "deploy":
		bin := r.param(req, node, "llama_server_bin", "llama-server")
		port := r.param(req, node, "port", "8081")
		ctxSize := r.param(req, node, "ctx_size", "512")
		model := filepath.Join(r.root, "models", "tinyllm-q4_k_m.gguf")
		binPath := bin
		if !filepath.IsAbs(binPath) {
			// 尝试 <llamacpp>/build/bin/ 下找
			dir := r.param(req, "export", "llamacpp_dir", "D:/tools/llama.cpp")
			for _, cand := range []string{
				filepath.Join(dir, "build", "bin", bin+".exe"),
				filepath.Join(dir, "build", "bin", bin),
				filepath.Join(dir, "bin", bin+".exe"),
			} {
				if _, err := os.Stat(cand); err == nil {
					binPath = cand
					break
				}
			}
		}
		if _, err := os.Stat(model); err != nil {
			r.emit(ch, wfEvent{Event: "log", Node: node, Line: "错误: 找不到量化模型 " + model + ", 请先运行量化节点"})
			r.emit(ch, wfEvent{Event: "status", Node: node, Status: "failed"})
			return false
		}
		ls := exec.CommandContext(ctx, binPath,
			"-m", model, "--host", "127.0.0.1", "--port", port, "--ctx-size", ctxSize)
		ls.Dir = r.root
		// 输出落到日志文件, 启动失败时回显末尾几行 (否则用户只看到"健康检查超时")
		logPath := filepath.Join(r.root, "out", "llama-server-deploy.log")
		_ = os.MkdirAll(filepath.Dir(logPath), 0o755)
		logFile, ferr := os.Create(logPath)
		if ferr == nil {
			defer logFile.Close()
			ls.Stdout = logFile
			ls.Stderr = logFile
		} else {
			ls.Stdout = io.Discard
			ls.Stderr = io.Discard
		}
		if err := ls.Start(); err != nil {
			r.emit(ch, wfEvent{Event: "log", Node: node, Line: "启动失败: " + err.Error()})
			r.emit(ch, wfEvent{Event: "status", Node: node, Status: "failed"})
			return false
		}
		r.llamaServer = ls
		exited := make(chan struct{})
		go func() { _ = ls.Wait(); close(exited) }() // 提前退出时用 exited 立即结束等待
		action = func() error {
			// 轮询健康检查, 最长 60s
			url := fmt.Sprintf("http://127.0.0.1:%s/health", port)
			client := &http.Client{Timeout: 2 * time.Second}
			deadline := time.Now().Add(60 * time.Second)
			for time.Now().Before(deadline) {
				select {
				case <-ctx.Done():
					return ctx.Err()
				case <-exited: // 进程自己退出了, 不必等满超时
					for _, l := range tailLines(logPath, 15) {
						r.emit(ch, wfEvent{Event: "log", Node: node, Line: "[llama-server] " + l})
					}
					return fmt.Errorf("llama-server 启动后立即退出, 完整日志: %s", logPath)
				default:
				}
				if resp, err := client.Get(url); err == nil {
					resp.Body.Close()
					if resp.StatusCode == http.StatusOK {
						r.emit(ch, wfEvent{Event: "log", Node: node,
							Line: fmt.Sprintf("llama-server 就绪: %s (端口 %s)", filepath.Base(binPath), port)})
						return nil
					}
				}
				time.Sleep(1 * time.Second)
			}
			for _, l := range tailLines(logPath, 15) {
				r.emit(ch, wfEvent{Event: "log", Node: node, Line: "[llama-server] " + l})
			}
			return fmt.Errorf("llama-server 未就绪 (%s), 完整日志: %s", url, logPath)
		}

	case "test":
		prompt := r.param(req, node, "prompt", "人工智能")
		np := r.param(req, node, "n_predict", "48")
		action = func() error {
			url := fmt.Sprintf("http://127.0.0.1:%s/completion", r.param(req, "deploy", "port", "8081"))
			// 用流式接口: 对小模型偶发的坏字节, 非流式会 500, 流式能拿到已生成的部分
			body := fmt.Sprintf(`{"prompt":%q,"n_predict":%s,"temperature":0.3,"stream":true}`, prompt, np)
			resp, err := http.Post(url, "application/json", strings.NewReader(body))
			if err != nil {
				return fmt.Errorf("请求推理服务失败: %w", err)
			}
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				return fmt.Errorf("推理服务返回 %s", resp.Status)
			}
			sc := bufio.NewScanner(resp.Body)
			var out string
			for sc.Scan() {
				line := strings.TrimSpace(sc.Text())
				if !strings.HasPrefix(line, "data: ") {
					continue
				}
				var chunk struct {
					Content  string `json:"content"`
					Stop     bool   `json:"stop"`
					Timings  map[string]any `json:"timings"`
				}
				if json.Unmarshal([]byte(line[6:]), &chunk) == nil {
					out += chunk.Content
					if chunk.Timings != nil {
						if v, ok := chunk.Timings["predicted_per_second"].(float64); ok {
							r.emit(ch, wfEvent{Event: "log", Node: node, Line: fmt.Sprintf("生成速度: %.1f tok/s", v)})
						}
					}
				}
			}
			r.emit(ch, wfEvent{Event: "log", Node: node, Line: fmt.Sprintf("生成结果: %s", out)})
			return nil
		}

	// ---------- 进阶阶段 (可选节点) ----------
	case "train_target":
		epochs := r.param(req, node, "epochs", "80")
		cmd = exec.CommandContext(ctx, r.pythonCmd, "pipeline/train.py",
			"--profile", "target", "--reuse-tokenizer", "--epochs", epochs)
		cmd.Dir = r.root

	case "sft":
		epochs := r.param(req, node, "epochs", "3")
		steps := r.param(req, node, "steps", "0")
		args := []string{"pipeline/sft.py", "--epochs", epochs}
		if steps != "" && steps != "0" {
			args = append(args, "--steps", steps)
		}
		cmd = exec.CommandContext(ctx, r.pythonCmd, args...)
		cmd.Dir = r.root

	case "sweep":
		dir := r.param(req, "export", "llamacpp_dir", "D:/tools/llama.cpp")
		schemes := r.param(req, node, "schemes", "Q8_0,Q4_K_M,Q4_K_S,IQ4_XS")
		imat := r.param(req, node, "imatrix", "auto")
		args := []string{"pipeline/quantize_sweep.py", "--llama-cpp", dir, "--schemes", schemes}
		if imat != "" && imat != "none" {
			args = append(args, "--imatrix", imat)
		}
		cmd = exec.CommandContext(ctx, r.pythonCmd, args...)
		cmd.Dir = r.root

	case "eval":
		dir := r.param(req, "export", "llamacpp_dir", "D:/tools/llama.cpp")
		rounds := r.param(req, node, "rounds", "3")
		maxTok := r.param(req, node, "max_tokens", "128")
		kv := r.param(req, node, "kv_quant", "1")
		args := []string{"pipeline/eval.py", "--llama-cpp", dir,
			"--rounds", rounds, "--max-tokens", maxTok}
		if kv == "1" {
			args = append(args, "--kv-quant")
		}
		cmd = exec.CommandContext(ctx, r.pythonCmd, args...)
		cmd.Dir = r.root

	case "spec":
		dir := r.param(req, "export", "llamacpp_dir", "D:/tools/llama.cpp")
		draft := r.param(req, node, "draft", "models/tinyllm-q4_k_m.gguf")
		target := r.param(req, node, "target", "models/tinyllm-target-q4_k_m.gguf")
		dmax := r.param(req, node, "draft_max", "8")
		cmd = exec.CommandContext(ctx, r.pythonCmd, "pipeline/spec_decode.py",
			"--llama-cpp", dir, "--draft", draft, "--target", target,
			"--draft-max", dmax)
		cmd.Dir = r.root

	default:
		r.emit(ch, wfEvent{Event: "log", Node: node, Line: "未知节点: " + node})
		r.emit(ch, wfEvent{Event: "status", Node: node, Status: "failed"})
		return false
	}

	var err error
	if action != nil {
		r.emit(ch, wfEvent{Event: "log", Node: node, Line: "▶ 开始执行 " + node})
		err = action()
	} else {
		r.emit(ch, wfEvent{Event: "log", Node: node, Line: "▶ " + cmd.Path + " " + joinArgs(cmd.Args[1:])})
		err = streamCmd(ctx, cmd, node, ch)
	}

	if ctx.Err() != nil {
		r.emit(ch, wfEvent{Event: "status", Node: node, Status: "skipped"})
		return false
	}
	if err != nil {
		r.emit(ch, wfEvent{Event: "log", Node: node, Line: "✗ 失败: " + err.Error()})
		r.emit(ch, wfEvent{Event: "status", Node: node, Status: "failed"})
		return false
	}
	r.emit(ch, wfEvent{Event: "status", Node: node, Status: "success"})
	return true
}

func streamCmd(ctx context.Context, cmd *exec.Cmd, node string, ch chan wfEvent) error {
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	cmd.Stderr = cmd.Stdout // 合并 stderr
	if err := cmd.Start(); err != nil {
		return err
	}
	sc := bufio.NewScanner(stdout)
	sc.Buffer(make([]byte, 1024*1024), 1024*1024)
	hinted := false
	for sc.Scan() {
		select {
		case <-ctx.Done():
			cmd.Process.Kill()
			return ctx.Err()
		default:
		}
		line := sc.Text()
		ch <- wfEvent{Event: "log", Node: node, Line: line}
		// 解释器依赖缺失时给出可操作提示 (否则用户只看到一长串 Traceback)
		if !hinted && strings.Contains(line, "No module named") {
			hinted = true
			ch <- wfEvent{Event: "log", Node: node, Line: "ℹ " + pythonHint(line)}
		}
	}
	return cmd.Wait()
}

// pythonHint 把 ModuleNotFoundError 翻译成"怎么修"。
func pythonHint(line string) string {
	mod := "依赖"
	if i := strings.Index(line, "No module named "); i >= 0 {
		mod = strings.Trim(strings.TrimSpace(line[i+len("No module named "):]), "'\"")
	}
	return fmt.Sprintf("缺模块 %s: 当前解释器没装流水线依赖。设置 PYTHON_CMD 指向装有依赖的虚拟环境 "+
		"(如 set PYTHON_CMD=D:\\envs\\llm_go\\Scripts\\python.exe) 后重启网关, 或执行 pip install -r requirements.txt", mod)
}

func (r *workflowRunner) topoOrder(def WFDef) ([]string, error) {
	// Kahn 拓扑排序
	indeg := map[string]int{}
	adj := map[string][]string{}
	for _, n := range def.Nodes {
		indeg[n.ID] += 0
	}
	for _, e := range def.Edges {
		adj[e.From] = append(adj[e.From], e.To)
		indeg[e.To]++
	}
	var queue []string
	for id, d := range indeg {
		if d == 0 {
			queue = append(queue, id)
		}
	}
	var order []string
	for len(queue) > 0 {
		id := queue[0]
		queue = queue[1:]
		order = append(order, id)
		for _, to := range adj[id] {
			indeg[to]--
			if indeg[to] == 0 {
				queue = append(queue, to)
			}
		}
	}
	if len(order) != len(def.Nodes) {
		return nil, fmt.Errorf("工作流存在环, 无法执行")
	}
	return order, nil
}

// ---------- HTTP 处理器 ----------

func (r *workflowRunner) handleGetWorkflow(c *gin.Context) {
	c.JSON(http.StatusOK, workflowDef())
}

func (r *workflowRunner) handleRun(c *gin.Context) {
	r.mu.Lock()
	if r.running {
		r.mu.Unlock()
		c.JSON(http.StatusConflict, gin.H{"error": "已有工作流在运行中"})
		return
	}
	r.running = true
	r.mu.Unlock()

	var req runRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		req = runRequest{}
	}

	ctx, cancel := context.WithCancel(context.Background())
	r.mu.Lock()
	r.cancel = cancel
	r.mu.Unlock()

	ch := make(chan wfEvent, 256)
	c.Writer.Header().Set("Content-Type", "text/event-stream")
	c.Writer.Header().Set("Cache-Control", "no-cache")
	flusher := c.Writer.(http.Flusher)

	go func() {
		defer close(ch)
		defer cancel()

		order, err := r.topoOrder(workflowDef())
		if err != nil {
			ch <- wfEvent{Event: "done", OK: false, Msg: err.Error()}
			return
		}
		wdef := workflowDef()
		// only 过滤 (调试): 保留指定节点及其**全部**前驱
		runSet := map[string]bool{}
		explicit := map[string]bool{}   // 用户在界面上明确要求运行的节点
		if len(req.Only) > 0 {
			need := map[string]bool{}
			for _, id := range req.Only {
				need[id] = true
				explicit[id] = true
			}
			// 反向遍历拓扑序求传递闭包: 一直标记到没有新增为止
			// (只做一趟的话链式前驱会漏, 例如 only=test 时 data/train/export/quantize 会被跳过)
			for changed := true; changed; {
				changed = false
				for i := len(order) - 1; i >= 0; i-- {
					id := order[i]
					if need[id] {
						continue
					}
					for _, t := range adjOf(wdef, id) {
						if need[t] {
							need[id] = true
							changed = true
							break
						}
					}
				}
			}
			for id := range need {
				runSet[id] = true
			}
		}

		def := wdef
		ok := true
		for _, id := range order {
			if len(runSet) > 0 && !runSet[id] {
				r.emit(ch, wfEvent{Event: "status", Node: id, Status: "skipped"})
				continue
			}
			// 可选节点 (SFT / 评测 / 投机解码等) 默认不参与「运行全部」,
			// 需要单独点「仅运行此节点」才会执行
			if isOptionalNode(def, id) && !explicit[id] {
				r.emit(ch, wfEvent{Event: "log", Node: id,
					Line: "ℹ 可选阶段, 未选择时跳过 (点「仅运行此节点」可单独执行)"})
				r.emit(ch, wfEvent{Event: "status", Node: id, Status: "skipped"})
				continue
			}
			if !ok { // 前序失败, 后续跳过
				r.emit(ch, wfEvent{Event: "status", Node: id, Status: "skipped"})
				continue
			}
			ok = r.runNode(ctx, id, &req, ch)
		}
		ch <- wfEvent{Event: "done", OK: ok}
	}()

	for e := range ch {
		b, _ := json.Marshal(e)
		fmt.Fprintf(c.Writer, "data: %s\n\n", b)
		flusher.Flush()
	}

	r.mu.Lock()
	r.running = false
	r.mu.Unlock()
}

func adjOf(def WFDef, from string) []string {
	var out []string
	for _, e := range def.Edges {
		if e.From == from {
			out = append(out, e.To)
		}
	}
	return out
}

// isOptionalNode 判断是否为可选阶段 (默认不参与「运行全部」)。
func isOptionalNode(def WFDef, id string) bool {
	for _, n := range def.Nodes {
		if n.ID == id {
			return n.Optional
		}
	}
	return false
}

func (r *workflowRunner) handleCancel(c *gin.Context) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.running || r.cancel == nil {
		c.JSON(http.StatusOK, gin.H{"ok": false, "msg": "当前没有运行中的工作流"})
		return
	}
	r.cancel()
	if r.llamaServer != nil && r.llamaServer.Process != nil {
		r.llamaServer.Process.Kill()
		r.llamaServer = nil
	}
	c.JSON(http.StatusOK, gin.H{"ok": true, "msg": "已发送取消信号"})
}

// ---------- 小工具 ----------

// tailLines 读取文件末尾 n 行 (文件不存在或读失败时返回 nil)。
func tailLines(path string, n int) []string {
	b, err := os.ReadFile(path)
	if err != nil || len(b) == 0 {
		return nil
	}
	lines := strings.Split(strings.TrimRight(string(b), "\r\n"), "\n")
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return lines
}

func joinArgs(args []string) string {
	out := ""
	for i, a := range args {
		if i > 0 {
			out += " "
		}
		out += a
	}
	return out
}
