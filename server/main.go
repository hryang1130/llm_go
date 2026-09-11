// llm-gateway —— 部署阶段 (4/4) 的 Go 推理网关
//
// 架构:
//
//	client ──HTTP──▶ Go 网关 (Gin) ──HTTP──▶ llama.cpp server (llama-server)
//	                 │
//                     ├─ /healthz              网关健康检查 (联动探测 llama-server)
//                     ├─ /v1/models            模型列表 (透传)
//                     └─ /v1/chat/completions  OpenAI 兼容对话接口 (透传, 支持流式)
//
// 环境变量:
//   LLAMA_SERVER_URL  llama-server 地址   (默认 http://127.0.0.1:8081)
//   GATEWAY_PORT      网关监听端口        (默认 8080)
//   DEFAULT_MAX_TOKENS 默认生成上限      (默认 256)
//   DEFAULT_TEMP      默认温度            (默认 0.8)
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
)

type Config struct {
	LlamaServerURL   string
	Port             string
	DefaultMaxTokens int
	DefaultTemp      float64
}

func loadConfig() Config {
	cfg := Config{
		LlamaServerURL:   envOr("LLAMA_SERVER_URL", "http://127.0.0.1:8081"),
		Port:             envOr("GATEWAY_PORT", "8080"),
		DefaultMaxTokens: envIntOr("DEFAULT_MAX_TOKENS", 256),
		DefaultTemp:      envFloatOr("DEFAULT_TEMP", 0.8),
	}
	return cfg
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envIntOr(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func envFloatOr(key string, def float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}

// ---------- 中间件 ----------

func loggerMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		c.Next()
		log.Printf("%s %s -> %d (%s)",
			c.Request.Method, c.Request.URL.Path, c.Writer.Status(),
			time.Since(start).Round(time.Millisecond))
	}
}

func corsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Content-Type, Authorization")
		if c.Request.Method == http.MethodOptions {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}
		c.Next()
	}
}

// ---------- 处理器 ----------

// healthz 网关自身健康检查, 同时探测上游 llama-server。
func healthz(cfg Config) gin.HandlerFunc {
	return func(c *gin.Context) {
		upstream := "down"
		client := http.Client{Timeout: 3 * time.Second}
		if resp, err := client.Get(cfg.LlamaServerURL + "/health"); err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				upstream = "ok"
			}
		}
		status := http.StatusOK
		if upstream != "ok" {
			status = http.StatusServiceUnavailable
		}
		c.JSON(status, gin.H{
			"gateway":  "ok",
			"upstream": upstream,
			"upstream_url": cfg.LlamaServerURL,
			"time":     time.Now().Format(time.RFC3339),
		})
	}
}

// chatCompletions OpenAI 兼容接口: 补全默认参数后透传给 llama-server。
// 支持 stream=true 的 SSE 流式透传。
func chatCompletions(cfg Config) gin.HandlerFunc {
	client := &http.Client{Timeout: 10 * time.Minute}
	return func(c *gin.Context) {
		body, err := io.ReadAll(c.Request.Body)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "读取请求体失败"})
			return
		}

		var req map[string]any
		if err := json.Unmarshal(body, &req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "请求体不是合法 JSON"})
			return
		}

		// 补全默认推理参数 (客户端未指定时)
		if _, ok := req["max_tokens"]; !ok {
			req["max_tokens"] = cfg.DefaultMaxTokens
		}
		if _, ok := req["temperature"]; !ok {
			req["temperature"] = cfg.DefaultTemp
		}
		stream, _ := req["stream"].(bool)

		out, err := json.Marshal(req)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "序列化请求失败"})
			return
		}

		upReq, err := http.NewRequestWithContext(c.Request.Context(),
			http.MethodPost, cfg.LlamaServerURL+"/v1/chat/completions",
			bytes.NewReader(out))
		if err != nil {
			c.JSON(http.StatusBadGateway, gin.H{"error": "构造上游请求失败"})
			return
		}
		upReq.Header.Set("Content-Type", "application/json")

		resp, err := client.Do(upReq)
		if err != nil {
			c.JSON(http.StatusBadGateway, gin.H{
				"error": "上游 llama-server 不可达, 请确认已启动: " + cfg.LlamaServerURL,
			})
			return
		}
		defer resp.Body.Close()

		if !stream {
			// 非流式: 原样透传 JSON
			c.DataFromReader(resp.StatusCode, resp.ContentLength,
				resp.Header.Get("Content-Type"), resp.Body, nil)
			return
		}

		// 流式: SSE 逐块透传
		c.Header("Content-Type", "text/event-stream")
		c.Header("Cache-Control", "no-cache")
		c.Header("X-Accel-Buffering", "no")
		flusher := c.Writer.(http.Flusher)
		buf := make([]byte, 4096)
		for {
			n, err := resp.Body.Read(buf)
			if n > 0 {
				if _, werr := c.Writer.Write(buf[:n]); werr != nil {
					return
				}
				flusher.Flush()
			}
			if err != nil {
				if err != io.EOF {
					log.Printf("stream read error: %v", err)
				}
				return
			}
		}
	}
}

// genericProxy 透传 /v1/models 等只读端点。
func genericProxy(cfg Config, path string) gin.HandlerFunc {
	target, err := url.Parse(cfg.LlamaServerURL)
	if err != nil {
		log.Fatalf("LLAMA_SERVER_URL 无效: %v", err)
	}
	proxy := httputil.NewSingleHostReverseProxy(target)
	return func(c *gin.Context) {
		c.Request.URL.Path = path
		proxy.ServeHTTP(c.Writer, c.Request)
	}
}

func main() {
	cfg := loadConfig()
	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	r.Use(gin.Recovery(), loggerMiddleware(), corsMiddleware())

	r.GET("/healthz", healthz(cfg))
	r.GET("/v1/models", genericProxy(cfg, "/v1/models"))
	r.POST("/v1/chat/completions", chatCompletions(cfg))
	r.POST("/v1/completions", chatCompletions(cfg))

	// 可视化工作流: 节点定义 + 执行引擎 (SSE 日志) + 取消
	runner := newWorkflowRunner()
	r.GET("/api/workflow", runner.handleGetWorkflow)
	r.POST("/api/run", runner.handleRun)
	r.POST("/api/cancel", runner.handleCancel)

	// 工作流前端 (节点编辑器)
	webDir := envOr("WEB_DIR", "../web")
	if abs, err := filepath.Abs(webDir); err == nil {
		webDir = abs
	}
	if info, err := os.Stat(webDir); err == nil && info.IsDir() {
		r.Static("/web", webDir)
		r.GET("/", func(c *gin.Context) {
			c.Redirect(http.StatusFound, "/web/")
		})
		r.GET("/web", func(c *gin.Context) {
			c.Redirect(http.StatusFound, "/web/")
		})
		log.Printf("工作流界面: http://localhost:%s/web/", cfg.Port)
	} else {
		log.Printf("未找到前端目录 %s, 工作流界面不可用", webDir)
	}

	srv := &http.Server{
		Addr:    ":" + cfg.Port,
		Handler: r,
	}

	go func() {
		log.Printf("llm-gateway 启动: http://localhost:%s  (上游: %s)",
			cfg.Port, cfg.LlamaServerURL)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("网关启动失败: %v", err)
		}
	}()

	// 优雅退出
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("正在关闭网关 ...")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		log.Printf("关闭时出错: %v", err)
	}
	fmt.Println("llm-gateway 已退出")
}
