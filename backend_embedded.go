//go:build llm

package gollum

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"

	llama "github.com/tcpipuk/llama-go"
)

// embeddedBackend runs inference locally using llama-go (CPU).
type embeddedBackend struct {
	model     *llama.Model
	ctx       *llama.Context
	cfg       Config
	closeOnce sync.Once
}

func newEmbeddedBackend(cfg Config) (Backend, error) {
	if cfg.ModelPath == "" {
		return nil, fmt.Errorf("%w: set model_path in config", ErrModelPathEmpty)
	}
	if _, err := os.Stat(cfg.ModelPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("%w: %s", ErrModelNotFound, cfg.ModelPath)
	}

	modelOpts := []llama.ModelOption{llama.WithGPULayers(0)}
	if !cfg.Verbose {
		os.Setenv("LLAMA_LOG", "error")
		llama.InitLogging()
		modelOpts = append(modelOpts, llama.WithSilentLoading())
	}

	model, err := llama.LoadModel(cfg.ModelPath, modelOpts...)
	if err != nil {
		return nil, fmt.Errorf("embedded: failed to load model %s: %w", cfg.ModelPath, err)
	}

	ctxOpts := []llama.ContextOption{llama.WithContext(cfg.ContextSize)}
	if cfg.Threads > 0 {
		ctxOpts = append(ctxOpts, llama.WithThreads(cfg.Threads))
	}

	llamaCtx, err := model.NewContext(ctxOpts...)
	if err != nil {
		model.Close()
		return nil, fmt.Errorf("embedded: failed to create context: %w", err)
	}

	return &embeddedBackend{model: model, ctx: llamaCtx, cfg: cfg}, nil
}

func (e *embeddedBackend) Name() string {
	return "embedded@" + e.cfg.ModelPath
}

func (e *embeddedBackend) Analyze(ctx context.Context, systemPrompt, userPrompt string) (string, error) {
	var sb strings.Builder
	err := e.AnalyzeStream(ctx, systemPrompt, userPrompt, func(token string) bool {
		sb.WriteString(token)
		return true
	})
	if err != nil {
		return "", err
	}
	return sb.String(), nil
}

// AnalyzeStream streams tokens via llama-go's GenerateStream callback.
//
// llama.cpp's decode loop has no built-in cancellation, so ctx is checked
// between tokens: returning false from the callback (either because onToken
// asked to stop or ctx was cancelled) halts generation at the next token
// boundary rather than running to completion.
func (e *embeddedBackend) AnalyzeStream(ctx context.Context, systemPrompt, userPrompt string, onToken func(string) bool) error {
	prompt := systemPrompt + "\n\n" + userPrompt

	maxTokens := e.cfg.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 512
	}

	err := e.ctx.GenerateStream(prompt, func(token string) bool {
		if ctx.Err() != nil {
			return false
		}
		return onToken(token)
	},
		llama.WithMaxTokens(maxTokens),
		llama.WithTemperature(0.3),
		llama.WithTopK(40),
		llama.WithTopP(0.9),
	)
	if err != nil {
		return fmt.Errorf("embedded: inference failed: %w", err)
	}
	return ctx.Err()
}

func (e *embeddedBackend) Available() bool {
	return e.model != nil && e.ctx != nil
}

func (e *embeddedBackend) Close() error {
	e.closeOnce.Do(func() {
		if e.ctx != nil {
			e.ctx.Close()
			e.ctx = nil
		}
		if e.model != nil {
			e.model.Close()
			e.model = nil
		}
	})
	return nil
}
