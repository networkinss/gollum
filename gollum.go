package gollum

import (
	"context"
	"errors"
	"fmt"
)

// Sentinel errors for distinguishing failure modes.
var (
	// ErrNoBackend means neither Ollama nor the embedded backend is available.
	ErrNoBackend = errors.New("no LLM backend available")
	// ErrNotCompiled means the binary was not built with -tags llm.
	ErrNotCompiled = errors.New("embedded LLM backend not available (build with -tags llm to enable)")
	// ErrModelNotFound means the configured model path does not exist on disk.
	ErrModelNotFound = errors.New("model file not found")
	// ErrModelPathEmpty means no model path was configured.
	ErrModelPathEmpty = errors.New("model path not configured")
)

// Backend is the interface for LLM inference backends.
type Backend interface {
	// Name returns a human-readable identifier for the backend (e.g. "ollama@localhost:11434").
	Name() string
	// Analyze sends a system prompt and user prompt to the LLM and returns the response.
	Analyze(ctx context.Context, systemPrompt, userPrompt string) (string, error)
	// Available reports whether the backend is ready to serve requests.
	Available() bool
	// Close releases any resources held by the backend.
	Close() error
}

// StreamingBackend is an additive capability: a Backend that can also deliver
// tokens incrementally as they're generated, instead of only the final result.
//
// Both built-in backends (Ollama, embedded) implement this; callers that need
// streaming should type-assert a Backend to StreamingBackend (or use
// AsStreamingBackend) rather than assuming it unconditionally, since a future
// or third-party Backend implementation may only implement the base interface.
type StreamingBackend interface {
	Backend
	// AnalyzeStream sends a system prompt and user prompt to the LLM and calls
	// onToken for each generated token as it arrives. onToken returns true to
	// continue generation or false to stop early.
	//
	// Cancelling ctx stops generation promptly: implementations must check ctx
	// between tokens and return ctx.Err() rather than running to completion.
	AnalyzeStream(ctx context.Context, systemPrompt, userPrompt string, onToken func(string) bool) error
}

// AsStreamingBackend type-asserts b to StreamingBackend, returning ok=false if
// the concrete backend doesn't support streaming.
func AsStreamingBackend(b Backend) (StreamingBackend, bool) {
	sb, ok := b.(StreamingBackend)
	return sb, ok
}

// Config holds LLM backend configuration.
type Config struct {
	// Backend selection: "auto", "embedded", "ollama", "remote".
	Backend string `json:"backend"`
	// ModelPath is the filesystem path to a .gguf model file (embedded backend).
	ModelPath string `json:"model_path"`
	// Model is the Ollama model name, e.g. "phi3:mini" or "llama3.2:3b".
	Model string `json:"model"`
	// Endpoint is the remote Ollama-compatible API URL, e.g. "http://host:11434".
	Endpoint string `json:"endpoint"`
	// MaxTokens limits the response length.
	MaxTokens int `json:"max_tokens"`
	// Threads sets CPU thread count for the embedded backend.
	Threads int `json:"threads"`
	// ContextSize sets the context window size in tokens for the embedded backend.
	// 0 means use the model's native maximum context length.
	ContextSize int `json:"context_size"`
	// Verbose enables detailed llama.cpp logging and model loading output.
	// When false (default), only errors are logged and loading progress is suppressed.
	Verbose bool `json:"-"`
	// Checksum is the expected SHA256 hex digest for the model file.
	Checksum string `json:"checksum"`
}

// DefaultConfig returns a Config with sensible defaults.
func DefaultConfig() Config {
	return Config{
		Backend:   "auto",
		Model:     "llama3.2:3b",
		MaxTokens: 512,
		Threads:   4,
	}
}

// ConfigFromValues creates a Config from individual values,
// allowing callers to bridge from their own config structs.
func ConfigFromValues(backend, modelPath, model, endpoint string, maxTokens, threads, contextSize int, checksum string) Config {
	return Config{
		Backend:     backend,
		ModelPath:   modelPath,
		Model:       model,
		Endpoint:    endpoint,
		MaxTokens:   maxTokens,
		Threads:     threads,
		ContextSize: contextSize,
		Checksum:    checksum,
	}
}

// New creates the best available Backend based on cfg.
func New(cfg Config) (Backend, error) {
	switch cfg.Backend {
	case "remote":
		if cfg.Endpoint == "" {
			return nil, fmt.Errorf("gollum: remote backend requires endpoint")
		}
		return newOllamaBackend(cfg.Endpoint, cfg.Model, cfg.MaxTokens)
	case "ollama":
		return newOllamaBackend("http://localhost:11434", cfg.Model, cfg.MaxTokens)
	case "embedded":
		return newEmbeddedBackend(cfg)
	case "auto", "":
		return autoDetect(cfg)
	default:
		return nil, fmt.Errorf("gollum: unknown backend %q", cfg.Backend)
	}
}

// autoDetect probes for a local Ollama instance, then falls back to embedded.
func autoDetect(cfg Config) (Backend, error) {
	if endpoint, ok := DiscoverLocalOllama(); ok {
		b, err := newOllamaBackend(endpoint, cfg.Model, cfg.MaxTokens)
		if err == nil {
			return b, nil
		}
	}
	b, err := newEmbeddedBackend(cfg)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrNoBackend, err)
	}
	return b, nil
}
