package gollum

import (
	"context"
	"errors"
	"fmt"
	"time"
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

// Sampling defaults for the embedded backend.
//
// Declared here rather than in backend_embedded.go, which is behind
// `-tags llm`: these are the library's sampling policy, and a test that can
// only run in a tagged build is a test that will not run in CI for the
// default one. The values are used only by the embedded backend. Low temperature on purpose:
// this library's callers analyse text rather than write fiction, and a model
// that invents detail about a document is worse than one that is dull.
const (
	defaultTemperature = 0.3
	defaultTopK        = 40
	defaultTopP        = 0.9
	defaultMaxTokens   = 512
	// llama.cpp defaults the repetition penalty to 1.0, which is to say off.
	// Combined with a low temperature that reliably produces degenerate
	// loops: the model finds a phrase it likes and emits it until the token
	// limit stops it. 1.1 over the last 64 tokens is llama.cpp's own
	// conventional setting and costs nothing when the model was not going to
	// repeat itself anyway.
	//
	// The Ollama backend sets no equivalent because Ollama already applies
	// its own repetition default server-side; this defect is embedded-only.
	defaultRepeatPenalty = 1.1
	defaultPenaltyLastN  = 64
)

// GenerateOptions carries per-request overrides.
//
// It exists because the alternative does not work: MaxTokens and the sampling
// parameters are bound when a backend is constructed, so varying them per
// request means building a new backend — which, on the embedded engine, means
// reloading the model from disk. A consumer that wants a short answer for one
// action and a long one for the next would pay several seconds for the
// privilege.
//
// A zero field means "use the value this backend was configured with", so the
// zero GenerateOptions is exactly the old behaviour.
type GenerateOptions struct {
	// MaxTokens caps this response. 0 uses the backend's configured value.
	MaxTokens int
	// Temperature, TopK and TopP override sampling for this request.
	// 0 uses the backend's default for each, independently.
	Temperature float32
	TopK        int
	TopP        float32
	// RepeatPenalty discourages the model from repeating itself. 1.0 is no
	// penalty; 1.1 is the conventional value. 0 uses the backend's default.
	//
	// This matters more than it sounds. llama.cpp defaults it to 1.0 —
	// disabled — and a small model at low temperature with no penalty will
	// happily emit the same sentence until it runs out of tokens. That is not
	// a hypothetical: a 3B model summarising an 8 KB document produced
	// "The updater is not built inside Edithor / The updater is separate from
	// Edithor" repeated until the limit stopped it.
	RepeatPenalty float32
	// PenaltyLastN is how many recent tokens the penalty considers.
	// 0 uses the backend's default.
	PenaltyLastN int
}

// IsZero reports whether every field is unset, in which case the backend's
// own configuration applies unchanged.
func (o GenerateOptions) IsZero() bool {
	return o == GenerateOptions{}
}

// OptionedBackend is an additive capability: a Backend that accepts
// per-request overrides as well as its constructor-time configuration.
//
// Additive for the same reason StreamingBackend is: `Backend` is the published
// contract and widening its method signatures would break every third-party
// implementation. Both built-in backends implement this; callers should
// type-assert (or use AsOptionedBackend) rather than assume it.
type OptionedBackend interface {
	Backend
	// AnalyzeWithOptions is Analyze with per-request overrides.
	AnalyzeWithOptions(ctx context.Context, systemPrompt, userPrompt string, opts GenerateOptions) (string, error)
	// AnalyzeStreamWithOptions is AnalyzeStream with per-request overrides.
	AnalyzeStreamWithOptions(ctx context.Context, systemPrompt, userPrompt string, onToken func(string) bool, opts GenerateOptions) error
}

// AsOptionedBackend type-asserts b to OptionedBackend, returning ok=false if
// the concrete backend does not accept per-request overrides.
func AsOptionedBackend(b Backend) (OptionedBackend, bool) {
	ob, ok := b.(OptionedBackend)
	return ob, ok
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
	// APIKey is a bearer token for an Ollama-compatible endpoint that needs
	// one (hosted services do; a local daemon does not). Never logged, and
	// never included in Backend.Name().
	APIKey string `json:"-"`
	// TimeoutSec bounds a single generation. 0 uses the default (120s), which
	// suits a small local model; a large or hosted one can need longer.
	TimeoutSec int `json:"timeout_sec"`
}

// WithAPIKey returns a copy of c carrying the given bearer token.
//
// A builder rather than a parameter on ConfigFromValues: that function's
// signature is part of the published API and consumers already call it
// positionally, so widening it would break them for a field most callers
// never set.
func (c Config) WithAPIKey(key string) Config {
	c.APIKey = key
	return c
}

// requestTimeout returns the configured generation timeout, or the default.
func (c Config) requestTimeout() time.Duration {
	if c.TimeoutSec <= 0 {
		return defaultRequestTimeout
	}
	return time.Duration(c.TimeoutSec) * time.Second
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
		return newOllamaBackend(cfg.Endpoint, cfg.Model, cfg.MaxTokens, cfg.APIKey, cfg.requestTimeout())
	case "ollama":
		return newOllamaBackend(defaultOllamaEndpoint, cfg.Model, cfg.MaxTokens, cfg.APIKey, cfg.requestTimeout())
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
		b, err := newOllamaBackend(endpoint, cfg.Model, cfg.MaxTokens, cfg.APIKey, cfg.requestTimeout())
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

// Compile-time assertions. Both built-in backends must satisfy every
// capability interface: a backend that silently stopped implementing one
// would fall back to configured-only behaviour at runtime with no error, and
// per-request options would be ignored rather than refused.
var (
	_ StreamingBackend = (*ollamaBackend)(nil)
	_ OptionedBackend  = (*ollamaBackend)(nil)
)
