package gollum

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// ollamaBackend talks to a local or remote Ollama-compatible API.
type ollamaBackend struct {
	endpoint  string
	model     string
	maxTokens int
	// apiKey, when non-empty, is sent as a bearer token. A local Ollama needs
	// none; hosted Ollama-compatible endpoints require one.
	apiKey string
	client *http.Client
}

// ollamaGenerateRequest is the request body for /api/generate.
type ollamaGenerateRequest struct {
	Model   string         `json:"model"`
	System  string         `json:"system,omitempty"`
	Prompt  string         `json:"prompt"`
	Stream  bool           `json:"stream"`
	Options map[string]any `json:"options,omitempty"`
}

// ollamaGenerateResponse is the response body from /api/generate (non-streaming).
type ollamaGenerateResponse struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
	Error    string `json:"error,omitempty"`
}

// defaultRequestTimeout bounds a single generation. Two minutes suits a small
// local model; a large or hosted one can need considerably longer, which is
// what Config.TimeoutSec is for.
const defaultRequestTimeout = 120 * time.Second

func newOllamaBackend(endpoint, model string, maxTokens int, apiKey string, timeout time.Duration) (*ollamaBackend, error) {
	if model == "" {
		model = "llama3.2:3b"
	}
	if maxTokens <= 0 {
		maxTokens = 512
	}
	if timeout <= 0 {
		timeout = defaultRequestTimeout
	}
	return &ollamaBackend{
		endpoint:  endpoint,
		model:     model,
		maxTokens: maxTokens,
		apiKey:    apiKey,
		client:    &http.Client{Timeout: timeout},
	}, nil
}

// genOptions builds Ollama's per-request options map, applying overrides on
// top of the backend's configured values. One place, so a parameter cannot be
// added to the non-streaming path and forgotten in the streaming one.
func (o *ollamaBackend) genOptions(opts GenerateOptions) map[string]any {
	m := map[string]any{"num_predict": o.maxTokens}
	if opts.MaxTokens > 0 {
		m["num_predict"] = opts.MaxTokens
	}
	// Sampling has no backend-level default here: Ollama applies the model's
	// own unless told otherwise, which is the better default than a number
	// this library invents.
	if opts.Temperature > 0 {
		m["temperature"] = opts.Temperature
	}
	if opts.TopK > 0 {
		m["top_k"] = opts.TopK
	}
	if opts.TopP > 0 {
		m["top_p"] = opts.TopP
	}
	return m
}

// setAuth applies the bearer token when one is configured. Every request path
// must call it, including the health check: a hosted endpoint answers an
// unauthenticated /api/version with 401, which would otherwise make Available()
// report a perfectly good backend as down.
func (o *ollamaBackend) setAuth(req *http.Request) {
	if o.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+o.apiKey)
	}
}

// setHeaders applies what a JSON request body needs, plus auth.
func (o *ollamaBackend) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	o.setAuth(req)
}

func (o *ollamaBackend) Name() string {
	return "ollama@" + o.endpoint
}

func (o *ollamaBackend) Analyze(ctx context.Context, systemPrompt, userPrompt string) (string, error) {
	return o.AnalyzeWithOptions(ctx, systemPrompt, userPrompt, GenerateOptions{})
}

// AnalyzeWithOptions is Analyze with per-request overrides.
func (o *ollamaBackend) AnalyzeWithOptions(ctx context.Context, systemPrompt, userPrompt string, opts GenerateOptions) (string, error) {
	reqBody := ollamaGenerateRequest{
		Model:   o.model,
		System:  systemPrompt,
		Prompt:  userPrompt,
		Stream:  false,
		Options: o.genOptions(opts),
	}
	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("ollama: failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.endpoint+"/api/generate", bytes.NewReader(jsonBody))
	if err != nil {
		return "", fmt.Errorf("ollama: failed to create request: %w", err)
	}
	o.setHeaders(req)

	resp, err := o.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("ollama: request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("ollama: failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("ollama: server returned %d: %s", resp.StatusCode, string(body))
	}

	var result ollamaGenerateResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return "", fmt.Errorf("ollama: failed to parse response: %w", err)
	}
	if result.Error != "" {
		return "", fmt.Errorf("ollama: %s", result.Error)
	}
	return result.Response, nil
}

// AnalyzeStream streams tokens from /api/generate as they're produced.
//
// Ollama's streaming response is newline-delimited JSON: one object per
// token/chunk, with a final object carrying Done=true. Returning false from
// onToken, or cancelling ctx, cancels the underlying HTTP request so the
// server stops generating rather than the client just stopping reading.
func (o *ollamaBackend) AnalyzeStream(ctx context.Context, systemPrompt, userPrompt string, onToken func(string) bool) error {
	return o.AnalyzeStreamWithOptions(ctx, systemPrompt, userPrompt, onToken, GenerateOptions{})
}

// AnalyzeStreamWithOptions is AnalyzeStream with per-request overrides.
func (o *ollamaBackend) AnalyzeStreamWithOptions(ctx context.Context, systemPrompt, userPrompt string, onToken func(string) bool, opts GenerateOptions) error {
	streamCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	reqBody := ollamaGenerateRequest{
		Model:   o.model,
		System:  systemPrompt,
		Prompt:  userPrompt,
		Stream:  true,
		Options: o.genOptions(opts),
	}
	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("ollama: failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(streamCtx, http.MethodPost, o.endpoint+"/api/generate", bytes.NewReader(jsonBody))
	if err != nil {
		return fmt.Errorf("ollama: failed to create request: %w", err)
	}
	o.setHeaders(req)

	resp, err := o.client.Do(req)
	if err != nil {
		return fmt.Errorf("ollama: request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("ollama: server returned %d: %s", resp.StatusCode, string(body))
	}

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		if ctx.Err() != nil {
			return ctx.Err()
		}

		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		var chunk ollamaGenerateResponse
		if err := json.Unmarshal(line, &chunk); err != nil {
			return fmt.Errorf("ollama: failed to parse stream chunk: %w", err)
		}
		if chunk.Error != "" {
			return fmt.Errorf("ollama: %s", chunk.Error)
		}

		if chunk.Response != "" && !onToken(chunk.Response) {
			return nil
		}
		if chunk.Done {
			return nil
		}
	}
	if err := scanner.Err(); err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return fmt.Errorf("ollama: failed to read stream: %w", err)
	}
	return nil
}

func (o *ollamaBackend) Available() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, o.endpoint+"/api/version", nil)
	if err != nil {
		return false
	}
	o.setAuth(req)
	resp, err := o.client.Do(req)
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

func (o *ollamaBackend) Close() error {
	return nil
}
