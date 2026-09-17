package gollum

import (
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
	client    *http.Client
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

func newOllamaBackend(endpoint, model string, maxTokens int) (*ollamaBackend, error) {
	if model == "" {
		model = "llama3.2:3b"
	}
	if maxTokens <= 0 {
		maxTokens = 512
	}
	return &ollamaBackend{
		endpoint:  endpoint,
		model:     model,
		maxTokens: maxTokens,
		client:    &http.Client{Timeout: 120 * time.Second},
	}, nil
}

func (o *ollamaBackend) Name() string {
	return "ollama@" + o.endpoint
}

func (o *ollamaBackend) Analyze(ctx context.Context, systemPrompt, userPrompt string) (string, error) {
	reqBody := ollamaGenerateRequest{
		Model:  o.model,
		System: systemPrompt,
		Prompt: userPrompt,
		Stream: false,
		Options: map[string]any{
			"num_predict": o.maxTokens,
		},
	}
	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("ollama: failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.endpoint+"/api/generate", bytes.NewReader(jsonBody))
	if err != nil {
		return "", fmt.Errorf("ollama: failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

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

func (o *ollamaBackend) Available() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, o.endpoint+"/api/version", nil)
	if err != nil {
		return false
	}
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
