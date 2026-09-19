package gollum

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// Compile-time check that ollamaBackend satisfies StreamingBackend.
var _ StreamingBackend = (*ollamaBackend)(nil)

// newStreamingTestServer serves an Ollama-style NDJSON /api/generate stream,
// one line per token in tokens plus a final {"done":true} line. It pauses
// briefly between writes so a client-side early stop or ctx cancellation has
// a chance to interrupt the stream before every token is sent, and reports
// via the returned *atomic.Bool whether the handler observed the request
// context being cancelled mid-stream.
func newStreamingTestServer(tokens []string, delay time.Duration) (*httptest.Server, *atomic.Bool) {
	cancelled := &atomic.Bool{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		flusher, _ := w.(http.Flusher)
		for _, tok := range tokens {
			select {
			case <-r.Context().Done():
				cancelled.Store(true)
				return
			default:
			}
			chunk, _ := json.Marshal(ollamaGenerateResponse{Response: tok})
			if _, err := w.Write(append(chunk, '\n')); err != nil {
				cancelled.Store(true)
				return
			}
			if flusher != nil {
				flusher.Flush()
			}
			if delay > 0 {
				time.Sleep(delay)
			}
		}
		done, _ := json.Marshal(ollamaGenerateResponse{Done: true})
		w.Write(append(done, '\n'))
		if flusher != nil {
			flusher.Flush()
		}
	}))
	return srv, cancelled
}

func waitFor(t *testing.T, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Error("condition not met within timeout")
}

func TestOllamaAnalyzeStream_TokensInOrder(t *testing.T) {
	srv, _ := newStreamingTestServer([]string{"Hello", ",", " world"}, 0)
	defer srv.Close()

	b, err := newOllamaBackend(srv.URL, "test-model", 10, "", 0)
	if err != nil {
		t.Fatalf("newOllamaBackend: %v", err)
	}

	var got []string
	err = b.AnalyzeStream(context.Background(), "system", "user", func(token string) bool {
		got = append(got, token)
		return true
	})
	if err != nil {
		t.Fatalf("AnalyzeStream: %v", err)
	}
	want := []string{"Hello", ",", " world"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("tokens = %v, want %v", got, want)
	}
}

func TestOllamaAnalyzeStream_EarlyStopViaOnToken(t *testing.T) {
	srv, cancelled := newStreamingTestServer([]string{"a", "b", "c", "d", "e"}, 20*time.Millisecond)
	defer srv.Close()

	b, err := newOllamaBackend(srv.URL, "test-model", 10, "", 0)
	if err != nil {
		t.Fatalf("newOllamaBackend: %v", err)
	}

	var got []string
	err = b.AnalyzeStream(context.Background(), "system", "user", func(token string) bool {
		got = append(got, token)
		return len(got) < 2
	})
	if err != nil {
		t.Fatalf("AnalyzeStream: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("got %d tokens, want exactly 2 (no further calls after stop): %v", len(got), got)
	}

	// The underlying request is cancelled once AnalyzeStream returns (via its
	// own context.WithCancel); the server should notice and stop generating
	// rather than run the remaining tokens to completion.
	waitFor(t, cancelled.Load)
}

func TestOllamaAnalyzeStream_ContextCancellation(t *testing.T) {
	srv, cancelled := newStreamingTestServer([]string{"a", "b", "c", "d", "e"}, 30*time.Millisecond)
	defer srv.Close()

	b, err := newOllamaBackend(srv.URL, "test-model", 10, "", 0)
	if err != nil {
		t.Fatalf("newOllamaBackend: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	var got []string
	go func() {
		time.Sleep(15 * time.Millisecond)
		cancel()
	}()

	err = b.AnalyzeStream(ctx, "system", "user", func(token string) bool {
		got = append(got, token)
		return true
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("AnalyzeStream error = %v, want context.Canceled", err)
	}
	if len(got) >= 5 {
		t.Errorf("expected cancellation to stop the stream before all tokens arrived, got %v", got)
	}
	waitFor(t, cancelled.Load)
}

func TestOllamaAnalyzeStream_ServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		chunk, _ := json.Marshal(ollamaGenerateResponse{Error: "boom"})
		w.Write(append(chunk, '\n'))
	}))
	defer srv.Close()

	b, err := newOllamaBackend(srv.URL, "test-model", 10, "", 0)
	if err != nil {
		t.Fatalf("newOllamaBackend: %v", err)
	}

	err = b.AnalyzeStream(context.Background(), "system", "user", func(string) bool { return true })
	if err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("AnalyzeStream error = %v, want error containing %q", err, "boom")
	}
}

func TestOllamaAnalyze_NonStreaming(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req ollamaGenerateRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}
		if req.Stream {
			t.Errorf("Analyze must not set Stream=true")
		}
		resp, _ := json.Marshal(ollamaGenerateResponse{Response: "hello there", Done: true})
		w.Write(resp)
	}))
	defer srv.Close()

	b, err := newOllamaBackend(srv.URL, "test-model", 10, "", 0)
	if err != nil {
		t.Fatalf("newOllamaBackend: %v", err)
	}

	got, err := b.Analyze(context.Background(), "system", "user")
	if err != nil {
		t.Fatalf("Analyze: %v", err)
	}
	if got != "hello there" {
		t.Errorf("Analyze = %q, want %q", got, "hello there")
	}
}

// captureAuth serves a minimal /api/generate and records the Authorization
// header it was sent.
func captureAuth(t *testing.T, seen *string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		*seen = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"response":"ok","done":true}`))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestOllamaSendsBearerTokenWhenConfigured(t *testing.T) {
	var seen string
	srv := captureAuth(t, &seen)

	b, err := newOllamaBackend(srv.URL, "test-model", 10, "sk-test-123", 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := b.Analyze(context.Background(), "sys", "user"); err != nil {
		t.Fatalf("Analyze: %v", err)
	}
	if seen != "Bearer sk-test-123" {
		t.Errorf("Authorization = %q, want %q", seen, "Bearer sk-test-123")
	}
}

// A local daemon needs no token, and sending an empty one would be wrong.
func TestOllamaSendsNoAuthHeaderWithoutAKey(t *testing.T) {
	var seen string
	srv := captureAuth(t, &seen)

	b, err := newOllamaBackend(srv.URL, "test-model", 10, "", 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := b.Analyze(context.Background(), "sys", "user"); err != nil {
		t.Fatalf("Analyze: %v", err)
	}
	if seen != "" {
		t.Errorf("Authorization = %q, want no header at all", seen)
	}
}

// The streaming path must authenticate too — it is a separate request.
func TestOllamaStreamSendsBearerToken(t *testing.T) {
	var seen string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.Write([]byte(`{"response":"hi","done":false}` + "\n" + `{"response":"","done":true}` + "\n"))
	}))
	defer srv.Close()

	b, err := newOllamaBackend(srv.URL, "test-model", 10, "sk-stream", 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := b.AnalyzeStream(context.Background(), "sys", "user", func(string) bool { return true }); err != nil {
		t.Fatalf("AnalyzeStream: %v", err)
	}
	if seen != "Bearer sk-stream" {
		t.Errorf("Authorization = %q, want %q", seen, "Bearer sk-stream")
	}
}

// Available() pings /api/version. A hosted endpoint answers an unauthenticated
// ping with 401, which would report a working backend as down.
func TestOllamaAvailableSendsBearerToken(t *testing.T) {
	var seen string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = r.Header.Get("Authorization")
		if r.Header.Get("Authorization") == "" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		w.Write([]byte(`{"version":"0.1.0"}`))
	}))
	defer srv.Close()

	b, err := newOllamaBackend(srv.URL, "test-model", 10, "sk-ping", 0)
	if err != nil {
		t.Fatal(err)
	}
	if !b.Available() {
		t.Error("Available() = false against a server that requires auth")
	}
	if seen != "Bearer sk-ping" {
		t.Errorf("Authorization = %q, want %q", seen, "Bearer sk-ping")
	}
}

// The token must never appear in the backend's identity, which callers log
// and display.
func TestOllamaNameDoesNotLeakTheToken(t *testing.T) {
	b, err := newOllamaBackend("https://example.invalid", "m", 10, "sk-secret-value", 0)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(b.Name(), "sk-secret-value") {
		t.Fatalf("Name() leaks the API key: %q", b.Name())
	}
}

func TestConfigWithAPIKeyAndTimeout(t *testing.T) {
	cfg := DefaultConfig().WithAPIKey("sk-abc")
	if cfg.APIKey != "sk-abc" {
		t.Errorf("WithAPIKey did not set the key: %+v", cfg)
	}

	if got := DefaultConfig().requestTimeout(); got != defaultRequestTimeout {
		t.Errorf("default requestTimeout() = %v, want %v", got, defaultRequestTimeout)
	}
	cfg.TimeoutSec = 300
	if got := cfg.requestTimeout(); got != 300*time.Second {
		t.Errorf("requestTimeout() = %v, want 5m", got)
	}
}

// The API key must not be serialised into a config file by accident.
func TestAPIKeyIsNotMarshalled(t *testing.T) {
	out, err := json.Marshal(DefaultConfig().WithAPIKey("sk-secret-value"))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(out), "sk-secret-value") {
		t.Fatalf("Config JSON contains the API key: %s", out)
	}
}

// captureOptions serves /api/generate and records the options map it was sent.
func captureOptions(t *testing.T, seen *map[string]any) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Options map[string]any `json:"options"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		*seen = body.Options
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"response":"ok","done":true}`))
	}))
	t.Cleanup(srv.Close)
	return srv
}

// The point of per-request options: a caller can cap one answer without
// rebuilding the backend, which on the embedded engine means reloading a
// multi-gigabyte model.
func TestPerRequestMaxTokensOverridesTheConfiguredValue(t *testing.T) {
	var seen map[string]any
	srv := captureOptions(t, &seen)

	b, err := newOllamaBackend(srv.URL, "test-model", 999, "", 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := b.AnalyzeWithOptions(context.Background(), "s", "u", GenerateOptions{MaxTokens: 42}); err != nil {
		t.Fatal(err)
	}
	if got := seen["num_predict"]; got != float64(42) {
		t.Fatalf("num_predict = %v, want 42 (the per-request override)", got)
	}
}

// A zero GenerateOptions must be exactly the old behaviour.
func TestZeroOptionsUsesTheConfiguredValue(t *testing.T) {
	var seen map[string]any
	srv := captureOptions(t, &seen)

	b, err := newOllamaBackend(srv.URL, "test-model", 77, "", 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := b.AnalyzeWithOptions(context.Background(), "s", "u", GenerateOptions{}); err != nil {
		t.Fatal(err)
	}
	if got := seen["num_predict"]; got != float64(77) {
		t.Fatalf("num_predict = %v, want the configured 77", got)
	}
	// Sampling keys must be absent entirely, so the model's own defaults
	// apply rather than numbers this library invented.
	for _, k := range []string{"temperature", "top_k", "top_p"} {
		if _, present := seen[k]; present {
			t.Errorf("zero options sent %q; the model's default should apply", k)
		}
	}
}

// The plain Analyze/AnalyzeStream must keep working untouched — they are the
// published contract and existing consumers call them.
func TestPlainAnalyzeStillUsesConfiguredValue(t *testing.T) {
	var seen map[string]any
	srv := captureOptions(t, &seen)

	b, err := newOllamaBackend(srv.URL, "test-model", 55, "", 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := b.Analyze(context.Background(), "s", "u"); err != nil {
		t.Fatal(err)
	}
	if got := seen["num_predict"]; got != float64(55) {
		t.Fatalf("num_predict = %v, want 55", got)
	}
}

// Options must reach the streaming path too — it is a separate request body,
// and that is exactly where a parameter gets forgotten.
func TestPerRequestOptionsReachTheStreamingPath(t *testing.T) {
	var seen map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Options map[string]any `json:"options"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		seen = body.Options
		w.Write([]byte(`{"response":"hi","done":true}` + "\n"))
	}))
	defer srv.Close()

	b, err := newOllamaBackend(srv.URL, "test-model", 999, "", 0)
	if err != nil {
		t.Fatal(err)
	}
	err = b.AnalyzeStreamWithOptions(context.Background(), "s", "u",
		func(string) bool { return true },
		GenerateOptions{MaxTokens: 13, Temperature: 0.9, TopK: 7, TopP: 0.5})
	if err != nil {
		t.Fatal(err)
	}
	if got := seen["num_predict"]; got != float64(13) {
		t.Errorf("num_predict = %v, want 13", got)
	}
	if got := seen["temperature"]; got != 0.9 {
		t.Errorf("temperature = %v, want 0.9", got)
	}
	if got := seen["top_k"]; got != float64(7) {
		t.Errorf("top_k = %v, want 7", got)
	}
}

func TestGenerateOptionsIsZero(t *testing.T) {
	if !(GenerateOptions{}).IsZero() {
		t.Error("the zero value does not report IsZero")
	}
	if (GenerateOptions{MaxTokens: 1}).IsZero() {
		t.Error("a populated value reports IsZero")
	}
}

func TestAsOptionedBackend(t *testing.T) {
	b, err := newOllamaBackend("http://example.invalid", "m", 10, "", 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := AsOptionedBackend(b); !ok {
		t.Fatal("the Ollama backend does not satisfy OptionedBackend")
	}
}
