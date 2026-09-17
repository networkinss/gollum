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

	b, err := newOllamaBackend(srv.URL, "test-model", 10)
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

	b, err := newOllamaBackend(srv.URL, "test-model", 10)
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

	b, err := newOllamaBackend(srv.URL, "test-model", 10)
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

	b, err := newOllamaBackend(srv.URL, "test-model", 10)
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

	b, err := newOllamaBackend(srv.URL, "test-model", 10)
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
