package gollum

import (
	"testing"
)

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.Backend != "auto" {
		t.Errorf("Backend = %q, want %q", cfg.Backend, "auto")
	}
	if cfg.Model != "llama3.2:3b" {
		t.Errorf("Model = %q, want %q", cfg.Model, "llama3.2:3b")
	}
	if cfg.MaxTokens != 512 {
		t.Errorf("MaxTokens = %d, want 512", cfg.MaxTokens)
	}
	if cfg.Threads != 4 {
		t.Errorf("Threads = %d, want 4", cfg.Threads)
	}
	if cfg.ModelPath != "" {
		t.Errorf("ModelPath = %q, want empty", cfg.ModelPath)
	}
	if cfg.Endpoint != "" {
		t.Errorf("Endpoint = %q, want empty", cfg.Endpoint)
	}
}

func TestConfigFromValues(t *testing.T) {
	cfg := ConfigFromValues("remote", "/path/to/model.gguf", "phi3:mini", "http://host:11434", 1024, 8, 16384, "abc123")
	if cfg.Backend != "remote" {
		t.Errorf("Backend = %q", cfg.Backend)
	}
	if cfg.ModelPath != "/path/to/model.gguf" {
		t.Errorf("ModelPath = %q", cfg.ModelPath)
	}
	if cfg.Model != "phi3:mini" {
		t.Errorf("Model = %q", cfg.Model)
	}
	if cfg.Endpoint != "http://host:11434" {
		t.Errorf("Endpoint = %q", cfg.Endpoint)
	}
	if cfg.MaxTokens != 1024 {
		t.Errorf("MaxTokens = %d", cfg.MaxTokens)
	}
	if cfg.Threads != 8 {
		t.Errorf("Threads = %d", cfg.Threads)
	}
	if cfg.ContextSize != 16384 {
		t.Errorf("ContextSize = %d", cfg.ContextSize)
	}
	if cfg.Checksum != "abc123" {
		t.Errorf("Checksum = %q", cfg.Checksum)
	}
}

func TestNew_RemoteRequiresEndpoint(t *testing.T) {
	_, err := New(Config{Backend: "remote", Endpoint: ""})
	if err == nil {
		t.Error("expected error when remote backend has no endpoint")
	}
}

func TestNew_RemoteWithEndpoint(t *testing.T) {
	b, err := New(Config{Backend: "remote", Endpoint: "http://fake:11434", Model: "test"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if b == nil {
		t.Error("expected non-nil backend")
	}
	if b.Name() == "" {
		t.Error("expected non-empty backend name")
	}
}

func TestNew_UnknownBackend(t *testing.T) {
	_, err := New(Config{Backend: "foobar"})
	if err == nil {
		t.Error("expected error for unknown backend")
	}
}

func TestNew_OllamaBackend(t *testing.T) {
	b, err := New(Config{Backend: "ollama", Model: "test"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if b == nil {
		t.Error("expected non-nil backend")
	}
}

// The embedded backend must NOT leave repetition unpenalised. llama.cpp
// defaults to 1.0 (off), and a small model at low temperature with no penalty
// loops until it runs out of tokens — observed in practice, not theory.
//
// Asserted on the constants because the backend itself needs -tags llm and a
// real model; this at least fails if someone "tidies" the default back to
// zero or to 1.0.
func TestEmbeddedRepetitionDefaultsAreActive(t *testing.T) {
	if defaultRepeatPenalty <= 1.0 {
		t.Errorf("defaultRepeatPenalty = %v; at or below 1.0 is no penalty at all", defaultRepeatPenalty)
	}
	if defaultPenaltyLastN <= 0 {
		t.Errorf("defaultPenaltyLastN = %d; a penalty over zero tokens does nothing", defaultPenaltyLastN)
	}
}
