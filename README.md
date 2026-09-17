# Gollum

A multi-backend LLM inference library for Go. Provides a unified `Backend` interface that abstracts away the differences between local embedded inference and Ollama HTTP APIs.

## Why Gollum instead of using llama-go directly?

[tcpipuk/llama-go](https://github.com/tcpipuk/llama-go) is an excellent Go binding for llama.cpp, but it only does local embedded inference via CGO. Gollum builds on top of it to provide:

- **Multiple backends** — Ollama (local or remote) and embedded llama-go behind a single `Backend` interface. Callers use `Analyze(ctx, systemPrompt, userPrompt)` regardless of which backend is active.
- **Auto-detection** — `gollum.New()` with `"auto"` probes for a running Ollama instance and falls back to the embedded backend. The caller doesn't need to know which backend is available.
- **No CGO by default** — `go get github.com/networkinss/gollum` gives you the Ollama backend with zero CGO dependencies. The embedded llama-go backend is opt-in via `-tags llm`.
- **Model download** — Download GGUF model files from Hugging Face with SHA256 integrity verification.

If you only need direct embedded inference, use `tcpipuk/llama-go` directly. Gollum is for when you want multiple backend options behind a single interface.

## Installation

```bash
go get github.com/networkinss/gollum
```

This gives you the Ollama backend out of the box (pure Go, no CGO).

### Embedded backend (optional)

To enable the embedded llama-go backend:

```bash
git clone https://github.com/networkinss/gollum
cd gollum
./build-llm.sh
go build -tags llm
```

`build-llm.sh` clones `tcpipuk/llama-go` at the pinned commit into the gitignored `third_party/` and builds `libbinding.a`; there is no submodule on this repo itself. Prerequisites: `gcc`, `g++`, `make` (`sudo apt install build-essential`).

## Usage

```go
package main

import (
	"context"
	"fmt"
	"time"

	"github.com/networkinss/gollum"
)

func main() {
	cfg := gollum.DefaultConfig()
	backend, err := gollum.New(cfg)
	if err != nil {
		panic(err)
	}
	defer backend.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	response, err := backend.Analyze(ctx, "You are a helpful assistant.", "What is Go?")
	if err != nil {
		panic(err)
	}
	fmt.Println(response)
}
```

## Backend selection

Set `Config.Backend` to control which backend is used:

| Value | Behavior |
|---|---|
| `"auto"` (default) | Probe local Ollama, fall back to embedded |
| `"ollama"` | Use Ollama at `localhost:11434` |
| `"remote"` | Use Ollama at the URL specified in `Config.Endpoint` |
| `"embedded"` | Use local llama-go inference (requires `-tags llm` build) |

## Configuration

```go
gollum.Config{
	Backend:   "auto",                    // "auto", "ollama", "remote", "embedded"
	ModelPath: "/path/to/model.gguf",     // Embedded backend: path to GGUF file
	Model:     "llama3.2:3b",             // Ollama backend: model name
	Endpoint:  "http://host:11434",       // Remote backend: Ollama API URL
	MaxTokens: 512,                       // Maximum response tokens
	Threads:   4,                         // CPU threads (embedded backend)
	Checksum:  "",                        // SHA256 for model download verification
}
```

## Model download

```go
path, err := gollum.DownloadModel("", "", "")  // downloads default model (Llama 3.2 3B Q4_K_M)
```

Custom model:
```go
path, err := gollum.DownloadModel("https://huggingface.co/..../model.gguf", "/path/to/models", "")
```

## License

Apache License 2.0, plus a non-negotiable field-of-use restriction (military, weapons systems, government mass surveillance, intelligence agencies) — see [LICENSE](LICENSE). Rationale and adoption notes in [LICENSING.md](LICENSING.md).
