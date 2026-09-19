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
	Backend:    "auto",                    // "auto", "ollama", "remote", "embedded"
	ModelPath:  "/path/to/model.gguf",     // Embedded backend: path to GGUF file
	Model:      "llama3.2:3b",             // Ollama backend: model name
	Endpoint:   "http://host:11434",       // Remote backend: Ollama API URL
	MaxTokens:  512,                       // Maximum response tokens
	Threads:    4,                         // CPU threads (embedded backend)
	Checksum:   "",                        // SHA256 for model download verification
	TimeoutSec: 0,                         // Per-request timeout; 0 = 120s
}
```

### Authenticated endpoints

A local Ollama needs no credentials. A hosted Ollama-compatible endpoint does,
and `APIKey` supplies it as a bearer token on every request — generation,
streaming and the `Available()` health check alike, since an unauthenticated
ping to such a service answers 401 and would report a working backend as down.

```go
cfg := gollum.DefaultConfig().WithAPIKey(os.Getenv("OLLAMA_API_KEY"))
cfg.Backend = "remote"
cfg.Endpoint = "https://example.com"
cfg.TimeoutSec = 300 // hosted models can be slower to first token
```

`WithAPIKey` is a builder rather than a parameter on `ConfigFromValues`,
because that function's signature is published API and consumers already call
it positionally. The key is `json:"-"`, so it is never written out with the
rest of a serialised config — storing it is the consumer's decision to make
deliberately.

## Per-request options

`MaxTokens` and the sampling parameters are bound when a backend is
constructed, so varying them per request would mean building a new backend —
and on the embedded engine that means reloading the model from disk. A caller
that wants a short answer for one action and a long one for the next should
not pay several seconds for it.

```go
if ob, ok := gollum.AsOptionedBackend(backend); ok {
	err = ob.AnalyzeStreamWithOptions(ctx, system, user, onToken,
		gollum.GenerateOptions{MaxTokens: 200})
}
```

A zero field means "use the value this backend was configured with", so the
zero `GenerateOptions` is exactly the behaviour of plain `Analyze` /
`AnalyzeStream` — which keep working unchanged and are still the right call
when there is nothing to override.

`OptionedBackend` is additive, like `StreamingBackend`: `Backend` is the
published contract and widening its signatures would break every third-party
implementation. Both built-in backends satisfy it, enforced by compile-time
assertions rather than by hope.

Note the sampling defaults differ by backend, on purpose. The embedded engine
applies temperature 0.3 / top-k 40 / top-p 0.9 because llama.cpp needs *some*
value; the Ollama backend sends no sampling keys at all unless you override
them, so the model's own defaults apply rather than numbers this library
invented.

## Model download

```go
path, err := gollum.DownloadModel("", "", "")  // downloads default model (Llama 3.2 3B Q4_K_M)
```

Custom model:
```go
path, err := gollum.DownloadModel("https://huggingface.co/..../model.gguf", "/path/to/models", "")
```

`DownloadModel` prints progress to stdout, which suits a CLI. **A GUI consumer
wants `DownloadModelContext`**: stdout is invisible in a windowed application,
and a multi-gigabyte transfer needs to be cancellable.

```go
path, err := gollum.DownloadModelContext(ctx, url, destDir, checksum,
	func(p gollum.DownloadProgress) {
		// p.Total is 0 when the server sends no Content-Length —
		// show bytes rather than a percentage in that case.
		emit(p.Downloaded, p.Total)
	})
```

Callbacks are throttled to roughly one per 100ms, because a read returns tens
of kilobytes at a time and an un-throttled callback fires tens of thousands of
times for a real model. The final state is always reported, throttle or not, so
a progress bar cannot stop short of the end.

Cancelling `ctx` aborts the transfer promptly and removes the partial file. The
download is written to `<dest>.tmp` and renamed into place only once the
checksum verifies, so an interrupted or corrupt download never leaves a
truncated file where a model discovery would find it.

## License

Apache License 2.0, plus a non-negotiable field-of-use restriction (military, weapons systems, government mass surveillance, intelligence agencies) — see [LICENSE](LICENSE). Rationale and adoption notes in [LICENSING.md](LICENSING.md).
