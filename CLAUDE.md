# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## What this is

Gollum is a standalone Go library: a multi-backend LLM inference abstraction (`Backend` interface) over Ollama's HTTP API and embedded llama.cpp inference via `tcpipuk/llama-go`. It was extracted on 2026-09-17 from two consumer projects — **edith** (Edithor, at `libs/gollum/`) and **agippy** — where it had lived as copy-pasted, unversioned source. Both consumers will move to `go get github.com/networkinss/gollum` once this repo is tagged; until then this repo is the source of truth and the copies in edith/agippy are stale forks that should not diverge further.

## Licence

Apache License 2.0 **plus** an appended, non-negotiable field-of-use restriction (military/paramilitary use, weapons systems, government mass-surveillance, intelligence-agency use) — see `LICENSE`. This is deliberate and not up for relaxation without the licensor's (Oliver Glas's) explicit sign-off: see `LICENSING.md` for the full rationale, the licences considered and rejected (SUL, PolyForm/Caldun-style), and the accepted cost (this is not OSI-approved and most SCA/licence-scanner tooling will flag it as "custom", not as recognised `Apache-2.0`).

Practical implications for anyone touching this repo:
- Don't silently drop the restriction section when reformatting `LICENSE`, and don't add an `SPDX-License-Identifier: Apache-2.0` header to source files without the addendum noted alongside it (there's no registered SPDX id for the addendum itself — see `LICENSING.md`'s checklist).
- `NOTICE` carries third-party attribution for `llama-go` and `llama.cpp` (both MIT). Keep it in sync if a new dependency behind `-tags llm` is added.

## Build tags

- Default build (no tags): pure-Go Ollama backend only, zero CGO. `backend_noembedded.go` provides a stub `newEmbeddedBackend` returning `ErrNotCompiled`.
- `-tags llm`: enables `backend_embedded.go`'s real CGO binding to llama-go, requiring `third_party/llama-go/libbinding.a` to exist first.

**`go build` without `-tags llm` must always work with `third_party/` absent** — this is the whole point of the tag split, and CI for the default build must never depend on running `build-llm.sh`.

## The llama-go pin

`build-llm.sh` clones `tcpipuk/llama-go` at a pinned short commit (`LLAMA_PIN`, currently `b8a6878`, 2026-05-03 — the last commit on Go 1.26.2 with static linkage by default; anything before it links dynamically and the resulting binary won't start, checkable via `ldd`) into the gitignored `third_party/llama-go`, then builds `libbinding.a`. `go.mod`'s `require github.com/tcpipuk/llama-go` line (pinned as pseudo-version `v0.0.0-20260503093322-b8a687898fa2`) and the local `replace ... => ./third_party/llama-go` must resolve to **that same commit** — a `go.mod` `require` pseudo-version that doesn't match `LLAMA_PIN` is a latent bug (masked locally by the `replace`, but a real risk for anyone who forks this repo, edits `go.mod`, or tries to consume `-tags llm` without running `build-llm.sh` first — `go mod tidy` without `third_party/` present would silently resolve `require` to whatever's actually on the module proxy, reintroducing drift). If you bump `LLAMA_PIN`, update the `require` line's pseudo-version to match in the same change (`git log -1 --format='%cI' <commit>` for the UTC timestamp, `git rev-parse <commit>` for the hash — Go pseudo-versions use `v0.0.0-yyyymmddhhmmss-<12-hex-hash>`), and check whether the new commit needs a newer Go toolchain (llama-go HEAD has required Go 1.27 in the past against this repo's Go 1.26.2).

## Consumers

edith and agippy both vendor a copy of this source under `libs/gollum/` (or similar) via a `replace github.com/networkinss/gollum => ./libs/gollum` in their own `go.mod`. Once this repo has a tagged release, the intended end state is both consumers dropping the `replace` and pinning a real version. Until then:
- A behavioural change here should be manually back-ported to both copies (or, better, prompt the user to move a consumer to `go get` this repo instead of continuing to hand-sync).
- Don't assume either consumer's copy is current — diff before trusting it as a reference.

## Interface stability

`Backend` (`Name()`, `Analyze(ctx, systemPrompt, userPrompt) (string, error)`, `Available()`, `Close()`) is the only public contract consumers depend on today. `backend_embedded.go`'s `Analyze` currently **discards its `context.Context` parameter** — cancellation is not honoured on the embedded backend. A `StreamingBackend` (`Backend` plus `AnalyzeStream(ctx, systemPrompt, userPrompt, onToken func(string) bool) error`) is planned as an additive interface for edith's Pro AI feature; land and freeze its shape here, in this repo, before the first tag — don't let it get designed ad hoc in a consumer.

## Testing

```bash
go test ./...              # default build, Ollama backend only
go test -tags llm ./...    # needs ./build-llm.sh run first
```

No test suite depends on a real Ollama server or real GGUF model being present — keep it that way; mock/stub the HTTP and CGO boundaries rather than requiring external services in CI.
