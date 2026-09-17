# Handover — streaming + publish-readiness

Read `CLAUDE.md` first for the repo's overall shape, licence, and build-tag rules. This file is the punch list for the session that picks this up.

## Why this repo needs work before edith's Pro AI feature can build on it

edith is building a Pro (paid) AI edition (`docs/PRO_EDITION.md` in the edith repo) that calls into gollum's `Backend` interface from `handler_ai_pro.go`. That handler needs **streaming tokens back to the UI** (so a summarise/ask/explain call shows incremental output, not a multi-second blocking wait) and **working cancellation** (so closing a tab or hitting Escape mid-generation actually stops the model). Neither exists in gollum today. This work has to land *here*, not in edith, because `Backend`'s concrete implementations (`backend_embedded.go`, `backend_ollama.go`) live in this repo and edith only sees the interface.

This repo was just extracted from edith/agippy as a standalone module (2026-09-17) and has never been tagged. That means: no external consumer depends on today's interface shape yet, so this is the only free window to change `Backend` without a breaking-change dance. Use it.

## Task 1 — `StreamingBackend` — DONE (2026-09-17)

Landed in `gollum.go` as an additive interface plus a helper:

```go
type StreamingBackend interface {
	Backend
	AnalyzeStream(ctx context.Context, systemPrompt, userPrompt string, onToken func(string) bool) error
}

func AsStreamingBackend(b Backend) (StreamingBackend, bool)
```

Both built-in backends implement it:

- **`backend_ollama.go`**: `AnalyzeStream` now sends `"stream": true` and reads the NDJSON response body line-by-line (`bufio.Scanner`). It wraps the request in its own `context.WithCancel(ctx)` so returning `false` from `onToken` or the caller cancelling `ctx` actually cancels the in-flight HTTP request (the server stops generating) rather than just stopping the client from reading further.
- **`backend_embedded.go`**: it turned out llama-go (pinned `b8a6878`) already exposes exactly this shape — `Context.GenerateStream(prompt, func(token string) bool, opts...)`. No pin bump was needed. `AnalyzeStream` wraps that callback to also check `ctx.Err()` per token (llama.cpp's decode loop has no native cancellation, so this is checked between tokens, not preemptively). **`Analyze` was rewritten in terms of `AnalyzeStream`** (accumulate tokens into a `strings.Builder`), which incidentally fixes the old bug where `Analyze(_ context.Context, ...)` discarded its context entirely — it's no longer discarded, `Analyze` now honours cancellation too.

**Still open:** no test coverage was added for either `AnalyzeStream` implementation. `gollum_test.go` currently has no case exercising streaming, early-stop-via-`onToken`, or ctx-cancellation-mid-stream for either backend. Next session should add:
- A mock `StreamingBackend` (or a lightweight fake HTTP server for `backend_ollama.go`, since it's the one testable without CGO/`-tags llm`) asserting tokens arrive in order.
- A case where `onToken` returns `false` after N tokens and asserts no further tokens/HTTP reads happen.
- A case with a pre-cancelled or short-deadline `ctx` asserting `AnalyzeStream`/`Analyze` return promptly with `ctx.Err()` (wrapped or not — check what's actually returned; the ollama path returns `ctx.Err()` directly when the scanner loop notices cancellation, the embedded path returns it via the final `return ctx.Err()`).
- `backend_embedded.go`'s tests need `-tags llm` and a real (small) GGUF model to run meaningfully — decide whether that's worth a CI job (see Task 4) or stays manual/local-only.

## Task 2 — fix the llama-go pin mismatch — DONE (2026-09-17)

`go.mod`'s `require` line was `v0.0.0-20260409130703-53d622fc7cd8` (9 Apr 2026), not matching `build-llm.sh`'s `LLAMA_PIN="b8a6878"` (actually **3 May 2026**, not 8 May as earlier notes assumed — verified via `git log -1 --format='%cI' b8a6878` against the local `third_party/llama-go` checkouts still present in edith's and agippy's repos). Updated `require` to `v0.0.0-20260503093322-b8a687898fa2`, matching `LLAMA_PIN` exactly. `build-llm.sh`'s header comment was also corrected (it wrongly said "Edithor pins..." and cited `docs/PRO_EDITION.md`, an edith-only file that doesn't exist in this repo).

**Still open:** this was a manual fix, not automated. If `LLAMA_PIN` is bumped again, nothing currently *enforces* that `go.mod`'s `require` line is updated in the same change — worth a small script or CI check (e.g. grep both files, extract the hash, compare) rather than relying on whoever edits `build-llm.sh` to remember `CLAUDE.md`'s instructions.

## Task 3 — resolve the README's build-instructions self-contradiction

Already partially fixed during extraction (the `git clone --recurse-submodules` line was wrong — there's no git submodule on *this* repo, `build-llm.sh` does its own clone of llama-go into `third_party/`). Double check nothing else in `README.md` still assumes the old two-repo-copy layout (e.g. any leftover reference to being vendored inside edith/agippy).

## Task 4 — decide the release/versioning story

This repo has never been tagged. Before edith's Pro AI handler can depend on a real version instead of a path-based `replace`:

- Pick a starting version (`v0.1.0` is reasonable — this is a young extraction, not a stable-for-years library).
- Decide whether `StreamingBackend` ships in that first tag or a `v0.2.0` right after — given edith's Pro feature is the reason this extraction happened, it probably makes sense to land `StreamingBackend` *before* cutting the first tag, so edith's `go.mod` update is a one-time `go get github.com/networkinss/gollum@v0.1.0` rather than an immediate bump.
- No CI exists yet in this repo (extraction was files-only, no `.github/workflows/`). At minimum: a workflow running `go test ./...` (default, no CGO) on push/PR. A `-tags llm` job needs `build-llm.sh` run first and will be slow (compiling llama.cpp) — worth caching `third_party/llama-go` keyed on `LLAMA_PIN`, mirroring the pattern edith's own `linux-pro` CI job already uses for the same clone (see `edith` repo's `.github/workflows/release.yml`, `linux-pro` job, for a working reference).

## Task 5 — update the two consumers once tagged

Not blocking for this session, but the reason all the above exists: once a tag lands, `edith`'s `go.mod` (`replace github.com/networkinss/gollum => ./libs/gollum` and the nested llama-go replace) and `agippy`'s equivalent should switch to a real versioned dependency, and `libs/gollum/` in both repos should be deleted rather than kept as a stale fork. Leave a note for whoever does that migration that `edith/CLAUDE.md`'s "Key dependencies" section and `docs/PRO_EDITION.md` both reference the vendored path and will need updating too.

## Non-goals for this session

- Don't touch the licence files (`LICENSE`, `NOTICE`, `LICENSING.md`) — that decision is settled; see `CLAUDE.md`'s licence section if tempted.
- Don't add a second inference backend (e.g. OpenAI-compatible HTTP) — out of scope, not requested.
- Don't try to make the embedded backend build without `gcc`/`g++`/`make`/CGO — that's an inherent constraint of linking llama.cpp, not a bug.
