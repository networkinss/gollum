//go:build llm

package gollum

// Compile-time check that embeddedBackend satisfies StreamingBackend.
//
// There's no test here exercising real generation: doing so needs a GGUF
// model file, which this repo doesn't ship and CI shouldn't need to
// download. If you're adding real embedded-backend test coverage, gate it
// behind an env var pointing at a local model path and skip when unset,
// rather than requiring one unconditionally under -tags llm.
var _ StreamingBackend = (*embeddedBackend)(nil)
