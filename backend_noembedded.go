//go:build !llm

package gollum

func newEmbeddedBackend(cfg Config) (Backend, error) {
	return nil, ErrNotCompiled
}
