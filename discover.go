package gollum

import (
	"net/http"
	"time"
)

const defaultOllamaEndpoint = "http://localhost:11434"

// DiscoverLocalOllama probes for a running Ollama instance on localhost.
// Returns the endpoint URL and true if Ollama responds, or ("", false) otherwise.
func DiscoverLocalOllama() (string, bool) {
	client := &http.Client{Timeout: 500 * time.Millisecond}
	resp, err := client.Get(defaultOllamaEndpoint + "/api/version")
	if err != nil {
		return "", false
	}
	resp.Body.Close()
	if resp.StatusCode == http.StatusOK {
		return defaultOllamaEndpoint, true
	}
	return "", false
}
