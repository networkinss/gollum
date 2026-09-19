package gollum

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// payload returns deterministic bytes plus their SHA256, so a test can supply
// the checksum DownloadModelContext will verify against.
func payload(size int) ([]byte, string) {
	b := make([]byte, size)
	for i := range b {
		b[i] = byte(i % 251)
	}
	sum := sha256.Sum256(b)
	return b, hex.EncodeToString(sum[:])
}

func serveBytes(t *testing.T, body []byte) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/octet-stream")
		w.Write(body)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestDownloadModelContextWritesAndVerifies(t *testing.T) {
	body, sum := payload(64 << 10)
	srv := serveBytes(t, body)
	dir := t.TempDir()

	var (
		mu      sync.Mutex
		updates []DownloadProgress
	)
	path, err := DownloadModelContext(context.Background(), srv.URL+"/model.gguf", dir, sum,
		func(p DownloadProgress) {
			mu.Lock()
			updates = append(updates, p)
			mu.Unlock()
		})
	if err != nil {
		t.Fatalf("DownloadModelContext: %v", err)
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read downloaded file: %v", err)
	}
	if len(got) != len(body) {
		t.Fatalf("downloaded %d bytes, want %d", len(got), len(body))
	}

	// The final update must report the complete size, or a progress bar
	// sticks short of the end.
	if len(updates) == 0 {
		t.Fatal("no progress reported")
	}
	last := updates[len(updates)-1]
	if last.Downloaded != int64(len(body)) {
		t.Errorf("final progress reports %d bytes, want %d", last.Downloaded, len(body))
	}

	// Nothing may be left behind at the temp path.
	if _, err := os.Stat(path + ".tmp"); !os.IsNotExist(err) {
		t.Error(".tmp file survived a successful download")
	}
}

func TestDownloadModelContextRejectsBadChecksum(t *testing.T) {
	body, _ := payload(32 << 10)
	srv := serveBytes(t, body)
	dir := t.TempDir()

	wrong := strings.Repeat("00", 32)
	path, err := DownloadModelContext(context.Background(), srv.URL+"/model.gguf", dir, wrong, nil)
	if err == nil {
		t.Fatal("a mismatched checksum was accepted")
	}
	if path != "" {
		t.Errorf("a failed download returned a path: %q", path)
	}

	// Neither the final file nor the temp file may survive: a corrupt model
	// left on disk would be picked up by the next model discovery.
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		t.Errorf("failed download left %q behind", e.Name())
	}
}

func TestDownloadModelContextCancels(t *testing.T) {
	// A server that dribbles bytes, so cancellation lands mid-transfer.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "10485760") // claim 10 MiB
		w.WriteHeader(http.StatusOK)
		chunk := make([]byte, 4096)
		for i := 0; i < 2560; i++ {
			if _, err := w.Write(chunk); err != nil {
				return
			}
			w.(http.Flusher).Flush()
			time.Sleep(time.Millisecond)
		}
	}))
	defer srv.Close()

	dir := t.TempDir()
	ctx, cancel := context.WithCancel(context.Background())

	// Cancel as soon as the transfer is demonstrably under way.
	done := make(chan struct{})
	go func() {
		defer close(done)
		_, err := DownloadModelContext(ctx, srv.URL+"/model.gguf", dir, "", func(p DownloadProgress) {
			if p.Downloaded > 0 {
				cancel()
			}
		})
		if err == nil {
			t.Error("a cancelled download returned no error")
			return
		}
		if !errors.Is(err, context.Canceled) {
			t.Errorf("error does not wrap context.Canceled: %v", err)
		}
	}()

	select {
	case <-done:
	case <-time.After(30 * time.Second):
		cancel()
		t.Fatal("cancelled download did not return promptly")
	}

	// The partial file must not survive — DiscoverModel-style lookups would
	// otherwise find a truncated model.
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		t.Errorf("cancelled download left %q behind", e.Name())
	}
}

func TestDownloadModelContextSkipsExistingFile(t *testing.T) {
	dir := t.TempDir()
	dest := filepath.Join(dir, "model.gguf")
	if err := os.WriteFile(dest, []byte("already here"), 0o644); err != nil {
		t.Fatal(err)
	}

	// A server that fails the test if it is contacted at all.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("server was contacted although the file already exists")
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	path, err := DownloadModelContext(context.Background(), srv.URL+"/model.gguf", dir, "", nil)
	if err != nil {
		t.Fatalf("DownloadModelContext: %v", err)
	}
	if path != dest {
		t.Errorf("path = %q, want the existing file %q", path, dest)
	}
}

// Progress must be throttled: an un-throttled callback fires once per Read,
// which is tens of thousands of times for a real model.
func TestProgressIsThrottled(t *testing.T) {
	body, sum := payload(4 << 20) // 4 MiB — many reads
	srv := serveBytes(t, body)

	var count int
	_, err := DownloadModelContext(context.Background(), srv.URL+"/model.gguf", t.TempDir(), sum,
		func(DownloadProgress) { count++ })
	if err != nil {
		t.Fatalf("DownloadModelContext: %v", err)
	}
	// 4 MiB at 32 KiB per read is ~128 reads; throttled to 100ms it should be
	// a small handful. The exact number depends on disk and CPU speed, so the
	// assertion is only that throttling happened at all.
	if count > 40 {
		t.Errorf("progress fired %d times for a 4 MiB download — throttling is not working", count)
	}
	if count == 0 {
		t.Error("progress never fired")
	}
}

// DownloadModel must keep working unchanged: it is the published signature.
func TestDownloadModelWrapperStillWorks(t *testing.T) {
	body, sum := payload(8 << 10)
	srv := serveBytes(t, body)

	path, err := DownloadModel(srv.URL+"/model.gguf", t.TempDir(), sum)
	if err != nil {
		t.Fatalf("DownloadModel: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("stat %s: %v", path, err)
	}
}
