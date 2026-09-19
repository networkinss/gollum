package gollum

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	// DefaultModelURL is a small, capable model suitable for security analysis on CPU.
	// Llama 3.2 3B, Q4_K_M quantization (~2.0 GB), 128K context.
	DefaultModelURL  = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
	DefaultModelName = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
	DefaultModelDir  = "/var/lib/agippy/models"

	// DefaultModelSHA256 is the known SHA256 of the default model file.
	DefaultModelSHA256 = "6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff"
)

// DefaultModelPath returns the full path where the default model is stored.
func DefaultModelPath() string {
	return filepath.Join(DefaultModelDir, DefaultModelName)
}

// DownloadProgress reports how far a model download has got. Total is 0 when
// the server sends no Content-Length, in which case only Downloaded is
// meaningful — show bytes, not a percentage.
type DownloadProgress struct {
	Downloaded int64
	Total      int64
}

// DownloadModel downloads a GGUF model file from the given URL to destDir.
//
// It prints progress to stdout, which suits a CLI. A GUI consumer wants
// DownloadModelContext instead: stdout is invisible in a windowed application,
// and this signature has no way to cancel a multi-gigabyte transfer.
func DownloadModel(url, destDir, configChecksum string) (string, error) {
	return DownloadModelContext(context.Background(), url, destDir, configChecksum, printProgress())
}

// printProgress returns the stdout reporter DownloadModel uses, matching the
// output this package produced before progress became a callback.
func printProgress() func(DownloadProgress) {
	lastPct := -1
	return func(p DownloadProgress) {
		if p.Total <= 0 {
			return
		}
		pct := int(p.Downloaded * 100 / p.Total)
		if pct != lastPct && pct%5 == 0 {
			lastPct = pct
			fmt.Printf("\r  Progress: %d%% (%.0f / %.0f MB)",
				pct,
				float64(p.Downloaded)/(1024*1024),
				float64(p.Total)/(1024*1024))
		}
	}
}

// DownloadModelContext downloads a GGUF model, reporting progress through
// onProgress (which may be nil) and honouring ctx.
//
// Cancelling ctx aborts the transfer promptly and removes the partial file, so
// a cancelled download leaves nothing behind for DiscoverModel-style lookups
// to trip over. The download is written to "<dest>.tmp" and renamed into place
// only after the checksum verifies, so an interrupted run can never leave a
// truncated file at the real path.
//
// If url is empty, the default model is downloaded. configChecksum is an
// optional SHA256 hex digest from the user's config; if empty, the hardcoded
// hash (for the default model) or the Hugging Face API is used.
func DownloadModelContext(ctx context.Context, url, destDir, configChecksum string, onProgress func(DownloadProgress)) (string, error) {
	if url == "" {
		url = DefaultModelURL
	}
	if destDir == "" {
		destDir = DefaultModelDir
	}

	if err := os.MkdirAll(destDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create model directory %s: %w", destDir, err)
	}

	destPath := filepath.Join(destDir, DefaultModelName)
	if url != DefaultModelURL {
		destPath = filepath.Join(destDir, filepath.Base(url))
	}

	// Check if already downloaded.
	if info, err := os.Stat(destPath); err == nil && info.Size() > 0 {
		fmt.Printf("Model already exists: %s (%.1f MB)\n", destPath, float64(info.Size())/(1024*1024))
		return destPath, nil
	}

	fmt.Printf("Downloading model from:\n  %s\n", url)
	fmt.Printf("Destination: %s\n", destPath)

	// The timeout is a backstop for a stalled transfer; ctx is what a caller
	// uses to cancel deliberately.
	client := &http.Client{Timeout: 30 * time.Minute}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", fmt.Errorf("download failed: %w", err)
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("download failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("download failed: HTTP %d", resp.StatusCode)
	}

	tmpPath := destPath + ".tmp"
	file, err := os.Create(tmpPath)
	if err != nil {
		return "", fmt.Errorf("failed to create file: %w", err)
	}

	totalSize := resp.ContentLength
	written, err := io.Copy(file, &progressWriter{
		ctx:      ctx,
		reader:   resp.Body,
		total:    totalSize,
		onUpdate: onProgress,
	})
	file.Close()
	if err != nil {
		os.Remove(tmpPath)
		if ctxErr := ctx.Err(); ctxErr != nil {
			return "", fmt.Errorf("download cancelled: %w", ctxErr)
		}
		return "", fmt.Errorf("download interrupted: %w", err)
	}

	// Determine expected checksum and verify integrity.
	if err := ctx.Err(); err != nil {
		os.Remove(tmpPath)
		return "", fmt.Errorf("download cancelled: %w", err)
	}

	expectedHash := resolveExpectedChecksum(url, configChecksum)
	if expectedHash != "" {
		fmt.Print("\nVerifying checksum...")
		if err := verifyChecksum(tmpPath, expectedHash); err != nil {
			os.Remove(tmpPath)
			return "", fmt.Errorf("integrity check failed: %w", err)
		}
		fmt.Println(" OK")
	} else {
		slog.Warn("no checksum available — skipping integrity verification")
	}

	if err := os.Rename(tmpPath, destPath); err != nil {
		os.Remove(tmpPath)
		return "", fmt.Errorf("failed to finalize download: %w", err)
	}

	fmt.Printf("\nDownload complete: %s (%.1f MB)\n", destPath, float64(written)/(1024*1024))
	return destPath, nil
}

// resolveExpectedChecksum determines the SHA256 to verify against.
// Priority: configChecksum > Hugging Face API > hardcoded default.
func resolveExpectedChecksum(url, configChecksum string) string {
	if configChecksum != "" {
		return configChecksum
	}

	// Try Hugging Face API for HF URLs.
	if strings.Contains(url, "huggingface.co/") {
		if hash, err := fetchHFChecksum(url); err == nil && hash != "" {
			return hash
		}
		slog.Debug("Hugging Face API unreachable, falling back to hardcoded checksum")
	}

	// Fall back to hardcoded hash for the default model.
	if url == DefaultModelURL {
		return DefaultModelSHA256
	}

	return ""
}

// verifyChecksum computes the SHA256 of the file at filePath and compares it
// against expectedHash (lowercase hex).
func verifyChecksum(filePath, expectedHash string) error {
	f, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("open failed: %w", err)
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return fmt.Errorf("read failed: %w", err)
	}

	got := fmt.Sprintf("%x", h.Sum(nil))
	if got != strings.ToLower(expectedHash) {
		return fmt.Errorf("checksum mismatch: expected %s, got %s", expectedHash, got)
	}
	return nil
}

// hfTreeEntry represents a file entry from the Hugging Face tree API.
type hfTreeEntry struct {
	Path string `json:"path"`
	LFS  *struct {
		OID  string `json:"oid"`
		Size int64  `json:"size"`
	} `json:"lfs"`
}

// fetchHFChecksum queries the Hugging Face tree API to get the SHA256 of a file.
// url must be a huggingface.co resolve URL like:
// https://huggingface.co/{org}/{repo}/resolve/main/{filename}
func fetchHFChecksum(url string) (string, error) {
	// Parse org/repo and filename from the URL.
	// Format: https://huggingface.co/{org}/{repo}/resolve/{ref}/{filename}
	const prefix = "huggingface.co/"
	idx := strings.Index(url, prefix)
	if idx < 0 {
		return "", fmt.Errorf("not a Hugging Face URL")
	}
	path := url[idx+len(prefix):]
	parts := strings.SplitN(path, "/", 5) // org, repo, "resolve", ref, filename
	if len(parts) < 5 || parts[2] != "resolve" {
		return "", fmt.Errorf("unexpected HF URL format")
	}
	org, repo, ref, filename := parts[0], parts[1], parts[3], parts[4]

	apiURL := fmt.Sprintf("https://huggingface.co/api/models/%s/%s/tree/%s", org, repo, ref)
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Get(apiURL)
	if err != nil {
		return "", fmt.Errorf("HF API request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("HF API returned HTTP %d", resp.StatusCode)
	}

	var entries []hfTreeEntry
	if err := json.NewDecoder(resp.Body).Decode(&entries); err != nil {
		return "", fmt.Errorf("failed to parse HF API response: %w", err)
	}

	for _, entry := range entries {
		if entry.Path == filename && entry.LFS != nil {
			return entry.LFS.OID, nil
		}
	}
	return "", fmt.Errorf("file %s not found in HF tree response", filename)
}

// progressWriter wraps an io.Reader to report download progress and to make
// the transfer cancellable.
//
// The ctx check lives here rather than only in the HTTP layer because
// io.Copy's loop is where a multi-gigabyte body is actually consumed: that is
// the point at which a cancelled download stops promptly instead of running to
// completion.
type progressWriter struct {
	ctx      context.Context
	reader   io.Reader
	total    int64
	written  int64
	onUpdate func(DownloadProgress)
	lastCall time.Time
}

// progressInterval throttles callbacks. A Read returns on the order of tens of
// kilobytes, so an un-throttled callback fires tens of thousands of times for
// a multi-gigabyte model — enough to swamp a GUI consumer's event bus with
// updates far finer than any progress bar can show. Time-based rather than
// percentage-based so a download with no Content-Length still reports.
const progressInterval = 100 * time.Millisecond

func (pw *progressWriter) Read(p []byte) (int, error) {
	if pw.ctx != nil {
		if err := pw.ctx.Err(); err != nil {
			return 0, err
		}
	}
	n, err := pw.reader.Read(p)
	pw.written += int64(n)

	if pw.onUpdate != nil {
		// Always report the final state, whatever the throttle says: a
		// progress bar that stops at 99% because the last update was rate
		// limited is worse than no bar at all.
		done := err != nil
		if done || time.Since(pw.lastCall) >= progressInterval {
			pw.lastCall = time.Now()
			pw.onUpdate(DownloadProgress{Downloaded: pw.written, Total: pw.total})
		}
	}
	return n, err
}
