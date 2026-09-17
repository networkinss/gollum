package gollum

import (
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

// DownloadModel downloads a GGUF model file from the given URL to destDir.
// If url is empty, the default model is downloaded.
// configChecksum is an optional SHA256 hex digest from the user's config; if empty,
// the hardcoded hash (for the default model) or Hugging Face API is used.
// It prints progress to stdout and returns the full path to the downloaded file.
func DownloadModel(url, destDir, configChecksum string) (string, error) {
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

	client := &http.Client{Timeout: 30 * time.Minute}
	resp, err := client.Get(url)
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
		reader: resp.Body,
		total:  totalSize,
	})
	file.Close()
	if err != nil {
		os.Remove(tmpPath)
		return "", fmt.Errorf("download interrupted: %w", err)
	}

	// Determine expected checksum and verify integrity.
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

// progressWriter wraps an io.Reader to print download progress.
type progressWriter struct {
	reader  io.Reader
	total   int64
	written int64
	lastPct int
}

func (pw *progressWriter) Read(p []byte) (int, error) {
	n, err := pw.reader.Read(p)
	pw.written += int64(n)
	if pw.total > 0 {
		pct := int(pw.written * 100 / pw.total)
		if pct != pw.lastPct && pct%5 == 0 {
			pw.lastPct = pct
			fmt.Printf("\r  Progress: %d%% (%.0f / %.0f MB)",
				pct,
				float64(pw.written)/(1024*1024),
				float64(pw.total)/(1024*1024))
		}
	}
	return n, err
}
