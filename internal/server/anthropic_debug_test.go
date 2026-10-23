package server

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/linguo2625469/workbuddy2api-panel/internal/auth"
)

// newDebugHandler 带排障转储目录的 Anthropic 测试 handler。
func newDebugHandler(t *testing.T, dir string, behavior func(string) (int, string, bool)) *Handler {
	t.Helper()
	up := newFakeUpstream(t, behavior)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	return NewHandler(Config{Pool: p, Upstream: up, APIKey: "k", MaxRotate: 1, AnthropicDebugDir: dir})
}

func lsDebugDir(t *testing.T, dir string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read debug dir: %v", err)
	}
	var out []string
	for _, e := range entries {
		out = append(out, e.Name())
	}
	return out
}

// readDebug 找到带指定后缀的唯一转储文件，返回（去后缀的文件名前缀, 内容）。
func readDebug(t *testing.T, dir, suffix string) (string, string) {
	t.Helper()
	for _, name := range lsDebugDir(t, dir) {
		if strings.HasSuffix(name, suffix) {
			raw, err := os.ReadFile(filepath.Join(dir, name))
			if err != nil {
				t.Fatalf("read %s: %v", name, err)
			}
			return strings.TrimSuffix(name, suffix), string(raw)
		}
	}
	t.Fatalf("no dump file with suffix %s: %v", suffix, lsDebugDir(t, dir))
	return "", ""
}

func TestAnthropicDebugDumpSuccess(t *testing.T) {
	dir := t.TempDir()
	h := newDebugHandler(t, dir, func(string) (int, string, bool) { return 200, sseOK, true })
	rec := anthropicPOST(t, h, "k", "/v1/messages",
		`{"model":"deepseek-v4","max_tokens":8,"stream":true,"messages":[{"role":"user","content":"你好"}]}`)
	if rec.Code != 200 {
		t.Fatalf("want 200 got %d: %s", rec.Code, rec.Body.String())
	}
	files := lsDebugDir(t, dir)
	if len(files) != 2 {
		t.Fatalf("success dumps exactly in+openai: %v", files)
	}
	pIn, in := readDebug(t, dir, ".anthropic.json")
	pOut, out := readDebug(t, dir, ".openai.json")
	if !strings.HasPrefix(pIn, "dbg-") || pIn != pOut {
		t.Fatalf("dbg- prefix and same request id expected: %q vs %q", pIn, pOut)
	}
	if !strings.Contains(in, `"max_tokens":8`) || !strings.Contains(in, "你好") {
		t.Fatalf("inbound body must be preserved verbatim: %s", in)
	}
	if !strings.Contains(out, `"model":"deepseek-v4"`) || !strings.Contains(out, `"messages"`) {
		t.Fatalf("converted body expected: %s", out)
	}
	// count_tokens 不转储（体积与 messages 重复）。
	rec = anthropicPOST(t, h, "k", "/v1/messages/count_tokens",
		`{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"x"}]}`)
	if rec.Code != 200 || len(lsDebugDir(t, dir)) != 2 {
		t.Fatalf("count_tokens must not dump: %d %v", rec.Code, lsDebugDir(t, dir))
	}
}

func TestAnthropicDebugDumpParseError(t *testing.T) {
	dir := t.TempDir()
	h := newDebugHandler(t, dir, func(string) (int, string, bool) { return 200, sseOK, true })
	rec := anthropicPOST(t, h, "k", "/v1/messages", `{"model":"m"}`) // 缺 max_tokens
	if rec.Code != 400 {
		t.Fatalf("want 400 got %d", rec.Code)
	}
	_, errTxt := readDebug(t, dir, ".error.txt")
	if !strings.Contains(errTxt, "max_tokens") {
		t.Fatalf("error note must carry the 400 reason: %s", errTxt)
	}
	for _, name := range lsDebugDir(t, dir) {
		if strings.HasSuffix(name, ".openai.json") {
			t.Fatalf("openai dump must be absent when conversion never ran: %v", name)
		}
	}
}

func TestAnthropicDebugDumpUpstreamError(t *testing.T) {
	dir := t.TempDir()
	h := newDebugHandler(t, dir, func(string) (int, string, bool) {
		return 429, `{"code":6004,"msg":"rate limited"}`, false
	})
	rec := anthropicPOST(t, h, "k", "/v1/messages", `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`)
	if rec.Code != 429 {
		t.Fatalf("want 429 got %d: %s", rec.Code, rec.Body.String())
	}
	_, errTxt := readDebug(t, dir, ".error.txt")
	if !strings.Contains(errTxt, "phase=") {
		t.Fatalf("error note must declare the failure phase: %s", errTxt)
	}
	if !strings.Contains(errTxt, "rate limited") {
		t.Fatalf("upstream message must be in the dump: %s", errTxt)
	}
}

func TestAnthropicDebugPrune(t *testing.T) {
	dir := t.TempDir()
	old := anthropicDebugKeepFiles
	anthropicDebugKeepFiles = 3
	t.Cleanup(func() { anthropicDebugKeepFiles = old })
	h := newDebugHandler(t, dir, func(string) (int, string, bool) { return 200, sseOK, true })
	for i := 0; i < 3; i++ {
		rec := anthropicPOST(t, h, "k", "/v1/messages",
			fmt.Sprintf(`{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"q%d"}]}`, i))
		if rec.Code != 200 {
			t.Fatalf("want 200 got %d", rec.Code)
		}
	}
	// 每请求 2 文件 × 3 次，prune 后封顶 anthropicDebugKeepFiles。
	if files := lsDebugDir(t, dir); len(files) > 3 {
		t.Fatalf("prune must cap file count at %d: %v", anthropicDebugKeepFiles, files)
	}
}
