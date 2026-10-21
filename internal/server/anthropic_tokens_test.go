package server

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestEstimateInputTokensDeterministic(t *testing.T) {
	body := []byte(`{"model":"m","messages":[{"role":"user","content":"hello world 你好世界"}],"metadata":{"conversation_id":"cc-abc"}}`)
	a := estimateInputTokens(body)
	b := estimateInputTokens(body)
	if a != b || a <= 0 {
		t.Fatalf("deterministic positive estimate expected: %d vs %d", a, b)
	}
	// 更长文本估得更多
	bigger := []byte(`{"model":"m","messages":[{"role":"user","content":"hello world 你好世界 hello world 你好世界 hello world"}],"metadata":{"conversation_id":"cc-abc"}}`)
	if got := estimateInputTokens(bigger); got <= a {
		t.Fatalf("monotonic expected: %d <= %d", got, a)
	}
}

func TestCountTokensEndpoint(t *testing.T) {
	h := newAnthropicTestHandler(t, func(string) (int, string, bool) { return 200, sseOK, true })
	rec := anthropicPOST(t, h, "k", "/v1/messages/count_tokens",
		`{"model":"deepseek-v4","max_tokens":8,"messages":[{"role":"user","content":"估算一下"}]}`)
	if rec.Code != 200 {
		t.Fatalf("want 200 got %d: %s", rec.Code, rec.Body.String())
	}
	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("json expected: %v", err)
	}
	if n, _ := out["input_tokens"].(float64); n <= 0 {
		t.Fatalf("positive input_tokens expected: %v", out)
	}
	if !strings.Contains(rec.Body.String(), "input_tokens") {
		t.Fatalf("anthropic count_tokens shape expected: %s", rec.Body.String())
	}
}

func TestCountTokensValidationShared(t *testing.T) {
	// 与 /v1/messages 同一套校验：缺 max_tokens → 400 invalid_request_error。
	h := newAnthropicTestHandler(t, func(string) (int, string, bool) { return 200, sseOK, true })
	rec := anthropicPOST(t, h, "k", "/v1/messages/count_tokens", `{"model":"m","messages":[{"role":"user","content":"x"}]}`)
	if rec.Code != 400 || !strings.Contains(rec.Body.String(), "max_tokens") {
		t.Fatalf("shared validation expected: %d %s", rec.Code, rec.Body.String())
	}
}
