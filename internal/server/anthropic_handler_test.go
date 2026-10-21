package server

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/linguo2625469/workbuddy2api-panel/internal/auth"
)

// newAnthropicTestHandler 构建 Anthropic 端点测试 handler：单账号池 + fake 上游。
func newAnthropicTestHandler(t *testing.T, behavior func(authz string) (int, string, bool)) *Handler {
	t.Helper()
	up := newFakeUpstream(t, behavior)
	p := testPoolWith(&auth.Auth{UID: "u1", AccessToken: "at1", ExpiresAt: 9999999999})
	return NewHandler(Config{Pool: p, Upstream: up, APIKey: "k", MaxRotate: 1})
}

// anthropicPOST 向 path 发 POST（x-api-key 鉴权形态）。
func anthropicPOST(t *testing.T, h *Handler, apiKey, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("x-api-key", apiKey)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func TestAnthropicMessagesAuth(t *testing.T) {
	h := newAnthropicTestHandler(t, func(string) (int, string, bool) { return 200, sseOK, true })
	rec := anthropicPOST(t, h, "wrong", "/v1/messages", `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`)
	if rec.Code != 401 {
		t.Fatalf("wrong key must 401, got %d", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), `"authentication_error"`) {
		t.Fatalf("anthropic error shape expected: %s", rec.Body.String())
	}
	// Bearer 形态（ANTHROPIC_AUTH_TOKEN 客户端）同放行。
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer k")
	rec2 := httptest.NewRecorder()
	h.ServeHTTP(rec2, req)
	if rec2.Code == 401 {
		t.Fatalf("bearer auth must pass")
	}
}

func TestAnthropicMessagesNonStream(t *testing.T) {
	h := newAnthropicTestHandler(t, func(string) (int, string, bool) {
		sse := "data: " + `{"id":"cc1","choices":[{"index":0,"delta":{"content":"你好"}}]}` + "\n\n" +
			"data: " + `{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":3}}` + "\n\n" +
			"data: [DONE]\n\n"
		return 200, sse, true
	})
	rec := anthropicPOST(t, h, "k", "/v1/messages", `{"model":"deepseek-v4","max_tokens":64,"stream":false,"messages":[{"role":"user","content":"hi"}]}`)
	if rec.Code != 200 {
		t.Fatalf("want 200 got %d: %s", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	for _, want := range []string{`"type":"message"`, `"text":"你好"`, `"stop_reason":"end_turn"`, `"input_tokens":7`} {
		if !strings.Contains(body, want) {
			t.Fatalf("missing %q: %s", want, body)
		}
	}
}

func TestAnthropicMessagesStream(t *testing.T) {
	h := newAnthropicTestHandler(t, func(string) (int, string, bool) {
		sse := "data: " + `{"id":"cc1","choices":[{"index":0,"delta":{"reasoning_content":"想"}}]}` + "\n\n" +
			"data: " + `{"choices":[{"index":0,"delta":{"content":"答"}}]}` + "\n\n" +
			"data: " + `{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n" +
			"data: [DONE]\n\n"
		return 200, sse, true
	})
	rec := anthropicPOST(t, h, "k", "/v1/messages", `{"model":"deepseek-v4","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	if rec.Code != 200 {
		t.Fatalf("want 200 got %d: %s", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	for _, want := range []string{
		"event: message_start", "event: ping", "thinking_delta", "text_delta",
		"event: message_delta", `"stop_reason":"end_turn"`, "event: message_stop",
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("missing %q:\n%s", want, body)
		}
	}
	if ct := rec.Header().Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		t.Fatalf("sse content-type expected, got %q", ct)
	}
}

func TestAnthropicMessagesToolLoop(t *testing.T) {
	// Claude Code 工具循环形态：请求带 tools + 历史 tool_use/tool_result 回传，
	// 上游回 tool_calls 增量 → 出站 tool_use 块 + stop_reason tool_use。
	h := newAnthropicTestHandler(t, func(string) (int, string, bool) {
		sse := "data: " + `{"id":"cc1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_9","function":{"name":"Read","arguments":"{\"path\":\""}}]}}]}` + "\n\n" +
			"data: " + `{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"main.go\"}"}}]}}]}` + "\n\n" +
			"data: " + `{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":50,"completion_tokens":6}}` + "\n\n" +
			"data: [DONE]\n\n"
		return 200, sse, true
	})
	body := `{"model":"deepseek-v4","max_tokens":64,"stream":true,
		"tools":[{"name":"Read","description":"read","input_schema":{"type":"object"}}],
		"messages":[
			{"role":"user","content":"读 main.go"},
			{"role":"assistant","content":[{"type":"tool_use","id":"toolu_01","name":"Read","input":{"path":"x.go"}}]},
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_01","content":"package main"}]},
			{"role":"user","content":"继续"}
		]}`
	rec := anthropicPOST(t, h, "k", "/v1/messages", body)
	if rec.Code != 200 {
		t.Fatalf("want 200 got %d: %s", rec.Code, rec.Body.String())
	}
	out := rec.Body.String()
	// json.Marshal 按键排序：块对象为 {"id":...,"input":{},"name":...,"type":"tool_use"}。
	for _, want := range []string{
		`"type":"tool_use"`, `"id":"call_9"`, `"name":"Read"`,
		`"partial_json":"{\"path\":\""`,
		`"stop_reason":"tool_use"`,
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q:\n%s", want, out)
		}
	}
}

func TestAnthropicMessagesValidationErrors(t *testing.T) {
	h := newAnthropicTestHandler(t, func(string) (int, string, bool) { return 200, sseOK, true })
	// 缺 max_tokens
	rec := anthropicPOST(t, h, "k", "/v1/messages", `{"model":"m","messages":[{"role":"user","content":"x"}]}`)
	if rec.Code != 400 || !strings.Contains(rec.Body.String(), "invalid_request_error") {
		t.Fatalf("validation 400 expected: %d %s", rec.Code, rec.Body.String())
	}
	// 不支持块类型：点名拒绝
	rec = anthropicPOST(t, h, "k", "/v1/messages", `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":[{"type":"document"}]}]}`)
	if rec.Code != 400 || !strings.Contains(rec.Body.String(), "document") {
		t.Fatalf("document reject expected: %d %s", rec.Code, rec.Body.String())
	}
	// server tool：点名拒绝
	rec = anthropicPOST(t, h, "k", "/v1/messages", `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"x"}],"tools":[{"type":"web_search_20250305","name":"web_search"}]}`)
	if rec.Code != 400 || !strings.Contains(rec.Body.String(), "web_search") {
		t.Fatalf("server tool reject expected: %d %s", rec.Code, rec.Body.String())
	}
}

func TestAnthropicMessagesUnavailable503(t *testing.T) {
	// 空池：全池不可用 → Anthropic overloaded_error 形态。
	up := newFakeUpstream(t, func(string) (int, string, bool) { return 200, sseOK, true })
	h := NewHandler(Config{Pool: testPoolWith(), Upstream: up, APIKey: "k"})
	rec := anthropicPOST(t, h, "k", "/v1/messages", `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`)
	if rec.Code != 503 || !strings.Contains(rec.Body.String(), "overloaded_error") {
		t.Fatalf("503 overloaded expected: %d %s", rec.Code, rec.Body.String())
	}
}

func TestAnthropicMessagesUpstreamErrorPassthrough(t *testing.T) {
	// 上游 429 限流文案（轮转耗尽）→ Anthropic rate_limit_error，message 带上游原文。
	h := newAnthropicTestHandler(t, func(string) (int, string, bool) {
		return 429, `{"code":6004,"msg":"rate limited, will reset at 12:00"}`, false
	})
	rec := anthropicPOST(t, h, "k", "/v1/messages", `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`)
	if rec.Code != 429 || !strings.Contains(rec.Body.String(), "rate_limit_error") {
		t.Fatalf("429 rate_limit expected: %d %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "6004") {
		t.Fatalf("upstream message passthrough expected: %s", rec.Body.String())
	}
}
