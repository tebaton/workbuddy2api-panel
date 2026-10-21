package server

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
)

// feedSSE 模拟 chatCompletions（StreamHint）向 translator 的写出：先设 SSE 头
// （真实调用方在首次 Write 前设置，翻译器据此判定流式模式），再逐帧 Write + Flush。
func feedSSE(t *testing.T, tr *anthropicTranslator, frames ...string) {
	t.Helper()
	tr.Header().Set("Content-Type", "text/event-stream")
	tr.Header().Set("Cache-Control", "no-cache")
	for _, f := range frames {
		if _, err := tr.Write([]byte("data: " + f + "\n\n")); err != nil {
			t.Fatalf("write frame: %v", err)
		}
		tr.Flush()
	}
}

func TestTranslatorStreamFullSequence(t *testing.T) {
	rec := httptest.NewRecorder()
	tr := newAnthropicTranslator(rec, "deepseek-v4")
	feedSSE(t, tr,
		`{"id":"chatcmpl-1","choices":[{"delta":{"reasoning_content":"让我想想"}}]}`,
		`{"choices":[{"delta":{"content":"你好"}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","function":{"name":"get_weather","arguments":"{\"city\":"}}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"北京\"}"}}]}}]}`,
		`{"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":100,"completion_tokens":20,"prompt_cache_hit_tokens":60}}`,
		`[DONE]`,
	)
	tr.Finish()
	joined := rec.Body.String()
	// 注意 json.Marshal 对 map 按键排序：块对象形如 {"thinking":"","type":"thinking"}、
	// {"id":...,"input":{},"name":...,"type":"tool_use"}，断言按该顺序书写。
	for _, want := range []string{
		"event: message_start", `"model":"deepseek-v4"`,
		"event: ping",
		`{"thinking":"","type":"thinking"}`, `"thinking_delta"`, `"signature_delta"`,
		`{"text":"","type":"text"}`, `"text_delta"`,
		`"type":"tool_use"`, `"id":"call_abc"`, `"name":"get_weather"`, `"input":{}`,
		`"partial_json":"{\"city\":"`,
		`"stop_reason":"tool_use"`,
		`"input_tokens":100`, `"output_tokens":20`, `"cache_read_input_tokens":60`,
		"event: message_stop",
	} {
		if !strings.Contains(joined, want) {
			t.Fatalf("missing %q in stream:\n%s", want, joined)
		}
	}
	// 事件顺序骨架：message_start 最先，message_stop 最后。
	if !strings.HasPrefix(joined, "event: message_start") {
		t.Fatalf("message_start must be first:\n%s", joined)
	}
	if !strings.HasSuffix(strings.TrimRight(joined, "\n"), "event: message_stop\ndata: {\"type\":\"message_stop\"}") {
		t.Fatalf("message_stop must be last:\n%s", joined)
	}
}

func TestTranslatorStreamSingleTextBlock(t *testing.T) {
	rec := httptest.NewRecorder()
	tr := newAnthropicTranslator(rec, "m")
	feedSSE(t, tr,
		`{"choices":[{"delta":{"content":"a"}}]}`,
		`{"choices":[{"delta":{"content":"b"}}]}`,
		`{"choices":[{"delta":{},"finish_reason":"stop"}]}`,
		`[DONE]`,
	)
	tr.Finish()
	body := rec.Body.String()
	if strings.Count(body, "event: content_block_start") != 1 {
		t.Fatalf("one text block expected:\n%s", body)
	}
	if strings.Contains(body, `"index":1`) {
		t.Fatalf("only block index 0 expected:\n%s", body)
	}
	if !strings.Contains(body, `"stop_reason":"end_turn"`) {
		t.Fatalf("stop->end_turn:\n%s", body)
	}
}

// TestTranslatorStreamInterleavedTools OpenAI 流允许不同 index 的 tool_calls
// 交错续传 arguments：tool 块按需打开、流尾统一关闭，增量永远落在自己的块上。
func TestTranslatorStreamInterleavedTools(t *testing.T) {
	rec := httptest.NewRecorder()
	tr := newAnthropicTranslator(rec, "m")
	feedSSE(t, tr,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c0","function":{"name":"f0","arguments":"{\"a\":"}}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"c1","function":{"name":"f1","arguments":"{\"b\":"}}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}`,
		`{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
		`[DONE]`,
	)
	tr.Finish()
	body := rec.Body.String()
	if strings.Count(body, `"type":"tool_use"`) != 2 {
		t.Fatalf("two tool blocks expected:\n%s", body)
	}
	if strings.Count(body, "event: content_block_start") != 2 {
		t.Fatalf("two block opens expected:\n%s", body)
	}
	if strings.Count(body, "event: content_block_stop") != 2 {
		t.Fatalf("both tool blocks must close at end:\n%s", body)
	}
	// 交错增量：块 0 的第二段 arguments 出现在块 1 开块之后（索引仍是 0）。
	if !strings.Contains(body, `"partial_json":"1}"`) {
		t.Fatalf("interleaved delta lost:\n%s", body)
	}
	if !strings.Contains(body, `"stop_reason":"tool_use"`) {
		t.Fatalf("finish tool_calls -> tool_use:\n%s", body)
	}
}

func TestTranslatorStreamErrorFrame(t *testing.T) {
	rec := httptest.NewRecorder()
	tr := newAnthropicTranslator(rec, "m")
	feedSSE(t, tr,
		`{"id":"c1","choices":[{"delta":{"content":"partial"}}]}`,
		`{"error":{"message":"rate limited","type":"api_error","code":"rate_limit"}}`,
		`[DONE]`,
	)
	tr.Finish()
	body := rec.Body.String()
	if !strings.Contains(body, "event: error") || !strings.Contains(body, `"message":"rate limited"`) {
		t.Fatalf("mid-stream error must map to anthropic error event:\n%s", body)
	}
	if strings.Contains(body, "event: message_stop") {
		t.Fatalf("error terminates stream without message_stop:\n%s", body)
	}
}

func TestTranslatorStreamSplitWrites(t *testing.T) {
	// 帧可能跨多次 Write 到达（半行）：行缓冲必须拼回完整帧。
	rec := httptest.NewRecorder()
	tr := newAnthropicTranslator(rec, "m")
	full := "data: " + `{"id":"c1","choices":[{"delta":{"content":"你好"}}]}` + "\n\n" +
		"data: [DONE]\n\n"
	half := len(full) / 2
	tr.Header().Set("Content-Type", "text/event-stream")
	if _, err := tr.Write([]byte(full[:half])); err != nil {
		t.Fatalf("write half: %v", err)
	}
	if _, err := tr.Write([]byte(full[half:])); err != nil {
		t.Fatalf("write rest: %v", err)
	}
	tr.Finish()
	body := rec.Body.String()
	if !strings.Contains(body, `"text_delta"`) || !strings.Contains(body, "event: message_stop") {
		t.Fatalf("split writes must reassemble:\n%s", body)
	}
}

func TestTranslatorAggregateJSON(t *testing.T) {
	rec := httptest.NewRecorder()
	tr := newAnthropicTranslator(rec, "deepseek-v4")
	resp := map[string]any{
		"id": "chatcmpl-9",
		"choices": []any{map[string]any{
			"index": 0,
			"message": map[string]any{
				"role":              "assistant",
				"content":           "答案",
				"reasoning_content": "推理",
				"tool_calls": []any{map[string]any{
					"id":   "call_x",
					"type": "function",
					"function": map[string]any{
						"name":      "f",
						"arguments": `{"a":1}`,
					},
				}},
			},
			"finish_reason": "tool_calls",
		}},
		"usage": map[string]any{"prompt_tokens": 10, "completion_tokens": 5},
	}
	raw, _ := json.Marshal(resp)
	// writeJSON 行为：先设头，再 WriteHeader(200)，再 Write(body)。
	tr.Header().Set("Content-Type", "application/json")
	tr.WriteHeader(200)
	if _, err := tr.Write(raw); err != nil {
		t.Fatalf("write: %v", err)
	}
	tr.Finish()
	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("non-stream output must be anthropic message json: %v\n%s", err, rec.Body.String())
	}
	if out["type"] != "message" || out["role"] != "assistant" || out["stop_reason"] != "tool_use" {
		t.Fatalf("message envelope wrong: %v", out)
	}
	content := out["content"].([]any)
	kinds := []string{}
	for _, c := range content {
		kinds = append(kinds, c.(map[string]any)["type"].(string))
	}
	if strings.Join(kinds, ",") != "thinking,text,tool_use" {
		t.Fatalf("content blocks wrong: %v", content)
	}
	// tool_use.input 必须是对象（arguments 字符串解析回对象）。
	tu := content[2].(map[string]any)
	if tu["input"].(map[string]any)["a"] != float64(1) {
		t.Fatalf("tool input wrong: %v", tu)
	}
	usage := out["usage"].(map[string]any)
	if usage["input_tokens"] != float64(10) || usage["output_tokens"] != float64(5) {
		t.Fatalf("usage wrong: %v", usage)
	}
	if rec.Header().Get("Content-Type") != "application/json" {
		t.Fatalf("content-type must be json")
	}
}

func TestTranslatorErrorJSON(t *testing.T) {
	rec := httptest.NewRecorder()
	tr := newAnthropicTranslator(rec, "m")
	tr.Header().Set("Content-Type", "application/json")
	tr.WriteHeader(429)
	_, _ = tr.Write([]byte(`{"error":{"message":"rate limited: all accounts cooling","type":"api_error","code":"rate_limit_exceeded","gateway_hint":"try later"}}`))
	tr.Finish()
	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("error output must be json: %v", err)
	}
	e := out["error"].(map[string]any)
	if out["type"] != "error" || e["type"] != "rate_limit_error" {
		t.Fatalf("error mapping wrong: %v", out)
	}
	if !strings.Contains(e["message"].(string), "rate limited") || !strings.Contains(e["message"].(string), "try later") {
		t.Fatalf("message passthrough + hint expected: %v", e)
	}
	if rec.Code != 429 {
		t.Fatalf("status preserved: %d", rec.Code)
	}
}

func TestAnthropicErrType(t *testing.T) {
	cases := map[int]string{
		400: "invalid_request_error", 401: "authentication_error",
		404: "not_found_error", 429: "rate_limit_error",
		500: "api_error", 502: "api_error", 503: "overloaded_error",
		501: "not_implemented_error",
	}
	for status, want := range cases {
		if got := anthropicErrType(status); got != want {
			t.Fatalf("status %d: got %s want %s", status, got, want)
		}
	}
}
