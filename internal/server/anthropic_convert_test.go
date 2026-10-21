package server

import (
	"encoding/json"
	"strings"
	"testing"
)

func mustParse(t *testing.T, body string) *anthropicRequest {
	t.Helper()
	req, err := parseAnthropicRequest([]byte(body))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	return req
}

func TestConvertBasicRoundtrip(t *testing.T) {
	req := mustParse(t, `{
		"model":"deepseek-v4","max_tokens":1024,"stream":true,
		"system":[{"type":"text","text":"You are helpful."},{"type":"text","text":"Be terse."}],
		"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],
		"temperature":0.7
	}`)
	out, err := req.ToOpenAI()
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("output not json: %v", err)
	}
	if m["model"] != "deepseek-v4" || m["max_tokens"] != float64(1024) || m["stream"] != true {
		t.Fatalf("scalar passthrough wrong: %v", m)
	}
	if m["temperature"] != 0.7 {
		t.Fatalf("temperature lost: %v", m["temperature"])
	}
	msgs := m["messages"].([]any)
	sys := msgs[0].(map[string]any)
	if sys["role"] != "system" || sys["content"] != "You are helpful.\n\nBe terse." {
		t.Fatalf("system join wrong: %v", sys)
	}
	usr := msgs[1].(map[string]any)
	if usr["role"] != "user" {
		t.Fatalf("user message wrong: %v", usr)
	}
	if _, ok := m["metadata"]; !ok {
		t.Fatal("metadata.conversation_id missing")
	}
	md := m["metadata"].(map[string]any)
	if cid, _ := md["conversation_id"].(string); !strings.HasPrefix(cid, "cc-") || len(cid) != 19 {
		t.Fatalf("conversation key wrong: %v", md)
	}
}

func TestConvertValidation(t *testing.T) {
	if _, err := parseAnthropicRequest([]byte(`{"model":"m"}`)); err == nil {
		t.Fatal("max_tokens required")
	}
	if _, err := parseAnthropicRequest([]byte(`not json`)); err == nil {
		t.Fatal("bad json must fail")
	}
	if _, err := parseAnthropicRequest([]byte(`{"model":"","max_tokens":8}`)); err == nil {
		t.Fatal("model required")
	}
	if _, err := parseAnthropicRequest([]byte(`{"model":"m","max_tokens":8}`)); err != nil {
		t.Fatalf("messages 不在解析层强校验（与 OpenAI 端点同口径交上游）: %v", err)
	}
	if _, err := parseAnthropicRequest([]byte(`{"model":"m","max_tokens":0,"messages":[{"role":"user","content":"x"}]}`)); err == nil {
		t.Fatal("max_tokens must be positive")
	}
}

func TestConvertAssistantBlocks(t *testing.T) {
	req := mustParse(t, `{"model":"m","max_tokens":8,"messages":[
		{"role":"assistant","content":[
			{"type":"thinking","thinking":"ponder","signature":"sig"},
			{"type":"text","text":"hello"},
			{"type":"tool_use","id":"call_1","name":"get_weather","input":{"city":"北京","unit":"C"}}
		]},
		{"role":"user","content":[
			{"type":"tool_result","tool_use_id":"call_1","content":"20度","is_error":false},
			{"type":"text","text":"thanks"}
		]}
	]}`)
	out, err := req.ToOpenAI()
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	var m map[string]any
	_ = json.Unmarshal(out, &m)
	msgs := m["messages"].([]any)
	asst := msgs[0].(map[string]any)
	if asst["reasoning_content"] != "ponder" {
		t.Fatalf("thinking -> reasoning_content wrong: %v", asst)
	}
	if asst["content"] != "hello" {
		t.Fatalf("text wrong: %v", asst)
	}
	tcs := asst["tool_calls"].([]any)
	tc := tcs[0].(map[string]any)
	if tc["id"] != "call_1" {
		t.Fatalf("tool id wrong: %v", tc)
	}
	fn := tc["function"].(map[string]any)
	// input 对象按键排序序列化（缓存前缀稳定性）
	if fn["arguments"] != `{"city":"北京","unit":"C"}` {
		t.Fatalf("arguments not sorted-stable: %v", fn["arguments"])
	}
	toolMsg := msgs[1].(map[string]any)
	if toolMsg["role"] != "tool" || toolMsg["tool_call_id"] != "call_1" || toolMsg["content"] != "20度" {
		t.Fatalf("tool_result -> tool message wrong: %v", toolMsg)
	}
	usr := msgs[2].(map[string]any)
	if usr["role"] != "user" {
		t.Fatalf("trailing user text lost: %v", usr)
	}
}

func TestConvertAssistantEmptyDropped(t *testing.T) {
	// 只含 signature 占位的空 assistant 轮：确定性丢弃，不出空消息。
	req := mustParse(t, `{"model":"m","max_tokens":8,"messages":[
		{"role":"assistant","content":[]},
		{"role":"user","content":"next"}
	]}`)
	out, err := req.ToOpenAI()
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	var m map[string]any
	_ = json.Unmarshal(out, &m)
	if len(m["messages"].([]any)) != 1 {
		t.Fatalf("empty assistant turn must be dropped: %v", m["messages"])
	}
}

func TestConvertToolsAndChoice(t *testing.T) {
	req := mustParse(t, `{"model":"m","max_tokens":8,
		"tools":[{"name":"f1","description":"d","input_schema":{"type":"object"}}],
		"tool_choice":{"type":"any"},"stop_sequences":["END"]}`)
	out, err := req.ToOpenAI()
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	var m map[string]any
	_ = json.Unmarshal(out, &m)
	tools := m["tools"].([]any)
	t0 := tools[0].(map[string]any)
	if t0["type"] != "function" {
		t.Fatalf("tool type wrong: %v", t0)
	}
	if m["tool_choice"] != "required" {
		t.Fatalf("tool_choice any->required wrong: %v", m["tool_choice"])
	}
	stop, _ := m["stop"].([]any)
	if len(stop) != 1 || stop[0] != "END" {
		t.Fatalf("stop_sequences -> stop wrong: %v", m["stop"])
	}
}

func TestConvertImageAndRejects(t *testing.T) {
	req := mustParse(t, `{"model":"m","max_tokens":8,"messages":[
		{"role":"user","content":[
			{"type":"text","text":"look"},
			{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aGk="}}
		]}]}`)
	out, err := req.ToOpenAI()
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	var m map[string]any
	_ = json.Unmarshal(out, &m)
	parts := m["messages"].([]any)[0].(map[string]any)["content"].([]any)
	img := parts[1].(map[string]any)
	iu := img["image_url"].(map[string]any)
	if iu["url"] != "data:image/png;base64,aGk=" {
		t.Fatalf("image data-url wrong: %v", iu)
	}
	// 不支持块类型：document / server tool
	bad := `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":[{"type":"document","source":{}}]}]}`
	if _, err := mustParse(t, bad).ToOpenAI(); err == nil || !strings.Contains(err.Error(), "document") {
		t.Fatalf("document block must be rejected with type named: %v", err)
	}
	badTool := `{"model":"m","max_tokens":8,"tools":[{"type":"web_search_20250305","name":"web_search"}]}`
	if _, err := mustParse(t, badTool).ToOpenAI(); err == nil || !strings.Contains(err.Error(), "web_search_20250305") {
		t.Fatalf("server tool must be rejected with type named: %v", err)
	}
}

func TestConvertThinkingParamPassthrough(t *testing.T) {
	req := mustParse(t, `{"model":"m","max_tokens":2048,
		"thinking":{"type":"enabled","budget_tokens":1024},
		"messages":[{"role":"user","content":"hi"}]}`)
	out, err := req.ToOpenAI()
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	var m map[string]any
	_ = json.Unmarshal(out, &m)
	th, ok := m["thinking"].(map[string]any)
	if !ok || th["type"] != "enabled" || th["budget_tokens"] != float64(1024) {
		t.Fatalf("thinking param must pass through verbatim: %v", m["thinking"])
	}
}

func TestConversationKeyStable(t *testing.T) {
	a := mustParse(t, `{"model":"m","max_tokens":8,"system":"S","messages":[{"role":"user","content":"u1"}]}`)
	b := mustParse(t, `{"model":"m","max_tokens":8,"system":"S","messages":[{"role":"user","content":"u1"}]}`)
	c := mustParse(t, `{"model":"m","max_tokens":8,"system":"S","messages":[{"role":"user","content":"DIFFERENT"}]}`)
	if a.ConversationKey() != b.ConversationKey() {
		t.Fatal("same input must derive same key")
	}
	if a.ConversationKey() == c.ConversationKey() {
		t.Fatal("different first user message must derive different key")
	}
}

// TestConvertDeterministic 纯函数约束：同输入两次转换字节一致。
func TestConvertDeterministic(t *testing.T) {
	body := `{"model":"m","max_tokens":8,"system":"S",
		"tools":[{"name":"f","input_schema":{"b":1,"a":2}}],
		"messages":[{"role":"user","content":"u"},{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"f","input":{"z":1,"a":2}}]}]}`
	a, err := mustParse(t, body).ToOpenAI()
	if err != nil {
		t.Fatalf("convert a: %v", err)
	}
	b, err := mustParse(t, body).ToOpenAI()
	if err != nil {
		t.Fatalf("convert b: %v", err)
	}
	if string(a) != string(b) {
		t.Fatalf("conversion must be deterministic:\n%s\n%s", a, b)
	}
}
