package server

import (
	"strings"
	"testing"
)

// TestAnthropicPrefixStability 缓存三约束之一（设计 §缓存命中保证）的可执行化：
// 同一会话第 2 轮请求（历史 = 第 1 轮 + assistant 回复 + tool_result + 新 user）
// 的转换产物，其 messages 数组内容必须与第 1 轮**字节级同前缀**——上游
// DeepSeek 系前缀缓存按请求前缀字节匹配，这里断了 = 每轮全量 cache miss。
//
// 字节级前缀可行的前提（由 ToOpenAI 的纯函数性质保证）：
//   - json.Marshal(map) 键排序固定 → messages 之前的 max_tokens 与之后的
//     metadata/model/stream 序列化逐字节相同（同会话 cc- 键恒同值）；
//   - 旧轮消息的转换是同一纯函数对同一输入的重复执行。
func TestAnthropicPrefixStability(t *testing.T) {
	turn1 := `{"model":"deepseek-v4","max_tokens":1024,"stream":true,
		"system":[{"type":"text","text":"You are Claude Code."}],
		"tools":[{"name":"Read","description":"read a file","input_schema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}],
		"messages":[{"role":"user","content":"读一下 main.go"}]}`
	turn2 := `{"model":"deepseek-v4","max_tokens":1024,"stream":true,
		"system":[{"type":"text","text":"You are Claude Code."}],
		"tools":[{"name":"Read","description":"read a file","input_schema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}],
		"messages":[
			{"role":"user","content":"读一下 main.go"},
			{"role":"assistant","content":[
				{"type":"thinking","thinking":"need to read the file","signature":"c2ln"},
				{"type":"tool_use","id":"toolu_01","name":"Read","input":{"path":"main.go"}}
			]},
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_01","content":"package main\nfunc main(){}"}]},
			{"role":"user","content":"总结一下"}
		]}`

	r1, r2 := mustParse(t, turn1), mustParse(t, turn2)
	// 约束二：同一会话的 cc- 键跨轮恒同值（粘性路由按它绑定账号）。
	if r1.ConversationKey() != r2.ConversationKey() {
		t.Fatal("conversation key must be stable across turns of one conversation")
	}
	b1, err := r1.ToOpenAI()
	if err != nil {
		t.Fatalf("turn1 convert: %v", err)
	}
	b2, err := r2.ToOpenAI()
	if err != nil {
		t.Fatalf("turn2 convert: %v", err)
	}
	seg1 := messagesSegment(t, string(b1))
	seg2 := messagesSegment(t, string(b2))
	if !strings.HasPrefix(seg2, seg1) {
		t.Fatalf("turn2 messages must byte-extend turn1 messages (cache prefix broken)\nturn1: %s\nturn2: %s", seg1, seg2)
	}
	if seg1 == seg2 {
		t.Fatal("turn2 must actually append new turns (sanity)")
	}
}

// messagesSegment 提取 body 里 "messages":[ ... ] 的数组内容字节段。
// 依赖 map 键排序：messages 之后紧跟的是 metadata（m-e-s < m-e-t），
// 用 `],"metadata"` 作右界（取首次出现即数组闭合处）。
func messagesSegment(t *testing.T, body string) string {
	t.Helper()
	const open = `"messages":[`
	i := strings.Index(body, open)
	if i < 0 {
		t.Fatal("messages segment missing")
	}
	start := i + len(open)
	j := strings.Index(body[start:], `],"metadata"`)
	if j < 0 {
		t.Fatal("messages array close marker missing")
	}
	return body[start : start+j]
}

// TestAnthropicPrefixStabilityToolsStable tools/system 的序列化跨轮字节一致
// （它们渲染在 messages 之前，前缀缓存的第一段）。
func TestAnthropicPrefixStabilityToolsStable(t *testing.T) {
	turn1 := `{"model":"m","max_tokens":8,"system":"S","tools":[{"name":"f","input_schema":{"type":"object"}}],"messages":[{"role":"user","content":"a"}]}`
	turn2 := `{"model":"m","max_tokens":8,"system":"S","tools":[{"name":"f","input_schema":{"type":"object"}}],"messages":[{"role":"user","content":"a"},{"role":"assistant","content":[{"type":"text","text":"b"}]}]}`
	b1, err := mustParse(t, turn1).ToOpenAI()
	if err != nil {
		t.Fatalf("turn1: %v", err)
	}
	b2, err := mustParse(t, turn2).ToOpenAI()
	if err != nil {
		t.Fatalf("turn2: %v", err)
	}
	seg1 := beforeMessages(t, string(b1))
	seg2 := beforeMessages(t, string(b2))
	if seg1 != seg2 {
		t.Fatalf("prefix before messages must be byte-identical\nturn1: %s\nturn2: %s", seg1, seg2)
	}
}

func beforeMessages(t *testing.T, body string) string {
	t.Helper()
	i := strings.Index(body, `"messages":[`)
	if i < 0 {
		t.Fatal("messages segment missing")
	}
	return body[:i]
}
