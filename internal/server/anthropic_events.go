// anthropic_events.go 翻译 ResponseWriter：把 chatCompletions 管线输出的
// OpenAI wire 形态（重建 SSE 帧或单 JSON）实时转写为 Anthropic Messages 形态。
//
// 三种工作模式（按管线行为自动判定，契约来自 sse.go 的白名单重建）：
//   - SSE：StreamHint 先经 Header() 设 text/event-stream 再 Write（隐式 200）——
//     逐帧解析 data: 行，驱动块序状态机输出 Anthropic 事件流；
//   - JSON：writeJSON 显式 WriteHeader(200)+application/json+单次 Write——
//     缓冲后整体转换为 Anthropic Message（非流式聚合路径）；
//   - ERROR：非 200 + application/json——缓冲后重渲染为 Anthropic 错误形态。
//
// 块序规则：thinking/text 顺序开闭（增量类型切换即闭块开新块，index 单调递增）；
// tool_use 块按需打开、**流尾统一关闭**——OpenAI 流允许不同 index 的 tool_calls
// 交错续传 arguments，提前闭块会让增量落到已关闭的块上（协议非法）。
package server

import (
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"strings"
)

// anthropicErrType HTTP 状态码 → Anthropic 错误类型词表。
func anthropicErrType(status int) string {
	switch {
	case status == http.StatusUnauthorized:
		return "authentication_error"
	case status == http.StatusNotFound:
		return "not_found_error"
	case status == http.StatusTooManyRequests:
		return "rate_limit_error"
	case status == http.StatusNotImplemented:
		return "not_implemented_error"
	case status == http.StatusServiceUnavailable:
		return "overloaded_error"
	case status >= 500:
		return "api_error"
	default:
		return "invalid_request_error"
	}
}

// 翻译器工作模式。
const (
	trModeUndecided = iota
	trModeSSE
	trModeJSON
	trModeError
)

// anthropicTranslator 包住真实 ResponseWriter 交给 chatCompletions。
type anthropicTranslator struct {
	real   http.ResponseWriter
	model  string
	mode   int
	status int
	buf    []byte // JSON/ERROR 模式缓冲
	sse    *sseState
}

func newAnthropicTranslator(real http.ResponseWriter, model string) *anthropicTranslator {
	return &anthropicTranslator{real: real, model: model}
}

func (t *anthropicTranslator) Header() http.Header { return t.real.Header() }

func (t *anthropicTranslator) Flush() {
	if fl, ok := t.real.(http.Flusher); ok {
		fl.Flush()
	}
}

func (t *anthropicTranslator) WriteHeader(status int) {
	t.status = status
	if status != http.StatusOK {
		t.mode = trModeError
		return // 延迟到 Finish 渲染（错误体可能还没写）
	}
	// 200：SSE 路径不显式 WriteHeader（首 Write 隐式提交），JSON 路径随后经
	// Content-Type 判定；这里不透传，等首 Write / Finish 决定。
}

func (t *anthropicTranslator) Write(p []byte) (int, error) {
	if t.mode == trModeUndecided {
		ct := t.real.Header().Get("Content-Type")
		if strings.Contains(ct, "text/event-stream") {
			t.mode = trModeSSE
			t.status = http.StatusOK
			t.sse = newSSEState(t.real, t.model)
		} else {
			t.mode = trModeJSON // writeJSON(application/json)+200
			t.status = http.StatusOK
		}
	}
	switch t.mode {
	case trModeSSE:
		return t.sse.feed(p)
	default: // trModeJSON / trModeError
		t.buf = append(t.buf, p...)
		return len(p), nil
	}
}

// Finish chatCompletions 返回后由 handler 调用：渲染最终 Anthropic 形态输出。
func (t *anthropicTranslator) Finish() {
	switch t.mode {
	case trModeSSE:
		if t.sse != nil {
			t.sse.finish() // 防御性收尾（[DONE] 缺失时不悬块）
		}
	case trModeJSON:
		writeAnthropicAggregated(t.real, t.buf, t.model)
	case trModeError:
		writeAnthropicErrorBody(t.real, t.status, t.buf)
	}
}

// writeAnthropicErrorBody 把 OpenAI 错误 JSON 重渲染为 Anthropic 错误形态。
// message 原文透传（error-passthrough 语义保留），gateway_hint 并列补充。
func writeAnthropicErrorBody(w http.ResponseWriter, status int, body []byte) {
	msg := string(body)
	var parsed struct {
		Error struct {
			Message    string `json:"message"`
			GatewayHit string `json:"gateway_hint"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &parsed); err == nil && parsed.Error.Message != "" {
		msg = parsed.Error.Message
		if parsed.Error.GatewayHit != "" {
			msg += " (" + parsed.Error.GatewayHit + ")"
		}
	}
	writeMessagesError(w, status, anthropicErrType(status), msg)
}

// writeAnthropicAggregated 非流式：chat.completion JSON → Anthropic Message。
func writeAnthropicAggregated(w http.ResponseWriter, body []byte, model string) {
	var resp map[string]any
	if err := json.Unmarshal(body, &resp); err != nil {
		writeMessagesError(w, http.StatusBadGateway, "api_error", "failed to parse aggregated response")
		return
	}
	content := []any{}
	finish := ""
	if chs, ok := resp["choices"].([]any); ok && len(chs) > 0 {
		if c, ok := chs[0].(map[string]any); ok {
			finish, _ = c["finish_reason"].(string)
			if msg, ok := c["message"].(map[string]any); ok {
				if rc, _ := msg["reasoning_content"].(string); rc != "" {
					content = append(content, map[string]any{
						"type": "thinking", "thinking": rc,
						"signature": thinkingSignature(rc),
					})
				}
				if txt, _ := msg["content"].(string); txt != "" {
					content = append(content, map[string]any{"type": "text", "text": txt})
				}
				if tcs, ok := msg["tool_calls"].([]any); ok {
					for _, tci := range tcs {
						tc, ok := tci.(map[string]any)
						if !ok {
							continue
						}
						input := any(map[string]any{})
						name, id := "", ""
						if fn, ok := tc["function"].(map[string]any); ok {
							name, _ = fn["name"].(string)
							if args, _ := fn["arguments"].(string); args != "" {
								var obj any
								if err := json.Unmarshal([]byte(args), &obj); err == nil {
									input = obj
								}
							}
						}
						id, _ = tc["id"].(string)
						content = append(content, map[string]any{
							"type": "tool_use", "id": id, "name": name, "input": input,
						})
					}
				}
			}
		}
	}
	if len(content) == 0 {
		content = append(content, map[string]any{"type": "text", "text": ""})
	}
	id, _ := resp["id"].(string)
	if id == "" {
		id = "msg_wb2api_" + randomHex(16)
	} else {
		id = "msg_" + id
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"id":            id,
		"type":          "message",
		"role":          "assistant",
		"model":         model,
		"content":       content,
		"stop_reason":   mapStopReason(finish),
		"stop_sequence": nil,
		"usage":         anthropicUsageFrom(resp["usage"]),
	})
}

// mapStopReason OpenAI finish_reason → Anthropic stop_reason。
func mapStopReason(finish string) string {
	switch finish {
	case "length":
		return "max_tokens"
	case "tool_calls", "function_call":
		return "tool_use"
	case "content_filter":
		return "refusal"
	default:
		return "end_turn"
	}
}

// anthropicUsageFrom 上游 usage 对象（prompt_tokens/completion_tokens/
// prompt_cache_hit_tokens，兼容 cached_tokens 别名）→ Anthropic usage。
// cache_read_input_tokens 是缓存命中量透出（约束三：命中可观测）。
func anthropicUsageFrom(raw any) map[string]any {
	out := map[string]any{"input_tokens": 0, "output_tokens": 0}
	u, ok := raw.(map[string]any)
	if !ok {
		return out
	}
	if v, ok := u["prompt_tokens"].(float64); ok {
		out["input_tokens"] = int(v)
	}
	if v, ok := u["completion_tokens"].(float64); ok {
		out["output_tokens"] = int(v)
	}
	hit := -1.0
	for _, k := range []string{"prompt_cache_hit_tokens", "cached_tokens"} {
		if v, ok := u[k].(float64); ok && v > hit {
			hit = v
		}
	}
	if hit >= 0 {
		out["cache_read_input_tokens"] = int(hit)
	}
	return out
}

// thinkingSignature 思维链签名：内容哈希派生。本网关即服务端，回传不校验（入站
// 转换剥掉 signature）；确定性派生（同 thinking 同签名）满足客户端 SDK 的非空
// 预期且保持本层幂等。
func thinkingSignature(thinking string) string {
	sum := sha256.Sum256([]byte(thinking))
	return base64.StdEncoding.EncodeToString(sum[:])
}

// ---- SSE 状态机 ----

// openTool 一个已打开的 tool_use 块（流尾统一关闭）。
type openTool struct {
	blockIdx int
}

// sseState 逐帧解析重建后的 OpenAI SSE，输出 Anthropic 事件流。
type sseState struct {
	w       http.ResponseWriter
	fl      http.Flusher
	model   string
	msgID   string
	started bool
	ended   bool
	line    []byte // 跨 Write 的半行缓冲

	curKind   string // "" | "thinking" | "text"（thinking/text 顺序开闭）
	curIdx    int    // 当前 thinking/text 块的 index
	nextIdx   int    // 块 index 分配器（单调递增）
	thinkBuf  strings.Builder
	openTools []*openTool          // 按开块顺序
	toolByUp  map[int]*openTool    // upstream tool_call index → 块

	stopReason string
	usage      map[string]any
}

func newSSEState(w http.ResponseWriter, model string) *sseState {
	fl, _ := w.(http.Flusher)
	return &sseState{w: w, fl: fl, model: model, toolByUp: map[int]*openTool{}}
}

// emit 写一个 Anthropic SSE 事件并 flush。
func (s *sseState) emit(name string, payload any) {
	b, err := json.Marshal(payload)
	if err != nil {
		return
	}
	_, _ = io.WriteString(s.w, "event: "+name+"\ndata: "+string(b)+"\n\n")
	if s.fl != nil {
		s.fl.Flush()
	}
}

// feed 处理管线写出的一段字节（可能含多帧 / 半帧）。
func (s *sseState) feed(p []byte) (int, error) {
	s.line = append(s.line, p...)
	for {
		i := bytes.IndexByte(s.line, '\n')
		if i < 0 {
			break
		}
		line := strings.TrimRight(string(s.line[:i]), "\r")
		s.line = s.line[i+1:]
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		s.handleFrame(strings.TrimPrefix(line, "data: "))
	}
	return len(p), nil
}

// handleFrame 单帧路由：[DONE] / error 帧 / chunk 帧。
func (s *sseState) handleFrame(payload string) {
	if s.ended {
		return
	}
	if payload == "[DONE]" {
		s.finish()
		return
	}
	var frame map[string]any
	if err := json.Unmarshal([]byte(payload), &frame); err != nil {
		return // 管线已保证白名单 JSON，解析失败防御性跳过
	}
	if e, ok := frame["error"]; ok && e != nil {
		// 中途 error 帧：Anthropic 形态 error 事件收尾（无 message_stop）。
		msg := payload
		if em, ok := e.(map[string]any); ok {
			if m, _ := em["message"].(string); m != "" {
				msg = m
			}
		}
		s.emit("error", map[string]any{
			"type":  "error",
			"error": map[string]any{"type": "api_error", "message": msg},
		})
		s.ended = true
		return
	}
	s.handleChunk(frame)
}

// handleChunk 规范重建后的 chunk 帧：delta 路由 + finish/usage 提取。
func (s *sseState) handleChunk(frame map[string]any) {
	if !s.started {
		if id, _ := frame["id"].(string); id != "" {
			s.msgID = "msg_" + id
		} else {
			s.msgID = "msg_wb2api_" + randomHex(16)
		}
		s.emit("message_start", map[string]any{
			"type": "message_start",
			"message": map[string]any{
				"id": s.msgID, "type": "message", "role": "assistant", "model": s.model,
				"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
				"usage": map[string]any{"input_tokens": 0, "output_tokens": 0},
			},
		})
		s.emit("ping", map[string]any{"type": "ping"})
		s.started = true
	}
	if u, ok := frame["usage"].(map[string]any); ok {
		s.usage = anthropicUsageFrom(u)
	}
	chs, _ := frame["choices"].([]any)
	if len(chs) == 0 {
		return
	}
	c, _ := chs[0].(map[string]any)
	if c == nil {
		return
	}
	delta, _ := c["delta"].(map[string]any)
	if delta != nil {
		if v, _ := delta["reasoning_content"].(string); v != "" {
			s.openBlock("thinking")
			s.emit("content_block_delta", map[string]any{
				"type": "content_block_delta", "index": s.curIdx,
				"delta": map[string]any{"type": "thinking_delta", "thinking": v},
			})
			s.thinkBuf.WriteString(v)
		}
		if v, _ := delta["content"].(string); v != "" {
			s.openBlock("text")
			s.emit("content_block_delta", map[string]any{
				"type": "content_block_delta", "index": s.curIdx,
				"delta": map[string]any{"type": "text_delta", "text": v},
			})
		}
		if tcs, ok := delta["tool_calls"].([]any); ok {
			for _, tci := range tcs {
				s.handleToolCallDelta(tci)
			}
		}
	}
	if fr, ok := c["finish_reason"].(string); ok && fr != "" && s.stopReason == "" {
		s.stopReason = mapStopReason(fr)
	}
}

// handleToolCallDelta 单条 delta.tool_calls 增量：首见开块（闭当前 thinking/text
// 块），后续增量（含交错）落在自己的块上。
func (s *sseState) handleToolCallDelta(tci any) {
	tc, ok := tci.(map[string]any)
	if !ok {
		return
	}
	upIdx := 0
	if v, ok := tc["index"].(float64); ok {
		upIdx = int(v)
	}
	arguments, name, id := "", "", ""
	if fn, ok := tc["function"].(map[string]any); ok {
		arguments, _ = fn["arguments"].(string)
		name, _ = fn["name"].(string)
	}
	id, _ = tc["id"].(string)
	ot := s.toolByUp[upIdx]
	if ot == nil {
		s.closeBlock() // 闭当前 thinking/text 块（工具块在其后）
		if id == "" {
			id = "call_" + randomHex(8)
		}
		if name == "" {
			name = "tool"
		}
		ot = &openTool{blockIdx: s.nextIdx}
		s.nextIdx++
		s.toolByUp[upIdx] = ot
		s.openTools = append(s.openTools, ot)
		s.emit("content_block_start", map[string]any{
			"type": "content_block_start", "index": ot.blockIdx,
			"content_block": map[string]any{
				"type": "tool_use", "id": id, "name": name, "input": map[string]any{},
			},
		})
	}
	if arguments != "" {
		s.emit("content_block_delta", map[string]any{
			"type": "content_block_delta", "index": ot.blockIdx,
			"delta": map[string]any{"type": "input_json_delta", "partial_json": arguments},
		})
	}
}

// openBlock 确保当前 open 块类型为 kind，否则闭旧开新。
func (s *sseState) openBlock(kind string) {
	if s.curKind == kind {
		return
	}
	s.closeBlock()
	s.curKind = kind
	s.curIdx = s.nextIdx
	s.nextIdx++
	var block map[string]any
	switch kind {
	case "thinking":
		s.thinkBuf.Reset()
		block = map[string]any{"type": "thinking", "thinking": ""}
	case "text":
		block = map[string]any{"type": "text", "text": ""}
	}
	s.emit("content_block_start", map[string]any{
		"type": "content_block_start", "index": s.curIdx, "content_block": block,
	})
}

// closeBlock 闭当前 thinking/text 块；thinking 块闭前补合成 signature_delta。
func (s *sseState) closeBlock() {
	if s.curKind == "" {
		return
	}
	if s.curKind == "thinking" {
		s.emit("content_block_delta", map[string]any{
			"type": "content_block_delta", "index": s.curIdx,
			"delta": map[string]any{"type": "signature_delta", "signature": thinkingSignature(s.thinkBuf.String())},
		})
	}
	s.emit("content_block_stop", map[string]any{"type": "content_block_stop", "index": s.curIdx})
	s.curKind = ""
}

// finish 流收尾：闭当前块 + 闭全部 tool 块（按开块顺序）+ message_delta + message_stop。
// error 帧路径不经过这里（直接 ended）。
func (s *sseState) finish() {
	if s.ended {
		return
	}
	s.ended = true
	if !s.started {
		return // 空流且从未开流：无事件可发（与「上游空流→error 帧」路径衔接）
	}
	s.closeBlock()
	for _, ot := range s.openTools {
		s.emit("content_block_stop", map[string]any{"type": "content_block_stop", "index": ot.blockIdx})
	}
	if s.stopReason == "" {
		s.stopReason = "end_turn"
	}
	usage := s.usage
	if usage == nil {
		usage = map[string]any{"input_tokens": 0, "output_tokens": 0}
	}
	s.emit("message_delta", map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": s.stopReason, "stop_sequence": nil},
		"usage": usage,
	})
	s.emit("message_stop", map[string]any{"type": "message_stop"})
}
