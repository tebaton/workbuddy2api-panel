// anthropic_convert.go Anthropic Messages 请求 → OpenAI chat/completions 请求的纯函数转换。
//
// 缓存三约束之一（设计 §缓存命中保证）：本文件必须是纯函数——同样的 Anthropic
// 历史字节，永远产出同样的 OpenAI 字节。只用 Go struct/map 序列化（map 键排序
// 固定），禁止注入任何逐请求变化的字节（时间戳 / 随机 id / nonce）。
// metadata 被整体替换为 cc- 会话键（同输入恒同值），供粘性路由与上游
// prompt_cache_key 使用；客户端原 metadata（如 Claude Code 跨会话稳定的
// user_id）不透传——粒度太粗，会话键按会话派生。
package server

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
)

// anthropicRequest Anthropic Messages 请求（只声明本层关心的字段；未知字段
// 整体丢弃 = 确定性剥离，含 cache_control / 各类 beta 能力字段）。
type anthropicRequest struct {
	Model         string               `json:"model"`
	MaxTokens     int                  `json:"max_tokens"`
	System        json.RawMessage      `json:"system"` // string 或 []block
	Messages      []anthropicMessage   `json:"messages"`
	Tools         []anthropicToolDef   `json:"tools"`
	ToolChoice    *anthropicToolChoice `json:"tool_choice"`
	StopSequences []string             `json:"stop_sequences"`
	Temperature   *float64             `json:"temperature"`
	TopP          *float64             `json:"top_p"`
	Thinking      json.RawMessage      `json:"thinking"` // 原样透传，payload.go 消费
	Stream        bool                 `json:"stream"`
}

type anthropicMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"` // string 或 []block
}

type anthropicToolDef struct {
	Type        string          `json:"type"` // "custom"/空 为自定义工具
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

type anthropicToolChoice struct {
	Type string `json:"type"` // auto|any|tool|none
	Name string `json:"name"`
}

// anthropicBlock content 数组里的块（多态字段拍平声明）。
type anthropicBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
	// image
	Source *anthropicSource `json:"source"`
	// thinking
	Thinking  string `json:"thinking"`
	Signature string `json:"signature"`
	// tool_use
	ID    string          `json:"id"`
	Name  string          `json:"name"`
	Input json.RawMessage `json:"input"`
	// tool_result
	ToolUseID string          `json:"tool_use_id"`
	Content   json.RawMessage `json:"content"` // string 或 []block
}

type anthropicSource struct {
	Type      string `json:"type"` // base64 | url
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
	URL       string `json:"url"`
}

// parseAnthropicRequest 解析并校验。max_tokens 是 Anthropic 协议必填项，缺失/非正
// 直接 400（与官方行为一致）；model 必填。messages 不在此强校验——空 messages
// 与 OpenAI 端点同口径交由上游处理（既有 chat 测试即以 messages:[] 打上游）。
func parseAnthropicRequest(raw []byte) (*anthropicRequest, error) {
	var req anthropicRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		return nil, fmt.Errorf("request body must be a JSON object: %v", err)
	}
	if strings.TrimSpace(req.Model) == "" {
		return nil, fmt.Errorf("model is required")
	}
	if req.MaxTokens <= 0 {
		return nil, fmt.Errorf("max_tokens is required and must be positive")
	}
	return &req, nil
}

// systemText 确定性提取 system 文本：string 原样；块数组取 text 块按 "\n\n" 拼接。
func (a *anthropicRequest) systemText() string {
	if len(a.System) == 0 {
		return ""
	}
	var plain string
	if err := json.Unmarshal(a.System, &plain); err == nil {
		return plain
	}
	var blocks []anthropicBlock
	if err := json.Unmarshal(a.System, &blocks); err != nil {
		return ""
	}
	parts := make([]string, 0, len(blocks))
	for _, b := range blocks {
		if b.Type == "text" && b.Text != "" {
			parts = append(parts, b.Text)
		}
	}
	return strings.Join(parts, "\n\n")
}

// firstUserText 提取首条 user 消息文本（会话键派生素材，与 systemText 同口径：
// string 原样，块数组取 text 块拼接）。
func (a *anthropicRequest) firstUserText() string {
	for _, m := range a.Messages {
		if m.Role != "user" {
			continue
		}
		var plain string
		if err := json.Unmarshal(m.Content, &plain); err == nil {
			return plain
		}
		var blocks []anthropicBlock
		if err := json.Unmarshal(m.Content, &blocks); err != nil {
			return ""
		}
		parts := make([]string, 0, len(blocks))
		for _, b := range blocks {
			if b.Type == "text" {
				parts = append(parts, b.Text)
			}
		}
		return strings.Join(parts, "\n\n")
	}
	return ""
}

// ConversationKey 派生会话键："cc-"+sha256(system+"\x00"+首条user)[:16hex]。
// Claude Code 不发 conversation_id，其 metadata.user_id 跨会话稳定（太粗）；
// 本键按会话粒度稳定，喂给粘性路由 → 同会话同账号 → 上游前缀缓存命名空间延续
// （缓存约束二）。确定性派生，同输入恒同值。
func (a *anthropicRequest) ConversationKey() string {
	sum := sha256.Sum256([]byte(a.systemText() + "\x00" + a.firstUserText()))
	return "cc-" + hex.EncodeToString(sum[:8])
}

// ToOpenAI 产出 OpenAI chat/completions 请求 body（确定性：同输入同字节）。
func (a *anthropicRequest) ToOpenAI() ([]byte, error) {
	msgs, err := a.convertMessages()
	if err != nil {
		return nil, err
	}
	if s := a.systemText(); s != "" {
		msgs = append([]map[string]any{{"role": "system", "content": s}}, msgs...)
	}
	out := map[string]any{
		"model":      a.Model,
		"max_tokens": a.MaxTokens,
		"stream":     a.Stream,
		"metadata":   map[string]any{"conversation_id": a.ConversationKey()},
		"messages":   msgs,
	}
	if len(a.StopSequences) > 0 {
		out["stop"] = a.StopSequences
	}
	if a.Temperature != nil {
		out["temperature"] = *a.Temperature
	}
	if a.TopP != nil {
		out["top_p"] = *a.TopP
	}
	if len(a.Thinking) > 0 && string(a.Thinking) != "null" {
		var th any
		if err := json.Unmarshal(a.Thinking, &th); err == nil {
			out["thinking"] = th
		}
	}
	if len(a.Tools) > 0 {
		tools := make([]any, 0, len(a.Tools))
		for _, t := range a.Tools {
			if err := rejectServerTool(t.Type); err != nil {
				return nil, err
			}
			schema := any(map[string]any{})
			if len(t.InputSchema) > 0 && string(t.InputSchema) != "null" {
				var s any
				if err := json.Unmarshal(t.InputSchema, &s); err == nil {
					schema = s
				}
			}
			tools = append(tools, map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        t.Name,
					"description": t.Description,
					"parameters":  schema,
				},
			})
		}
		out["tools"] = tools
	}
	if a.ToolChoice != nil {
		switch a.ToolChoice.Type {
		case "auto":
			out["tool_choice"] = "auto"
		case "any":
			out["tool_choice"] = "required"
		case "none":
			out["tool_choice"] = "none"
		case "tool":
			out["tool_choice"] = map[string]any{
				"type":     "function",
				"function": map[string]any{"name": a.ToolChoice.Name},
			}
		}
	}
	return json.Marshal(out)
}

// serverToolPrefixes Anthropic server tool 的 type 前缀（模型侧执行，本网关无对应物）。
var serverToolPrefixes = []string{
	"web_search", "web_fetch", "computer", "text_editor", "bash",
	"code_execution", "memory", "tool_search",
}

// rejectServerTool 自定义工具的 type 恒为 "custom" 或空；其余是 server tool，
// 网关无对应能力，400 点名拒绝（响亮失败优于静默丢）。
func rejectServerTool(typ string) error {
	if typ == "" || typ == "custom" {
		return nil
	}
	return fmt.Errorf("server tool type %q is not supported by this gateway", typ)
}

// convertMessages Anthropic 消息数组 → OpenAI messages。
// assistant 的 text/thinking/tool_use 合成单条 assistant 消息；user 的 tool_result
// 逐条独立 role:"tool"，其余 text/image 归入其后一条 user 消息（Anthropic 允许
// 混排，OpenAI 要求分离）。
func (a *anthropicRequest) convertMessages() ([]map[string]any, error) {
	var out []map[string]any
	for _, m := range a.Messages {
		// content 纯字符串简写。
		var plain string
		if err := json.Unmarshal(m.Content, &plain); err == nil {
			out = append(out, map[string]any{"role": m.Role, "content": plain})
			continue
		}
		var blocks []anthropicBlock
		if err := json.Unmarshal(m.Content, &blocks); err != nil {
			return nil, fmt.Errorf("message content must be a string or an array of blocks")
		}
		switch m.Role {
		case "assistant":
			msg, err := convertAssistantBlocks(blocks)
			if err != nil {
				return nil, err
			}
			if msg != nil {
				out = append(out, msg)
			}
		case "user":
			msgs, err := convertUserBlocks(blocks)
			if err != nil {
				return nil, err
			}
			out = append(out, msgs...)
		case "system":
			// Claude Code 会内联 role:"system" 消息。原位转成 OpenAI system
			// 消息（挪位/前并会改变既有轮序列化字节，破坏前缀缓存）。
			text, err := systemBlocksText(blocks)
			if err != nil {
				return nil, err
			}
			if text != "" {
				out = append(out, map[string]any{"role": "system", "content": text})
			}
		default:
			return nil, fmt.Errorf("unsupported message role %q", m.Role)
		}
	}
	return out, nil
}

// systemBlocksText 内联 system 消息的块数组 → 文本（text 块按 "\n\n" 拼接），
// 产出与纯字符串透传形态对齐（同为 {"role":"system","content":"..."}）。
// 空文本轮确定性丢弃。
func systemBlocksText(blocks []anthropicBlock) (string, error) {
	parts := make([]string, 0, len(blocks))
	for _, b := range blocks {
		if b.Type != "text" {
			return "", fmt.Errorf("unsupported system content block type %q", b.Type)
		}
		if b.Text != "" {
			parts = append(parts, b.Text)
		}
	}
	return strings.Join(parts, "\n\n"), nil
}

// convertAssistantBlocks thinking → reasoning_content（剥 signature）；text →
// content；tool_use → tool_calls（input 对象经 map 重排按键排序序列化——缓存
// 前缀稳定）。全空轮确定性丢弃（不出空消息）。
func convertAssistantBlocks(blocks []anthropicBlock) (map[string]any, error) {
	var textParts, thinkParts []string
	var toolCalls []any
	for _, b := range blocks {
		switch b.Type {
		case "text":
			if b.Text != "" {
				textParts = append(textParts, b.Text)
			}
		case "thinking", "redacted_thinking":
			if b.Thinking != "" {
				thinkParts = append(thinkParts, b.Thinking)
			}
		case "tool_use":
			args := "{}"
			if len(b.Input) > 0 && string(b.Input) != "null" {
				var obj any
				if err := json.Unmarshal(b.Input, &obj); err != nil {
					return nil, fmt.Errorf("tool_use input for %q is not valid JSON", b.Name)
				}
				raw, err := json.Marshal(obj) // map 键排序 → 确定性
				if err != nil {
					return nil, err
				}
				args = string(raw)
			}
			id := b.ID
			if id == "" {
				id = "call_" + b.Name
			}
			toolCalls = append(toolCalls, map[string]any{
				"id":   id,
				"type": "function",
				"function": map[string]any{
					"name":      b.Name,
					"arguments": args,
				},
			})
		default:
			return nil, fmt.Errorf("unsupported assistant content block type %q", b.Type)
		}
	}
	if len(textParts) == 0 && len(thinkParts) == 0 && len(toolCalls) == 0 {
		return nil, nil // 空轮（如只含 signature 的占位）：确定性丢弃
	}
	msg := map[string]any{"role": "assistant"}
	if len(textParts) > 0 {
		msg["content"] = strings.Join(textParts, "\n\n")
	} else {
		msg["content"] = nil
	}
	if len(thinkParts) > 0 {
		msg["reasoning_content"] = strings.Join(thinkParts, "\n\n")
	}
	if len(toolCalls) > 0 {
		msg["tool_calls"] = toolCalls
	}
	return msg, nil
}

// convertUserBlocks tool_result → 独立 role:"tool" 消息（先），text/image → 一条
// user 消息（后，保持块内原始顺序）。单个纯 text 块折叠为字符串形态（与 OpenAI
// 客户端常规形态一致）。tool_result 内 image 块以上游不支持的占位文本替代（确定性）。
func convertUserBlocks(blocks []anthropicBlock) ([]map[string]any, error) {
	var out []map[string]any
	var parts []any
	for _, b := range blocks {
		switch b.Type {
		case "tool_result":
			content, err := toolResultContent(b)
			if err != nil {
				return nil, err
			}
			out = append(out, map[string]any{
				"role":         "tool",
				"tool_call_id": b.ToolUseID,
				"content":      content,
			})
		case "text":
			if b.Text != "" {
				parts = append(parts, map[string]any{"type": "text", "text": b.Text})
			}
		case "image":
			url, err := imageURL(b.Source)
			if err != nil {
				return nil, err
			}
			parts = append(parts, map[string]any{
				"type":      "image_url",
				"image_url": map[string]any{"url": url},
			})
		default:
			return nil, fmt.Errorf("unsupported user content block type %q", b.Type)
		}
	}
	// 单个纯 text part：字符串形态；其余（含图片/多块）保序数组形态。
	if len(parts) == 1 {
		if tp, ok := parts[0].(map[string]any); ok && tp["type"] == "text" {
			out = append(out, map[string]any{"role": "user", "content": tp["text"]})
			return out, nil
		}
	}
	if len(parts) > 0 {
		out = append(out, map[string]any{"role": "user", "content": parts})
	}
	return out, nil
}

// toolResultContent tool_result.content（string 或块数组）→ OpenAI tool 消息的
// content 字符串。image 块替换为占位文本（上游 tool content 只吃字符串）。
func toolResultContent(b anthropicBlock) (string, error) {
	if len(b.Content) == 0 || string(b.Content) == "null" {
		return "", nil
	}
	var plain string
	if err := json.Unmarshal(b.Content, &plain); err == nil {
		return plain, nil
	}
	var blocks []anthropicBlock
	if err := json.Unmarshal(b.Content, &blocks); err != nil {
		return "", fmt.Errorf("tool_result content must be a string or an array of blocks")
	}
	var parts []string
	for _, cb := range blocks {
		switch cb.Type {
		case "text":
			parts = append(parts, cb.Text)
		case "image":
			parts = append(parts, "[image attachment omitted]")
		default:
			return "", fmt.Errorf("unsupported tool_result content block type %q", cb.Type)
		}
	}
	return strings.Join(parts, "\n"), nil
}

// imageURL Anthropic image source → OpenAI image_url.url。
func imageURL(s *anthropicSource) (string, error) {
	if s == nil {
		return "", fmt.Errorf("image block is missing source")
	}
	switch s.Type {
	case "base64":
		if s.MediaType == "" || s.Data == "" {
			return "", fmt.Errorf("base64 image source requires media_type and data")
		}
		return "data:" + s.MediaType + ";base64," + s.Data, nil
	case "url":
		if s.URL == "" {
			return "", fmt.Errorf("url image source requires url")
		}
		return s.URL, nil
	default:
		return "", fmt.Errorf("unsupported image source type %q", s.Type)
	}
}
