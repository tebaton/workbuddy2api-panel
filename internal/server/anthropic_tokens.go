// anthropic_tokens.go count_tokens 本地估算器。
//
// 上游无 token 计数端点，本地启发式估算：ascii 字节 ÷4 + CJK 字符 ×1，按转换后
// 的 OpenAI body 全文（含 tools/system/messages）计算，向上取整。刻意略微高估
// ——Claude Code 用该值判断自动压缩阈值，高估只会提前压缩（安全侧）。
package server

import (
	"encoding/json"
	"unicode/utf8"
)

// estimateInputTokens 对转换后的 OpenAI body 估算输入 token 数。
// 解析后剔除 metadata（网关注入的会话键，非请求内容）与 stream（协议位）再
// 序列化为定序 JSON（map 键排序），保证同一请求体恒同估值。
func estimateInputTokens(openAIBody []byte) int {
	var body map[string]any
	if err := json.Unmarshal(openAIBody, &body); err != nil {
		return (len(openAIBody) + 3) / 4
	}
	delete(body, "metadata")
	delete(body, "stream")
	raw, err := json.Marshal(body) // map 键排序 → 确定性
	if err != nil {
		return (len(openAIBody) + 3) / 4
	}
	ascii, cjk := 0, 0
	for i := 0; i < len(raw); {
		r, size := utf8.DecodeRune(raw[i:])
		if r >= 0x4E00 && r <= 0x9FFF {
			cjk++
		} else {
			ascii += size
		}
		i += size
	}
	return (ascii+3)/4 + cjk
}
