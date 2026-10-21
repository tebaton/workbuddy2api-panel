// anthropic_handler.go Anthropic Messages 兼容端点（POST /v1/messages）。
//
// 方案一（设计 §总体架构）：请求纯函数转换为 OpenAI body → 构造合成
// *http.Request（保留原始头与 context，供日志/reqlog/粘性与 trace 复用）→
// 翻译 ResponseWriter 包住真实 writer → 直调 chatCompletions。现有管线零改动。
package server

import (
	"bytes"
	"io"
	"net/http"

	"github.com/linguo2625469/workbuddy2api-panel/internal/httpauth"
)

// withAuthAnthropic Anthropic 端点鉴权：Bearer（ANTHROPIC_AUTH_TOKEN 形态）或
// x-api-key（Anthropic SDK 标准形态）任一命中即放行；401 用 Anthropic 错误形态。
// anthropic-version 头不校验、anthropic-beta 忽略（行为固定，无需能力协商）。
func (h *Handler) withAuthAnthropic(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		key := h.loadLive().APIKey
		if !httpauth.VerifyBearer(r, key) && !httpauth.VerifyXAPIKey(r, key) {
			writeMessagesError(w, http.StatusUnauthorized, "authentication_error", "missing or invalid API key")
			return
		}
		next(w, r)
	}
}

// anthropicMessages POST /v1/messages 主入口。
func (h *Handler) anthropicMessages(w http.ResponseWriter, r *http.Request) {
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		writeMessagesError(w, http.StatusBadRequest, "invalid_request_error", "failed to read request body")
		return
	}
	areq, cerr := parseAnthropicRequest(raw)
	if cerr != nil {
		writeMessagesError(w, http.StatusBadRequest, "invalid_request_error", cerr.Error())
		return
	}
	obody, cerr := areq.ToOpenAI()
	if cerr != nil {
		writeMessagesError(w, http.StatusBadRequest, "invalid_request_error", cerr.Error())
		return
	}
	// 合成请求：同 context（requestTrace 共享，单请求单条记录）、同头（UA/XFF →
	// 日志与 reqlog 同源；X-Conversation-Request-ID 透传语义保留）。
	sr := r.Clone(r.Context())
	sr.Method = http.MethodPost
	sr.URL.Path = "/v1/chat/completions"
	sr.Body = io.NopCloser(bytes.NewReader(obody))
	sr.ContentLength = int64(len(obody))
	tr := newAnthropicTranslator(w, areq.Model)
	h.chatCompletions(tr, sr)
	tr.Finish()
}

// anthropicCountTokens POST /v1/messages/count_tokens：本地确定性估算，不打上游。
func (h *Handler) anthropicCountTokens(w http.ResponseWriter, r *http.Request) {
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		writeMessagesError(w, http.StatusBadRequest, "invalid_request_error", "failed to read request body")
		return
	}
	areq, cerr := parseAnthropicRequest(raw)
	if cerr != nil {
		writeMessagesError(w, http.StatusBadRequest, "invalid_request_error", cerr.Error())
		return
	}
	obody, cerr := areq.ToOpenAI()
	if cerr != nil {
		writeMessagesError(w, http.StatusBadRequest, "invalid_request_error", cerr.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"input_tokens": estimateInputTokens(obody)})
}

// anthropicNotImplemented 明确不实现的 Anthropic 端点：501 + 规范错误形态。
func (h *Handler) anthropicNotImplemented(w http.ResponseWriter, r *http.Request) {
	writeMessagesError(w, http.StatusNotImplemented, "not_implemented_error",
		"this endpoint is not implemented by the gateway; chat is served at /v1/messages")
}
