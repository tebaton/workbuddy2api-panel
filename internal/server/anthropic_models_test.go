package server

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// anthropicGET 向 path 发 GET（Bearer 鉴权形态）。
func anthropicGET(t *testing.T, h *Handler, path string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, path, nil)
	req.Header.Set("Authorization", "Bearer k")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

// resetDynamicModelsCache 测试前后清包级动态模型缓存（fake 域名拉取失败会写
// 5min 负缓存，污染后续依赖真实拉取的测试——与 TestModelsDynamic 同款处理）。
func resetDynamicModelsCache(t *testing.T) {
	t.Helper()
	reset := func() {
		dynamicModelsCache.Lock()
		dynamicModelsCache.ids = nil
		dynamicModelsCache.fetched = time.Time{}
		dynamicModelsCache.lastFail = time.Time{}
		dynamicModelsCache.Unlock()
	}
	reset()
	t.Cleanup(reset)
}

func TestModelsDualShape(t *testing.T) {
	resetDynamicModelsCache(t)
	h := newAnthropicTestHandler(t, func(string) (int, string, bool) { return 200, sseOK, true })
	rec := anthropicGET(t, h, "/v1/models")
	if rec.Code != 200 {
		t.Fatalf("want 200 got %d: %s", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	// OpenAI 形态保留
	if !strings.Contains(body, `"object":"list"`) {
		t.Fatalf("openai shape must survive: %s", body)
	}
	// Anthropic 形态字段在场（空目录时 has_more/first_id/last_id 也必须在）。
	if !strings.Contains(body, `"has_more":false`) {
		t.Fatalf("anthropic has_more missing: %s", body)
	}
}

func TestModelsRetrieve(t *testing.T) {
	resetDynamicModelsCache(t)
	h := newAnthropicTestHandler(t, func(string) (int, string, bool) { return 200, sseOK, true })
	rec := anthropicGET(t, h, "/v1/models/__nope__")
	if rec.Code != 404 || !strings.Contains(rec.Body.String(), "not_found_error") {
		t.Fatalf("unknown model 404 expected: %d %s", rec.Code, rec.Body.String())
	}
}

func TestModelsRetrieveHit(t *testing.T) {
	resetDynamicModelsCache(t)
	// 假上游返回动态目录，retrieve 命中真实条目（双形态字段在场）。
	h := newAnthropicTestHandler(t, func(string) (int, string, bool) {
		return 200, `{"code":0,"data":{"models":[{"id":"dyn-a","name":"Dyn A"}],"agents":[{"name":"cli","models":["dyn-a"]}]}}`, false
	})
	rec := anthropicGET(t, h, "/v1/models/cn:dyn-a")
	if rec.Code != 200 {
		t.Fatalf("want 200 got %d: %s", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	for _, want := range []string{`"id":"cn:dyn-a"`, `"type":"model"`, `"display_name":"Dyn A"`, `"created_at"`} {
		if !strings.Contains(body, want) {
			t.Fatalf("dual shape missing %q: %s", want, body)
		}
	}
}

func TestNotImplementedEndpoints(t *testing.T) {
	h := newAnthropicTestHandler(t, func(string) (int, string, bool) { return 200, sseOK, true })
	for _, path := range []string{"/v1/messages/batches", "/v1/messages/batches/msgb_1", "/v1/files", "/v1/files/file_1", "/v1/complete"} {
		req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(`{}`))
		req.Header.Set("x-api-key", "k")
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != 501 || !strings.Contains(rec.Body.String(), "not_implemented_error") {
			t.Fatalf("%s: 501 not_implemented expected: %d %s", path, rec.Code, rec.Body.String())
		}
	}
}
