// anthropic_debug.go /v1/messages 排障转储（logging.anthropic_debug_dir，空 = 关闭）。
//
// 开启后每个请求把三样材料落盘到该目录，文件名前缀 dbg-<request_id>（与
// request-logs/*.jsonl 里的 request_id 同源，指标行 ↔ 报文可互查）：
//
//	dbg-<rid>.anthropic.json  进站 Anthropic 原始请求体
//	dbg-<rid>.openai.json     转换后的 OpenAI 请求体（解析/转换失败时省略）
//	dbg-<rid>.error.txt       以错误收尾时的失败形态说明 + 上游原始报文
//
// 含完整提示词与会话历史（敏感），只排障开启；目录自管理：单文件截断上限
// 8 MiB（base64 图片能把请求体撑得很大），文件数超过保留上限时删最旧。
package server

import (
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/linguo2625469/workbuddy2api-panel/internal/reqlog"
)

// anthropicDebugMaxFileBytes 单文件截断上限。
const anthropicDebugMaxFileBytes = 8 << 20

// anthropicDebugKeepFiles 目录内 dbg-* 文件保留上限（变量供测试覆写）。
var anthropicDebugKeepFiles = 300

// debugWarnOnce 写盘失败只告警一次：转储是排障附属品，不能刷屏盖过业务日志。
var debugWarnOnce sync.Once

// debugRequestID 转储文件名用的请求标识：优先复用 reqlog 同源的 request_id，
// 拿不到（非 traced 路径）时现生成一个。
func debugRequestID(r *http.Request) string {
	if tr := requestTraceFrom(r); tr != nil && tr.id != "" {
		return tr.id
	}
	return reqlog.NewRequestID()
}

// dumpAnthropicDebug 落盘一次请求的排障材料。dir 为空整体 no-op；in 为进站
// Anthropic body；out 为转换后 OpenAI body（nil = 转换未完成，省略该文件）；
// errNote 非空时写 error.txt（首行形态说明，errBody 原样追加）。
func dumpAnthropicDebug(dir, rid string, in, out []byte, errNote string, errBody []byte) {
	if dir == "" {
		return
	}
	if rid == "" {
		rid = reqlog.NewRequestID()
	}
	_ = os.MkdirAll(dir, 0o700)
	base := filepath.Join(dir, "dbg-"+rid)
	writeDebugFile(base+".anthropic.json", in)
	if out != nil {
		writeDebugFile(base+".openai.json", out)
	}
	if errNote != "" {
		writeDebugFile(base+".error.txt", append([]byte(errNote+"\n"), errBody...))
	}
	pruneAnthropicDebug(dir)
}

// writeDebugFile 原子性足够（temp 目录/单写者场景）的直接写；超限截断。
func writeDebugFile(path string, body []byte) {
	if len(body) > anthropicDebugMaxFileBytes {
		truncated := make([]byte, 0, anthropicDebugMaxFileBytes+len("\n...[truncated]"))
		truncated = append(truncated, body[:anthropicDebugMaxFileBytes]...)
		truncated = append(truncated, "\n...[truncated]"...)
		body = truncated
	}
	if err := os.WriteFile(path, body, 0o600); err != nil {
		debugWarnOnce.Do(func() {
			log.Printf("WARN: [server] anthropic debug dump 写盘失败（后续不再重复告警）: %v", err)
		})
	}
}

// pruneAnthropicDebug 目录内 dbg-* 文件数超上限时按 mtime（同名序）删最旧。
func pruneAnthropicDebug(dir string) {
	entries, err := os.ReadDir(dir)
	if err != nil || len(entries) <= anthropicDebugKeepFiles {
		return
	}
	type item struct {
		name string
		mod  time.Time
	}
	items := make([]item, 0, len(entries))
	for _, e := range entries {
		if e.IsDir() || !strings.HasPrefix(e.Name(), "dbg-") {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		items = append(items, item{name: e.Name(), mod: info.ModTime()})
	}
	sort.Slice(items, func(i, j int) bool {
		if items[i].mod.Equal(items[j].mod) {
			return items[i].name < items[j].name
		}
		return items[i].mod.Before(items[j].mod)
	})
	for i := 0; i < len(items)-anthropicDebugKeepFiles; i++ {
		_ = os.Remove(filepath.Join(dir, items[i].name))
	}
}
