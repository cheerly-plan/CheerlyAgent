#!/usr/bin/env bash
# 开发模式：API（热更新）+ 知识库 ingest-worker。
# 用法：./scripts/dev.sh [--host HOST] [--port PORT]
set -euo pipefail

cd "$(dirname "$0")/.."

# API 默认不跑解析（KB_INGEST_INLINE=false）；worker 抢 uploaded 任务。
# 退出 API 时一并停掉 worker，避免残留抢任务。
uv run almost-agent ingest-worker &
WORKER_PID=$!

cleanup() {
  if kill -0 "$WORKER_PID" 2>/dev/null; then
    kill "$WORKER_PID" 2>/dev/null || true
    wait "$WORKER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

uv run almost-agent dev "$@"
