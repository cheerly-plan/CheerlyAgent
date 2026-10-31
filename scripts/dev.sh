#!/usr/bin/env bash
# 开发模式启动后端：代码改动后 uvicorn 自动重启（热更新）。
# 用法：./scripts/dev.sh [--host HOST] [--port PORT]
set -euo pipefail

# 切到仓库根目录（脚本可能在任意 cwd 下被调用）
cd "$(dirname "$0")/.."

exec uv run almost-agent dev "$@"
