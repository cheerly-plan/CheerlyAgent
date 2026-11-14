#!/usr/bin/env bash
# 本地评测闭环：连本机 .env 的模型网关与 pgvector，不碰 CI。
#
# 用法：
#   ./scripts/eval-local.sh                # T0 静态门禁（秒级、零费用、零依赖）
#   ./scripts/eval-local.sh health         # 只读探测：DB + 模型网关连通性
#   ./scripts/eval-local.sh t1             # T1 连库评测（检索/图谱/记忆健康度，不调模型）
#   ./scripts/eval-local.sh t2             # T2 调模型评测（工具选择 + 记忆抽取，会花钱）
#   ./scripts/eval-local.sh all            # T0 + T1 + T2 全量（会连库、调模型）
#
# 说明：
#   - T2 会真调模型（eval_tool_route 走主模型、eval_memory 走小模型），几块钱以内；
#     带 --allow-external 才连库/调模型，防止误触。
#   - 结果报告写 data/eval/runs/，T2 趋势追加 data/eval/trend.jsonl（本地、gitignored）。
set -euo pipefail

cd "$(dirname "$0")/.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache-wb}"

MODE="${1:-t0}"

case "$MODE" in
  t0)
    echo "== T0 静态门禁 =="
    uv run python scripts/eval_run.py
    ;;
  health)
    echo "== 只读连通性探测 =="
    uv run python - <<'PY'
from ai_agent.core.config import settings
import psycopg
print("数据库:", settings.database_url.split("@")[-1])
try:
    with psycopg.connect(settings.database_url, connect_timeout=5) as conn:
        v = conn.execute("SHOW server_version").fetchone()[0]
        ext = conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchall()
        n = conn.execute("SELECT count(*) FROM pg_tables WHERE schemaname='public'").fetchone()[0]
    print(f"  连通 OK · PostgreSQL {v} · vector {ext[0][0] if ext else '缺失'} · {n} 张表")
except Exception as e:
    print(f"  连接失败: {type(e).__name__}: {str(e)[:160]}")

from ai_agent.llm.client import _client, small_client
for label, model, c in [
    ("主模型", settings.model, _client),
    ("小模型", settings.small_model or settings.model, small_client()),
]:
    try:
        c.chat.completions.create(model=model, messages=[{"role": "user", "content": "ping"}], max_tokens=1)
        print(f"  {label} {model} · 连通 OK")
    except Exception as e:
        print(f"  {label} {model} · 失败: {type(e).__name__}: {str(e)[:160]}")
PY
    ;;
  t1)
    echo "== T1 连库评测 =="
    uv run python scripts/eval_run.py --tiers 1 --allow-external
    ;;
  t2)
    echo "== T2 调模型评测 =="
    uv run python scripts/eval_run.py --tiers 2 --allow-external
    ;;
  all)
    echo "== T0 + T1 + T2 全量 =="
    uv run python scripts/eval_run.py --tiers 0,1,2 --allow-external
    ;;
  *)
    echo "未知模式: $MODE（可用 t0 / health / t1 / t2 / all）" >&2
    exit 1
    ;;
esac
