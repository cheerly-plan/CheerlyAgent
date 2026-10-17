import argparse
from pathlib import Path

import uvicorn

from ai_agent.agents.loop import run_stream
from ai_agent.core.config import settings
from ai_agent.core.logging import setup_logging


def _print_stream(query: str) -> None:
    for event in run_stream(query):
        if event["type"] == "text":
            print(event.get("content") or "", end="", flush=True)
        elif event["type"] == "tool_start":
            print(f"\n[工具] {event.get('name')}", flush=True)
        elif event["type"] == "error":
            print(event.get("message") or "ERROR")
            return
        elif event["type"] == "done":
            print()


def _chat(message: str | None) -> None:
    setup_logging()
    if message:
        _print_stream(message)
        return

    print("直接提问。输入 exit 退出。")
    while True:
        try:
            query = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            return
        _print_stream(query)


def _serve(host: str, port: int, reload: bool) -> None:
    uvicorn.run(
        "ai_agent.api.app:app",
        host=host,
        port=port,
        reload=reload,
        reload_dirs=[str(Path(__file__).resolve().parents[1])],
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="almost-agent")
    sub = parser.add_subparsers(dest="command", required=True)

    chat_parser = sub.add_parser("chat", help="本地对话")
    chat_parser.add_argument("message", nargs="*", help="一次性提问；不传则进入交互")

    serve_parser = sub.add_parser("serve", help="启动 HTTP 服务")
    serve_parser.add_argument("--host", default=settings.host)
    serve_parser.add_argument("--port", type=int, default=settings.port)
    serve_parser.add_argument(
        "--reload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="代码变更后自动重启，生产环境请加 --no-reload",
    )

    args = parser.parse_args()
    if args.command == "chat":
        _chat(" ".join(args.message).strip() or None)
        return
    _serve(args.host, args.port, args.reload)


if __name__ == "__main__":
    main()
