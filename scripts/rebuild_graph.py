"""已并入 scripts/kb.py，这里只转发。

    uv run python scripts/kb.py reparse --name 安克 --with-graph
    uv run python scripts/kb.py graph --all --yes
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kb import main  # noqa: E402


def _forward(argv: list[str]) -> list[str]:
    rest = [a for a in argv if a != "--graph-only"]
    if "--graph-only" in argv:
        print("[kb] rebuild_graph.py → kb.py graph", file=sys.stderr)
        return ["graph", *rest]
    print("[kb] rebuild_graph.py → kb.py reparse --with-graph", file=sys.stderr)
    return ["reparse", "--with-graph", *rest]


if __name__ == "__main__":
    raise SystemExit(main(_forward(sys.argv[1:])))
