"""已并入 scripts/kb.py，这里只转发。

    uv run python scripts/kb.py list
    uv run python scripts/kb.py coverage --name 招股
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kb import main  # noqa: E402

if __name__ == "__main__":
    argv = ["coverage"]
    for arg in sys.argv[1:]:
        argv.append("--full" if arg in ("--all",) else arg)
    print("[kb] kb_page_coverage.py → kb.py coverage", file=sys.stderr)
    raise SystemExit(main(argv))
