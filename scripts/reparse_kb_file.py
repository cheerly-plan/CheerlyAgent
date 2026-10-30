"""已并入 scripts/kb.py，这里只转发。

    uv run python scripts/kb.py reparse --name 南岭
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kb import main  # noqa: E402

if __name__ == "__main__":
    print("[kb] reparse_kb_file.py → kb.py reparse", file=sys.stderr)
    raise SystemExit(main(["reparse", *sys.argv[1:]]))
