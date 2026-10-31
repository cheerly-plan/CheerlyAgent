"""已并入 scripts/kb.py，这里只转发。

    uv run python scripts/kb.py reembed
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kb import main  # noqa: E402

if __name__ == "__main__":
    print("[kb] reembed_chunks.py → kb.py reembed", file=sys.stderr)
    raise SystemExit(main(["reembed", *sys.argv[1:]]))
