"""已并入 scripts/kb.py，这里只转发。

    uv run python scripts/kb.py reembed --target all
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kb import main  # noqa: E402

if __name__ == "__main__":
    print("[kb] reembed_vectors.py → kb.py reembed --target all", file=sys.stderr)
    raise SystemExit(main(["reembed", "--target", "all", *sys.argv[1:]]))
