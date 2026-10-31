"""知识库维护入口的命令拼装，不连库。"""

from scripts.kb import _parser
from scripts.rebuild_graph import _forward


def test_kb_help_lists_daily_commands():
    text = _parser().format_help()
    for cmd in ("list", "reparse", "graph", "reembed", "coverage"):
        assert cmd in text


def test_reparse_uses_env_embedding_by_default():
    args = _parser().parse_args(["reparse", "--name", "中行"])
    assert args.embedding_model == ""


def test_rebuild_graph_forwards_full_pipeline():
    assert _forward(["--name", "安克", "--yes"]) == [
        "reparse",
        "--with-graph",
        "--name",
        "安克",
        "--yes",
    ]


def test_rebuild_graph_forwards_graph_only():
    assert _forward(["--all", "--yes", "--graph-only"]) == [
        "graph",
        "--all",
        "--yes",
    ]
