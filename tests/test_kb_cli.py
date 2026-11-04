"""知识库维护入口的命令拼装，不连库。"""

from scripts.kb import _parser


def test_kb_help_lists_daily_commands():
    text = _parser().format_help()
    for cmd in ("list", "reparse", "graph", "reembed", "coverage", "lexicon"):
        assert cmd in text


def test_reparse_uses_env_embedding_by_default():
    args = _parser().parse_args(["reparse", "--name", "中行"])
    assert args.embedding_model == ""
