"""查询改写（多轮对话改写 A + 检索增强 multi-query B）的确定性行为。

这两块都依赖 LLM，但正确性不靠 LLM——靠的是「降级链」：开关关 / 无上文 /
失败 / 超时 / 结果退化时一律退回原 query，绝不挡检索主流程。本文件用
monkeypatch 把 complete_chat 换掉，只验逻辑分支，不烧真实调用。

背景（git 时间线）：改写曾绑定在「预检索 prefetch」上（e8997ec/1684c05），
prefetch 被移除（185fa6e/bb47db6）后连带丢失，effective_kb_query 退化成
原样返回。本次重建挂回 effective_kb_query，覆盖 kb_search / kg_search 两个
工具入口。
"""

from ai_agent.core.config import settings
from ai_agent.kb import scope
from ai_agent.kb.scope import (
    decompose_queries,
    effective_kb_query,
    expand_queries,
    extract_prev_turn,
)


# ——— A 多轮对话改写 ———


def test_effective_kb_query_returns_original_without_context(monkeypatch):
    """contextvar 未设置（无上一轮）时，不调 LLM，直接原样返回。"""
    monkeypatch.setattr(settings, "kb_rewrite_enabled", True)
    called = {"n": 0}

    def fake_chat(messages, **kwargs):
        called["n"] += 1
        return "不该被调用"

    monkeypatch.setattr(scope, "complete_chat", fake_chat)
    # 确保 contextvar 干净
    from ai_agent.kb.scope import rewrite_context

    assert rewrite_context() == []
    out = effective_kb_query("它去年营收是多少")
    assert out == "它去年营收是多少"
    assert called["n"] == 0


def test_effective_kb_query_uses_rewrite_context(monkeypatch):
    """设置了上一轮对话后，改写真正生效并把补全结果返回。"""
    monkeypatch.setattr(settings, "kb_rewrite_enabled", True)
    from ai_agent.kb.scope import reset_rewrite_context, set_rewrite_context

    def fake_chat(messages, **kwargs):
        return "中国能源建设股份有限公司2025年营业收入"

    monkeypatch.setattr(scope, "complete_chat", fake_chat)
    token = set_rewrite_context(["中国能源建设股份有限公司前三季度营收", "前三季度营收 3092 亿"])
    try:
        out = effective_kb_query("那它去年呢")
    finally:
        reset_rewrite_context(token)
    assert out == "中国能源建设股份有限公司2025年营业收入"


def test_effective_kb_query_disabled_returns_original(monkeypatch):
    """开关关闭时直接退回原话，不碰 LLM。"""
    monkeypatch.setattr(settings, "kb_rewrite_enabled", False)
    from ai_agent.kb.scope import reset_rewrite_context, set_rewrite_context

    called = {"n": 0}

    def fake_chat(messages, **kwargs):
        called["n"] += 1
        return "x"

    monkeypatch.setattr(scope, "complete_chat", fake_chat)
    token = set_rewrite_context(["上一轮说了点啥"])
    try:
        out = effective_kb_query("追问")
    finally:
        reset_rewrite_context(token)
    assert out == "追问"
    assert called["n"] == 0


def test_effective_kb_query_falls_back_on_llm_error(monkeypatch):
    """LLM 抛错（超时/限流）时退回原话，不向上抛。"""
    monkeypatch.setattr(settings, "kb_rewrite_enabled", True)
    from ai_agent.kb.scope import reset_rewrite_context, set_rewrite_context
    from ai_agent.llm.client import LLMError

    def boom(messages, **kwargs):
        raise LLMError("timeout")

    monkeypatch.setattr(scope, "complete_chat", boom)
    token = set_rewrite_context(["上一轮"])
    try:
        out = effective_kb_query("它怎么样")
    finally:
        reset_rewrite_context(token)
    assert out == "它怎么样"


def test_effective_kb_query_falls_back_when_unchanged(monkeypatch):
    """模型返回空或等于原句时，退回原话（避免空查询或无效改写）。"""
    monkeypatch.setattr(settings, "kb_rewrite_enabled", True)
    from ai_agent.kb.scope import reset_rewrite_context, set_rewrite_context

    monkeypatch.setattr(scope, "complete_chat", lambda messages, **kwargs: "   ")
    token = set_rewrite_context(["上一轮"])
    try:
        assert effective_kb_query("它怎么样") == "它怎么样"
    finally:
        reset_rewrite_context(token)

    monkeypatch.setattr(scope, "complete_chat", lambda messages, **kwargs: "它怎么样")
    token = set_rewrite_context(["上一轮"])
    try:
        assert effective_kb_query("它怎么样") == "它怎么样"
    finally:
        reset_rewrite_context(token)


def test_effective_kb_query_strips_quotes(monkeypatch):
    """模型输出带引号包裹时，剥掉引号。"""
    monkeypatch.setattr(settings, "kb_rewrite_enabled", True)
    from ai_agent.kb.scope import reset_rewrite_context, set_rewrite_context

    monkeypatch.setattr(scope, "complete_chat", lambda messages, **kwargs: '"中国银行营收"')
    token = set_rewrite_context(["中国银行2026三季报"])
    try:
        out = effective_kb_query("它的营收")
    finally:
        reset_rewrite_context(token)
    assert out == "中国银行营收"


# ——— extract_prev_turn ———


def test_extract_prev_turn_skips_tool_call_intermediate():
    """跳过带 tool_calls 的 assistant 中间态，只取最终回答正文。"""
    messages = [
        {"role": "user", "content": "中国能源建设营收多少"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "name": "kb_search", "arguments": "{}"}],
        },
        {"role": "tool", "content": "检索结果：3092 亿"},
        {"role": "assistant", "content": "前三季度营收 3092 亿元。"},
        {"role": "user", "content": "那它去年呢"},  # 当前句
    ]
    turns = extract_prev_turn(messages, max_turns=1)
    assert turns == ["中国能源建设营收多少", "前三季度营收 3092 亿元。"]


def test_extract_prev_turn_collects_multiple_turns():
    """多轮收集：主体在最前一轮，中间隔无主体追问，也能补到。"""
    messages = [
        {"role": "user", "content": "顺丰控股营收多少"},
        {"role": "assistant", "content": "2377 亿"},
        {"role": "user", "content": "同比呢"},
        {"role": "assistant", "content": "增长 5.53%"},
        {"role": "user", "content": "那净利润呢"},  # 当前句
    ]
    turns = extract_prev_turn(messages, max_turns=2)
    assert turns == ["顺丰控股营收多少", "2377 亿", "同比呢", "增长 5.53%"]


def test_extract_prev_turn_no_prior_user_returns_empty():
    """只有一句 user（当前句）没有上文时，返回空。"""
    messages = [{"role": "user", "content": "就这一句"}]
    assert extract_prev_turn(messages) == []


def test_extract_prev_turn_ignores_empty_content():
    """空 user 内容不作为当前句或上文。"""
    messages = [
        {"role": "user", "content": ""},
        {"role": "user", "content": "中国银行营收"},
        {"role": "assistant", "content": "526365 百万"},
        {"role": "user", "content": "那净利润"},  # 当前句
    ]
    turns = extract_prev_turn(messages, max_turns=1)
    assert turns == ["中国银行营收", "526365 百万"]


# ——— B multi-query 扩展 ———


def test_expand_queries_disabled_returns_single(monkeypatch):
    monkeypatch.setattr(settings, "kb_multiquery_enabled", False)
    called = {"n": 0}

    def fake_chat(messages, **kwargs):
        called["n"] += 1
        return ""

    monkeypatch.setattr(scope, "complete_chat", fake_chat)
    assert expand_queries("中国银行营收") == ["中国银行营收"]
    assert called["n"] == 0


def test_expand_queries_returns_variants_with_original(monkeypatch):
    monkeypatch.setattr(settings, "kb_multiquery_enabled", True)
    monkeypatch.setattr(settings, "kb_multiquery_count", 3)
    monkeypatch.setattr(
        scope,
        "complete_chat",
        lambda messages, **kwargs: "中国银行营业收入\n中国银行营收额\n中国银行营收",
    )
    out = expand_queries("中国银行营收")
    # 原 query 一定在首位，且变体去重后不超过 count
    assert out[0] == "中国银行营收"
    assert "中国银行营业收入" in out
    assert "中国银行营收额" in out
    assert len(out) <= 3


def test_expand_queries_dedupes_and_caps_count(monkeypatch):
    monkeypatch.setattr(settings, "kb_multiquery_enabled", True)
    monkeypatch.setattr(settings, "kb_multiquery_count", 2)
    monkeypatch.setattr(
        scope,
        "complete_chat",
        lambda messages, **kwargs: "变体一\n变体一\n变体二\n变体三",
    )
    out = expand_queries("原始查询")
    assert out == ["原始查询", "变体一"]  # 去重 + 截到 count=2


def test_expand_queries_falls_back_on_failure(monkeypatch):
    monkeypatch.setattr(settings, "kb_multiquery_enabled", True)

    def boom(messages, **kwargs):
        raise RuntimeError("llm down")

    monkeypatch.setattr(scope, "complete_chat", boom)
    assert expand_queries("中国银行营收") == ["中国银行营收"]


def test_expand_queries_empty_query_returns_empty():
    assert expand_queries("") == []
    assert expand_queries("   ") == []


# ——— search_knowledge_multi 融合 ———


def test_search_knowledge_multi_single_query_delegates(monkeypatch):
    """单条 query 退化为 search_knowledge，行为一致。"""
    from ai_agent.kb import retrieval

    def fake_single(query, *, file_ids=None, top_k=3, filters=None):
        assert query == "唯一查询"
        return ["hit"]

    monkeypatch.setattr(retrieval, "search_knowledge", fake_single)
    assert retrieval.search_knowledge_multi(["唯一查询"], top_k=3) == ["hit"]


def test_search_knowledge_multi_skips_failed_embedding(monkeypatch):
    """某个变体向量化失败只跳过该变体，不影响其他。"""
    from ai_agent.kb import retrieval

    def fake_embed(q):
        if q == "坏变体":
            raise RuntimeError("embed fail")
        return [0.1]

    calls = []

    def fake_candidates(q, embedding, *, file_ids=None, top_k=3, filters=None):
        calls.append(q)
        return []

    monkeypatch.setattr(retrieval, "embed_query", fake_embed)
    monkeypatch.setattr(retrieval, "_retrieve_candidates", fake_candidates)
    monkeypatch.setattr(settings, "kb_rerank_enabled", False)

    retrieval.search_knowledge_multi(["好变体", "坏变体"], top_k=3)
    assert calls == ["好变体"]


# ——— C 子问题分解（sub-question）———


def test_decompose_queries_disabled_returns_single(monkeypatch):
    """开关关闭时直接退回原话，不碰 LLM。"""
    monkeypatch.setattr(settings, "kb_decompose_enabled", False)
    called = {"n": 0}

    def fake_chat(messages, **kwargs):
        called["n"] += 1
        return ""

    monkeypatch.setattr(scope, "complete_chat", fake_chat)
    assert decompose_queries("中国银行和长安汽车营收") == ["中国银行和长安汽车营收"]
    assert called["n"] == 0


def test_decompose_queries_no_signal_returns_single(monkeypatch):
    """无并列/对比信号词时不拆、不碰 LLM（如单主体单指标）。"""
    monkeypatch.setattr(settings, "kb_decompose_enabled", True)
    called = {"n": 0}

    def fake_chat(messages, **kwargs):
        called["n"] += 1
        return ""

    monkeypatch.setattr(scope, "complete_chat", fake_chat)
    assert decompose_queries("中国银行前三季度营业收入") == ["中国银行前三季度营业收入"]
    assert called["n"] == 0


def test_decompose_queries_single_subject_returns_single(monkeypatch):
    """有信号词但只点名一个主体（「营收和净利润」）→ 不拆、不碰 LLM。"""
    from ai_agent.kb import retrieval

    monkeypatch.setattr(settings, "kb_decompose_enabled", True)
    called = {"n": 0}

    def fake_chat(messages, **kwargs):
        called["n"] += 1
        return ""

    monkeypatch.setattr(scope, "complete_chat", fake_chat)
    monkeypatch.setattr(
        retrieval, "mentioned_subjects", lambda q, file_ids=None: {"中国银行"}
    )
    assert decompose_queries("中国银行的营收和净利润") == ["中国银行的营收和净利润"]
    assert called["n"] == 0


def test_decompose_queries_splits_multi_subject(monkeypatch):
    """多主体对比问句拆成单主体子问题，逐条返回。"""
    from ai_agent.kb import retrieval

    monkeypatch.setattr(settings, "kb_decompose_enabled", True)
    monkeypatch.setattr(settings, "kb_decompose_max", 4)
    monkeypatch.setattr(
        retrieval, "mentioned_subjects",
        lambda q, file_ids=None: {"中国银行", "长安汽车"},
    )
    monkeypatch.setattr(
        scope,
        "complete_chat",
        lambda m, **k: "中国银行前三季度营业收入\n长安汽车前三季度营业收入",
    )
    out = decompose_queries("中国银行和长安汽车的前三季度营业收入分别是多少")
    assert out == ["中国银行前三季度营业收入", "长安汽车前三季度营业收入"]


def test_decompose_queries_under_split_returns_single(monkeypatch):
    """模型只返回一条（没拆开）时退回原话，避免单条子问题顶替整问。"""
    from ai_agent.kb import retrieval

    monkeypatch.setattr(settings, "kb_decompose_enabled", True)
    monkeypatch.setattr(
        retrieval, "mentioned_subjects",
        lambda q, file_ids=None: {"中国银行", "长安汽车"},
    )
    monkeypatch.setattr(
        scope, "complete_chat", lambda m, **k: "中国银行和长安汽车营业收入"
    )
    assert decompose_queries("中国银行和长安汽车营业收入") == [
        "中国银行和长安汽车营业收入"
    ]


def test_decompose_queries_falls_back_on_failure(monkeypatch):
    """LLM 抛错时退回原话，不向上抛。"""
    from ai_agent.kb import retrieval

    monkeypatch.setattr(settings, "kb_decompose_enabled", True)
    monkeypatch.setattr(
        retrieval, "mentioned_subjects",
        lambda q, file_ids=None: {"中国银行", "长安汽车"},
    )

    def boom(messages, **kwargs):
        raise RuntimeError("llm down")

    monkeypatch.setattr(scope, "complete_chat", boom)
    assert decompose_queries("中国银行和长安汽车营收") == ["中国银行和长安汽车营收"]


def test_decompose_queries_caps_and_dedupes(monkeypatch):
    """子问题去重 + 截到 kb_decompose_max 上限。"""
    from ai_agent.kb import retrieval

    monkeypatch.setattr(settings, "kb_decompose_enabled", True)
    monkeypatch.setattr(settings, "kb_decompose_max", 2)
    monkeypatch.setattr(
        retrieval, "mentioned_subjects", lambda q, file_ids=None: {"A", "B", "C"}
    )
    monkeypatch.setattr(
        scope, "complete_chat",
        lambda m, **k: "A的营收\nB的营收\nA的营收\nC的营收",
    )
    assert decompose_queries("A和B和C的营收") == ["A的营收", "B的营收"]


def test_decompose_queries_empty_returns_empty():
    assert decompose_queries("") == []
    assert decompose_queries("   ") == []


# ——— mentioned_subjects ———


def _ready_file(file_id, filename, subject, aliases=None):
    return {
        "file_id": file_id,
        "filename": filename,
        "status": "ready",
        "subject": subject,
        "subject_aliases": aliases or [],
    }


def test_mentioned_subjects_detects_multiple(monkeypatch):
    """对比问句点名两个主体，返回两个主体核心名。"""
    import uuid

    from ai_agent.kb import retrieval

    a, b = uuid.uuid4(), uuid.uuid4()
    monkeypatch.setattr(
        retrieval.store,
        "list_kb_files",
        lambda **k: [
            _ready_file(a, "中国银行2026三季报.pdf", "中国银行股份有限公司", ["中国银行"]),
            _ready_file(b, "长安汽车2026三季报.pdf", "长安汽车股份有限公司", []),
        ],
    )
    subjects = retrieval.mentioned_subjects("中国银行和长安汽车的营收")
    assert subjects == {"中国银行", "长安汽车"}


def test_mentioned_subjects_single(monkeypatch):
    """单主体问句只返回一个主体。"""
    import uuid

    from ai_agent.kb import retrieval

    monkeypatch.setattr(
        retrieval.store,
        "list_kb_files",
        lambda **k: [
            _ready_file(uuid.uuid4(), "中国银行2026三季报.pdf", "中国银行股份有限公司", ["中国银行"]),
        ],
    )
    assert retrieval.mentioned_subjects("中国银行的营收") == {"中国银行"}


def test_mentioned_subjects_respects_file_ids(monkeypatch):
    """file_ids 收窄后只在这些文件里数主体。"""
    import uuid

    from ai_agent.kb import retrieval

    a, b = uuid.uuid4(), uuid.uuid4()
    monkeypatch.setattr(
        retrieval.store,
        "list_kb_files",
        lambda **k: [
            _ready_file(a, "中国银行2026三季报.pdf", "中国银行股份有限公司", ["中国银行"]),
            _ready_file(b, "长安汽车2026三季报.pdf", "长安汽车股份有限公司", []),
        ],
    )
    subjects = retrieval.mentioned_subjects(
        "中国银行和长安汽车的营收", file_ids=[a]
    )
    assert subjects == {"中国银行"}


# ——— search_knowledge_subqueries 轮转合并 ———


def _hit(fid, seq):
    from ai_agent.kb import retrieval

    return retrieval.Hit(
        file_id=fid, filename="f", seq=seq, page=1, kind="text",
        text="t", distance=0.1,
    )


def test_search_knowledge_subqueries_single_delegates(monkeypatch):
    """单条子问题退化为 search_knowledge。"""
    from ai_agent.kb import retrieval

    def fake_single(query, *, file_ids=None, top_k=3, filters=None):
        assert query == "唯一子问题"
        return ["hit"]

    monkeypatch.setattr(retrieval, "search_knowledge", fake_single)
    assert retrieval.search_knowledge_subqueries(["唯一子问题"], top_k=3) == ["hit"]


def test_search_knowledge_subqueries_round_robin_merge(monkeypatch):
    """轮转交织：A0, B0, A1 → 每主体至少一条，共 top_k 条。"""
    from ai_agent.kb import retrieval

    results = {
        "A营收": [_hit("a", 1), _hit("a", 2), _hit("a", 3)],
        "B营收": [_hit("b", 1), _hit("b", 2)],
    }
    monkeypatch.setattr(
        retrieval,
        "search_knowledge",
        lambda query, *, file_ids=None, top_k=3, filters=None: results[query],
    )
    hits = retrieval.search_knowledge_subqueries(["A营收", "B营收"], top_k=3)
    assert [(h.file_id, h.seq) for h in hits] == [("a", 1), ("b", 1), ("a", 2)]


def test_search_knowledge_subqueries_dedupes(monkeypatch):
    """两个子问题命中同一片段时去重，不重复占位。"""
    from ai_agent.kb import retrieval

    results = {
        "A营收": [_hit("a", 1)],
        "B营收": [_hit("a", 1), _hit("b", 1)],
    }
    monkeypatch.setattr(
        retrieval,
        "search_knowledge",
        lambda query, *, file_ids=None, top_k=3, filters=None: results[query],
    )
    hits = retrieval.search_knowledge_subqueries(["A营收", "B营收"], top_k=3)
    assert [(h.file_id, h.seq) for h in hits] == [("a", 1), ("b", 1)]


def test_search_knowledge_subqueries_skips_failed(monkeypatch):
    """某条子问题检索失败只跳过，不影响其余。"""
    from ai_agent.kb import retrieval

    def fake_single(query, *, file_ids=None, top_k=3, filters=None):
        if query == "坏":
            raise RuntimeError("boom")
        return [_hit("a", 1)]

    monkeypatch.setattr(retrieval, "search_knowledge", fake_single)
    hits = retrieval.search_knowledge_subqueries(["坏", "好"], top_k=3)
    assert [(h.file_id, h.seq) for h in hits] == [("a", 1)]


def test_search_knowledge_subqueries_empty():
    from ai_agent.kb import retrieval

    assert retrieval.search_knowledge_subqueries([]) == []
    assert retrieval.search_knowledge_subqueries(["  "]) == []
