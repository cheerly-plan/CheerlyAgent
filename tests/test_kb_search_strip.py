"""files 已指定时，检索 query 应确定性剥掉被点名文件的主体名/文件名。

对应问题：模型在 files 已圈定文件时仍把公司全称写进 query，占掉关键词 6 个名额、
并把向量嵌入拉向反复出现公司名的页眉页，导致「全称 + 营业收入」被页眉压过数字行。
"""

import uuid

from ai_agent import tools as tools_pkg
from ai_agent.tools.kb import kb_search
from ai_agent.storage import store as kb_store


def _fake_file(subject, aliases, filename):
    return {
        "subject": subject,
        "subject_aliases": aliases,
        "filename": filename,
    }


def _patch_file(monkeypatch, fake):
    # kb_search 模块内 `from ai_agent.storage.store import store`，直接替 store 的该方法
    monkeypatch.setattr(kb_store.store, "get_kb_file", lambda _id: fake)


def test_strip_removes_full_subject_name(monkeypatch):
    fid = uuid.uuid4()
    _patch_file(
        monkeypatch,
        _fake_file(
            "中化装备科技(青岛)股份有限公司",
            ["中化装备", "中化装备科技"],
            "中化装备科技(青岛)股份有限公司2026年第三季度报告.pdf",
        ),
    )
    out = kb_search._strip_scope_subject(
        "中化装备科技(青岛)股份有限公司 证券代码", [fid]
    )
    assert "中化装备" not in out
    assert out.strip() == "证券代码"


def test_strip_uses_alias_when_model_abbreviates(monkeypatch):
    # 模型只写了简称「中化装备」，用 subject 标准名硬删删不到，必须靠别名
    fid = uuid.uuid4()
    _patch_file(
        monkeypatch,
        _fake_file(
            "中化装备科技(青岛)股份有限公司",
            ["中化装备"],
            "中化装备科技(青岛)股份有限公司2026年第三季度报告.pdf",
        ),
    )
    out = kb_search._strip_scope_subject("中化装备 营业收入", [fid])
    assert "中化装备" not in out
    assert "营业收入" in out


def test_strip_removes_filename_too(monkeypatch):
    # 模型把文件名也带进了 query，应一并剥掉
    fid = uuid.uuid4()
    _patch_file(
        monkeypatch,
        _fake_file(
            "中化装备科技(青岛)股份有限公司",
            ["中化装备"],
            "中化装备科技(青岛)股份有限公司2026年第三季度报告.pdf",
        ),
    )
    out = kb_search._strip_scope_subject(
        "中化装备科技(青岛)股份有限公司2026年第三季度报告 营业收入", [fid]
    )
    assert "报告" not in out
    assert "营业收入" in out


def test_strip_falls_back_when_only_subject_remains(monkeypatch):
    # 问的就是全称本身，剥完几乎没剩 → 回退原句，别把真问题删没了
    fid = uuid.uuid4()
    _patch_file(
        monkeypatch,
        _fake_file(
            "中化装备科技(青岛)股份有限公司",
            ["中化装备"],
            "中化装备科技(青岛)股份有限公司2026年第三季度报告.pdf",
        ),
    )
    original = "中化装备科技(青岛)股份有限公司"
    assert kb_search._strip_scope_subject(original, [fid]) == original


def test_no_strip_when_file_unknown(monkeypatch):
    # 解析到的文件在库里查不到元数据，原句不动
    fid = uuid.uuid4()
    _patch_file(monkeypatch, None)
    q = "中化装备科技(青岛)股份有限公司 营业收入"
    assert kb_search._strip_scope_subject(q, [fid]) == q


def test_modules_importable():
    # 确保整链导入无副作用（prompts 改动后仍能加载）
    assert tools_pkg is not None
