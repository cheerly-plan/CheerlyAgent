"""文档配置（通用底座 + 可插拔配置块）的单测。

核心要守住两件事：
1. 换文档类型只换配置块，底座与输出契约一字不变——否则「一套抽取器跑全部文档」
   这个前提就没了；
2. 配置有问题时宁可回落到通用模式，也不能让建图整个挂掉（抽取是长任务，
   一次运行几十分钟，配置里一个笔误不该把它全部作废）。
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from ai_agent.kb import doc_profiles
from ai_agent.kb.doc_profiles import (
    _BASE_SYSTEM,
    DEFAULT_LEVEL,
    LEVELS,
    build_system_prompt,
    list_profiles,
    normalize_level,
    render_config,
    resolve_profile,
)


def test_builtin_profiles_have_required_fields():
    profiles = list_profiles()
    keys = [p.key for p in profiles]
    # 常用资本市场文档都得覆盖到，缺一个就意味着换文档时只能走通用模式
    assert {"prospectus", "annual_report", "legal_opinion", "general"} <= set(keys)
    assert len(keys) == len(set(keys)), "key 不能重复"
    for profile in profiles:
        assert profile.label and profile.doc_name
        assert profile.key == profile.key.strip().lower()


def test_general_profile_is_last_and_fallback():
    profiles = list_profiles()
    assert profiles[-1].key == "general"
    # 未知类型、空类型都回落到通用，而不是报错
    assert resolve_profile("no_such_type").key == "general"
    assert resolve_profile("").key == "general"
    assert resolve_profile(None).key == "general"


def test_suffix_prompt_keeps_base_contract_intact():
    """换配置块不改底座：输出契约、片段序号规则必须原样保留。"""
    for profile in list_profiles():
        prompt = build_system_prompt(profile.key, "full")
        assert doc_profiles.CONFIG_SLOT not in prompt, "占位符必须被替换掉"
        # 底座里的关键约束一条都不能少（少了模型就会改输出形状）
        assert "只输出一行 JSON" in prompt
        assert "片段序号从 0 开始" in prompt
        assert "person / org / product / technology / metric / event / location / concept / other" in prompt
        assert f"文档类型：{profile.doc_name}" in prompt


def test_base_prompt_has_no_document_specific_words():
    """底座不许出现行业专有词，否则「通用」就是空话。"""
    for word in ("招股", "发行人", "募投", "年报", "法律意见书"):
        assert word not in _BASE_SYSTEM


def test_scope_and_predicates_are_injected():
    prompt = build_system_prompt("prospectus", "full")
    assert "募投项目" in prompt
    assert "拥有资质" in prompt
    # 通用配置不该带别家的白名单，否则陌生文档会被带偏
    general = build_system_prompt("general", "full")
    assert "募投项目" not in general
    assert "不限定白名单" in general


def test_light_level_adds_constraint_and_full_does_not():
    light = build_system_prompt("prospectus", "light")
    full = build_system_prompt("prospectus", "full")
    assert "抽取粒度：轻量" in light
    assert "抽取粒度：轻量" not in full
    assert light != full
    # 轻量只是加一段约束，不是换一整套 prompt：实体范围照旧在
    assert "重点实体范围" in light


def test_normalize_level_falls_back_to_full():
    assert normalize_level("light") == "light"
    assert normalize_level("FULL") == "full"
    assert normalize_level("") == DEFAULT_LEVEL
    assert normalize_level("超轻量") == DEFAULT_LEVEL
    assert set(LEVELS) == {"full", "light"}


def test_aliases_rendered_as_mapping():
    config = render_config(
        doc_profiles.DocProfile(
            key="x",
            label="X",
            doc_name="测试文档",
            aliases=(("本公司", "某某股份有限公司"),),
        )
    )
    assert "本公司 → 某某股份有限公司" in config


def _write_profile_file(content: str) -> tuple[Path, Path]:
    """临时目录里放一份配置。不用 pytest 的 tmp_path：本机沙箱下它偶发
    mkdir 竞争（同一个 pytest-of-* 目录被并发创建），会让用例环境性失败。"""
    root = Path(tempfile.mkdtemp(prefix="kb-doc-profiles-"))
    path = root / "profiles.json"
    path.write_text(content, encoding="utf-8")
    return root, path


def test_json_overrides_add_and_replace(monkeypatch):
    """JSON 能覆盖内置、也能追加新类型，且不用改代码。"""
    root, path = _write_profile_file(
        json.dumps(
            {
                "profiles": [
                    {"key": "bond", "label": "债券募集说明书", "doc_name": "债券募集说明书",
                     "entity_scope": ["发行人", "担保人"], "predicates": ["发行", "担保"]},
                    {"key": "general", "label": "通用", "doc_name": "通用商业文档"},
                ]
            },
            ensure_ascii=False,
        )
    )
    monkeypatch.setattr(doc_profiles.settings, "kb_doc_profiles_path", path)
    doc_profiles.reload_profiles()
    try:
        keys = [p.key for p in list_profiles()]
        assert "bond" in keys
        assert keys[-1] == "general"
        assert "担保人" in build_system_prompt("bond", "full")
    finally:
        doc_profiles.reload_profiles()
        shutil.rmtree(root, ignore_errors=True)


def test_broken_json_falls_back_to_builtin(monkeypatch):
    """配置写坏了只回退，不能让建图抛异常。"""
    root, path = _write_profile_file("{ not json")
    monkeypatch.setattr(doc_profiles.settings, "kb_doc_profiles_path", path)
    doc_profiles.reload_profiles()
    try:
        assert resolve_profile("prospectus").key == "prospectus"
    finally:
        doc_profiles.reload_profiles()
        shutil.rmtree(root, ignore_errors=True)


def test_profile_without_key_is_skipped(monkeypatch):
    root, path = _write_profile_file(
        json.dumps({"profiles": [{"label": "没有 key"}]}, ensure_ascii=False)
    )
    monkeypatch.setattr(doc_profiles.settings, "kb_doc_profiles_path", path)
    doc_profiles.reload_profiles()
    try:
        assert {p.key for p in list_profiles()} >= {"general", "prospectus"}
    finally:
        doc_profiles.reload_profiles()
        shutil.rmtree(root, ignore_errors=True)
