"""记忆抽取的领域中立性：换业务场景不能失效。

背景：早期禁止清单是按金融+知识库写的（股东、招股书、年报、股票代码），
换到医疗/法律/电商客服这些场景一条都对不上。现在规则分成两层：

- 通用模式：本轮动作、任务状态、短期安排、工具输出、对话纪要
- 行业词：股票代码/行情/招股披露（MEMORY_DOMAIN_NOISE_ENABLED 可关）
"""

import pytest

from ai_agent.core.config import settings
from ai_agent.memory.long_term import (
    ExtractedMemory,
    _keep_extracted,
    is_transient_text,
)


def _episode(text: str) -> ExtractedMemory:
    return ExtractedMemory(kind="episode", content=text)


# ——— 跨领域的「任务对象」，都不该记 ———

@pytest.mark.parametrize(
    "text",
    [
        # 医疗
        "用户正在查询该患者的用药记录，已确认存在青霉素过敏史。",
        "患者在2026年10月入院，诊断为急性阑尾炎，已提交出院小结。",
        # 法律
        "用户正在对比两份合同中的违约金条款，已确认甲方为某某科技公司。",
        "该案已于11月10日开庭，判决书已归档。",
        # 电商客服
        "订单 A20261115 已发货，预计3-5分钟更新物流状态。",
        "用户正在确认该商品的退换货政策。",
        # 通用任务状态
        "任务已提交到后台，job_id=abc123，预计5-8分钟完成。",
    ],
)
def test_task_objects_are_rejected_in_any_domain(text):
    assert not _keep_extracted(_episode(text))


# ——— 跨领域的「用户本人长期事实」，都该记 ———

@pytest.mark.parametrize(
    "text",
    [
        "我在做一家社区诊所的慢病管理系统，已经上线两个月。",
        "我在准备法律职业资格考试，每天复习三小时。",
        "我在维护一套电商客服的话术库，覆盖退换货场景。",
        "我在用 Koa 做 Node.js BFF 数据聚合。",
    ],
)
def test_user_own_projects_are_kept_in_any_domain(text):
    assert _keep_extracted(_episode(text))


# ——— 行业词开关 ———

def test_generic_rules_work_even_with_domain_noise_off(monkeypatch):
    """关掉行业词后，通用规则（本轮动作）依然要拦住。"""
    monkeypatch.setattr(settings, "memory_domain_noise_enabled", False)
    assert is_transient_text("用户正在确认证券代码为688361")
    assert is_transient_text("患者在2026年10月入院，诊断为急性阑尾炎")


def test_domain_noise_off_lets_pure_domain_text_through(monkeypatch):
    """纯行业特征（无通用噪声信号）关掉行业词后应当放行。"""
    pure_domain = "融资融券余额 688361"
    assert is_transient_text(pure_domain)          # 默认开着 → 拦
    monkeypatch.setattr(settings, "memory_domain_noise_enabled", False)
    assert not is_transient_text(pure_domain)      # 关掉 → 放行


def test_tool_markers_are_domain_neutral():
    """工具输出标记在任何领域都应被识别为「查来的」，不是用户事实。"""
    for text in (
        "search_history 返回了一些旧对话",
        "read_tool_result 读到了原文",
        "kb_search 命中了三份文件",
    ):
        assert is_transient_text(text)
