"""记忆抽取的领域中立性：换业务场景不能失效。

背景：早期禁止清单是按金融+知识库写的（股东、招股书、年报、股票代码），
换到医疗/法律/电商客服这些场景一条都对不上。现在规则只用结构特征：
本轮动作、任务对象主语句、文档事实、祈使任务、工具输出——不含任何行业词，
医疗/法律/教育/客服与金融走同一套规则。
"""

import pytest

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


# ——— 文档事实：通用规则，不再绑行业词 ———

@pytest.mark.parametrize(
    "text",
    [
        # 金融
        "招股书里提到的主要客户",
        "年报披露的应收账款",
        # 医疗
        "病历记载患者既往有高血压史",
        # 法律
        "判决书载明合同已于2025年解除",
        # 客服
        "工单显示客户已申请退款",
    ],
)
def test_document_facts_are_noise_across_domains(text):
    """主语是文档本身时，讲的是文件内容而非用户本人——任何领域同理。

    这条规则替代了原先硬编码的「招股」「年报披露」行业词。
    """
    assert is_transient_text(text)


def test_topic_object_lists_are_not_preferences():
    """「关注公司：A、B」是查询对象清单，不是用户偏好。"""
    from ai_agent.memory.long_term import ExtractedMemory, _keep_extracted

    for text in ("关注公司：绿的谐波、中大力德", "关注公司财务数据和业务状况"):
        item = ExtractedMemory(
            kind="profile", facet="preference", content=text, op="add"
        )
        assert _keep_extracted(item) is False, text


def test_tool_markers_are_domain_neutral():
    """工具输出标记在任何领域都应被识别为「查来的」，不是用户事实。"""
    for text in (
        "search_history 返回了一些旧对话",
        "read_tool_result 读到了原文",
        "kb_search 命中了三份文件",
    ):
        assert is_transient_text(text)
