"""研报完成后的企业微信文本通知。

把「由 owner(user_id) 定位接收人 → 拼通知文本 → 发 text 消息」串成一个
不抛异常、绝不阻塞研报主流程的入口：任何一步失败（凭据未配、用户查不到、
企业微信报错）都只记日志并静默跳过，让通知成为「尽力而为」的旁路能力。

touser 约定取用户 username（当前账号体系下 username == 企业微信 userid），
不是 display_name（可能是中文显示名，企业微信 touser 不接受）。
"""

from __future__ import annotations

import threading

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.wecom import send_text_to_user

logger = get_logger(__name__)

# 通知里塞的下载链接是相对路径（/v1/report/files/...），企业微信端无法直接点。
# 这里只带一个「研报已完成，回系统查看下载链接」的提示，不伪造绝对 URL。
_NOTIFY_TEMPLATE = "研报已生成完成，请回系统查看并下载。"


def _build_text(title: str) -> str:
    """拼通知正文：有标题就带上，让用户一眼知道是哪份研报。"""
    if title:
        return f"研报「{title}」已生成完成，请回系统查看并下载。"
    return _NOTIFY_TEMPLATE


def _notify_worker(owner_user_id: str, title: str) -> None:
    """在独立线程里发通知：不阻塞任务线程，也不让通知失败拖累研报。"""
    try:
        from ai_agent.storage.store import store

        user = store.get_user_by_id(owner_user_id)
        if user is None:
            logger.warning("研报完成通知跳过：查不到用户 user_id=%s", owner_user_id)
            return
        touser = (user.username or "").strip()
        if not touser:
            logger.warning("研报完成通知跳过：用户无 username user_id=%s", owner_user_id)
            return

        text = _build_text(title)
        send_text_to_user(touser=touser, text=text)
        logger.info("研报完成通知已发 user=%s", touser)
    except Exception:
        # 通知是旁路能力，任何异常都吞掉，绝不向上抛影响研报主流程。
        logger.exception("研报完成通知发送失败 user_id=%s", owner_user_id)


def notify_report_done(owner_user_id: str, title: str = "") -> None:
    """研报跑完后异步发一条完成通知（fire-and-forget）。

    ``owner_user_id`` 是服务端 UUID（``current_user_id()``），``title`` 是
    研报标题（可空）。开关 ``wecom_report_notify_enabled`` 关闭时直接返回。
    """
    if not settings.wecom_report_notify_enabled:
        return
    threading.Thread(
        target=_notify_worker,
        args=(owner_user_id, title),
        name=f"wecom-notify-{owner_user_id[:8]}",
        daemon=True,
    ).start()
