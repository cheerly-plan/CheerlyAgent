"""企业微信自建应用推送（主动推消息 / 文件给指定成员）。

能力边界：只做「主动推送」——文本消息直接发，文件走「上传临时素材换
media_id 再发」两跳。不做消息回调、不做通讯录同步、不做群管理。

凭据从 env 读（``WECOM_CORP_ID`` / ``WECOM_AGENT_SECRET`` / ``WECOM_AGENT_ID``），
缺一即视为未配置。文件消息上限 ``wecom_max_file_bytes``（默认 20MB），超限
由调用方降级为图文（带下载链接），本层不越权决定降级策略。
"""

from ai_agent.wecom.client import (
    WecomError,
    WecomNotConfigured,
    send_file_to_user,
    send_text_to_user,
)

__all__ = ["WecomError", "WecomNotConfigured", "send_file_to_user", "send_text_to_user"]
