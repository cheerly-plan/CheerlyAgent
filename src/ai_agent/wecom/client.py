"""企业微信自建应用客户端：access_token 缓存 + 上传临时素材 + 发送文件消息。

设计约束（与 web_search / gate 同策略）：

* 零第三方依赖：标准库 urllib（经 ``core.http``）手拼请求与 multipart；
* fail-fast 且错误可分类：企业微信返回的是 ``{"errcode": N, "errmsg": "..."}``，
  即使 HTTP 200 也可能业务失败。这里把 errcode != 0 统一转成 ``WecomError``
  并带 errcode，调用方能据此决定「重试 / 降级 / 报错」；
* access_token 进程内缓存：企业微信 token 有效期 7200 秒，逐次请求 gettoken
  会撞频率限制。缓存带锁防并发刷爆，并按 ``wecom_token_expire_buffer_seconds``
  提前失效主动刷新；
* 凭据只走 env，绝不进日志与异常文本。

企业微信接口（2026-11 核对，见文档 path/90248 与 90135/90253）：

* gettoken:   GET  /cgi-bin/gettoken?corpid=&corpsecret= → {"access_token","expires_in"}
* upload:     POST /cgi-bin/media/upload?access_token=&type=file → {"type","media_id","created_at"}
* message:    POST /cgi-bin/message/send?access_token=  body={"touser","msgtype":"file","agentid","file":{"media_id"}}
"""

from __future__ import annotations

import threading
import time

from ai_agent.core import http
from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger

logger = get_logger(__name__)

_API_BASE = "https://qyapi.weixin.qq.com/cgi-bin"

# access_token 过期后，用旧 token 调接口会返回 40014。
_ERRCODE_TOKEN_INVALID = 40014
_ERRCODE_TOKEN_EXPIRED = 42001


class WecomError(RuntimeError):
    """企业微信接口返回的业务错误（errcode != 0）。"""

    def __init__(self, errcode: int, errmsg: str) -> None:
        self.errcode = errcode
        self.errmsg = errmsg
        super().__init__(f"企业微信 errcode={errcode}: {errmsg}")


class WecomNotConfigured(RuntimeError):
    """未配置企业微信凭据，推送不可用。"""


def _credentials() -> tuple[str, str, str] | None:
    corp_id = (settings.wecom_corp_id or "").strip()
    secret = (settings.wecom_agent_secret or "").strip()
    agent_id = (settings.wecom_agent_id or "").strip()
    if not (corp_id and secret and agent_id):
        return None
    return corp_id, secret, agent_id


class _TokenCache:
    """进程内 access_token 缓存：带锁 + 提前失效。

    提前失效用 ``settings.wecom_token_expire_buffer_seconds`` 做缓冲，避免拿一个
    「还剩 1 秒」的 token 去发消息时刚好过期。企业微信 gettoken 对同一 corpsecret
    有频率限制（约 2000 次/天），缓存是刚需不是优化。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._token = ""
        self._expire_at = 0.0

    def get(self, corp_id: str, secret: str) -> str:
        now = time.time()
        buffer = max(0, int(settings.wecom_token_expire_buffer_seconds))
        if self._token and now < self._expire_at - buffer:
            return self._token
        with self._lock:
            # 双重检查：等锁期间别的线程可能已刷好
            if self._token and now < self._expire_at - buffer:
                return self._token
            url = f"{_API_BASE}/gettoken?corpid={corp_id}&corpsecret={secret}"
            data = http.request_json(url, timeout=10)
            token = data.get("access_token")
            expires_in = data.get("expires_in")
            if not token:
                raise WecomError(
                    int(data.get("errcode", -1)),
                    str(data.get("errmsg", "gettoken 响应缺少 access_token")),
                )
            self._token = token
            self._expire_at = now + max(0, int(expires_in or 0))
            return token

    def invalidate(self) -> None:
        with self._lock:
            self._token = ""
            self._expire_at = 0.0


_token_cache = _TokenCache()


def _upload_media(access_token: str, filename: str, content: bytes, content_type: str) -> str:
    """上传临时素材换 media_id（3 天有效）。返回 media_id。"""
    url = f"{_API_BASE}/media/upload?access_token={access_token}&type=file"
    data = http.request_upload(
        url,
        field_name="media",
        filename=filename,
        content=content,
        content_type=content_type,
        timeout=60,
    )
    media_id = data.get("media_id")
    if not media_id:
        raise WecomError(
            int(data.get("errcode", -1)),
            str(data.get("errmsg", "media/upload 响应缺少 media_id")),
        )
    return str(media_id)


def _send_file(access_token: str, agent_id: str, touser: str, media_id: str) -> dict:
    """发送文件消息给指定成员。返回完整响应（含 invaliduser 等）。"""
    url = f"{_API_BASE}/message/send?access_token={access_token}"
    body = {
        "touser": touser,
        "msgtype": "file",
        "agentid": int(agent_id),
        "file": {"media_id": media_id},
        "safe": 0,
    }
    data = http.request_json(url, body=body, timeout=15)
    errcode = int(data.get("errcode", -1))
    if errcode != 0:
        raise WecomError(errcode, str(data.get("errmsg", "未知错误")))
    return data


def send_text_to_user(*, touser: str, text: str) -> dict:
    """把一条文本消息推给指定成员 userid（可 ``|`` 分隔多个）。

    与 ``send_file_to_user`` 同契约：返回 ``message/send`` 原始响应（含
    ``invaliduser`` / ``invalidparty``），失败抛 ``WecomError`` / ``WecomNotConfigured``。

    企业微信 ``text`` 消息上限 2048 字节（UTF-8），超长会由企业微信报错或截断；
    调用方如需长文应自行拆条或改发 markdown 消息（msgtype=markdown，需 agentid
    有对应权限）。
    """
    creds = _credentials()
    if creds is None:
        raise WecomNotConfigured("未配置 WECOM_CORP_ID / WECOM_AGENT_SECRET / WECOM_AGENT_ID")
    corp_id, secret, agent_id = creds

    token = _token_cache.get(corp_id, secret)
    try:
        return _send_text(token, agent_id, touser, text)
    except WecomError as exc:
        if exc.errcode in {_ERRCODE_TOKEN_INVALID, _ERRCODE_TOKEN_EXPIRED}:
            logger.warning("企业微信 token 失效（errcode=%s），刷新后重试", exc.errcode)
            _token_cache.invalidate()
            token = _token_cache.get(corp_id, secret)
            return _send_text(token, agent_id, touser, text)
        raise


def _send_text(access_token: str, agent_id: str, touser: str, text: str) -> dict:
    url = f"{_API_BASE}/message/send?access_token={access_token}"
    body = {
        "touser": touser,
        "msgtype": "text",
        "agentid": int(agent_id),
        "text": {"content": text},
        "safe": 0,
    }
    data = http.request_json(url, body=body, timeout=15)
    errcode = int(data.get("errcode", -1))
    if errcode != 0:
        raise WecomError(errcode, str(data.get("errmsg", "未知错误")))
    return data


def send_file_to_user(
    *,
    filename: str,
    content: bytes,
    content_type: str,
    touser: str,
) -> dict:
    """把一份文件推给指定成员 userid（可逗号分隔多个，如 ``"zhangsan|lisi"``）。

    返回企业微信 ``message/send`` 的原始响应 dict（``errcode=0`` 时含
    ``invaliduser`` / ``invalidparty`` 等字段，调用方据此提示哪些人没发到）。

    抛出的异常：
    * ``WecomNotConfigured`` —— 未配凭据；
    * ``WecomError`` —— 企业微信业务失败（errcode 可在异常上取）。
    """
    creds = _credentials()
    if creds is None:
        raise WecomNotConfigured("未配置 WECOM_CORP_ID / WECOM_AGENT_SECRET / WECOM_AGENT_ID")
    corp_id, secret, agent_id = creds

    token = _token_cache.get(corp_id, secret)
    try:
        media_id = _upload_media(token, filename, content, content_type)
        return _send_file(token, agent_id, touser, media_id)
    except WecomError as exc:
        # token 失效/过期：刷一次再整体重试一次（上传的 media_id 是 3 天有效的
        # 临时素材，重试时重新上传无害；避免用坏 token 反复重试）。
        if exc.errcode in {_ERRCODE_TOKEN_INVALID, _ERRCODE_TOKEN_EXPIRED}:
            logger.warning("企业微信 token 失效（errcode=%s），刷新后重试", exc.errcode)
            _token_cache.invalidate()
            token = _token_cache.get(corp_id, secret)
            media_id = _upload_media(token, filename, content, content_type)
            return _send_file(token, agent_id, touser, media_id)
        raise
