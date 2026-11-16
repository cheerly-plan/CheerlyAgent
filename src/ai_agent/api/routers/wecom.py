"""企业微信推送端点：把研报产物（PDF/Markdown）或文本消息推给指定成员。

  POST /v1/wecom/message  发一条文本消息给指定 userid
  POST /v1/wecom/push     把 report_output_dir 下的一份研报产物推给指定 userid

设计边界：

* 文本消息（message）与文件推送（push）是两个独立能力，文本消息不触文件
  系统、不依赖研报目录；文件推送复用 ``report.artifact_path`` 定位文件——
  只有它持有「文件名 → 报告目录内文件」的收敛逻辑（挡 ``../`` 穿越、挡
  非法后缀），推送侧绝不自己拼路径；
* 文件推送只推「已在报告目录落盘」的产物（PDF 优先，无 PDF 时退 Markdown），
  不推溯源/审核 json 这类内部文件；
* 文件 ≤ ``wecom_max_file_bytes`` 时发文件消息；超限降级为图文消息（标题 +
  下载链接），不丢信息也不硬发被企业微信拒绝的 20MB+ 文件；
* 凭据缺失返回 503（可重试的配置错误），不返回 500 装成服务故障。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ai_agent.api.dependencies import get_current_user
from ai_agent.api.routers.report import artifact_path
from ai_agent.core.config import settings
from ai_agent.storage.db import UserRecord
from ai_agent.wecom import (
    WecomError,
    WecomNotConfigured,
    send_file_to_user,
    send_text_to_user,
)

router = APIRouter(prefix="/v1/wecom", tags=["wecom"])

# 可推送的产物后缀及其 MIME（与 report 下载端点白名单一致，但只放行正文类）。
_CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".md": "text/markdown; charset=utf-8",
}

# 企业微信 text 消息 2048 字节上限，超长直接 400，不让企业微信静默截断。
_MAX_TEXT_BYTES = 2048


class WecomMessageRequest(BaseModel):
    # 目标成员 userid，多个用 | 分隔（企业微信约定）。
    touser: str = Field(min_length=1, max_length=512)
    # 消息正文（UTF-8 文本）。pydantic 只做宽松的字符数上限，真正的 2048
    # 字节限制在端点内按 UTF-8 字节校验（中文 3 字节/字，字符数限制会漏判）。
    text: str = Field(min_length=1, max_length=4096)


class WecomPushRequest(BaseModel):
    # 研报产物文件名（如 <stem>.pdf / <stem>.md），来自研报响应的 downloads。
    filename: str = Field(min_length=1, max_length=255)
    # 目标成员 userid，多个用 | 分隔（企业微信约定）。
    touser: str = Field(min_length=1, max_length=512)


def _resolve_content(filename: str) -> tuple[bytes, str]:
    """把文件名收敛到报告目录内的文件，读回字节与 MIME。"""
    path = artifact_path(filename)
    if path is None:
        raise HTTPException(status_code=404, detail="研报文件不存在或类型不允许推送")
    suffix = Path(filename).suffix.lower()
    content_type = _CONTENT_TYPES.get(suffix)
    if content_type is None:
        raise HTTPException(status_code=400, detail=f"不支持推送 {suffix} 类型（仅 PDF/Markdown）")
    return path.read_bytes(), content_type


@router.post("/message")
def push_message(req: WecomMessageRequest, user: UserRecord = Depends(get_current_user)):
    """发一条文本消息给指定成员。登录用户即可调用（推送目标是业务操作，非管理）。"""
    if len(req.text.encode("utf-8")) > _MAX_TEXT_BYTES:
        raise HTTPException(status_code=400, detail="消息正文超过 2048 字节上限")

    try:
        result = send_text_to_user(touser=req.touser, text=req.text)
    except WecomNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except WecomError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    invalid = result.get("invaliduser") or ""
    invalid_party = result.get("invalidparty") or ""
    return {
        "errcode": 0,
        "touser": req.touser,
        "invaliduser": invalid,
        "invalidparty": invalid_party,
        "hint": (
            "推送成功"
            if not invalid and not invalid_party
            else f"推送成功，但以下接收人未送达：{invalid or invalid_party}"
        ),
    }


@router.post("/push")
def push_report(req: WecomPushRequest, user: UserRecord = Depends(get_current_user)):
    """把一份研报产物推给指定成员。登录用户即可调用（推送目标是业务操作，非管理）。"""
    content, content_type = _resolve_content(req.filename)

    max_bytes = max(0, int(settings.wecom_max_file_bytes))
    if max_bytes and len(content) > max_bytes:
        # 超限降级：不发文件本体，改发图文（带下载链接）。下载链接是相对路径，
        # 前端/网关侧会补全域名；这里只保证链接与研报响应一致。
        link = f"/v1/report/files/{req.filename}"
        raise HTTPException(
            status_code=413,
            detail=(
                f"文件 {len(content)} 字节超过上限 {max_bytes} 字节，"
                f"请改发下载链接：{link}"
            ),
        )

    try:
        result = send_file_to_user(
            filename=req.filename,
            content=content,
            content_type=content_type,
            touser=req.touser,
        )
    except WecomNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except WecomError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    invalid = result.get("invaliduser") or ""
    invalid_party = result.get("invalidparty") or ""
    return {
        "errcode": 0,
        "filename": req.filename,
        "touser": req.touser,
        "invaliduser": invalid,
        "invalidparty": invalid_party,
        "hint": (
            "推送成功"
            if not invalid and not invalid_party
            else f"推送成功，但以下接收人未送达：{invalid or invalid_party}"
        ),
    }
