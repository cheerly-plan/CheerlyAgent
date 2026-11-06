"""研报接口：一次提交确定性跑完六步，返回正文 + 溯源 + 校验 + 审核队列。

  POST   /v1/report                              生成研报（登录用户）
  GET    /v1/report/skills                       可用报告类型（skill 注册表）
  GET    /v1/report/files/{name}                 附件下载；PDF 需已定稿（M2 门禁）
  GET    /v1/report/reviews                      审核队列（可按 status 过滤）
  GET    /v1/report/reviews/{id}                 单份审核详情
  POST   /v1/report/reviews/{id}/decision        批准 / 驳回 / 重置（管理员）
  POST   /v1/report/reviews/{id}/finalize        定稿（管理员，过门禁才行）
  GET    /v1/report/reviews/{id}/final           定稿文件下载（已定稿才放行）

工具分层（见 tools/registry.py）：抽数 / 算增减 / 校验三块确定性积木以
report_extract_indicators / report_compare / report_validate 注册进聊天循环，
均无路径参数、不写盘；report_pdf 仍不注册——渲染含落盘与 M2 审核门禁，
只能由服务端流水线触发，模型不参与落盘路径。

M2 门禁：校验有 error 的报告在生成阶段就不渲染 PDF；审核项未全部批准、
或被驳回、或定稿后文件被改动，PDF 一律 409。Markdown 正文与溯源附件仍可
下载，供审核员核对草稿。审批/定稿要求管理员白名单，且不留「强制放行」后门。

流水线本身不调用 LLM（见 report/pipeline.py 的说明），所以这里没有
流式输出、没有超时/限流配置，也没有 token 成本；耗时主要是知识库检索
与 PDF 渲染，同步路由由 FastAPI 丢进线程池，不阻塞事件循环。
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from ai_agent.api.dependencies import get_current_user
from ai_agent.api.schemas import ReportDecisionRequest, ReportRequest
from ai_agent.core.config import settings
from ai_agent.memory.db import UserRecord
from ai_agent.report.pipeline import ReportResult, run_report
from ai_agent.report import review as review_store
from ai_agent.report.review import FINALIZED, ReviewError, ReviewForbidden, ReviewNotFound
from ai_agent.report.skills import SKILLS

router = APIRouter(prefix="/v1/report", tags=["report"])

# 下载端点只放行这几种后缀，且文件必须直接位于 report_output_dir 下
_ALLOWED_SUFFIXES = {".pdf", ".md", ".json"}
_MEDIA = {
    ".pdf": "application/pdf",
    ".md": "text/markdown; charset=utf-8",
    ".json": "application/json; charset=utf-8",
}


def artifact_path(name: str) -> Path | None:
    """把下载请求的文件名收敛到报告目录内，挡住 ../ 与子目录穿越。"""
    if not name or Path(name).name != name or ".." in name:
        return None
    if Path(name).suffix.lower() not in _ALLOWED_SUFFIXES:
        return None
    root = settings.report_output_dir.resolve()
    candidate = (root / name).resolve()
    if candidate.parent != root or not candidate.is_file():
        return None
    return candidate


def _payload(result: ReportResult) -> dict:
    """把编排结果摊平成可序列化响应。数字相关结论一律以校验结果为准。"""
    pdf = None
    if result.pdf is not None:
        pdf = {
            "filename": result.artifacts.pdf,
            "pages": result.pdf.pages,
            "gaps": result.pdf.gaps,
            "download_url": f"/v1/report/files/{result.artifacts.pdf}",
        }
    artifacts = asdict(result.artifacts)
    downloads = {
        key: (f"/v1/report/files/{name}" if name else "")
        for key, name in artifacts.items()
    }
    return {
        "report_id": result.report_id,
        "status": result.status,
        "deliverable": result.deliverable,
        "pdf_status": result.pdf_status,
        "pdf_error": result.pdf_error,
        "request": result.request,
        "title": result.document.title,
        "plan": {
            "kind": result.plan.kind,
            "skill_id": result.plan.skill_id,
            "period": result.plan.period,
            "subjects": result.plan.subjects,
            "industry": result.plan.industry,
            "indicators": result.plan.indicators,
            "dimensions": result.plan.dimensions,
            "outline": [item.title for item in result.plan.outline],
            "sources": [asdict(item) for item in result.plan.files],
        },
        "markdown": result.document.markdown,
        "provenance": result.validation.provenance.markdown,
        "fact_count": len(result.facts),
        "validation": {
            "passed": result.validation.passed,
            "summary": result.validation.summary,
            "findings": [asdict(item) for item in result.validation.findings],
            "review_queue": [asdict(item) for item in result.validation.review_queue],
        },
        "gaps": result.gaps,
        "missing": result.missing,
        "artifacts": artifacts,
        "downloads": downloads,
        "pdf": pdf,
        "review": _review_payload(result.report_id, result.review_status, result.output_dir),
    }


def _review_payload(
    report_id: str, review_status: str, output_dir: Path | None = None
) -> dict:
    """生成响应里的审核摘要：能否定稿、去哪审、定稿件在哪。"""
    record = review_store.get_record(report_id, output_dir)
    _, reason = review_store.gate(record, output_dir)
    return {
        "report_id": report_id,
        "status": review_status or (record.status if record else "pending"),
        "finalized": bool(record and record.status == FINALIZED),
        "gate_reason": reason,
        "queue_url": f"/v1/report/reviews/{report_id}",
        "final_url": f"/v1/report/reviews/{report_id}/final",
    }


def _review_http_error(exc: ReviewError) -> HTTPException:
    if isinstance(exc, ReviewNotFound):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ReviewForbidden):
        return HTTPException(status_code=409, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))


@router.get("/skills")
def list_report_skills(
    _user: UserRecord = Depends(get_current_user),
) -> dict:
    """可用报告类型。POST /v1/report 的 report_type 从这里取值。"""
    return {
        "total": len(SKILLS),
        "skills": [
            {
                "skill_id": skill.skill_id,
                "name": skill.name,
                "kind": skill.kind,
                "description": skill.description,
                "default_indicators": list(skill.indicators),
                "sections": [title for title, _brief in skill.sections],
            }
            for skill in SKILLS
        ],
    }


@router.post("")
def create_report(
    req: ReportRequest, user: UserRecord = Depends(get_current_user)
) -> dict:
    """生成一份研报。校验未通过不是 4xx：草稿照常返回，但 PDF 会被门禁挡住。"""
    from ai_agent.core.logging import get_logger

    result = run_report(
        req.request,
        skill_id=req.report_type or "",
        deliverable=req.deliverable,
        write_pdf=req.write_pdf,
    )
    get_logger(__name__).info(
        "研报生成 user=%s report_id=%s status=%s facts=%d passed=%s pdf=%s review=%s",
        user.username,
        result.report_id,
        result.status,
        len(result.facts),
        result.validation.passed,
        result.pdf_status,
        result.review_status,
    )
    return _payload(result)


@router.get("/files/{name}")
def download_report(
    name: str, _user: UserRecord = Depends(get_current_user)
) -> FileResponse:
    path = artifact_path(name)
    if path is None:
        raise HTTPException(status_code=404, detail="文件不存在或不允许下载")
    record = review_store.record_for_artifact(name)
    if path.suffix.lower() == ".pdf":
        allowed, reason = review_store.pdf_download_gate(name)
        if not allowed:
            raise HTTPException(status_code=409, detail=f"PDF 未放行：{reason}")
    headers = {"X-Report-Review-Status": record.status if record else "unknown"}
    return FileResponse(
        path,
        media_type=_MEDIA[path.suffix.lower()],
        filename=path.name,
        headers=headers,
    )


# ---------------------------------------------------------------------------
# M2：人工审核队列与定稿硬门禁
# ---------------------------------------------------------------------------


def _require_reviewer(user: UserRecord) -> None:
    """审批/定稿的管理员守卫。放在函数体内，直接调用路由也会执行。"""
    if not settings.is_admin_username(user.username):
        raise HTTPException(status_code=403, detail="只有管理员可以审核与定稿研报")


def _record_payload(record: review_store.ReviewRecord) -> dict:
    allowed, reason = review_store.gate(record)
    payload = record.to_dict()
    payload["finalizable"] = allowed
    payload["gate_reason"] = reason
    payload["pending_items"] = len(record.pending_items)
    payload["queue_url"] = f"/v1/report/reviews/{record.report_id}"
    payload["final_url"] = f"/v1/report/reviews/{record.report_id}/final"
    return payload


@router.get("/reviews")
def list_reviews(
    status: str | None = Query(default=None, pattern=r"^(pending|approved|rejected|finalized)$"),
    _user: UserRecord = Depends(get_current_user),
) -> dict:
    """审核队列。不传 status 看全部；pending 即待人工处置的报告。"""
    items = review_store.list_records(status=status)
    return {"total": len(items), "items": items}


@router.get("/reviews/{report_id}")
def get_review(report_id: str, _user: UserRecord = Depends(get_current_user)) -> dict:
    record = review_store.get_record(report_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"没有 report_id={report_id} 的审核记录")
    return _record_payload(record)


@router.post("/reviews/{report_id}/decision")
def decide_review(
    report_id: str,
    req: ReportDecisionRequest,
    user: UserRecord = Depends(get_current_user),
) -> dict:
    """逐条或整批处置审核项。权限在函数体内显式校验，不依赖 DI 注入。"""
    _require_reviewer(user)
    try:
        record = review_store.decide(
            report_id,
            req.action,
            user.username,
            item_ids=req.item_ids or None,
            note=req.note,
        )
    except ReviewError as exc:
        raise _review_http_error(exc) from exc
    return _record_payload(record)


@router.post("/reviews/{report_id}/finalize")
def finalize_review(
    report_id: str, user: UserRecord = Depends(get_current_user)
) -> dict:
    """定稿。校验未过 / 有待审项 / 被驳回 / 内容被改动，一律 409。"""
    _require_reviewer(user)
    try:
        record = review_store.finalize(report_id, user.username)
    except ReviewError as exc:
        raise _review_http_error(exc) from exc
    return _record_payload(record)


@router.get("/reviews/{report_id}/final")
def download_final(
    report_id: str, _user: UserRecord = Depends(get_current_user)
) -> FileResponse:
    """定稿件下载。没有 PDF（渲染降级）时返回 Markdown 定稿。"""
    record = review_store.get_record(report_id)
    if record is None:
        raise HTTPException(status_code=404, detail="没有该报告的审核记录")
    allowed, reason = review_store.gate(record)
    if not allowed or record.status != FINALIZED:
        raise HTTPException(status_code=409, detail=f"报告未定稿：{reason}")
    # 只认记录里登记的文件名，且再走一遍路径收敛，避免索引被塞入绝对路径。
    name = record.artifacts.get("pdf") or record.artifacts.get("markdown") or ""
    path = artifact_path(name)
    if path is None:
        raise HTTPException(status_code=404, detail="定稿文件不存在或不允许下载")
    return FileResponse(path, media_type=_MEDIA[path.suffix.lower()], filename=path.name)
