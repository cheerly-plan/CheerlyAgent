"""研报接口：一次提交确定性跑完六步，返回正文 + 溯源 + 校验 + 审核队列。

  POST   /v1/report                              生成研报（同步，登录用户）
  POST   /v1/report/jobs                         提交后台研报任务（202 + job_id）
  GET    /v1/report/jobs/{id}                    轮询任务状态 / 阶段 / 进度
  GET    /v1/report/jobs/{id}/result             取已完成任务的完整 payload
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
双 PDF 档的 ``<stem>.deep.pdf`` 映射到 ``<report_id>-deep`` 这条独立审核记录，
须人工批准**并**定稿后才放行；事实层底稿仍按主记录自动定稿，互不影响。

事实层流水线本身不调用 LLM，但开启深度旁路后会走模型通道，``full`` 档耗时
可达数分钟。同步路由由 FastAPI 丢进线程池，不阻塞事件循环；网关超时较短的
客户端应改用 ``POST /v1/report/jobs`` 后台任务 + 轮询，两条入口共用
``_run_request`` / ``_payload``，产物与门禁完全一致。
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from ai_agent.api.dependencies import get_current_user
from ai_agent.api.schemas import ReportDecisionRequest, ReportRequest
from ai_agent.core.config import settings
from ai_agent.storage.db import UserRecord, user_scope
from ai_agent.report.pipeline import ReportResult, run_report
from ai_agent.report.jobs import job_manager
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
    payload = {
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
    # 未启用时**不新增任何响应键**，既有客户端与契约测试看到的 payload 逐字节不变。
    if result.deep_analysis is not None or result.deep_error:
        analysis = result.deep_analysis
        payload["deep"] = {
            "enabled": analysis is not None,
            "error": result.deep_error,
            "artifacts": result.deep_artifacts,
            "downloads": {
                key: f"/v1/report/files/{name}"
                for key, name in result.deep_artifacts.items()
            },
            "llm_calls": analysis.execution.llm_calls if analysis else 0,
            "estimated_input_tokens": (
                analysis.execution.estimated_input_tokens if analysis else 0
            ),
            "estimated_output_tokens": (
                analysis.execution.estimated_output_tokens if analysis else 0
            ),
            "enabled_chapters": (
                analysis.execution.enabled_chapters if analysis else []
            ),
            "gaps": [asdict(gap) for gap in analysis.all_gaps] if analysis else [],
            "pdf_merged": result.deep_pdf_merged,
            "pdf_mode": result.deep_pdf_mode,
            # 历史键只在 merge 档有意义，双 PDF 档统一看 pdf_error。
            "pdf_merge_error": (
                result.deep_pdf_error if result.deep_pdf_mode == "merge" else ""
            ),
            "pdf": (
                {
                    "filename": result.deep_artifacts["pdf"],
                    "pages": result.deep_pdf.pages if result.deep_pdf else 0,
                    "download_url": f"/v1/report/files/{result.deep_artifacts['pdf']}",
                }
                if result.deep_artifacts.get("pdf")
                else None
            ),
            "pdf_error": result.deep_pdf_error,
            "review": _deep_review_payload(result),
        }
    return payload


def _deep_review_payload(result: ReportResult) -> dict | None:
    """独立深度 PDF 的审核摘要。两份 PDF 各有一条记录，前端要分别展示。"""
    if not result.deep_pdf_review_id:
        return None
    record = review_store.get_record(result.deep_pdf_review_id, result.output_dir)
    _, reason = review_store.gate(record, result.output_dir)
    return {
        "report_id": result.deep_pdf_review_id,
        "status": record.status if record else "pending",
        "finalized": bool(record and record.status == FINALIZED),
        "gate_reason": reason,
        "queue_url": f"/v1/report/reviews/{result.deep_pdf_review_id}",
        "final_url": f"/v1/report/reviews/{result.deep_pdf_review_id}/final",
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


def _run_request(req: ReportRequest) -> ReportResult:
    """同步端点与后台任务共用的生成链路，保证两者 payload 逐字一致。"""
    # 默认 full+separate：底稿与深度两份 PDF；off 时连模块都不 import。
    deep_options = None
    if req.deep != "off":
        from ai_agent.report.deep import (
            ENTRY_CHAPTER_CHANNEL,
            ENTRY_CHAPTER_CONCURRENCY,
            ENTRY_REVIEW_ROUND_CONCURRENCY,
            default_facts_dir,
        )
        from ai_agent.report.deep.bridge import DeepOptions

        deep_options = DeepOptions(
            enabled=req.deep == "full",
            pdf_mode=req.deep_pdf,
            # 对外入口显式开启章节并行；库级 run_report 默认仍是串行。
            chapter_concurrency=ENTRY_CHAPTER_CONCURRENCY,
            # 单章评审轮次并发：3 轮互不依赖，串行只为了推进度，不改评审语义。
            review_round_concurrency=ENTRY_REVIEW_ROUND_CONCURRENCY,
            # 对外入口章节/评审直接主模型（决策 ④-B）；库级默认仍是小模型优先。
            chapter_channel=ENTRY_CHAPTER_CHANNEL,
            # 显式挂载结构化事实清单目录；库级 DeepOptions 默认 None 不启用，
            # 目录缺失时由 LocalFactsProvider 记缺口，绝不假装有外部数据。
            facts_dir=default_facts_dir(),
        )
    return run_report(
        req.request,
        skill_id=req.report_type or "",
        deliverable=req.deliverable,
        write_pdf=req.write_pdf,
        deep=deep_options,
    )


@router.post("")
def create_report(
    req: ReportRequest, user: UserRecord = Depends(get_current_user)
) -> dict:
    """生成一份研报。校验未通过不是 4xx：草稿照常返回，但 PDF 会被门禁挡住。"""
    from ai_agent.core.logging import get_logger

    # 审核记录要按生成者收敛，这里必须带上用户作用域（后台任务由 job 线程设置）
    with user_scope(user.user_id):
        result = _run_request(req)
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


# ---------------------------------------------------------------------------
# 可选异步入口：同步 POST /v1/report 的响应契约一行不动，长任务改用轮询
# ---------------------------------------------------------------------------


@router.post("/jobs", status_code=202)
def submit_report_job(
    req: ReportRequest, user: UserRecord = Depends(get_current_user)
) -> dict:
    """提交后台研报任务，立刻返回 job_id；用 poll_url 轮询阶段与进度。

    ``full`` 档会跑数分钟，网关超时短的场景应走这里。任务表只在进程内存里，
    进程重启后未完成任务会丢失，但已落盘的报告仍可从审核队列/文件端点取回。

    聊天入口的 ``generate_report`` 也提交到这张表（结果存 ``{"kind": "chat_report",
    "text": …}``）：断线的聊天用户可以用同一个 job_id 在这里轮询与取回。
    """
    job = job_manager.submit(
        # 任务归属用 user_id（UUID）而不是登录名：聊天入口的 generate_report
        # 也提交到这张表，它只拿得到 current_user_id()；两边必须同一个键，
        # 否则聊天任务在前端轮询 /v1/report/jobs/{id} 时会被判成「不是你的任务」。
        owner=user.user_id,
        request_text=req.request,
        runner=lambda: _payload(_run_request(req)),
    )
    return job.to_dict()


@router.get("/jobs/{job_id}/result")
def get_report_job_result(
    job_id: str, _user: UserRecord = Depends(get_current_user)
) -> dict:
    """只取已完成任务的完整 payload；未完成返回 409。"""
    job = job_manager.get(job_id, _user.user_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"没有 job_id={job_id} 的任务")
    if job.status != "succeeded":
        raise HTTPException(
            status_code=409, detail=f"任务尚未完成：{job.stage}（{job.status}）"
        )
    return job.result or {}


@router.get("/jobs/{job_id}")
def get_report_job(
    job_id: str, _user: UserRecord = Depends(get_current_user)
) -> dict:
    """轮询任务状态。完成时顺带返回 payload，省一次请求。"""
    job = job_manager.get(job_id, _user.user_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"没有 job_id={job_id} 的任务")
    return job.to_dict(include_result=job.status == "succeeded")


@router.get("/files/{name}")
def download_report(
    name: str,
    _user: UserRecord = Depends(get_current_user),
    preview: int = 0,
) -> FileResponse:
    """附件下载；``preview=1`` 只做在线预览，不触发下载门禁。

    ``preview=0``（默认）的整条路径逐字保持原样：PDF 仍须过
    ``pdf_download_gate``，Markdown / 溯源附件仍可下载。预览通道只对
    **已登录用户 + 有审核记录的产物**开放，返回 inline 且不带
    ``Content-Disposition: attachment``；前端据 ``X-Report-Download-Allowed``
    决定是否显示下载按钮。它不是绕过门禁的后门：未定稿的深度层 PDF
    仍不可作为附件外发，只是允许审核员在页面内查看红框标注的推断内容。

    ``preview`` 故意用裸 ``int`` 默认值而非 ``Query(...)``：路由函数会被引擎
    外部直接调用（测试/聊天入口），默认值必须真的是 0，不能是 Query 对象。
    """
    if preview not in (0, 1):
        raise HTTPException(status_code=422, detail="preview 只接受 0 或 1")
    path = artifact_path(name)
    if path is None:
        raise HTTPException(status_code=404, detail="文件不存在或不允许下载")
    record = review_store.record_for_artifact(name)
    headers = {"X-Report-Review-Status": record.status if record else "unknown"}
    if preview:
        # 没有审核记录的产物一律不许在线预览，避免端点变成任意 PDF 的读取口。
        if record is None:
            raise HTTPException(status_code=409, detail="这份文件没有审核记录，不能在线预览")
        download_allowed = True
        if path.suffix.lower() == ".pdf":
            download_allowed, _reason = review_store.pdf_download_gate(name)
        headers["X-Report-Download-Allowed"] = "1" if download_allowed else "0"
        headers["Content-Disposition"] = "inline"
        return FileResponse(
            path,
            media_type=_MEDIA[path.suffix.lower()],
            headers=headers,
        )
    if path.suffix.lower() == ".pdf":
        allowed, reason = review_store.pdf_download_gate(name)
        if not allowed:
            raise HTTPException(status_code=409, detail=f"PDF 未放行：{reason}")
    return FileResponse(
        path,
        media_type=_MEDIA[path.suffix.lower()],
        filename=path.name,
        headers=headers,
    )


# ---------------------------------------------------------------------------
# M2：人工审核队列与定稿硬门禁
# ---------------------------------------------------------------------------


def _is_review_admin(user: UserRecord) -> bool:
    return settings.is_admin_username(user.username)


def _require_review_access(record: review_store.ReviewRecord | None, user: UserRecord) -> None:
    """审批权：管理员可审全部；普通用户只能审自己生成的那份。

    放在函数体内显式校验，直接调用路由也会执行。``record.owner`` 为空的旧记录
    只有管理员能动（不知道当初是谁生成的，不能默认放给所有人）。
    """
    if _is_review_admin(user):
        return
    if record is not None and record.owner and record.owner == user.user_id:
        return
    raise HTTPException(status_code=403, detail="只能审核自己生成的研报")


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
    user: UserRecord = Depends(get_current_user),
) -> dict:
    """审核队列。管理员看全部，普通用户只看自己生成的那批。

    审核记录是全局索引，但按 ``owner`` 收敛：谁生成的谁审。``pending`` 即
    待人工处置的报告；默认按创建时间倒序。
    """
    owner = None if _is_review_admin(user) else user.user_id
    items = review_store.list_records(status=status, owner=owner)
    return {"total": len(items), "items": items}


@router.get("/reviews/{report_id}")
def get_review(report_id: str, user: UserRecord = Depends(get_current_user)) -> dict:
    """单条审核详情：管理员或该报告的生成者。"""
    record = review_store.get_record(report_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"没有 report_id={report_id} 的审核记录")
    _require_review_access(record, user)
    return _record_payload(record)


@router.post("/reviews/{report_id}/decision")
def decide_review(
    report_id: str,
    req: ReportDecisionRequest,
    user: UserRecord = Depends(get_current_user),
) -> dict:
    """逐条或整批处置审核项（管理员或该报告的生成者）。"""
    record = review_store.get_record(report_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"没有 report_id={report_id} 的审核记录")
    _require_review_access(record, user)
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
    """定稿（管理员或该报告的生成者）。校验未过 / 有待审项 / 被驳回 / 内容被改动，一律 409。"""
    record = review_store.get_record(report_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"没有 report_id={report_id} 的审核记录")
    _require_review_access(record, user)
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
