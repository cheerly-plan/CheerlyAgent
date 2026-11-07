"""从巨潮拉取定期报告 PDF，并走现有 KB 入库链路。

研报流水线通过 ``report.ensure_filings`` 调用本模块；管理端 API 亦可直接调。
"""

from __future__ import annotations

import hashlib
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.kb import cninfo
from ai_agent.kb.pipeline import kb_pipeline
from ai_agent.storage.store import store
from ai_agent.tools.progress import emit_progress

logger = get_logger(__name__)

_SYSTEM_USER = "system:cninfo-fetch"


def ingest_wait_budget(file_count: int) -> float:
    """按本次补齐份数估算入库等待上限（秒）。"""
    base = max(float(settings.cninfo_ingest_wait_seconds), 1.0)
    per = max(float(settings.cninfo_ingest_wait_per_file_seconds), 0.0)
    cap = max(float(settings.cninfo_ingest_wait_cap_seconds), base)
    scaled = max(base, per * max(int(file_count), 0))
    return min(scaled, cap)


def requeue_uploaded_files(file_ids: list[str] | None = None) -> list[str]:
    """把仍停在 uploaded、磁盘上已有 PDF 的文件重新塞进解析队列。

    进程热重载会丢掉 ThreadPool 任务，文件会永远停在 uploaded；
    等待入库时主动重投，避免干等到超时。
    """
    rows = store.list_kb_files()
    wanted = {str(item) for item in (file_ids or []) if item}
    restarted: list[str] = []
    for row in rows:
        file_id = str(row.get("file_id") or "")
        if not file_id:
            continue
        if wanted and file_id not in wanted:
            continue
        if str(row.get("status") or "") != "uploaded":
            continue
        try:
            uid = uuid.UUID(file_id)
        except ValueError:
            continue
        folder = Path(settings.kb_upload_dir) / file_id
        pdfs = sorted(folder.glob("*.pdf")) if folder.is_dir() else []
        if not pdfs:
            continue
        user_id = str(row.get("user_id") or _SYSTEM_USER)
        kb_pipeline.submit(uid, user_id, pdfs[0])
        restarted.append(file_id)
        logger.info("kb 重投 uploaded 文件 file_id=%s name=%s", file_id, row.get("filename"))
    return restarted


@dataclass
class FetchedFile:
    file_id: str
    filename: str
    status: str
    announcement_id: str
    title: str
    duplicate_of: str = ""


@dataclass
class FetchResult:
    code: str
    period: str
    company: str
    files: list[FetchedFile] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"巨潮补齐 {self.code} {self.period}"
            + (f"（{self.company}）" if self.company else "")
        ]
        if self.files:
            lines.append(f"已提交入库 {len(self.files)} 份：")
            for item in self.files:
                extra = (
                    f"，与已有《{item.duplicate_of}》相同已跳过建新档"
                    if item.duplicate_of
                    else f"，status={item.status}"
                )
                lines.append(
                    f"- {item.filename} file_id={item.file_id} "
                    f"公告 {item.announcement_id}{extra}"
                )
        if self.skipped:
            lines.append("跳过：")
            lines.extend(f"- {item}" for item in self.skipped)
        if self.errors:
            lines.append("失败：")
            lines.extend(f"- {item}" for item in self.errors)
        if not self.files and not self.errors and not self.skipped:
            lines.append("未找到可下载的正文 PDF。")
        elif not self.files and self.skipped and not self.errors:
            lines.append("没有新文件需要入库。")
        return "\n".join(lines)


def fetch_missing_reports(
    code: str,
    period: str,
    *,
    user_id: str = "",
    max_files: int = 1,
) -> FetchResult:
    """查公告 → 下载校验 → create_kb_file → submit。默认只取 1 份正文。"""
    if not settings.enable_cninfo_fetch:
        raise RuntimeError(
            "巨潮补齐已关闭。在 .env 设置 ENABLE_CNINFO_FETCH=true 后重试。"
        )
    code = cninfo.normalize_code(code)
    period = cninfo.normalize_period(period)
    max_files = max(1, min(int(max_files), 10))
    stock = cninfo.lookup_stock(code)
    result = FetchResult(code=code, period=period, company=stock.name)
    try:
        announcements = cninfo.search_announcements(code, period)
    except Exception as exc:
        logger.exception("巨潮公告查询失败 code=%s period=%s", code, period)
        result.errors.append(f"查询失败：{exc}")
        return result
    if not announcements:
        result.skipped.append("公告列表为空（可能尚未披露或过滤后无正文）")
        return result

    owner = (user_id or "").strip() or _SYSTEM_USER
    for announcement in announcements[:max_files]:
        try:
            downloaded = cninfo.download_pdf(
                announcement,
                company_name=_company_from_announcement(announcement, stock.name),
                period=period,
            )
        except Exception as exc:
            logger.warning(
                "巨潮 PDF 下载失败 id=%s err=%s", announcement.announcement_id, exc
            )
            result.errors.append(
                f"{announcement.title} 下载失败：{exc}"
            )
            continue
        try:
            fetched = _ingest_bytes(
                owner,
                downloaded.content,
                downloaded.filename_stem,
                tags=_tags(announcement, code, period),
            )
        except Exception as exc:
            logger.exception(
                "巨潮 PDF 入库失败 id=%s", announcement.announcement_id
            )
            result.errors.append(f"{downloaded.filename_stem} 入库失败：{exc}")
            continue
        fetched.announcement_id = announcement.announcement_id
        fetched.title = announcement.title
        if fetched.duplicate_of:
            result.skipped.append(
                f"《{fetched.filename}》与已有《{fetched.duplicate_of}》内容相同"
                + (
                    f"（当前 status={fetched.status}，需等入库）"
                    if fetched.status and fetched.status != "ready"
                    else ""
                )
            )
            # 重复档若尚未 ready，仍放入 files 以便上层等待，避免「未解析完就出稿」。
            if fetched.status != "ready":
                result.files.append(fetched)
        else:
            result.files.append(fetched)
    return result


def ingest_pdf_url(
    url: str,
    *,
    filename_stem: str,
    user_id: str = "",
    tags: list[str] | None = None,
) -> FetchedFile:
    """管理员贴白名单 PDF 直链入库（仅 static.cninfo.com.cn）。"""
    if not settings.enable_cninfo_fetch:
        raise RuntimeError(
            "巨潮补齐已关闭。在 .env 设置 ENABLE_CNINFO_FETCH=true 后重试。"
        )
    stem = (filename_stem or "").strip()
    if not stem:
        raise ValueError("filename_stem 不能为空")
    cninfo.assert_allowed_pdf_url(url)
    max_bytes = min(settings.kb_max_file_mb, 50) * 1024 * 1024
    from ai_agent.core import http

    content, content_type = http.request_bytes(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.cninfo.com.cn/",
        },
        timeout=60,
        max_bytes=max_bytes,
    )
    if not content.startswith(b"%PDF"):
        raise ValueError(f"响应不是 PDF：content_type={content_type!r}")
    return _ingest_bytes(
        (user_id or "").strip() or _SYSTEM_USER,
        content,
        stem,
        tags=tags or ["source:url", "host:static.cninfo.com.cn"],
    )


def _company_from_announcement(announcement: cninfo.Announcement, fallback: str) -> str:
    title = announcement.title or ""
    for marker in ("股份有限公司", "有限公司"):
        idx = title.find(marker)
        if idx >= 0:
            return title[: idx + len(marker)]
    return announcement.sec_name or fallback


def _tags(announcement: cninfo.Announcement, code: str, period: str) -> list[str]:
    tags = [
        "source:cninfo",
        f"cninfo:{announcement.announcement_id}",
        f"code:{code}",
        f"period:{period}",
    ]
    if "摘要" in (announcement.title or ""):
        tags.append("doc:summary")
    if announcement.announce_date:
        tags.append(f"date:{announcement.announce_date}")
    return tags[:8]


def _ingest_bytes(
    user_id: str,
    content: bytes,
    filename_stem: str,
    *,
    tags: list[str],
) -> FetchedFile:
    stem = Path(filename_stem.replace("\\", "/")).name.strip()
    if stem.lower().endswith(".pdf"):
        stem = stem[:-4]
    if not stem:
        raise ValueError("文件名为空")
    digest = hashlib.sha256(content).hexdigest()
    duplicate = store.find_kb_file_by_hash(digest)
    if duplicate is not None:
        # 如实带回库内状态：若仍在 parsing，调用方必须纳入 wait_until_ready，
        # 不能假装已 ready 而跳过等待。
        return FetchedFile(
            file_id=str(duplicate["file_id"]),
            filename=str(duplicate["filename"]),
            status=str(duplicate.get("status") or ""),
            announcement_id="",
            title="",
            duplicate_of=str(duplicate["filename"]),
        )

    file_id = store.create_kb_file(
        user_id, stem, len(content), ".pdf", tags=tags
    )
    dest_dir = Path(settings.kb_upload_dir) / str(file_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{stem}.pdf"
    try:
        path.write_bytes(content)
        store.update_kb_file_hash(file_id, digest)
        kb_pipeline.submit(file_id, user_id, path)
    except Exception:
        store.delete_kb_file(file_id)
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise
    return FetchedFile(
        file_id=str(file_id),
        filename=stem,
        status="uploaded",
        announcement_id="",
        title="",
    )


def _display_filing_name(filename: str) -> str:
    """进度条用短名：去掉「股份有限公司」等套话，保留公司+报告期。"""
    text = (filename or "").strip()
    for noise in ("股份有限公司", "有限公司", "集团"):
        text = text.replace(noise, "")
    return text or "财报"


def _user_status(status: str) -> str:
    return {
        "uploaded": "排队中",
        "parsing": "解析中",
        "embedding": "整理中",
        "failed": "失败",
        "empty": "失败",
        "ready": "已完成",
    }.get(status, "处理中")


def wait_until_ready(
    file_ids: list[str],
    *,
    timeout_seconds: float | None = None,
    poll_seconds: float = 2.0,
) -> list[str]:
    """轮询到 ready / failed / 超时。返回仍未 ready 的 file_id（含失败与超时）。"""
    parsing = [item for item in file_ids if item]
    if not parsing:
        return []
    budget = (
        ingest_wait_budget(len(parsing))
        if timeout_seconds is None
        else max(float(timeout_seconds), 1.0)
    )
    deadline = time.monotonic() + budget
    interval = max(float(poll_seconds), 0.2)
    unfinished: list[str] = []
    last_emit = 0.0
    # 热重载后可能丢队：先把 uploaded 重投一次，再进入轮询。
    requeued = requeue_uploaded_files(parsing)
    if requeued:
        logger.info("kb 重投 uploaded %d 份后继续等待入库", len(requeued))
    emit_progress(f"正在准备财报原文（共 {len(parsing)} 份）…")
    while parsing and time.monotonic() < deadline:
        next_parsing: list[str] = []
        progressed = False
        waiting_labels: list[str] = []
        for raw in parsing:
            try:
                file_id = uuid.UUID(str(raw))
            except ValueError:
                unfinished.append(raw)
                progressed = True
                continue
            row = store.get_kb_file(file_id)
            status = str((row or {}).get("status") or "")
            name = str((row or {}).get("filename") or file_id)
            if status == "ready":
                progressed = True
                continue
            if status == "empty":
                unfinished.append(raw)
                progressed = True
                continue
            if status == "failed":
                # 误标「源文件已不在」但 PDF 还在：退回队列继续等，避免研报直接跳过
                err = str((row or {}).get("error") or "")
                folder = Path(settings.kb_upload_dir) / str(file_id)
                has_pdf = folder.is_dir() and any(folder.glob("*.pdf"))
                if has_pdf and (
                    "源文件已不在" in err or "解析子进程异常退出" in err
                ):
                    store.update_kb_status(
                        file_id, "uploaded", error="", chunk_count=0, text_count=0, table_count=0
                    )
                    requeue_uploaded_files([str(file_id)])
                    waiting_labels.append(
                        f"{_display_filing_name(name)}（已重新入队）"
                    )
                    next_parsing.append(raw)
                    progressed = True
                    continue
                unfinished.append(raw)
                progressed = True
                continue
            # uploaded 只在进入轮询前统一重投一次；这里再投会叠多个 _process，
            # 后写的切片撞上 (file_id, seq) 唯一约束。
            waiting_labels.append(
                f"{_display_filing_name(name)}（{_user_status(status)}）"
            )
            next_parsing.append(raw)
        finished = len(file_ids) - len(next_parsing)
        now = time.monotonic()
        if now - last_emit >= 3.0 or not next_parsing:
            if next_parsing:
                sample = "、".join(waiting_labels[:2])
                more = f" 等 {len(waiting_labels)} 份" if len(waiting_labels) > 2 else ""
                emit_progress(
                    f"正在解析财报原文 {finished}/{len(file_ids)}"
                    + (f"：{sample}{more}" if sample else "")
                )
            else:
                emit_progress(f"财报原文已就绪 {finished}/{len(file_ids)}")
            last_emit = now
        parsing = next_parsing
        if not parsing:
            break
        time.sleep(interval)
        if not progressed and time.monotonic() >= deadline:
            break
    unfinished.extend(parsing)
    if unfinished:
        logger.warning(
            "入库等待超时 unfinished=%d budget=%.0fs", len(unfinished), budget
        )
        emit_progress(
            f"部分财报仍在处理（{len(unfinished)} 份），"
            "将先用已就绪的原文继续生成"
        )
    else:
        emit_progress(f"财报原文已全部就绪（{len(file_ids)} 份）")
    return unfinished
