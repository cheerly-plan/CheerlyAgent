"""研报产物清理：同标题只留最新版 + 按保留天数删过期件。

研报落在 ``report_output_dir``，与知识库 uploads **分开**，不会自动进 RAG；
但重复生成会占磁盘、也让下载链接难辨认哪个是终稿。清理只管报告目录与
``_review/index.json``，不动 uploads。
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ai_agent.core.config import settings
from ai_agent.core.logging import get_logger
from ai_agent.report import review as review_store

logger = get_logger(__name__)

# title_reportid.md / .pdf / .provenance.md / .review.json
# 以及深度分析旁路的 .deep.md / .deep.json —— 不列进来旧版清理会留下孤儿文件。
# 长后缀放前面，避免 .md / .json 先抢匹配（尽管 $ 锚定已能保证正确性）。
_ARTIFACT = re.compile(
    r"^(?P<title>.+)_(?P<id>[0-9a-f]{10})"
    r"(?:\.provenance\.md|\.review\.json|\.deep\.md|\.deep\.json|\.md|\.pdf)$"
)


@dataclass
class CleanupResult:
    deleted_report_ids: list[str] = field(default_factory=list)
    deleted_files: list[str] = field(default_factory=list)
    kept_report_ids: list[str] = field(default_factory=list)


@dataclass
class _Edition:
    report_id: str
    title: str
    mtime: float
    files: list[Path] = field(default_factory=list)


def cleanup_reports(
    output_dir: Path | str | None = None,
    *,
    keep_latest: int | None = None,
    retention_days: int | None = None,
    preserve_ids: set[str] | frozenset[str] | None = None,
) -> CleanupResult:
    """清理研报目录。

    * ``keep_latest``：同一标题（去掉 ``_report_id``）只留最新 N 版；``<=0`` 关闭。
    * ``retention_days``：mtime 超过 N 天的版本删除；``<=0`` 关闭。
    * ``preserve_ids``：强制保留（例如刚生成的那一版）。
    """
    root = Path(output_dir) if output_dir else settings.report_output_dir
    keep = settings.report_keep_latest_per_title if keep_latest is None else keep_latest
    days = settings.report_retention_days if retention_days is None else retention_days
    preserve = set(preserve_ids or ())
    if not root.is_dir():
        return CleanupResult()
    if keep <= 0 and days <= 0:
        return CleanupResult()

    editions = _scan(root)
    doomed = _select_doomed(editions, keep=keep, retention_days=days, preserve=preserve)
    result = CleanupResult(kept_report_ids=sorted(set(editions) - doomed))
    if not doomed:
        return result

    for report_id in sorted(doomed):
        edition = editions[report_id]
        for path in edition.files:
            try:
                path.unlink(missing_ok=True)
                result.deleted_files.append(path.name)
            except OSError:
                logger.exception("删除研报文件失败：%s", path)
        result.deleted_report_ids.append(report_id)

    removed = review_store.remove_records(result.deleted_report_ids, root)
    if removed != len(result.deleted_report_ids):
        logger.warning(
            "研报文件已删，审核索引同步 %s/%s 条",
            removed,
            len(result.deleted_report_ids),
        )
    return result


def prune_older_editions(
    title: str,
    *,
    keep_report_id: str,
    output_dir: Path | str | None = None,
) -> CleanupResult:
    """生成成功后调用：同标题只留刚写出的这一版（不影响其他标的）。"""
    keep = settings.report_keep_latest_per_title
    if keep <= 0:
        return CleanupResult(kept_report_ids=[keep_report_id])
    root = Path(output_dir) if output_dir else settings.report_output_dir
    cleaned_title = _normalize_title(title)
    editions = {
        rid: ed
        for rid, ed in _scan(root).items()
        if _normalize_title(ed.title) == cleaned_title
    }
    if not editions:
        return CleanupResult(kept_report_ids=[keep_report_id])
    doomed = _select_doomed(
        editions, keep=keep, retention_days=0, preserve={keep_report_id}
    )
    result = CleanupResult(kept_report_ids=sorted(set(editions) - doomed))
    for report_id in sorted(doomed):
        for path in editions[report_id].files:
            try:
                path.unlink(missing_ok=True)
                result.deleted_files.append(path.name)
            except OSError:
                logger.exception("删除旧版研报失败：%s", path)
        result.deleted_report_ids.append(report_id)
    if result.deleted_report_ids:
        review_store.remove_records(result.deleted_report_ids, root)
    return result


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", (title or "").strip())


def _scan(root: Path) -> dict[str, _Edition]:
    editions: dict[str, _Edition] = {}
    for path in root.iterdir():
        if not path.is_file():
            continue
        match = _ARTIFACT.match(path.name)
        if match is None:
            continue
        report_id = match.group("id")
        title = match.group("title")
        edition = editions.get(report_id)
        if edition is None:
            edition = _Edition(report_id=report_id, title=title, mtime=0.0)
            editions[report_id] = edition
        edition.files.append(path)
        try:
            edition.mtime = max(edition.mtime, path.stat().st_mtime)
        except OSError:
            continue
    # 审核索引里有、磁盘已空的残留，也纳入可删集合（仅 id，无文件）
    try:
        for report_id, data in review_store.list_index_records(root).items():
            if report_id in editions:
                continue
            editions[report_id] = _Edition(
                report_id=report_id,
                title=str(data.get("title") or report_id),
                mtime=0.0,
            )
    except review_store.ReviewError:
        logger.exception("读取研报审核索引失败，本次只按磁盘文件清理")
    return editions


def _select_doomed(
    editions: dict[str, _Edition],
    *,
    keep: int,
    retention_days: int,
    preserve: set[str],
) -> set[str]:
    doomed: set[str] = set()
    now = time.time()
    cutoff = now - retention_days * 86400 if retention_days > 0 else None

    if keep > 0:
        by_title: dict[str, list[_Edition]] = defaultdict(list)
        for edition in editions.values():
            by_title[_normalize_title(edition.title)].append(edition)
        for group in by_title.values():
            ordered = sorted(group, key=lambda item: item.mtime, reverse=True)
            # preserve 的版本永远进保留池；其余按新→旧填满 keep 名额
            kept: list[str] = []
            for edition in ordered:
                if edition.report_id in preserve and edition.report_id not in kept:
                    kept.append(edition.report_id)
            for edition in ordered:
                if len(kept) >= keep:
                    break
                if edition.report_id not in kept:
                    kept.append(edition.report_id)
            for edition in ordered:
                if edition.report_id not in kept:
                    doomed.add(edition.report_id)

    if cutoff is not None:
        for edition in editions.values():
            if edition.report_id in preserve:
                continue
            # 无文件的索引残留 mtime=0，一律当作过期清掉
            if edition.mtime == 0.0 or edition.mtime < cutoff:
                doomed.add(edition.report_id)

    return doomed
