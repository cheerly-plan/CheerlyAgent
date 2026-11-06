"""M2：研报人工审核队列 + 定稿硬门禁。

设计要点（对应 next.md 第 6 步「异常数据打入人工审核队列」）：

* **唯一权威源**是 ``report_output_dir/_review/index.json``。每份报告旁边的
  ``<stem>.review.json`` 只保留为生成时的**只读快照**（兼容 M1 的产物格式），
  改快照不会改变门禁结论，避免出现「两套状态」。
* 审核粒度是**每条异常**（缺少出处页码、正文数字对不上、推导链不完整等），
  记录审批人、时间、结论与备注；整份报告的状态由各条状态推导。
* 状态机：``pending → approved → finalized``，可 ``rejected``（驳回后不许定稿），
  可 ``reset`` 回到 pending 重新处置。定稿后不可再改，重新生成请用新 report_id。
* **内容指纹**绑定正文 / 溯源 / PDF 字节。审批后任何人替换文件，状态自动退回
  pending 并留一条审计记录——防止「审批旧稿、外发新稿」。
* 门禁是硬的：``validation.passed=False`` 的报告**不渲染 PDF**；PDF 下载端点
  只放行 ``finalized`` 的定稿。无异常且校验通过的报告在生成时自动定稿，
  保证聊天侧正常链路不用人工多点一次。

并发：进程内用 RLock 串行化「读-改-写」，落盘用同目录临时文件 + os.replace
原子替换。多进程 / 多副本部署时请换成 Postgres 表 + 行锁（见交付文档风险项）。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from ai_agent.core.config import settings

if TYPE_CHECKING:  # 避免 pipeline <-> review 循环导入
    from ai_agent.report.pipeline import ReportResult

PENDING = "pending"
APPROVED = "approved"
REJECTED = "rejected"
FINALIZED = "finalized"

_DECISIONS = {"approve", "reject", "reset"}
_INDEX_DIR = "_review"
_INDEX_FILE = "index.json"
_LOCK = threading.RLock()
# 报告产物名末尾是 10 位十六进制 report_id（见 pipeline._stem）
_ID_IN_NAME = re.compile(r"_([0-9a-f]{10})(?:\.[^.]+)+$")


class ReviewError(ValueError):
    """审核操作的业务错误，路由层按类型映射成 HTTP 状态码。"""


class ReviewNotFound(ReviewError):
    pass


class ReviewForbidden(ReviewError):
    """状态不允许该操作（未批准就定稿、驳回后下载等）。"""


@dataclass
class ReviewEntry:
    """一条待人工核对的异常。字段与 validate.ReviewItem 对齐，另加处置结果。"""

    id: str
    reason: str
    value: str = ""
    subject: str = ""
    name: str = ""
    period: str = ""
    file_id: str = ""
    quote: str = ""
    state: str = PENDING
    reviewer: str = ""
    note: str = ""
    decided_at: str = ""


@dataclass
class ReviewRecord:
    report_id: str
    title: str = ""
    report_status: str = ""  # pipeline 的 ok / no_data / no_request
    status: str = PENDING
    created_at: str = ""
    updated_at: str = ""
    content_hash: str = ""
    validation_passed: bool = False
    error_count: int = 0
    artifacts: dict[str, str] = field(default_factory=dict)
    items: list[ReviewEntry] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    finalized_by: str = ""
    finalized_at: str = ""
    auto_finalized: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ReviewRecord:
        items = [ReviewEntry(**item) for item in data.get("items", [])]
        known = set(cls.__dataclass_fields__)
        payload = {key: value for key, value in data.items() if key in known}
        payload["items"] = items
        return cls(**payload)

    @property
    def pending_items(self) -> list[ReviewEntry]:
        return [item for item in self.items if item.state == PENDING]

    @property
    def rejected_items(self) -> list[ReviewEntry]:
        return [item for item in self.items if item.state == REJECTED]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _root(output_dir: Path | str | None = None) -> Path:
    return Path(output_dir) if output_dir else settings.report_output_dir


def index_path(output_dir: Path | str | None = None) -> Path:
    return _root(output_dir) / _INDEX_DIR / _INDEX_FILE


def _read_index(output_dir: Path | str | None = None) -> dict[str, dict]:
    path = index_path(output_dir)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        # 索引损坏时宁可让写操作失败，也不能静默重建覆盖掉历史审批记录。
        raise ReviewError(f"审核索引不可读，拒绝覆盖：{exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("records", {}), dict):
        raise ReviewError("审核索引结构不合法，拒绝覆盖")
    return data["records"]


def _write_index(records: dict[str, dict], output_dir: Path | str | None = None) -> None:
    path = index_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 同目录临时文件 + os.replace：写入过程不会留下半个 JSON。
    fd, tmp = tempfile.mkstemp(prefix="index-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"version": 1, "records": records}, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _write_snapshot(record: ReviewRecord, output_dir: Path | str | None = None) -> None:
    name = record.artifacts.get("review")
    if not name:
        return
    path = _root(output_dir) / name
    path.write_text(
        json.dumps(record.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _save_record(record: ReviewRecord, output_dir: Path | str | None = None) -> ReviewRecord:
    records = _read_index(output_dir)
    records[record.report_id] = record.to_dict()
    _write_index(records, output_dir)
    _write_snapshot(record, output_dir)
    return record


def list_index_records(output_dir: Path | str | None = None) -> dict[str, dict]:
    """给清理任务读审核索引；索引损坏时抛 ReviewError，由调用方决定是否继续。"""
    with _LOCK:
        return dict(_read_index(output_dir))


def remove_records(
    report_ids: list[str] | tuple[str, ...] | set[str],
    output_dir: Path | str | None = None,
) -> int:
    """从审核索引里摘掉已删除产物对应的记录。返回实际移除条数。"""
    wanted = {str(item).strip() for item in report_ids if str(item).strip()}
    if not wanted:
        return 0
    with _LOCK:
        records = _read_index(output_dir)
        removed = 0
        for report_id in wanted:
            if records.pop(report_id, None) is not None:
                removed += 1
        if removed:
            _write_index(records, output_dir)
        return removed


def content_hash(markdown: str, provenance: str, pdf: bytes | None) -> str:
    """对本报告的可交付内容取指纹。PDF 可选：渲染降级时只绑定正文与溯源。"""
    digest = hashlib.sha256()
    for part in (markdown.encode("utf-8"), provenance.encode("utf-8"), pdf or b""):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def _current_hash(record: ReviewRecord, output_dir: Path | str | None = None) -> str:
    """按记录里登记的文件名重算指纹；文件缺失也算变更（返回空串）。"""
    root = _root(output_dir)
    names = record.artifacts
    try:
        markdown = (root / names["markdown"]).read_text(encoding="utf-8")
        provenance = (root / names["provenance"]).read_text(encoding="utf-8")
        pdf_name = names.get("pdf", "")
        pdf = (root / pdf_name).read_bytes() if pdf_name else None
    except (OSError, KeyError):
        return ""
    return content_hash(markdown, provenance, pdf)


def _decision(
    record: ReviewRecord, action: str, actor: str, note: str, item_ids: list[str], status: str
) -> None:
    record.decisions.append(
        {
            "at": _now(),
            "actor": actor,
            "action": action,
            "note": note[:500],
            "item_ids": item_ids,
            "status": status,
        }
    )
    record.updated_at = _now()


def create_record(
    result: ReportResult, output_dir: Path | str | None = None, *, auto_finalize: bool = True
) -> ReviewRecord:
    """生成完成后登记一份审核记录。重复 report_id 不覆盖已有审批。"""
    with _LOCK:
        existing = _read_index(output_dir).get(result.report_id)
        if existing:
            return ReviewRecord.from_dict(existing)

        items = [
            ReviewEntry(
                id=f"r{index + 1:03d}",
                reason=item.reason,
                value=item.value,
                subject=item.subject,
                name=item.name,
                period=item.period,
                file_id=item.file_id,
                quote=item.quote,
            )
            for index, item in enumerate(result.validation.review_queue)
        ]
        error_count = sum(
            1 for item in result.validation.findings if item.severity == "error"
        )
        now = _now()
        record = ReviewRecord(
            report_id=result.report_id,
            title=result.document.title,
            report_status=result.status,
            created_at=now,
            updated_at=now,
            validation_passed=result.validation.passed,
            error_count=error_count,
            artifacts=asdict(result.artifacts),
            items=items,
        )
        # 指纹在产物全部落盘之后计算，见 pipeline.run_report 的调用顺序。
        record.content_hash = _current_hash(record, output_dir)
        clean = (
            auto_finalize
            and result.status == "ok"
            and result.validation.passed
            and not items
        )
        if clean:
            record.status = FINALIZED
            record.finalized_by = "system"
            record.finalized_at = now
            record.auto_finalized = True
            _decision(record, "auto_finalize", "system", "校验通过且无待审异常，自动定稿", [], FINALIZED)
        return _save_record(record, output_dir)


def get_record(
    report_id: str, output_dir: Path | str | None = None, *, verify: bool = True
) -> ReviewRecord | None:
    data = _read_index(output_dir).get(report_id)
    if not data:
        return None
    record = ReviewRecord.from_dict(data)
    if verify:
        return _verify(record, output_dir)
    return record


def _verify(record: ReviewRecord, output_dir: Path | str | None = None) -> ReviewRecord:
    """指纹失配即作废定稿，留审计记录，绝不带着旧审批继续放行。"""
    if record.status == FINALIZED and _current_hash(record, output_dir) != record.content_hash:
        with _LOCK:
            # 锁内重读一次，避免并发下覆盖别人的审批
            latest = get_record(record.report_id, output_dir, verify=False)
            if latest is None or latest.status != FINALIZED:
                return latest or record
            if _current_hash(latest, output_dir) == latest.content_hash:
                return latest
            latest.status = PENDING
            latest.finalized_by = ""
            latest.finalized_at = ""
            latest.auto_finalized = False
            _decision(latest, "invalidate", "system", "定稿后内容发生变化，审批自动作废", [], PENDING)
            return _save_record(latest, output_dir)
    return record


def list_records(
    output_dir: Path | str | None = None, status: str | None = None
) -> list[dict]:
    """队列列表。按创建时间倒序，附带门禁结论，方便前端直接渲染。"""
    records = [ReviewRecord.from_dict(data) for data in _read_index(output_dir).values()]
    if status:
        records = [item for item in records if item.status == status]
    records.sort(key=lambda item: item.created_at, reverse=True)
    output = []
    for record in records:
        checked = get_record(record.report_id, output_dir) or record
        allowed, reason = gate(checked, output_dir)
        payload = checked.to_dict()
        payload["finalizable"] = allowed
        payload["gate_reason"] = reason
        payload["pending_items"] = len(checked.pending_items)
        payload["queue_url"] = f"/v1/report/reviews/{checked.report_id}"
        payload["final_url"] = f"/v1/report/reviews/{checked.report_id}/final"
        output.append(payload)
    return output


def decide(
    report_id: str,
    action: str,
    actor: str,
    *,
    item_ids: list[str] | None = None,
    note: str = "",
    output_dir: Path | str | None = None,
) -> ReviewRecord:
    """处置审核项。approve/reject 可指定条目，不传即整批；reset 整批回退。"""
    if action not in _DECISIONS:
        raise ReviewError(f"不支持的审核动作：{action}")
    with _LOCK:
        record = get_record(report_id, output_dir)
        if record is None:
            raise ReviewNotFound(f"没有 report_id={report_id} 的审核记录")
        if record.status == FINALIZED and action != "reset":
            raise ReviewForbidden("报告已定稿，如需改动请重新生成研报")
        if action == "reset" and record.status == FINALIZED:
            raise ReviewForbidden("报告已定稿，不能重置；请重新生成研报")
        if _current_hash(record, output_dir) != record.content_hash:
            raise ReviewForbidden("报告内容已变更，请重新生成研报后再审核")

        targets = record.items
        if action != "reset" and item_ids is not None:
            wanted = set(item_ids)
            known = {item.id for item in record.items}
            unknown = wanted - known
            if unknown:
                raise ReviewError(f"审核项不存在：{', '.join(sorted(unknown))}")
            targets = [item for item in record.items if item.id in wanted]

        now = _now()
        for item in targets:
            if action == "reset":
                item.state = PENDING
            elif action == "approve":
                item.state = APPROVED
            else:
                item.state = REJECTED
            item.reviewer = "" if action == "reset" else actor
            item.note = "" if action == "reset" else note[:500]
            item.decided_at = "" if action == "reset" else now

        if action == "reject":
            record.status = REJECTED
        elif action == "reset":
            record.status = PENDING
        elif record.rejected_items:
            record.status = REJECTED
        elif record.pending_items:
            record.status = PENDING
        elif record.validation_passed and record.report_status == "ok":
            record.status = APPROVED
        else:
            # 校验有 error 时不因人工批准而放行：必须修数据、重新生成。
            record.status = PENDING
        _decision(record, action, actor, note, [item.id for item in targets], record.status)
        return _save_record(record, output_dir)


def finalize(
    report_id: str, actor: str, *, output_dir: Path | str | None = None
) -> ReviewRecord:
    """定稿。门禁不通过一律拒绝，不提供「管理员强制放行」的后门。"""
    with _LOCK:
        record = get_record(report_id, output_dir)
        if record is None:
            raise ReviewNotFound(f"没有 report_id={report_id} 的审核记录")
        if record.status == FINALIZED:
            return record
        allowed, reason = gate(record, output_dir)
        if not allowed:
            raise ReviewForbidden(reason)
        record.status = FINALIZED
        record.finalized_by = actor
        record.finalized_at = _now()
        _decision(record, "finalize", actor, "人工定稿", [], FINALIZED)
        return _save_record(record, output_dir)


def gate(record: ReviewRecord | None, output_dir: Path | str | None = None) -> tuple[bool, str]:
    """能否作为定稿交付。返回 (是否放行, 原因)，下载端点与列表共用。"""
    if record is None:
        return False, "没有审核记录，请重新生成研报"
    if not record.validation_passed:
        return False, f"校验未通过（{record.error_count} 条错误），须修数据后重新生成"
    if record.report_status != "ok":
        return False, f"研报状态为 {record.report_status}，没有有效内容可定稿"
    if record.status == REJECTED or record.rejected_items:
        return False, "审核被驳回，不得定稿"
    if record.pending_items:
        return False, f"仍有 {len(record.pending_items)} 条待人工审核"
    if record.status not in {APPROVED, FINALIZED}:
        return False, f"报告状态为 {record.status}，尚未批准定稿"
    if _current_hash(record, output_dir) != record.content_hash:
        return False, "定稿内容已被改动，审批失效"
    if record.status == FINALIZED:
        return True, "已定稿"
    return True, "审核通过，可以定稿"


def record_for_artifact(
    name: str, output_dir: Path | str | None = None
) -> ReviewRecord | None:
    """从产物文件名反查审核记录（report_id 在文件名末尾）。"""
    match = _ID_IN_NAME.search(name)
    if not match:
        return None
    return get_record(match.group(1), output_dir)


def pdf_download_gate(
    name: str, output_dir: Path | str | None = None
) -> tuple[bool, str]:
    """PDF 下载的硬门禁：只有记录里登记的那份、且已定稿，才放行。"""
    record = record_for_artifact(name, output_dir)
    if record is None:
        return False, "这份文件没有审核记录（M2 之前生成或文件名不含 report_id），请重新生成研报"
    if record.artifacts.get("pdf") != name:
        return False, "文件与审核记录不匹配"
    allowed, reason = gate(record, output_dir)
    if not allowed:
        return False, f"报告未定稿：{reason}"
    return True, reason
