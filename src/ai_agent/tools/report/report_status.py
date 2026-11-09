"""聊天侧的研报任务查询：generate_report 提交后台任务后，用它取进度与结果。

为什么单独一个工具：深度层 5–8 分钟，同步等待会把 SSE 拖到网关超时。提交即返回
job_id 后，聊天流几秒结束；用户问进度/要结果时再查这里，取回与同步链路**逐字
相同**的摘要与下载链接。

边界：

* 只读，不触发任何模型调用或流水线；
* 任务表是内存表，按 owner 隔离：查到别人的 job_id 一律当不存在，不回显正文；
* 进程重启后任务表丢失，但已落盘的报告仍可从审核队列 / 文件端点取回，文案里
  要如实说明这一点。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ai_agent.prompts.tools import REPORT_JOB_STATUS_DESC
from ai_agent.report.jobs import job_manager
from ai_agent.storage.db import current_user_id
from ai_agent.tools.base import Tool

# 失败原因只前缀这么多字符，避免把堆栈/路径整段塞回模型上下文
_MAX_ERROR_CHARS = 300
_MISSING = (
    "没有找到该 job_id 对应的研报任务（任务表只在内存里，服务重启会丢失）。"
    "不要重复提交同一份研报；已生成的文件可以从 /v1/report/reviews 审核队列找回，"
    "请如实告知用户「任务记录已失效，但报告文件可能仍在审核队列」。"
)


class ReportJobStatusArgs(BaseModel):
    job_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="generate_report 返回的 job_id，原样传入，不要改写",
    )


def _normalize_job_id(raw: str) -> str:
    """模型常把整句 "job_id=abc123" 原样传回来，这里兜底剥掉前缀。"""
    text = (raw or "").strip().strip("`'\"")
    if text.lower().startswith("job_id"):
        text = text.split("=", 1)[1].strip() if "=" in text else text
    return text.strip().strip("`'\"")


def _report_job_status(job_id: str) -> str:
    job = job_manager.get(_normalize_job_id(job_id), current_user_id())
    if job is None:
        return _MISSING
    if job.status == "succeeded":
        payload = job.result if isinstance(job.result, dict) else {}
        text = str(payload.get("text") or "").strip()
        if text:
            return text
        return (
            f"任务 {job.job_id} 已完成，但没有可转述的结果文本；"
            "请让用户到 /v1/report/reviews 审核队列查看产物。"
        )
    if job.status == "failed":
        return (
            f"研报生成失败（job_id={job.job_id}）：{job.error[:_MAX_ERROR_CHARS]}\n"
            "请如实告知用户失败原因，不要补写正文、不要说「稍后自动完成」。"
        )
    if job.status == "queued":
        return (
            f"任务在排队（前面还有在跑的研报任务，已等 {job.elapsed_seconds():.0f} 秒，"
            f"job_id={job.job_id}）。请如实告诉用户「排队中」，不要当成失败。"
        )
    return (
        f"任务仍在生成：{job.stage}（进度 {job.progress}%，"
        f"已用时 {job.elapsed_seconds():.0f} 秒，job_id={job.job_id}）。"
        "请如实告诉用户还在生成、稍后再查；不要说失败，也不要自己撰写正文或补数。"
    )


report_job_status_tool = Tool(
    name="report_job_status",
    description=REPORT_JOB_STATUS_DESC,
    func=_report_job_status,
    args_model=ReportJobStatusArgs,
)
