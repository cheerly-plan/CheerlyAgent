"""工具结构化事件：通过同一队列下发成独立 SSE 事件，不混进进度文案。"""

from __future__ import annotations

import ai_agent.agents.loop as loop
from ai_agent.tools.progress import emit_event, emit_progress


def test_structured_event_is_forwarded_as_its_own_sse_event(monkeypatch):
    def fake_execute(_name: str, _arguments: dict, **_kwargs) -> str:
        emit_progress("正在提交研报任务…")
        emit_event({"type": "report_job", "job_id": "abc12345", "poll_url": "/v1/report/jobs/abc12345"})
        return "工具正文"

    monkeypatch.setattr(loop, "execute_tool", fake_execute)
    events = list(
        loop._run_tool_with_progress(
            "generate_report",
            {},
            user_id="u1",
            kb_file_ids=None,
            messages=[],
        )
    )
    kinds = [event.get("type") for event in events]
    # 进度文案仍走 tool_progress；结构化事件原样下发；结果照旧最后一个
    assert "tool_progress" in kinds
    assert "report_job" in kinds
    assert kinds[-1] == "_tool_result"
    job_event = next(event for event in events if event.get("type") == "report_job")
    assert job_event["job_id"] == "abc12345"
    result = events[-1]
    assert result["content"] == "工具正文"


def test_emit_event_without_subscriber_is_silent():
    emit_event({"type": "report_job", "job_id": "nobody-listening"})
    emit_event({})  # 缺 type 直接丢弃，不抛错
