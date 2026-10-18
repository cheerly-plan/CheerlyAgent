import json
from collections.abc import Iterator

from ai_agent.agents.loop import run_stream
from ai_agent.api.schemas import ChatRequest
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/v1", tags=["chat"])


def _sse(req: ChatRequest) -> Iterator[str]:
    for event in run_stream(
        req.message, user_id=req.user_id, session_id=req.resolved_session_id()
    ):
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _sse(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
