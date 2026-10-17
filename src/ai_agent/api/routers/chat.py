import json
from collections.abc import Iterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ai_agent.agents.loop import run_stream
from ai_agent.api.schemas import ChatRequest

router = APIRouter(prefix="/v1", tags=["chat"])


def _sse(message: str) -> Iterator[str]:
    for event in run_stream(message):
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _sse(req.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
