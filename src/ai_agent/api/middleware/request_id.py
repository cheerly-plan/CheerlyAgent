import re
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

# 只接受安全的 request id，防止把异常字符回写进响应头
_REQUEST_ID_RE = re.compile(r"^[\w.-]{1,64}$")


class RequestIdMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = uuid.uuid4().hex
        for key, value in scope.get("headers", []):
            if key == b"x-request-id" and value:
                raw = value.decode("ascii", errors="ignore").strip()
                if _REQUEST_ID_RE.fullmatch(raw):
                    request_id = raw
                break

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_id)
