import hmac
from urllib.parse import parse_qs

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from ai_agent.core.config import settings


class AuthTokenMiddleware:
    """可选的服务端鉴权。

    配置 AUTH_TOKEN 后，所有 /v1/** 请求必须携带令牌，优先级：
    ``X-API-Key`` 头 > ``Authorization: Bearer <token>`` 头 > ``?api_key=`` 查询参数。
    查询参数是为原生 EventSource（SSE 只支持 GET、不能带 header）准备的兜底，
    令牌会进访问日志，能用 header 就不要用。
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not settings.auth_token:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if not path.startswith("/v1"):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        provided = ""
        if b"x-api-key" in headers:
            provided = headers[b"x-api-key"].decode("latin-1").strip()
        elif b"authorization" in headers:
            auth = headers[b"authorization"].decode("latin-1").strip()
            if auth[:7].lower() == "bearer ":
                provided = auth[7:].strip()

        if not provided:
            query = parse_qs(scope.get("query_string", b"").decode("latin-1"))
            provided = (query.get("api_key") or [""])[0].strip()

        if not provided or not hmac.compare_digest(provided, settings.auth_token):
            # 静态令牌不匹配时，回退尝试登录会话令牌（Bearer 位置），二者任一有效即放行
            from ai_agent.memory.store import store

            if provided and store.resolve_token(provided) is not None:
                await self.app(scope, receive, send)
                return
            response = JSONResponse(status_code=401, content={"detail": "未授权"})
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
