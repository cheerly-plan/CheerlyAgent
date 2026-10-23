from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ai_agent.api.middleware.auth import AuthTokenMiddleware
from ai_agent.api.middleware.request_id import RequestIdMiddleware
from ai_agent.api.routers.auth import router as auth_router
from ai_agent.api.routers.chat import router as chat_router
from ai_agent.api.routers.health import router as health_router
from ai_agent.api.routers.session import router as session_router
from ai_agent.api.routers.user import router as user_router
from ai_agent.core.exceptions import AgentError
from ai_agent.core.logging import setup_logging
from ai_agent.memory.short_term import short_term_memory
from ai_agent.memory.store import store
from ai_agent.tasks.cleanup import empty_session_cleaner
from ai_agent.tasks.title import shutdown_title_jobs


@asynccontextmanager
async def lifespan(_app: FastAPI):
    store.ensure_schema()
    short_term_memory.start_cleanup()
    empty_session_cleaner.start()
    yield
    empty_session_cleaner.stop()
    short_term_memory.stop_cleanup()
    shutdown_title_jobs()
    store.close()


def create_app() -> FastAPI:
    setup_logging()
    app = FastAPI(title="almost-agent", lifespan=lifespan)
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(AuthTokenMiddleware)
    app.add_middleware(
        CORSMiddleware,
            allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(user_router)
    app.include_router(session_router)
    app.include_router(chat_router)

    @app.exception_handler(AgentError)
    async def agent_error(_request: Request, exc: AgentError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    return app


app = create_app()