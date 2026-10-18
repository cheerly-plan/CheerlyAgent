from pydantic import BaseModel, Field


class UserInfo(BaseModel):
    user_id: str
    sessionsid: str
    user_name: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    user_id: str = Field(default="default")
    session_id: str = "default"
    sessionsid: str | None = None

    def resolved_session_id(self) -> str:
        return self.sessionsid or self.session_id
