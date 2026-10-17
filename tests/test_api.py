from unittest.mock import patch

from fastapi.testclient import TestClient

from ai_agent.api.app import app


def test_chat_sse():
    def fake_stream(user_input: str):
        yield {"type": "text", "content": "你"}
        yield {"type": "text", "content": "好"}
        yield {"type": "done"}

    with patch("ai_agent.api.routers.chat.run_stream", side_effect=fake_stream):
        client = TestClient(app)
        with client.stream("POST", "/v1/chat", json={"message": "hi"}) as response:
            body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert '"content": "你"' in body
    assert '"content": "好"' in body
    assert '"type": "done"' in body
