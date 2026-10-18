from types import SimpleNamespace
from unittest.mock import patch

from ai_agent.tools.registry import execute, schemas
from ai_agent.tools.weather import get_weather


def test_get_weather():
    assert "北京" in get_weather("北京")
    assert "20度" in get_weather("北京")


def test_schemas_include_weather():
    names = [item["function"]["name"] for item in schemas()]
    assert "get_weather" in names


def test_execute_weather():
    result = execute("get_weather", {"city": "上海"})
    assert "上海" in result


def test_execute_unknown_tool():
    result = execute("not_exist", {})
    assert result.startswith("ERROR:")


def test_execute_missing_city():
    result = execute("get_weather", {})
    assert result.startswith("ERROR:")


def test_write_memory_profile_kind():
    item = SimpleNamespace(text="喜欢 pytest")
    with (
        patch("ai_agent.tools.memory.store.write_profile", return_value=item) as write_profile,
        patch("ai_agent.tools.memory.store.write") as write,
    ):
        result = execute("write_memory", {"content": "喜欢 pytest", "kind": "profile"})
    assert "用户画像" in result
    write_profile.assert_called_once_with("喜欢 pytest")
    write.assert_not_called()


def test_write_memory_defaults_to_episode():
    item = SimpleNamespace(text="上周修过登录")
    with (
        patch("ai_agent.tools.memory.store.write_profile") as write_profile,
        patch("ai_agent.tools.memory.store.write", return_value=item) as write,
    ):
        result = execute("write_memory", {"content": "上周修过登录"})
    assert "情景记忆" in result
    write.assert_called_once_with("上周修过登录")
    write_profile.assert_not_called()
