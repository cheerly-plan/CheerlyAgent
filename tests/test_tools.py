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
