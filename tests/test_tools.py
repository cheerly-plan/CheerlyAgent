from types import SimpleNamespace
from unittest.mock import patch

from ai_agent.tools.registry import execute_tool, tool_schemas
from ai_agent.tools import weather as wt


def _fake_geocode(name="北京"):
    return {
        "name": name,
        "admin1": "北京市" if name == "北京" else "",
        "country": "中国",
        "latitude": 39.9,
        "longitude": 116.4,
    }


def _fake_forecast(days=2):
    return {
        "current": {
            "temperature_2m": 20.5,
            "apparent_temperature": 19.0,
            "relative_humidity_2m": 45,
            "weather_code": 2,
            "wind_speed_10m": 3.2,
        },
        "daily": {
            "time": ["2026-10-25", "2026-10-26"][:days],
            "weather_code": [3, 61][:days],
            "temperature_2m_max": [22.0, 18.0][:days],
            "temperature_2m_min": [12.0, 10.0][:days],
            "precipitation_probability_max": [10, 80][:days],
        },
    }


def test_get_weather_formats_report():
    with (
        patch.object(wt, "_geocode", return_value=_fake_geocode()),
        patch.object(wt, "_fetch_forecast", return_value=_fake_forecast()),
    ):
        out = wt.get_weather("北京")
    assert "北京（北京市、中国）" in out
    assert "气温 20.5°C" in out
    assert "局部多云" in out
    assert "今日（10-25）：阴，12.0~22.0°C，降水概率 10%。" in out


def test_get_weather_multi_day():
    with (
        patch.object(wt, "_geocode", return_value=_fake_geocode()),
        patch.object(wt, "_fetch_forecast", return_value=_fake_forecast(days=2)) as fetch,
    ):
        out = wt.get_weather("北京", days=2)
    assert "明日（10-26）：小雨，10.0~18.0°C，降水概率 80%。" in out
    assert fetch.call_args.args[2] == 2


def test_get_weather_city_not_found():
    with patch.object(wt, "_geocode", return_value=None):
        out = wt.get_weather("不存在的城")
    assert out.startswith("ERROR:")
    assert "未找到城市" in out


def test_get_weather_network_error():
    with patch.object(wt, "_geocode", side_effect=TimeoutError("timeout")):
        out = wt.get_weather("北京")
    assert out.startswith("ERROR:")
    assert "天气查询失败" in out


def test_schemas_include_weather():
    names = [item["function"]["name"] for item in tool_schemas()]
    assert "get_weather" in names


def test_execute_weather():
    with (
        patch.object(wt, "_geocode", return_value=_fake_geocode("上海")),
        patch.object(wt, "_fetch_forecast", return_value=_fake_forecast()),
    ):
        result = execute_tool("get_weather", {"city": "上海"})
    assert "上海" in result


def test_execute_unknown_tool():
    result = execute_tool("not_exist", {})
    assert result.startswith("ERROR:")


def test_execute_missing_city():
    result = execute_tool("get_weather", {})
    assert result.startswith("ERROR:")


def test_write_memory_profile_kind():
    with (
        patch(
            "ai_agent.tools.memory.apply_profile_text",
            return_value="已写入用户画像: 喜欢 pytest",
        ) as apply_profile,
        patch("ai_agent.tools.memory.store.write_episode") as write_episode,
    ):
        result = execute_tool("write_memory", {"content": "喜欢 pytest", "kind": "profile"})
    assert "用户画像" in result
    apply_profile.assert_called_once_with("喜欢 pytest")
    write_episode.assert_not_called()


def test_write_memory_defaults_to_profile():
    with (
        patch(
            "ai_agent.tools.memory.apply_profile_text",
            return_value="已写入用户画像: 喜欢 pytest",
        ) as apply_profile,
        patch("ai_agent.tools.memory.store.write_episode") as write_episode,
    ):
        result = execute_tool("write_memory", {"content": "喜欢 pytest"})
    assert "用户画像" in result
    apply_profile.assert_called_once_with("喜欢 pytest")
    write_episode.assert_not_called()


def test_write_memory_episode_kind():
    item = SimpleNamespace(text="上周修过登录")
    with (
        patch("ai_agent.tools.memory.store.write_profile") as write_profile,
        patch("ai_agent.tools.memory.store.write_episode", return_value=item) as write_episode,
    ):
        result = execute_tool("write_memory", {"content": "上周修过登录", "kind": "episode"})
    assert "情景记忆" in result
    write_episode.assert_called_once_with("上周修过登录")
    write_profile.assert_not_called()


def test_search_memory_uses_episode_store():
    item = SimpleNamespace(text="上周修过登录")
    with patch("ai_agent.tools.memory.store.search_episode", return_value=[item]) as search:
        result = execute_tool("search_memory", {"query": "登录"})
    assert "上周修过登录" in result
    search.assert_called_once_with("登录")
