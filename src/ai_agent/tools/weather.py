"""天气查询工具。

数据源 Open-Meteo（https://open-meteo.com）：免费、无需 API key，
全球覆盖。城市名先经其 geocoding 接口转经纬度，再查实时天气与
每日预报。零第三方依赖：两次 REST 调用走 core.http 共享客户端
（与 web_search / gate 同策略）。

错误处理：execute_tool 会把工具异常吞成通用 ERROR，模型拿不到原因，
所以这里把「城市找不到」「网络失败」主动转成带原因的 ERROR 文案返回。
"""

from pydantic import BaseModel, Field

from ai_agent.core import http
from ai_agent.core.logging import get_logger
from ai_agent.tools.base import Tool

logger = get_logger(__name__)

_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO 天气现象代码 → 中文描述（Open-Meteo 文档标准码表）
_WMO = {
    0: "晴",
    1: "基本晴",
    2: "局部多云",
    3: "阴",
    45: "雾",
    48: "雾凇",
    51: "毛毛雨",
    53: "毛毛雨",
    55: "浓毛毛雨",
    56: "冻毛毛雨",
    57: "浓冻毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "冻雨",
    67: "大冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "雪粒",
    80: "小阵雨",
    81: "阵雨",
    82: "强阵雨",
    85: "小阵雪",
    86: "大阵雪",
    95: "雷阵雨",
    96: "雷阵雨伴冰雹",
    99: "强雷阵雨伴冰雹",
}


class WeatherArgs(BaseModel):
    city: str = Field(
        description="城市名称，可用中文。重名城市可带国家或省份，如「剑桥 英国」"
    )
    days: int = Field(
        default=1,
        ge=1,
        le=7,
        description="预报天数：1=只看今天，2=到明天，最多 7 天。用户问周末/未来安排时适当加大",
    )


def _geocode(city: str) -> dict | None:
    """城市名 → 经纬度等位置信息；找不到返回 None。"""
    url = f"{_GEOCODING_URL}?name={http.quote(city)}&count=1&language=zh&format=json"
    results = http.request_json(url).get("results") or []
    return results[0] if results else None


def _fetch_forecast(lat: float, lon: float, days: int) -> dict:
    url = (
        f"{_FORECAST_URL}?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,apparent_temperature,relative_humidity_2m,"
        "weather_code,wind_speed_10m"
        "&daily=weather_code,temperature_2m_max,temperature_2m_min,"
        "precipitation_probability_max"
        f"&timezone=auto&forecast_days={days}"
    )
    return http.request_json(url)


def _describe(code: object) -> str:
    return _WMO.get(code, "未知") if isinstance(code, (int, float)) else "未知"


def get_weather(city: str, days: int = 1) -> str:
    try:
        location = _geocode(city)
    except Exception as e:
        logger.warning("天气地理编码失败 city=%s: %s", city, e)
        return f"ERROR: 天气查询失败（{e}），请稍后重试。"
    if location is None:
        return f"ERROR: 未找到城市「{city}」的位置，请确认名称，重名城市可带国家或省份（如「剑桥 英国」）。"

    place = location["name"]
    region = "、".join(
        part for part in (location.get("admin1"), location.get("country")) if part
    )
    try:
        data = _fetch_forecast(
            location["latitude"], location["longitude"], days
        )
    except Exception as e:
        logger.warning("天气查询失败 city=%s: %s", city, e)
        return f"ERROR: 天气查询失败（{e}），请稍后重试。"

    current = data.get("current") or {}
    if not current:
        return f"ERROR: 天气数据为空（{place}）。"
    lines = [
        f"{place}（{region}）当前：{_describe(current.get('weather_code'))}，"
        f"气温 {current.get('temperature_2m')}°C"
        f"（体感 {current.get('apparent_temperature')}°C），"
        f"湿度 {current.get('relative_humidity_2m')}%，"
        f"风速 {current.get('wind_speed_10m')} m/s。"
    ]

    daily = data.get("daily") or {}
    dates = daily.get("time") or []
    labels = ["今日", "明日"] + [f"第{i + 1}天" for i in range(2, len(dates))]
    for i, date in enumerate(dates):
        desc = _describe((daily.get("weather_code") or [None] * len(dates))[i])
        t_max = (daily.get("temperature_2m_max") or [None] * len(dates))[i]
        t_min = (daily.get("temperature_2m_min") or [None] * len(dates))[i]
        rain = (daily.get("precipitation_probability_max") or [None] * len(dates))[i]
        rain_part = f"，降水概率 {rain}%" if rain is not None else ""
        lines.append(f"{labels[i]}（{date[5:]}）：{desc}，{t_min}~{t_max}°C{rain_part}。")
    return "\n".join(lines)


weather_tool = Tool(
    name="get_weather",
    display_name="天气查询",
    description=(
        "查询指定城市的实时天气和每日预报（温度、体感、湿度、风速、降水概率）。"
        "用户问天气、穿什么、要不要带伞、出行安排时调用。"
    ),
    func=get_weather,
    args_model=WeatherArgs,
)
