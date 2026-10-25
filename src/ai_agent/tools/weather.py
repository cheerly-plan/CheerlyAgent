from pydantic import BaseModel, Field

from ai_agent.tools.base import Tool


class WeatherArgs(BaseModel):
    city: str = Field(description="城市名称")


def get_weather(city: str) -> str:
    return f"今天{city}天气晴朗，气温20度。"


weather_tool = Tool(
    name="get_weather",
    display_name="天气查询",
    description="获取天气信息",
    func=get_weather,
    args_model=WeatherArgs,
)