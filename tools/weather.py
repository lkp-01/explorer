"""封装和风天气实时天气查询工具。"""

import logging
import os
from typing import Any

import httpx
from dotenv import load_dotenv

from tools.registry import register_tool

logger = logging.getLogger(__name__)
load_dotenv()

DEFAULT_QWEATHER_API_HOST = "devapi.qweather.com"
QWEATHER_NOW_PATH = "/v7/weather/now"

WEATHER_TOOL_DESCRIPTION = """
查询指定经纬度位置的实时天气。参数 lat 是纬度，lng 是经度。
工具会调用和风天气实时天气接口，返回天气现象、温度、湿度和风速。
推荐城市漫步地点前应先调用本工具，雨天需要优先考虑室内场所。
""".strip()

WEATHER_TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "lat": {"type": "number", "description": "当前位置纬度"},
        "lng": {"type": "number", "description": "当前位置经度"},
    },
    "required": ["lat", "lng"],
    "additionalProperties": False,
}

WEATHER_FALLBACK: dict[str, object] = {
    "error": "天气服务暂时不可用",
    "fallback": True,
}


def _get_weather_url() -> str:
    """根据 .env 中的 QWEATHER_API_HOST 生成实时天气接口地址。"""

    host = os.getenv("QWEATHER_API_HOST") or DEFAULT_QWEATHER_API_HOST
    host = host.strip().removeprefix("https://").removeprefix("http://").rstrip("/")
    return f"https://{host}{QWEATHER_NOW_PATH}"


@register_tool(
    name="get_weather",
    description=WEATHER_TOOL_DESCRIPTION,
    parameters=WEATHER_TOOL_PARAMETERS,
)
async def get_weather(lat: float, lng: float) -> dict[str, object]:
    """根据经纬度查询实时天气，失败时返回友好的降级结果。"""

    api_key = os.getenv("QWEATHER_API_KEY")
    if not api_key:
        logger.warning("QWEATHER_API_KEY 未配置，天气工具返回降级结果")
        return WEATHER_FALLBACK.copy()

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                _get_weather_url(),
                params={"location": f"{lng},{lat}", "key": api_key},
            )
            response.raise_for_status()

        payload = response.json()
        if payload.get("code") != "200":
            logger.warning("和风天气接口返回失败 code=%s", payload.get("code"))
            return WEATHER_FALLBACK.copy()

        now = payload.get("now") or {}
        return {
            "condition": str(now["text"]),
            "temp": float(now["temp"]),
            "humidity": int(now["humidity"]) if now.get("humidity") else None,
            "wind_speed": float(now["windSpeed"]),
        }
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "天气服务暂时不可用，HTTP 状态码：%s",
            exc.response.status_code,
        )
        return WEATHER_FALLBACK.copy()
    except httpx.RequestError as exc:
        logger.warning("天气服务请求失败：%s", exc.__class__.__name__)
        return WEATHER_FALLBACK.copy()
    except Exception as exc:
        logger.warning("天气服务暂时不可用：%s", exc.__class__.__name__)
        return WEATHER_FALLBACK.copy()
