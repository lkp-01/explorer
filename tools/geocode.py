"""封装腾讯位置服务地址解析（地名 -> 经纬度）工具（第 6 步新增工具）。

有了它，用户可以直接说"我在东京塔附近"而不必手输经纬度——这是把 agent 从"能用"
推向"好用"的关键一步。它和 places.py 用的是同一家 provider（腾讯），密钥也复用，
正好示范了"照着已有工具的接口，再加一个新工具"的标准流程。
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from config import config
from tools.registry import register_tool
from utils.parser import clean_text

logger = logging.getLogger(__name__)

TENCENT_GEOCODER_URL = "https://apis.map.qq.com/ws/geocoder/v1/"

GEOCODE_TOOL_DESCRIPTION = """
把一个地名或地址解析为经纬度坐标。参数 query 是地点名称或地址，比如"东京塔""北京国贸"。
当用户用文字描述自己的位置、或希望以某个地标为中心来搜索时，先调用本工具拿到经纬度，
再把经纬度传给天气查询或地点搜索工具。
""".strip()

GEOCODE_TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "要解析的地名或地址"},
    },
    "required": ["query"],
    "additionalProperties": False,
}


@register_tool(
    name="resolve_location",
    description=GEOCODE_TOOL_DESCRIPTION,
    parameters=GEOCODE_TOOL_PARAMETERS,
)
async def resolve_location(query: str) -> dict[str, object]:
    """把地名解析为经纬度，失败时返回结构化错误（与其他工具保持一致的降级约定）。"""

    api_key = config.tencent_map_key
    if not api_key:
        logger.warning("TENCENT_MAP_KEY 未配置，地址解析返回降级结果")
        return {
            "error": "腾讯位置服务 Key 未配置，请在 .env 中设置 TENCENT_MAP_KEY",
            "fallback": True,
            "provider": "tencent",
            "retryable": False,
        }

    params: dict[str, Any] = {"address": query, "key": api_key, "output": "json"}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(TENCENT_GEOCODER_URL, params=params)
            response.raise_for_status()

        payload = response.json()
        status = int(payload.get("status") or 0)
        if status != 0:
            message = clean_text(payload.get("message")) or "地址解析暂时不可用"
            logger.warning("腾讯地址解析失败 status=%s message=%s", status, message)
            return {
                "error": message,
                "fallback": True,
                "provider": "tencent",
                "retryable": False,
            }

        result = payload.get("result") or {}
        location = result.get("location") or {}
        lat = location.get("lat")
        lng = location.get("lng")
        if lat is None or lng is None:
            return {
                "error": f"没能解析出“{query}”的坐标",
                "fallback": True,
                "provider": "tencent",
                "retryable": False,
            }

        return {
            "query": query,
            "lat": float(lat),
            "lng": float(lng),
            "address": clean_text(result.get("title")) or query,
        }
    except httpx.HTTPStatusError as exc:
        logger.warning("腾讯地址解析 HTTP 失败：%s", exc.response.status_code)
    except httpx.RequestError as exc:
        logger.warning("腾讯地址解析请求失败：%s", exc.__class__.__name__)
    except Exception as exc:
        logger.warning("腾讯地址解析暂时不可用：%s", exc.__class__.__name__)

    return {
        "error": "地址解析暂时不可用",
        "fallback": True,
        "provider": "tencent",
        "retryable": True,
    }
