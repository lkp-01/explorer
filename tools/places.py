"""封装高德地图周边地点搜索工具。"""

import logging
import os
from typing import Any

import httpx
from dotenv import load_dotenv

from tools.registry import register_tool

logger = logging.getLogger(__name__)
load_dotenv()

AMAP_AROUND_URL = "https://restapi.amap.com/v5/place/around"

CATEGORY_TYPE_CODES: dict[str, str] = {
    "restaurant": "050000",
    "cafe": "050000",
    "park": "110101",
    "shopping": "060000",
    "attraction": "110200",
    "bar": "050000",
}

PLACES_TOOL_DESCRIPTION = """
搜索指定经纬度附近的城市漫步地点，最多返回 10 个结果。
category 可选值为 restaurant、cafe、park、shopping、attraction、bar。
radius 单位是米，默认 800，最大 5000；如果用户说“走路 10 分钟内”，请设置为约 800。
keyword 用于细化搜索，比如“拉面”“咖啡”“书店”。返回结果包含地点 ID、名称、类别、距离、评分、简介和地址。
""".strip()

PLACES_TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "lat": {"type": "number", "description": "当前位置纬度"},
        "lng": {"type": "number", "description": "当前位置经度"},
        "category": {
            "type": "string",
            "enum": list(CATEGORY_TYPE_CODES.keys()),
            "description": "地点类别",
        },
        "radius": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5000,
            "default": 800,
            "description": "搜索半径，单位为米",
        },
        "keyword": {
            "type": ["string", "null"],
            "description": "可选搜索关键词，用于细化地点类型",
        },
    },
    "required": ["lat", "lng", "category"],
    "additionalProperties": False,
}


def _optional_float(value: object) -> float | None:
    """把接口返回的可选数值转换为 float。"""

    if value in (None, "", "[]"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_or_zero(value: object) -> float:
    """把接口返回的距离转换为 float，失败时返回 0。"""

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _text(value: object) -> str:
    """把高德接口中可能出现的空数组或空值转换为文本。"""

    if value in (None, "", "[]") or isinstance(value, list):
        return ""
    return str(value)


@register_tool(
    name="search_places",
    description=PLACES_TOOL_DESCRIPTION,
    parameters=PLACES_TOOL_PARAMETERS,
)
async def search_places(
    lat: float,
    lng: float,
    category: str,
    radius: int = 800,
    keyword: str | None = None,
) -> list[dict[str, object]]:
    """根据经纬度、类别和关键词搜索附近地点，失败时返回空列表。"""

    api_key = os.getenv("AMAP_API_KEY")
    if not api_key:
        logger.warning("AMAP_API_KEY 未配置，地点工具返回空列表")
        return []

    type_code = CATEGORY_TYPE_CODES.get(category)
    if type_code is None:
        logger.warning("未知地点类别: %s", category)
        return []

    bounded_radius = max(1, min(int(radius), 5000))
    params: dict[str, Any] = {
        "key": api_key,
        "location": f"{lng},{lat}",
        "radius": str(bounded_radius),
        "types": type_code,
        "page_size": "10",
        "show_fields": "business",
        "output": "json",
    }
    if keyword:
        params["keywords"] = keyword

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(AMAP_AROUND_URL, params=params)
            response.raise_for_status()

        payload = response.json()
        if payload.get("status") != "1":
            logger.warning(
                "高德周边搜索失败 infocode=%s info=%s",
                payload.get("infocode"),
                payload.get("info"),
            )
            return []

        results: list[dict[str, object]] = []
        for poi in payload.get("pois") or []:
            name = _text(poi.get("name"))
            if not name:
                continue

            business = poi.get("business")
            if not isinstance(business, dict):
                business = {}

            address = _text(poi.get("address"))
            poi_type = _text(poi.get("type"))
            brief = address or poi_type or "暂无简介"

            results.append(
                {
                    "place_id": _text(poi.get("id")),
                    "name": name,
                    "category": category,
                    "distance_meters": _float_or_zero(poi.get("distance")),
                    "rating": _optional_float(business.get("rating")),
                    "brief": brief,
                    "address": address,
                }
            )

        return results[:10]
    except Exception:
        logger.warning("高德周边搜索暂时不可用", exc_info=True)
        return []
