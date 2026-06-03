"""封装腾讯位置服务周边地点搜索工具。"""

import asyncio
import logging
from typing import Any

import httpx

from config import config
from tools.registry import register_tool
from utils.parser import clean_text, float_or_zero

logger = logging.getLogger(__name__)

TENCENT_PLACE_SEARCH_URL = "https://apis.map.qq.com/ws/place/v1/search"
TENCENT_DAILY_LIMIT_STATUS = 121

# 并发闸门（阶段五）：阶段五的并行框架会同时跑多个子任务，各自调用 search_places，
# 对腾讯接口形成并发请求。用一个信号量把同时在飞的请求数压到配置上限以内，
# 既保护日调用额度，也避免短时打爆触发限流。单点/路线等非并发路径几乎不受影响。
_TENCENT_GATE = asyncio.Semaphore(config.tencent_max_concurrency)

CATEGORY_KEYWORDS: dict[str, str] = {
    "restaurant": "餐厅",
    "cafe": "咖啡",
    "park": "公园",
    "shopping": "商场",
    "attraction": "景点",
    "bar": "酒吧",
}

PLACES_TOOL_DESCRIPTION = """
搜索指定经纬度附近的城市漫步地点，最多返回 10 个结果。
category 可选值为 restaurant、cafe、park、shopping、attraction、bar。
radius 单位是米，默认 800，最大 1000；如果用户说"走路 10 分钟内"，请设置为约 800。
keyword 用于细化搜索，比如"拉面""咖啡""书店"。返回结果包含地点 ID、名称、类别、距离、简介和地址。
""".strip()

PLACES_TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "lat": {"type": "number", "description": "当前位置纬度"},
        "lng": {"type": "number", "description": "当前位置经度"},
        "category": {
            "type": "string",
            "enum": list(CATEGORY_KEYWORDS.keys()),
            "description": "地点类别",
        },
        "radius": {
            "type": "integer",
            "minimum": 1,
            "maximum": 1000,
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
) -> list[dict[str, object]] | dict[str, object]:
    """根据经纬度、类别和关键词搜索附近地点，失败时返回结构化错误。"""

    api_key = config.tencent_map_key
    if not api_key:
        logger.warning("TENCENT_MAP_KEY 未配置，地点工具返回降级结果")
        return {
            "error": "腾讯位置服务 Key 未配置，请在 .env 中设置 TENCENT_MAP_KEY",
            "fallback": True,
            "provider": "tencent",
            "status": "missing_key",
            "retryable": False,
        }

    default_keyword = CATEGORY_KEYWORDS.get(category)
    if default_keyword is None:
        logger.warning("未知地点类别: %s", category)
        return []

    bounded_radius = max(10, min(int(radius), 1000))
    params: dict[str, Any] = {
        "key": api_key,
        "keyword": keyword or default_keyword,
        "boundary": f"nearby({lat},{lng},{bounded_radius},0)",
        "page_size": "10",
        "page_index": "1",
        "orderby": "_distance",
        "output": "json",
    }

    try:
        # 信号量限并发：超出上限的搜索请求在此排队，待前面的放行后再发，避免并发触顶额度。
        async with _TENCENT_GATE:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(TENCENT_PLACE_SEARCH_URL, params=params)
                response.raise_for_status()

        payload = response.json()
        status = int(payload.get("status") or 0)
        if status != 0:
            message = str(payload.get("message") or "腾讯周边搜索暂时不可用")
            logger.warning("腾讯周边搜索失败 status=%s message=%s", status, message)
            user_message = (
                "腾讯位置服务今日查询额度已用完"
                if status == TENCENT_DAILY_LIMIT_STATUS
                else message
            )
            return {
                "error": user_message,
                "fallback": True,
                "provider": "tencent",
                "status": str(status),
                "message": message,
                "retryable": False,
            }

        results: list[dict[str, object]] = []
        for poi in payload.get("data") or []:
            name = clean_text(poi.get("title"))
            if not name:
                continue

            address = clean_text(poi.get("address"))
            poi_category = clean_text(poi.get("category"))
            brief = address or poi_category or "暂无简介"

            results.append(
                {
                    "place_id": clean_text(poi.get("id")),
                    "name": name,
                    "category": category,
                    "distance_meters": float_or_zero(poi.get("_distance")),
                    "rating": None,
                    "brief": brief,
                    "address": address,
                    "provider_category": poi_category,
                }
            )

        return results[:10]
    except httpx.HTTPStatusError as exc:
        logger.warning("腾讯周边搜索 HTTP 失败：%s", exc.response.status_code)
    except httpx.RequestError as exc:
        logger.warning("腾讯周边搜索请求失败：%s", exc.__class__.__name__)
    except Exception as exc:
        logger.warning("腾讯周边搜索暂时不可用：%s", exc.__class__.__name__)

    return {
        "error": "腾讯周边搜索暂时不可用",
        "fallback": True,
        "provider": "tencent",
        "retryable": True,
    }
