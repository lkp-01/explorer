"""负责把工具调用结果同步回结构化 AgentState。"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent.state import AgentState, Candidate, WeatherInfo

logger = logging.getLogger(__name__)


def _optional_float(value: object) -> float | None:
    """把可选数值转换为 float。"""

    if value in (None, "", "[]"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_or_zero(value: object) -> float:
    """把数值转换为 float，失败时返回 0。"""

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _sync_weather_state(state: AgentState, payload: dict[str, Any]) -> None:
    """把天气工具结果同步到 AgentState。"""

    if payload.get("error"):
        return

    try:
        humidity = payload.get("humidity")
        state.weather = WeatherInfo(
            condition=str(payload["condition"]),
            temp=float(payload["temp"]),
            humidity=int(humidity) if humidity is not None else None,
        )
    except (KeyError, TypeError, ValueError):
        logger.warning("天气结果无法写入状态: %s", payload)


def _sync_candidates_state(
    state: AgentState,
    arguments: dict[str, Any],
    payload: list[Any],
) -> None:
    """把地点搜索结果同步到 AgentState 的候选列表。"""

    source_query = arguments.get("keyword") or arguments.get("category")
    by_place_id = {candidate.place_id: candidate for candidate in state.candidates}

    for item in payload:
        if not isinstance(item, dict):
            continue

        place_id = str(item.get("place_id") or "")
        name = str(item.get("name") or "")
        if not place_id or not name or place_id in state.rejected_candidates:
            continue

        try:
            by_place_id[place_id] = Candidate(
                place_id=place_id,
                name=name,
                category=str(item.get("category") or arguments.get("category") or ""),
                distance_meters=_float_or_zero(item.get("distance_meters")),
                rating=_optional_float(item.get("rating")),
                brief=str(item.get("brief") or ""),
                source_query=str(source_query) if source_query else None,
            )
        except (TypeError, ValueError):
            logger.warning("地点结果无法写入状态: %s", item)

    state.candidates = list(by_place_id.values())


def sync_state_from_tool_result(
    state: AgentState,
    tool_name: str,
    arguments: dict[str, Any],
    result_json: str,
) -> None:
    """把已知工具的 JSON 结果同步到结构化状态。"""

    try:
        payload = json.loads(result_json)
    except json.JSONDecodeError:
        logger.warning("工具结果不是合法 JSON: %s", result_json)
        return

    if tool_name == "get_weather" and isinstance(payload, dict):
        _sync_weather_state(state, payload)
    elif tool_name == "search_places" and isinstance(payload, list):
        _sync_candidates_state(state, arguments, payload)
