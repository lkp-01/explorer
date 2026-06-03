"""Plan-and-Solve 路线规划（阶段二）：先成计划，再逐站执行。

和现有 ReAct"走一步看一步"互补：
- generate_plan：让 LLM 一次性产出结构化路线计划（JSON），这一步不调任何工具。
- execute_plan：按计划逐站调用搜索工具，把每一站回填成真实地点，最后再让 LLM 合成回复。
- run_route_plan：把上面两步串起来，并把成形的路线写回 state，供持久化与后续轮次使用。

刻意把"计划"和"执行"拆开：计划是纯 JSON，能被序列化给用户看、被用户否决修改——
这正是阶段三 Reflexion（反馈学习）要接的口子。
"""

from __future__ import annotations

import json
import logging

from agent.messages import extract_text
from agent.prompts import (
    PLAN_GENERATION_PROMPT,
    ROUTE_SYNTHESIS_PROMPT,
    format_preference_block,
)
from agent.state import AgentState, RoutePlan
from agent.state_sync import sync_state_from_tool_result
from models.client import LLMClient
from tools.registry import dispatch
from utils.parser import safe_json_loads
from utils.parser import strip_json_fence as _strip_json_fence

logger = logging.getLogger(__name__)

# 单站搜索半径：路线场景给得比单点推荐略大一些，给排序留出余地。
_ROUTE_SEARCH_RADIUS = 1000


def _plan_context(state: AgentState) -> str:
    """把规划需要的位置、天气、时间约束、以及已知偏好/否决压缩成一段上下文喂给 LLM。

    注意只带规划相关字段（不带 candidates / 历史），让计划阶段的 prompt 尽量短、聚焦。
    阶段三起额外带上偏好与本会话否决，让排路线时就直接避开用户不想要的类别/地点。
    """

    snapshot = state.model_dump(
        mode="json",
        include={"location", "weather", "time_constraints"},
    )
    context = (
        "当前位置与天气（供你规划，不必复述）：\n"
        f"{json.dumps(snapshot, ensure_ascii=False, default=str)}"
    )

    # 阶段三：把长期偏好 + 本会话否决拼进规划上下文，让计划阶段就遵守
    constraints = format_preference_block(state.preferences, state.session_feedback)
    if constraints:
        context = f"{context}\n\n{constraints}"
    return context


async def generate_plan(
    client: LLMClient,
    state: AgentState,
    user_message: str,
) -> RoutePlan:
    """第一段：让 LLM 产出结构化路线计划。失败时返回空计划，由执行阶段兜底。"""

    messages = [
        {"role": "system", "content": PLAN_GENERATION_PROMPT},
        {"role": "user", "content": f"{_plan_context(state)}\n\n用户需求：{user_message}"},
    ]

    # 计划阶段不传 tools：这一步只要 JSON，不需要模型去调天气/地点。
    response = await client.chat(messages)
    raw = extract_text(response.choices[0].message)

    payload = safe_json_loads(_strip_json_fence(raw))
    if not isinstance(payload, dict):
        logger.warning("路线计划不是合法 JSON 对象，使用空计划: %s", raw)
        return RoutePlan()

    try:
        # pydantic 负责把 JSON 校验成 RoutePlan；字段缺失或类型不对就抛异常。
        plan = RoutePlan.model_validate(payload)
    except Exception:
        logger.warning("路线计划字段不合法，使用空计划: %s", payload)
        return RoutePlan()

    logger.info("route_plan_generated steps=%s summary=%s", len(plan.steps), plan.summary)
    return plan


async def _ensure_weather(state: AgentState) -> None:
    """执行前确认天气：缺了就调一次天气工具补上，用于室内/室外取舍与最终提示。"""

    if state.weather is not None or state.location is None:
        return

    result_json = await dispatch(
        "get_weather",
        {"lat": state.location.lat, "lng": state.location.lng},
    )
    # 复用既有的状态同步逻辑，把天气结果写回 state.weather
    sync_state_from_tool_result(state, "get_weather", {}, result_json)


async def _fill_step_with_place(
    state: AgentState,
    step,  # RouteStep；不标注类型是为了避免在签名里反复 import
    used_place_ids: set[str],
) -> None:
    """给计划里的一站搜索真实地点，并回填到该 step。

    用 used_place_ids 记录已选地点，避免整条路线里两站撞到同一家店。
    搜索失败（如未配 Key、额度用完）时，这一站就保持"只有计划、没有实际地点"，不影响其他站。
    """

    assert state.location is not None  # 调用方已保证有位置，这里给类型检查一个提示
    arguments = {
        "lat": state.location.lat,
        "lng": state.location.lng,
        "category": step.category,
        "radius": _ROUTE_SEARCH_RADIUS,
    }
    if step.keyword:
        arguments["keyword"] = step.keyword

    result_json = await dispatch("search_places", arguments)
    # 顺便把搜索结果同步进 candidates，复用既有逻辑，状态保持一致
    sync_state_from_tool_result(state, "search_places", arguments, result_json)

    places = safe_json_loads(result_json)
    if not isinstance(places, list):
        return  # 返回的是错误对象（dict）而非列表，说明搜索没成功，这一站留空

    # 选第一个还没被前面占用的地点（搜索已按距离排序，第一个通常最近）
    for place in places:
        if not isinstance(place, dict):
            continue
        place_id = str(place.get("place_id") or "")
        if not place_id or place_id in used_place_ids:
            continue

        used_place_ids.add(place_id)
        step.place_id = place_id
        step.name = str(place.get("name") or "")
        step.brief = str(place.get("brief") or "")
        step.distance_meters = place.get("distance_meters")
        break


async def _synthesize_route(
    client: LLMClient,
    state: AgentState,
    plan: RoutePlan,
) -> str:
    """第二段收尾：把回填好的路线交给 LLM，合成一段自然的中文路线推荐。"""

    payload = {
        "weather": state.weather.model_dump() if state.weather else None,
        "route": plan.model_dump(),
    }
    messages = [
        {"role": "system", "content": ROUTE_SYNTHESIS_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
    ]

    response = await client.chat(messages)
    return extract_text(response.choices[0].message)


async def execute_plan(
    client: LLMClient,
    state: AgentState,
    plan: RoutePlan,
) -> str:
    """第二段：按计划逐站回填真实地点，再合成最终回复。

    几处主动兜底，保证即使计划不理想也能给用户一个明确的回应，而不是抛异常。
    """

    if not plan.steps:
        return "我没能为你排出合适的路线，要不要换个区域，或说说你想逛什么类型的地方？"

    # 路线必须有出发点；没有位置就先问，不猜测（和单点推荐保持一致的原则）
    if state.location is None:
        return "我还不知道你的出发位置，告诉我大概在哪，我就帮你把路线排出来。"

    await _ensure_weather(state)

    used_place_ids: set[str] = set()
    for step in plan.steps:
        await _fill_step_with_place(state, step, used_place_ids)

    # 把回填后的计划写回 state：既用于持久化，也为阶段三"用户否决-修改"留住可引用的路线
    state.route_plan = plan

    return await _synthesize_route(client, state, plan)


async def run_route_plan(
    client: LLMClient,
    state: AgentState,
    user_message: str,
) -> str:
    """路线规划入口：编排"生成计划 → 执行计划"两段，返回给用户的最终回复。

    供 loop 在意图判定为 ROUTE_PLAN 时直接调用，替代通用 ReAct 循环。
    """

    plan = await generate_plan(client, state, user_message)
    return await execute_plan(client, state, plan)
