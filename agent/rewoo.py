"""REWOO 快速出方案模式（阶段六）：先一次性规划全部工具调用，再批量执行，最后一次合成。

和 ReAct"调一个工具→看结果→再决定"的多轮循环互补：
- generate_tool_plan：让 LLM 一次性产出一份结构化的工具调用计划（JSON），这一步不真正调工具。
- execute_plan：按计划逐步批量执行（无中间观察），支持用 "$id.field" 引用前序步骤结果。
- synthesize：把全部工具结果一次性交给 LLM 合成最终回复。
- run_fast_mode：把三步串起来，供 loop 在 FAST_MODE 意图下调用。

为什么这样更快：ReAct 每调一次工具就要一次 LLM 往返（N+1 次），REWOO 把它压成"规划 1 次 +
合成 1 次"共 2 次。代价是缺乏中间纠错——这一点由 loop 侧的轻量评估 + 回退标准 ReAct 兜底。

刻意不持久化计划：REWOO 计划是瞬态的，工具结果经 sync_state_from_tool_result 同步进
location/weather/candidates 即可，不新增 AgentState 字段（与阶段四瞬态评估保持一致）。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from agent.messages import extract_text
from agent.prompts import REWOO_PLAN_PROMPT, REWOO_SYNTHESIS_PROMPT
from agent.state import AgentState
from agent.state_sync import sync_state_from_tool_result
from models.client import LLMClient
from tools.registry import dispatch, get_openai_tools
from utils.parser import safe_json_loads, strip_json_fence

logger = logging.getLogger(__name__)

# 计划步数上限：赶时间场景不该规划太多步，也守住批量调用的成本/额度。
MAX_REWOO_STEPS = 5


@dataclass
class ToolCallStep:
    """计划里的一步工具调用：id（供后续步骤引用）、工具名、参数（可能含 "$id.field" 引用）。"""

    id: int
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)


def _tool_catalog() -> str:
    """从注册表生成一份紧凑的"可用工具"清单，注入规划 prompt，使其跟随工具注册表变化。"""

    lines = []
    for spec in get_openai_tools():
        fn = spec.get("function") or {}
        name = fn.get("name")
        params = (fn.get("parameters") or {}).get("properties") or {}
        param_names = "、".join(params.keys()) or "无"
        desc_lines = (fn.get("description") or "").splitlines()
        summary = desc_lines[0] if desc_lines else ""
        lines.append(f"- {name}（参数：{param_names}）：{summary}")
    return "\n".join(lines)


def _plan_context(state: AgentState) -> str:
    """规划所需上下文：当前位置（含经纬度，若已知）、天气，以及可用工具清单。

    位置已知时把经纬度直接给模型，使其内联坐标、跳过 resolve_location（常见情形零依赖）。
    """

    snapshot = state.model_dump(mode="json", include={"location", "weather"})
    return (
        "当前位置与天气（位置已知则直接用其经纬度，不必再解析地址）：\n"
        f"{json.dumps(snapshot, ensure_ascii=False, default=str)}\n\n"
        f"可用工具：\n{_tool_catalog()}"
    )


async def generate_tool_plan(
    client: LLMClient,
    state: AgentState,
    user_message: str,
) -> list[ToolCallStep]:
    """第一段：让 LLM 一次性产出工具调用计划。解析失败或拆不出步骤时返回空列表，由上层回退。"""

    messages = [
        {"role": "system", "content": REWOO_PLAN_PROMPT},
        {"role": "user", "content": f"{_plan_context(state)}\n\n用户需求：{user_message}"},
    ]

    # 规划阶段不传 tools：这一步只要 JSON 计划，不需要模型真的去调工具。
    try:
        response = await client.chat(messages)
        payload = safe_json_loads(strip_json_fence(extract_text(response.choices[0].message)))
    except Exception:
        logger.exception("REWOO 计划生成调用失败")
        return []

    if not isinstance(payload, dict):
        logger.warning("REWOO 计划不是 JSON 对象，回退标准 ReAct：%s", payload)
        return []

    steps: list[ToolCallStep] = []
    for item in payload.get("steps") or []:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or "").strip()
        if not tool:
            continue
        arguments = item.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        # id 缺失/非法时按出现顺序补一个，保证引用与执行顺序稳定
        try:
            step_id = int(item.get("id"))
        except (TypeError, ValueError):
            step_id = len(steps) + 1
        steps.append(ToolCallStep(id=step_id, tool=tool, arguments=arguments))
        if len(steps) >= MAX_REWOO_STEPS:
            logger.info("REWOO 计划步数达到上限 %s，截断其余", MAX_REWOO_STEPS)
            break

    logger.info("rewoo_plan steps=%s", [(s.id, s.tool) for s in steps])
    return steps


def _resolve_one(value: Any, results_by_id: dict[int, Any]) -> Any:
    """解析单个参数值：形如 "$1.lat" 的引用替换为前序步骤结果的字段；其余原样返回。

    引用缺失、越界、或结果不是 dict / 字段不存在时返回 None——让该参数缺省，
    由工具自身的校验决定如何降级，而不是在这里硬抛异常打断整批执行。
    """

    if not isinstance(value, str) or not value.startswith("$"):
        return value

    ref = value[1:]
    id_part, _, field_path = ref.partition(".")
    try:
        ref_id = int(id_part)
    except ValueError:
        logger.warning("REWOO 引用格式不合法，置空：%s", value)
        return None

    source = results_by_id.get(ref_id)
    if not field_path:
        return source
    if not isinstance(source, dict):
        logger.warning("REWOO 引用的步骤 %s 结果不是对象，置空：%s", ref_id, value)
        return None
    return source.get(field_path)


def _resolve_references(
    arguments: dict[str, Any],
    results_by_id: dict[int, Any],
) -> dict[str, Any]:
    """把一步参数里所有 "$id.field" 引用替换成前序步骤的实际结果值。"""

    return {key: _resolve_one(value, results_by_id) for key, value in arguments.items()}


async def execute_plan(
    state: AgentState,
    steps: list[ToolCallStep],
) -> dict[int, Any]:
    """第二段：按 id 顺序串行批量执行各步，回填 state，并返回 {id: 解析后的结果}。

    单步出错（解析/调用异常）只记录并把该步结果置 None，不中断其余步骤——
    REWOO 没有中间纠错，靠"尽量多跑通几步 + 下游轻量评估兜底"来保证可用性。
    """

    results_by_id: dict[int, Any] = {}

    for step in sorted(steps, key=lambda s: s.id):
        arguments = _resolve_references(step.arguments, results_by_id)
        logger.info(
            "rewoo_exec id=%s tool=%s arguments=%s",
            step.id,
            step.tool,
            json.dumps(arguments, ensure_ascii=False, default=str),
        )
        try:
            result_json = await dispatch(step.tool, arguments)
        except Exception:
            logger.exception("REWOO 步骤执行失败 id=%s tool=%s", step.id, step.tool)
            results_by_id[step.id] = None
            continue

        # 复用既有同步逻辑，把天气/地点/坐标结果写回结构化状态
        sync_state_from_tool_result(state, step.tool, arguments, result_json)
        results_by_id[step.id] = safe_json_loads(result_json)

    return results_by_id


async def synthesize(
    client: LLMClient,
    state: AgentState,
    user_message: str,
    results_by_id: dict[int, Any],
) -> str:
    """第三段：把全部工具结果一次性交给 LLM，合成给用户的中文快速推荐。"""

    payload = {
        "user_message": user_message,
        "weather": state.weather.model_dump() if state.weather else None,
        "candidates": [c.model_dump() for c in state.candidates],
        "tool_results": results_by_id,
    }
    messages = [
        {"role": "system", "content": REWOO_SYNTHESIS_PROMPT},
        {
            "role": "user",
            "content": (
                "批量执行得到的工具结果如下，请据此直接给出推荐：\n"
                f"{json.dumps(payload, ensure_ascii=False, default=str)}"
            ),
        },
    ]
    response = await client.chat(messages)
    return extract_text(response.choices[0].message)


async def run_fast_mode(
    client: LLMClient,
    state: AgentState,
    user_message: str,
) -> str:
    """REWOO 入口：规划 → 批量执行 → 合成。计划为空时返回空串，让 loop 回退标准 ReAct。

    供 loop 在意图判定为 FAST_MODE 时调用。state 会被就地更新（工具结果同步进各字段）。
    """

    steps = await generate_tool_plan(client, state, user_message)
    if not steps:
        logger.info("REWOO 未规划出可执行步骤，交由上层回退")
        return ""

    results_by_id = await execute_plan(state, steps)
    return await synthesize(client, state, user_message, results_by_id)
