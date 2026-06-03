"""实现城市漫步 Agent 的核心 ReAct 工具调用循环。"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent.messages import (
    assistant_message_to_dict,
    build_messages,
    extract_text,
    extract_tool_calls,
)
from agent.planner import run_route_plan
from agent.prompts import build_system_prompt
from agent.reflexion import detect_veto
from agent.router import Intent, classify
from agent.state import AgentState
from agent.state_sync import sync_state_from_tool_result
from config import config
from guardrails.input_filter import check_input
from guardrails.output_filter import clean_output
from models.client import LLMClient
from tools.registry import dispatch, get_openai_tools

logger = logging.getLogger(__name__)


def _parse_tool_arguments(arguments_json: str) -> dict[str, Any]:
    """把 OpenAI tool call 的 JSON 参数字符串解析为 dict。"""

    try:
        arguments = json.loads(arguments_json or "{}")
    except json.JSONDecodeError:
        logger.warning("工具参数不是合法 JSON: %s", arguments_json)
        return {}

    if not isinstance(arguments, dict):
        logger.warning("工具参数不是对象: %s", arguments)
        return {}

    return arguments


def _non_retryable_tool_reply(tool_name: str, result_json: str) -> str | None:
    """把不可重试的工具错误转换为直接回复，避免模型重复调用。"""

    try:
        payload = json.loads(result_json)
    except json.JSONDecodeError:
        return None

    if tool_name != "search_places" or not isinstance(payload, dict):
        return None

    if payload.get("provider") == "tencent" and not payload.get("retryable", True):
        return str(payload.get("error") or "腾讯位置服务暂时不可用，请检查 TENCENT_MAP_KEY。")

    return None


async def run_turn(
    client: LLMClient,
    state: AgentState,
    user_message: str,
) -> tuple[str, AgentState]:
    """执行一轮用户消息处理，返回助手回复和更新后的状态。

    client 由调用方（main）构造一次后注入，循环本身不关心 provider 细节。
    输入先过护栏，输出再过护栏——一进一出守住 agent 与外界的边界。
    """

    updated_state = state.model_copy(deep=True)

    # —— 输入护栏：异常输入直接拦下，不进入昂贵的模型调用 ——
    checked = check_input(user_message)
    if not checked.ok:
        updated_state.add_message("user", user_message)
        updated_state.add_message("assistant", checked.message)
        return checked.message, updated_state
    user_message = checked.message

    # —— 意图分流（阶段一）：先判断这句话该走哪条路径，再决定用哪套 prompt、是否进工具循环 ——
    intent = await classify(client, user_message)

    # —— Reflexion 反馈捕获（阶段三）：在有上一轮推荐、且本轮是推荐类意图时，
    #    先看用户是不是在否决某个推荐。是就记一条会话级反馈，下面生成时由 prompt 注入规避。——
    if intent in (Intent.SINGLE_SPOT, Intent.ROUTE_PLAN) and (
        updated_state.candidates or updated_state.route_plan is not None
    ):
        veto = await detect_veto(client, updated_state, user_message)
        if veto is not None:
            updated_state.session_feedback.append(veto)

    # —— 路线规划（阶段二）：走独立的 Plan-and-Solve 路径，不进通用 ReAct 循环 ——
    if intent is Intent.ROUTE_PLAN:
        try:
            final_reply = await run_route_plan(client, updated_state, user_message)
        except Exception:
            logger.exception("路线规划执行失败")
            final_reply = "我这边暂时排不出这条路线，请稍后再试或换个区域。"
        # 复用同一套输出护栏与历史记录逻辑，保持各路径行为一致
        final_reply = clean_output(final_reply)
        updated_state.add_message("user", user_message)
        updated_state.add_message("assistant", final_reply)
        return final_reply, updated_state

    # 阶段三：把长期偏好 + 本会话否决注入系统提示词，单点推荐生成时遵守
    system_prompt = build_system_prompt(
        intent,
        preferences=updated_state.preferences,
        session_feedback=updated_state.session_feedback,
    )

    messages = build_messages(state, user_message, system_prompt)
    tools = get_openai_tools()
    final_reply = ""

    try:
        # —— 闲聊路径：只回一句，不进工具循环。调用 chat 时不传 tools，模型就不会去调天气/地点工具 ——
        if intent is Intent.CHITCHAT:
            response = await client.chat(messages)
            final_reply = extract_text(response.choices[0].message)
        else:
            for turn_index in range(config.max_tool_turns):
                response = await client.chat(messages, tools)

                assistant_message = response.choices[0].message
                tool_calls = extract_tool_calls(assistant_message)
                assistant_text = extract_text(assistant_message)
                logger.info(
                    "model_turn turn=%s model=%s tool_calls=%s content=%s",
                    turn_index + 1,
                    client.model,
                    len(tool_calls),
                    assistant_text,
                )
                messages.append(assistant_message_to_dict(assistant_message))

                if not tool_calls:
                    final_reply = assistant_text
                    break

                for tool_call in tool_calls:
                    function = tool_call.get("function") or {}
                    tool_name = str(function.get("name") or "")
                    arguments = _parse_tool_arguments(str(function.get("arguments") or "{}"))

                    logger.info(
                        "tool_call turn=%s name=%s arguments=%s",
                        turn_index + 1,
                        tool_name,
                        json.dumps(arguments, ensure_ascii=False, default=str),
                    )
                    result_json = await dispatch(tool_name, arguments)
                    logger.info(
                        "tool_result turn=%s name=%s result=%s",
                        turn_index + 1,
                        tool_name,
                        result_json,
                    )
                    sync_state_from_tool_result(
                        updated_state,
                        tool_name,
                        arguments,
                        result_json,
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(tool_call.get("id") or ""),
                            "content": result_json,
                        }
                    )
                    final_reply = _non_retryable_tool_reply(tool_name, result_json) or ""
                    if final_reply:
                        break
                if final_reply:
                    break
            else:
                final_reply = (
                    "我连续调用工具的次数太多了，先停一下。"
                    "请你补充一个更具体的地点类型、时间范围或位置。"
                )
    except Exception:
        logger.exception("Agent 循环执行失败")
        final_reply = "我这边暂时无法完成本轮推荐，请稍后再试或检查 API 配置。"

    # —— 输出护栏：兜底空回复、拦截内部信息泄露、截断超长 ——
    final_reply = clean_output(final_reply)

    updated_state.add_message("user", user_message)
    updated_state.add_message("assistant", final_reply)
    return final_reply, updated_state
