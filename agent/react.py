"""共享的 ReAct 工具调用循环（阶段五抽取）。

原先这段循环内嵌在 loop.run_turn 里。阶段五新增并行执行框架后，"跑一遍单点 ReAct"
成了会被多处复用的能力：单点推荐走它一次，并行调度里的每个子任务也各走它一次。
于是把它抽成独立模块——给定已组装好的 messages 与 tools，跑"调用模型 → 执行工具 →
回填状态"的循环，返回最终回复。

这一层刻意只关心"把一轮 ReAct 跑完"：意图分流、护栏、质量自检、历史记录都仍由上层
loop 负责。state 会被就地更新（工具结果同步进 candidates/weather 等），因此并行场景下
调用方必须传入各自的 state 副本，避免多个子任务并发写同一个状态对象。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent.messages import (
    assistant_message_to_dict,
    extract_text,
    extract_tool_calls,
)
from agent.state import AgentState
from agent.state_sync import sync_state_from_tool_result
from config import config
from models.client import LLMClient
from tools.registry import dispatch

logger = logging.getLogger(__name__)


def parse_tool_arguments(arguments_json: str) -> dict[str, Any]:
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


def non_retryable_tool_reply(tool_name: str, result_json: str) -> str | None:
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


async def react_generate(
    client: LLMClient,
    state: AgentState,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> tuple[str, bool]:
    """跑一轮 ReAct 工具循环，返回 (最终回复, 是否为正常推荐回复)。

    state 会被就地更新——工具结果通过 sync_state_from_tool_result 同步进候选/天气等字段。
    并行场景下调用方应传入 state 的深拷贝，避免多个子任务并发写同一个状态对象。

    第二个返回值 is_normal_reply 标记本轮是否产出了"正常的推荐回复"（而非错误兜底 /
    循环上限提示）。上层据此决定是否进阶段四质量自检——对错误提示做评估/重做没有意义。
    """

    final_reply = ""
    is_normal_reply = False

    try:
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
                is_normal_reply = True
                break

            for tool_call in tool_calls:
                function = tool_call.get("function") or {}
                tool_name = str(function.get("name") or "")
                arguments = parse_tool_arguments(str(function.get("arguments") or "{}"))

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
                    state,
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
                final_reply = non_retryable_tool_reply(tool_name, result_json) or ""
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
        logger.exception("ReAct 循环执行失败")
        final_reply = "我这边暂时无法完成本轮推荐，请稍后再试或检查 API 配置。"

    return final_reply, is_normal_reply
