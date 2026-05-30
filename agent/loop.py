"""实现城市漫步 Agent 的核心 ReAct 工具调用循环。"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from openai import AsyncOpenAI
from dotenv import load_dotenv

from agent.messages import (
    assistant_message_to_dict,
    build_messages,
    extract_text,
    extract_tool_calls,
)
from agent.prompts import SYSTEM_PROMPT
from agent.state import AgentState
from agent.state_sync import sync_state_from_tool_result
from tools.registry import dispatch, get_openai_tools

import tools.places  # noqa: F401 触发工具注册
import tools.weather  # noqa: F401 触发工具注册

logger = logging.getLogger(__name__)
load_dotenv()

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL_NAME = "deepseek-v4-pro"
MAX_TOOL_TURNS = 8


def _get_api_key() -> str | None:
    """读取 DeepSeek 或通用 OpenAI 兼容 API Key。"""

    return os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")


def _get_base_url() -> str:
    """读取 OpenAI 兼容服务地址，默认指向 DeepSeek。"""

    return (
        os.getenv("DEEPSEEK_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or DEFAULT_BASE_URL
    )


def _get_model_name() -> str:
    """读取模型名，默认使用 DeepSeek 当前 OpenAI 兼容模型。"""

    return (
        os.getenv("DEEPSEEK_MODEL")
        or os.getenv("OPENAI_MODEL")
        or DEFAULT_MODEL_NAME
    )


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


async def run_turn(state: AgentState, user_message: str) -> tuple[str, AgentState]:
    """执行一轮用户消息处理，返回助手回复和更新后的状态。"""

    updated_state = state.model_copy(deep=True)
    messages = build_messages(state, user_message, SYSTEM_PROMPT)
    api_key = _get_api_key()

    if not api_key:
        reply = "DeepSeek API key 未配置，请先在环境变量或 .env 中设置 DEEPSEEK_API_KEY。"
        updated_state.add_message("user", user_message)
        updated_state.add_message("assistant", reply)
        return reply, updated_state

    client = AsyncOpenAI(api_key=api_key, base_url=_get_base_url())
    model_name = _get_model_name()
    tools = get_openai_tools()
    final_reply = ""

    try:
        for turn_index in range(MAX_TOOL_TURNS):
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                max_tokens=1200,
            )

            assistant_message = response.choices[0].message
            messages.append(assistant_message_to_dict(assistant_message))
            tool_calls = extract_tool_calls(assistant_message)

            if not tool_calls:
                final_reply = extract_text(assistant_message)
                break

            for tool_call in tool_calls:
                function = tool_call.get("function") or {}
                tool_name = str(function.get("name") or "")
                arguments = _parse_tool_arguments(str(function.get("arguments") or "{}"))

                logger.info(
                    "tool_call turn=%s name=%s arguments=%s",
                    turn_index + 1,
                    tool_name,
                    arguments,
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
        else:
            final_reply = (
                "我连续调用工具的次数太多了，先停一下。"
                "请你补充一个更具体的地点类型、时间范围或位置。"
            )
    except Exception:
        logger.exception("Agent 循环执行失败")
        final_reply = "我这边暂时无法完成本轮推荐，请稍后再试或检查 API 配置。"

    if not final_reply:
        final_reply = "我现在还没能整理出明确建议，请你补充想去的地点类型、位置或时间。"

    updated_state.add_message("user", user_message)
    updated_state.add_message("assistant", final_reply)
    return final_reply, updated_state
