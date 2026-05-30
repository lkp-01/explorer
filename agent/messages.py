"""负责构造 OpenAI 兼容 messages 并解析模型返回的工具调用。"""

from __future__ import annotations

import json
from typing import Any

from agent.state import AgentState


def _append_text_message(
    messages: list[dict[str, Any]],
    role: str,
    content: str,
) -> None:
    """追加文本消息；相邻同角色文本消息会合并，避免历史格式过碎。"""

    if not content:
        return

    if (
        messages
        and messages[-1]["role"] == role
        and isinstance(messages[-1]["content"], str)
    ):
        messages[-1]["content"] = f"{messages[-1]['content']}\n\n{content}"
        return

    messages.append({"role": role, "content": content})


def _history_to_messages(state: AgentState) -> list[dict[str, Any]]:
    """把状态中的历史记录转换为 OpenAI 兼容 messages。"""

    messages: list[dict[str, Any]] = []
    for item in state.conversation_history:
        role = item.get("role")
        content = str(item.get("content", ""))

        if role == "assistant":
            _append_text_message(messages, "assistant", content)
        elif role == "user":
            _append_text_message(messages, "user", content)
        elif role == "tool":
            _append_text_message(messages, "user", f"历史工具结果：{content}")

    return messages


def _format_state_context(state: AgentState) -> str:
    """把当前状态压缩为给 LLM 决策使用的上下文。"""

    snapshot = state.model_dump(mode="json", exclude={"conversation_history"})
    return (
        "当前状态快照（供你决策，不要逐字复述）：\n"
        f"{json.dumps(snapshot, ensure_ascii=False, default=str)}"
    )


def build_messages(
    state: AgentState,
    user_message: str,
    system_prompt: str,
) -> list[dict[str, Any]]:
    """组装系统提示词、历史消息、状态快照和当前用户消息。"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(_history_to_messages(state))
    current_content = f"{_format_state_context(state)}\n\n当前用户消息：{user_message}"
    _append_text_message(messages, "user", current_content)
    return messages


def _get_value(source: Any, key: str, default: Any = None) -> Any:
    """兼容 dict 和 SDK 对象两种取值方式。"""

    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def _tool_call_to_dict(tool_call: Any) -> dict[str, Any]:
    """把 OpenAI SDK 的 tool_call 对象转换为普通 dict。"""

    function = _get_value(tool_call, "function", {})
    return {
        "id": str(_get_value(tool_call, "id", "")),
        "type": str(_get_value(tool_call, "type", "function")),
        "function": {
            "name": str(_get_value(function, "name", "")),
            "arguments": str(_get_value(function, "arguments", "{}")),
        },
    }


def assistant_message_to_dict(message: Any) -> dict[str, Any]:
    """把 OpenAI SDK 的 assistant message 转换为可继续传入 messages 的 dict。"""

    content = _get_value(message, "content")
    tool_calls = _get_value(message, "tool_calls") or []
    result: dict[str, Any] = {"role": "assistant", "content": content or ""}

    if tool_calls:
        result["tool_calls"] = [_tool_call_to_dict(tool_call) for tool_call in tool_calls]

    return result


def extract_text(message: Any) -> str:
    """提取 OpenAI 兼容 assistant message 中的文本内容。"""

    return str(_get_value(message, "content") or "").strip()


def extract_tool_calls(message: Any) -> list[dict[str, Any]]:
    """提取 OpenAI 兼容 assistant message 中的 tool_calls。"""

    tool_calls = _get_value(message, "tool_calls") or []
    return [_tool_call_to_dict(tool_call) for tool_call in tool_calls]
