"""提供 Agent 工具的注册、暴露和分发能力。"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolSpec:
    """描述一个可供 LLM 调用的工具。"""

    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable[..., Any]


TOOL_REGISTRY: dict[str, ToolSpec] = {}


def register_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """注册工具函数，并在模块导入时写入全局工具表。"""

    def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
        TOOL_REGISTRY[name] = ToolSpec(
            name=name,
            description=description,
            parameters=parameters,
            function=function,
        )
        return function

    return decorator


def get_openai_tools() -> list[dict[str, Any]]:
    """把内部工具注册表转换为 OpenAI 兼容的 tools 参数格式。"""

    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        }
        for spec in TOOL_REGISTRY.values()
    ]


def get_tools_for_llm() -> list[dict[str, Any]]:
    """返回 OpenAI 兼容工具列表，保留旧函数名方便调用方迁移。"""

    return get_openai_tools()


async def dispatch(tool_name: str, arguments: dict) -> str:
    """按工具名调用对应函数，并把结果序列化为 JSON 字符串。"""

    spec = TOOL_REGISTRY.get(tool_name)
    if spec is None:
        return json.dumps(
            {"error": f"未知工具: {tool_name}", "fallback": True},
            ensure_ascii=False,
        )

    try:
        result = spec.function(**(arguments or {}))
        if inspect.isawaitable(result):
            result = await result
    except Exception as exc:
        logger.exception("工具调用失败: %s", tool_name)
        return json.dumps(
            {"error": f"工具调用失败: {exc}", "fallback": True},
            ensure_ascii=False,
        )

    return json.dumps(result, ensure_ascii=False, default=str)
