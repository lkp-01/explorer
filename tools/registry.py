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


def register_tool_spec(spec: ToolSpec) -> bool:
    """直接登记一个已经构造好的 ToolSpec（feature/mcp）。

    与 register_tool 装饰器的区别：
    - register_tool 用于"写代码时就知道工具长什么样"的本地工具（天气/地点/地址解析）；
    - register_tool_spec 用于"运行时才从外部拿到"的工具——典型就是 MCP 客户端连上 server、
      list_tools() 之后，把每个远程工具包成 ToolSpec 在这里登记进来。

    两条路径登记进的是同一个 TOOL_REGISTRY，所以下游的 get_openai_tools() / dispatch()
    对工具来源完全无感知——这正是让 agent 决策逻辑（ReAct/并行/REWOO）一行都不用改的关键。

    返回是否登记成功：名字已存在时跳过并返回 False，避免外部工具悄悄覆盖掉同名的本地工具。
    """

    if spec.name in TOOL_REGISTRY:
        logger.warning("工具名已存在，跳过注册：%s", spec.name)
        return False

    TOOL_REGISTRY[spec.name] = spec
    return True


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
