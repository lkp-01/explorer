"""LLM 客户端封装：把 provider（DeepSeek / OpenAI 兼容）的细节挡在循环之外。

为什么单独抽一层：
1. 之前 run_turn 每一轮都 new 一个 AsyncOpenAI、每一轮都重读环境变量。客户端是有
   连接池的对象，应该构造一次、整个会话复用。
2. agent 的循环逻辑不该关心"用的是哪家模型、base_url 是什么"。循环只需要一个能
   接收 messages + tools、返回回复的 chat() 方法。换模型时只动这一个文件。

这正是 agent 架构里"模型层"该负责的边界：稳定的调用接口 + 易替换的实现。
"""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from config import config


class LLMClient:
    """对 OpenAI 兼容 Chat Completions 接口的最小封装。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        max_tokens: int,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        """发起一次对话补全请求，返回原始响应对象。

        只有在确实有工具时才传 tools 字段，避免给某些兼容服务传空列表。
        """

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        return await self._client.chat.completions.create(**kwargs)


def build_client() -> LLMClient | None:
    """按当前配置构造客户端；没有 API Key 时返回 None，交给调用方处理。"""

    if not config.has_llm_key:
        return None

    return LLMClient(
        api_key=config.api_key or "",
        base_url=config.base_url,
        model=config.model,
        max_tokens=config.max_tokens,
    )
