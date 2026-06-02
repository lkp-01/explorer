"""路由分类的端到端测试（阶段一）。

为什么先给 router 写测试：它是后续所有决策模式的入口，一旦分类逻辑被改坏，
后面阶段的行为会跟着错乱却很难定位。这里覆盖三件事：
1. 规则层能否对"明显"的句子直接判定（不调用模型）；
2. LLM 层对模糊句子的兜底，以及调用出错时是否安全退回 SINGLE_SPOT；
3. build_system_prompt 是否为每种意图返回对应的 prompt。

本文件不依赖 pytest，直接 `python3 tests/test_router.py` 即可运行。
"""

from __future__ import annotations

import asyncio
import os
import sys

# 让测试无论从哪个目录运行，都能 import 到项目包（把项目根目录加入 sys.path）。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.prompts import (  # noqa: E402  （上面手动改了 sys.path，import 必须放在其后）
    CHITCHAT_PROMPT,
    ROUTE_PLAN_PROMPT,
    SYSTEM_PROMPT,
    build_system_prompt,
)
from agent.router import Intent, classify  # noqa: E402


class _FakeMessage:
    """模拟 OpenAI 返回的 message 对象，只需要一个 content 字段。"""

    def __init__(self, content: str) -> None:
        self.content = content
        self.tool_calls = None


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeClient:
    """假的 LLMClient：不真的联网，按预设文本返回，用来测 LLM 兜底分支。

    record 记录 chat 被调了几次，用来验证"规则命中时不应该调用模型"。
    """

    def __init__(self, reply: str = "single_spot", *, raises: bool = False) -> None:
        self.model = "fake-model"
        self._reply = reply
        self._raises = raises
        self.call_count = 0

    async def chat(self, messages, tools=None):  # 签名与真实 LLMClient.chat 对齐
        self.call_count += 1
        if self._raises:
            raise RuntimeError("模拟模型调用失败")
        return _FakeResponse(self._reply)


def _run(coro):
    """同步地跑一个协程，省去每个用例都写 asyncio.run。"""

    return asyncio.run(coro)


def test_rule_layer_hits_route_without_calling_model() -> None:
    """带"路线/逛一天"等词的句子，应由规则层判为路线，且完全不调用模型。"""

    client = _FakeClient(reply="chitchat")  # 故意设成错的，命中规则就不该用到它
    intent = _run(classify(client, "帮我排一条逛一天的路线"))
    assert intent is Intent.ROUTE_PLAN, intent
    assert client.call_count == 0, "规则命中时不应调用 LLM"


def test_rule_layer_hits_chitchat_for_short_greeting() -> None:
    """很短的问候语应由规则层判为闲聊，同样不调用模型。"""

    client = _FakeClient(reply="single_spot")
    intent = _run(classify(client, "你好呀"))
    assert intent is Intent.CHITCHAT, intent
    assert client.call_count == 0


def test_long_message_with_greeting_is_not_chitchat() -> None:
    """带问候但其实有需求的长句，不能被误判成闲聊——规则层放手，交给 LLM。"""

    client = _FakeClient(reply="single_spot")
    intent = _run(classify(client, "你好，帮我看看附近有什么好吃的餐厅"))
    assert intent is Intent.SINGLE_SPOT, intent
    assert client.call_count == 1, "规则不表态时应调用一次 LLM"


def test_llm_layer_parses_enum_value() -> None:
    """规则拿不准时走 LLM；模型返回带噪声也应能解析出正确枚举。"""

    client = _FakeClient(reply="意图是 single_spot。")
    intent = _run(classify(client, "附近有什么好玩的地方"))
    assert intent is Intent.SINGLE_SPOT, intent
    assert client.call_count == 1


def test_llm_failure_falls_back_to_single_spot() -> None:
    """LLM 调用抛异常时，必须安全退回 SINGLE_SPOT，不能让整轮崩溃。"""

    client = _FakeClient(raises=True)
    intent = _run(classify(client, "随便聊聊吧今天天气不错啊朋友"))
    assert intent is Intent.SINGLE_SPOT, intent


def test_build_system_prompt_per_intent() -> None:
    """每种意图都应拿到对应的系统提示词。"""

    assert build_system_prompt(Intent.SINGLE_SPOT) == SYSTEM_PROMPT
    assert build_system_prompt(Intent.ROUTE_PLAN) == ROUTE_PLAN_PROMPT
    assert build_system_prompt(Intent.CHITCHAT) == CHITCHAT_PROMPT


def main() -> None:
    """依次跑完所有用例，全部通过则打印汇总。"""

    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n全部 {len(tests)} 个用例通过。")


if __name__ == "__main__":
    main()
