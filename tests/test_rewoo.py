"""REWOO 快速模式的端到端测试（阶段六）。

覆盖四件事：
1. generate_tool_plan 能把（可能带代码块围栏的）JSON 解析成步骤，并在超量时截断到上限；
2. 坏 JSON 时返回空列表（交由 loop 回退标准 ReAct）；
3. _resolve_references 能把 "$id.field" 引用替换成前序步骤结果，缺失引用置 None；
4. run_fast_mode 端到端：resolve_location→search_places 依赖链按序执行、参数被解引用、
   工具结果回填 state、最终合成回复。

工具调用通过替换 rewoo.dispatch 打桩，不真的联网。本文件不依赖 pytest，直接
`python tests/test_rewoo.py` 即可运行。
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools  # noqa: E402,F401  导入以填充工具注册表，让 _tool_catalog 拿到真实工具
import agent.rewoo as rewoo  # noqa: E402
from agent.rewoo import (  # noqa: E402
    MAX_REWOO_STEPS,
    ToolCallStep,
    _resolve_references,
    generate_tool_plan,
    run_fast_mode,
)
from agent.state import AgentState, Location  # noqa: E402


class _FakeMessage:
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
    """按预设顺序返回若干段文本：第一段当计划 JSON，之后当合成回复。"""

    model = "fake-model"

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.call_count = 0

    async def chat(self, messages, tools=None, temperature=None):
        self.call_count += 1
        return _FakeResponse(self._replies.pop(0) if self._replies else "")


# 一份带 ```json 围栏的计划，含 resolve_location → get_weather/search_places 的变量引用
_PLAN_JSON = """```json
{
  "steps": [
    {"id": 1, "tool": "resolve_location", "arguments": {"query": "东京塔"}},
    {"id": 2, "tool": "get_weather", "arguments": {"lat": "$1.lat", "lng": "$1.lng"}},
    {"id": 3, "tool": "search_places",
     "arguments": {"lat": "$1.lat", "lng": "$1.lng", "category": "restaurant"}}
  ]
}
```"""

_SYNTHESIS_TEXT = "赶时间的话，附近这家拉面店最近，走两步就到。"


def _plan_json_with_steps(n: int) -> str:
    items = ", ".join(
        f'{{"id": {i}, "tool": "search_places", "arguments": {{"lat": 1, "lng": 2, "category": "cafe"}}}}'
        for i in range(1, n + 1)
    )
    return f'{{"steps": [{items}]}}'


def _fake_dispatch_factory():
    """造一个假的 dispatch：按工具名返回不同 JSON，并记录每次调用的工具名与（解引用后的）参数。"""

    calls: list[tuple[str, dict]] = []

    async def fake_dispatch(tool_name: str, arguments: dict) -> str:
        calls.append((tool_name, dict(arguments)))
        if tool_name == "resolve_location":
            return '{"query": "东京塔", "lat": 35.66, "lng": 139.7, "address": "东京塔"}'
        if tool_name == "get_weather":
            return '{"condition": "晴", "temp": 20.0, "humidity": 50}'
        if tool_name == "search_places":
            return (
                '[{"place_id": "r1", "name": "拉面店", "category": "restaurant",'
                ' "distance_meters": 150.0, "rating": null, "brief": "好吃"}]'
            )
        return "{}"

    return fake_dispatch, calls


def _run(coro):
    return asyncio.run(coro)


# ——————————————————————————————————————————————————————————————

def test_generate_tool_plan_parses_steps() -> None:
    """generate_tool_plan 应把（含围栏的）JSON 解析成 3 步计划，保留工具名与参数。"""

    client = _FakeClient([_PLAN_JSON])
    state = AgentState(location=Location(lat=35.6, lng=139.7))
    steps = _run(generate_tool_plan(client, state, "我赶时间，附近找个吃饭的"))
    assert len(steps) == 3, steps
    assert steps[0].tool == "resolve_location"
    assert steps[2].arguments["lat"] == "$1.lat", steps[2].arguments


def test_generate_tool_plan_caps_steps() -> None:
    """计划步数超过上限时应截断到 MAX_REWOO_STEPS。"""

    client = _FakeClient([_plan_json_with_steps(MAX_REWOO_STEPS + 2)])
    state = AgentState(location=Location(lat=35.6, lng=139.7))
    steps = _run(generate_tool_plan(client, state, "随便快点"))
    assert len(steps) == MAX_REWOO_STEPS, len(steps)


def test_generate_tool_plan_bad_json_returns_empty() -> None:
    """模型没吐出合法 JSON 时，应返回空列表交给上层回退。"""

    client = _FakeClient(["抱歉我规划不了"])
    state = AgentState(location=Location(lat=35.6, lng=139.7))
    steps = _run(generate_tool_plan(client, state, "随便"))
    assert steps == [], steps


def test_resolve_references_substitutes_and_nulls() -> None:
    """$id.field 应替换成前序结果；引用不存在的步骤/字段时置 None，字面值原样保留。"""

    results = {1: {"lat": 35.66, "lng": 139.7}}
    args = {"lat": "$1.lat", "lng": "$1.lng", "category": "restaurant", "bad": "$2.x"}
    resolved = _resolve_references(args, results)
    assert resolved["lat"] == 35.66 and resolved["lng"] == 139.7, resolved
    assert resolved["category"] == "restaurant"
    assert resolved["bad"] is None, resolved


def test_run_fast_mode_end_to_end() -> None:
    """端到端：依赖链按序执行、参数被解引用、结果回填 state、返回合成回复。"""

    client = _FakeClient([_PLAN_JSON, _SYNTHESIS_TEXT])
    state = AgentState(location=Location(lat=35.6, lng=139.7))

    fake_dispatch, calls = _fake_dispatch_factory()
    original = rewoo.dispatch
    rewoo.dispatch = fake_dispatch
    try:
        reply = _run(run_fast_mode(client, state, "我赶时间，东京塔附近找个吃饭的"))
    finally:
        rewoo.dispatch = original

    assert reply == _SYNTHESIS_TEXT
    # 三步按序执行
    assert [name for name, _ in calls] == ["resolve_location", "get_weather", "search_places"], calls
    # get_weather / search_places 的 lat/lng 应已由 $1.lat/$1.lng 解引用成 resolve_location 的真实坐标
    assert calls[1][1]["lat"] == 35.66 and calls[1][1]["lng"] == 139.7, calls[1]
    assert calls[2][1]["lat"] == 35.66, calls[2]
    # 工具结果应回填进结构化 state
    assert state.weather is not None and state.weather.condition == "晴"
    assert state.location is not None and state.location.lat == 35.66
    names = [c.name for c in state.candidates]
    assert "拉面店" in names, names


def main() -> None:
    tests = [value for name, value in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n全部 {len(tests)} 个用例通过。")


if __name__ == "__main__":
    main()
