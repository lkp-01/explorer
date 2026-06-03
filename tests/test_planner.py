"""Plan-and-Solve 路线规划的端到端测试（阶段二）。

覆盖三件事：
1. generate_plan 能把（可能带代码块围栏的）JSON 解析成 RoutePlan；
2. run_route_plan 端到端：逐站回填真实地点、写回 state.route_plan、产出回复；
3. 几处兜底：没有位置、计划为空时给出明确回应而不崩溃。

工具调用（天气 / 地点搜索）通过替换 planner.dispatch 来打桩，不真的联网。
本文件不依赖 pytest，直接 `python3 tests/test_planner.py` 即可运行。
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent.planner as planner  # noqa: E402
from agent.planner import generate_plan, run_route_plan  # noqa: E402
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

    async def chat(self, messages, tools=None):
        self.call_count += 1
        # 用完就一直返回最后一段，避免 IndexError
        content = self._replies.pop(0) if self._replies else (self._replies or [""])[0]
        return _FakeResponse(content)


# 一份带 ```json 围栏的计划，顺便测试围栏剥离
_PLAN_JSON = """```json
{
  "summary": "雨天室内漫步：先咖啡再逛公园",
  "start_time": "10:00",
  "total_minutes": 180,
  "steps": [
    {"order": 1, "category": "cafe", "keyword": "手冲", "stay_minutes": 45,
     "transport_hint": "步行约 8 分钟", "reason": "先喝杯咖啡热热身"},
    {"order": 2, "category": "park", "keyword": null, "stay_minutes": 60,
     "transport_hint": "步行约 10 分钟", "reason": "饭后散步消食"}
  ]
}
```"""

_SYNTHESIS_TEXT = "今天给你排了一条轻松路线：先去……再去……记得带把伞。"


def _fake_dispatch_factory():
    """造一个假的 dispatch：按工具名返回不同的 JSON 字符串，并记录调用。"""

    calls: list[str] = []

    async def fake_dispatch(tool_name: str, arguments: dict) -> str:
        calls.append(tool_name)
        if tool_name == "get_weather":
            return '{"condition": "小雨", "temp": 18.0, "humidity": 80}'
        if tool_name == "search_places":
            # 用类别拼出独一无二的 place_id，方便验证两站不会撞店
            category = arguments.get("category", "x")
            return (
                f'[{{"place_id": "{category}-1", "name": "{category}的店",'
                f' "category": "{category}", "distance_meters": 200.0,'
                f' "rating": null, "brief": "示例简介"}}]'
            )
        return "{}"

    return fake_dispatch, calls


def _run(coro):
    return asyncio.run(coro)


def test_strip_json_fence_removes_code_block() -> None:
    """围栏剥离：```json 包裹的内容应被还原成纯 JSON 文本。"""

    out = planner._strip_json_fence(_PLAN_JSON)
    assert out.startswith("{") and out.endswith("}"), out


def test_generate_plan_parses_steps() -> None:
    """generate_plan 应把 JSON 解析成含 2 站的 RoutePlan。"""

    client = _FakeClient([_PLAN_JSON])
    state = AgentState(location=Location(lat=35.6, lng=139.7))
    plan = _run(generate_plan(client, state, "帮我排条雨天路线"))
    assert len(plan.steps) == 2, plan
    assert plan.steps[0].category == "cafe"
    assert plan.start_time == "10:00"


def test_generate_plan_bad_json_returns_empty() -> None:
    """模型没吐出合法 JSON 时，应返回空计划而不是抛异常。"""

    client = _FakeClient(["抱歉我不会"])
    state = AgentState(location=Location(lat=35.6, lng=139.7))
    plan = _run(generate_plan(client, state, "随便排排"))
    assert plan.steps == [], plan


def test_run_route_plan_end_to_end() -> None:
    """端到端：回填真实地点、写回 state.route_plan、返回合成回复。"""

    client = _FakeClient([_PLAN_JSON, _SYNTHESIS_TEXT])
    state = AgentState(location=Location(lat=35.6, lng=139.7))

    fake_dispatch, calls = _fake_dispatch_factory()
    planner.dispatch = fake_dispatch  # 打桩：替换 planner 里引用的 dispatch
    try:
        reply = _run(run_route_plan(client, state, "帮我排条雨天路线"))
    finally:
        # 还原，避免污染同进程内的其他测试
        from tools.registry import dispatch as real_dispatch

        planner.dispatch = real_dispatch

    assert reply == _SYNTHESIS_TEXT
    assert state.route_plan is not None
    assert len(state.route_plan.steps) == 2
    # 两站都应回填到真实地点，且 place_id 不重复
    names = [s.name for s in state.route_plan.steps]
    assert names == ["cafe的店", "park的店"], names
    # 应先查过天气，再查两次地点
    assert calls == ["get_weather", "search_places", "search_places"], calls
    # 天气也应被同步进 state
    assert state.weather is not None and state.weather.condition == "小雨"


def test_execute_plan_without_location_asks_user() -> None:
    """没有出发位置时，应明确询问而不是猜测或报错。"""

    client = _FakeClient([_PLAN_JSON, _SYNTHESIS_TEXT])
    state = AgentState()  # location 为空
    reply = _run(run_route_plan(client, state, "帮我排条路线"))
    assert "出发位置" in reply, reply


def main() -> None:
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n全部 {len(tests)} 个用例通过。")


if __name__ == "__main__":
    main()
