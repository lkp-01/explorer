"""并行执行框架的端到端测试（阶段五）。

覆盖五件事：
1. _decompose 能把 JSON 拆解结果解析成子任务，并在超量时截断到 MAX_SUBTASKS；
2. 解析失败时返回空列表（交由上层回退单点推荐）；
3. 多方案：并发子结果被机械并排呈现，且候选被并回主 state；
4. 复杂请求：子结果经 LLM 聚合成一份回复，候选同样并回主 state；
5. 拆不出多个子任务时，回退到单点推荐路径。

子任务执行（react_generate → 工具）通过替换 parallel._run_subtask / _fallback_single
打桩，不真的联网、也不依赖并发顺序。本文件不依赖 pytest，直接
`python tests/test_parallel.py` 即可运行。
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent.parallel as parallel  # noqa: E402
from agent.parallel import (  # noqa: E402
    MAX_SUBTASKS,
    SubResult,
    SubTask,
    _decompose,
    _format_multi_plan,
    _merge_candidates,
    run_complex_task,
    run_multi_plan,
)
from agent.prompts import MULTI_PLAN_DECOMPOSE_PROMPT  # noqa: E402
from agent.state import AgentState, Candidate, Location  # noqa: E402


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


class _ScriptedClient:
    """按系统提示词内容路由返回，避免依赖并发调用顺序。

    - 命中拆解 prompt → 返回预设的拆解 JSON；
    - 命中聚合 prompt → 返回预设的聚合文本；
    - 其余（子任务/兜底）→ 不应被这个 client 直接处理，测试里都打了桩。
    """

    model = "fake-model"

    def __init__(self, *, decompose_json: str, aggregate_text: str = "合并后的统一推荐") -> None:
        self._decompose_json = decompose_json
        self._aggregate_text = aggregate_text
        self.calls = 0

    async def chat(self, messages, tools=None, temperature=None):
        self.calls += 1
        system = str(messages[0]["content"])
        if "方案变体" in system or "分别检索的子需求" in system:
            return _FakeResponse(self._decompose_json)
        if "交叉合并" in system:
            return _FakeResponse(self._aggregate_text)
        return _FakeResponse("（未预期的调用）")


def _run(coro):
    return asyncio.run(coro)


def _decompose_json(n: int) -> str:
    items = ", ".join(
        f'{{"label": "方案{i}", "instruction": "在附近找适合方案{i}的地方"}}'
        for i in range(1, n + 1)
    )
    return f'{{"subtasks": [{items}]}}'


def _stub_subtask(label_suffix: str = "店"):
    """造一个假的 _run_subtask：给每个子任务塞一个以 label 命名的候选，便于验证合并。"""

    async def fake_run_subtask(client, base_state, subtask: SubTask) -> SubResult:
        sub_state = base_state.model_copy(deep=True)
        sub_state.candidates.append(
            Candidate(
                place_id=subtask.label,
                name=f"{subtask.label}{label_suffix}",
                category="cafe",
                distance_meters=100.0,
                brief="示例简介",
            )
        )
        return SubResult(
            label=subtask.label,
            reply=f"{subtask.label}的推荐内容",
            state=sub_state,
        )

    return fake_run_subtask


# ——————————————————————————————————————————————————————————————

def test_decompose_parses_and_caps() -> None:
    """拆解结果超过上限时，应截断到 MAX_SUBTASKS。"""

    client = _ScriptedClient(decompose_json=_decompose_json(MAX_SUBTASKS + 2))
    state = AgentState(location=Location(lat=35.6, lng=139.7))
    tasks = _run(_decompose(client, MULTI_PLAN_DECOMPOSE_PROMPT, state, "随便给我几套"))
    assert len(tasks) == MAX_SUBTASKS, len(tasks)
    assert all(isinstance(t, SubTask) and t.instruction for t in tasks)


def test_decompose_bad_json_returns_empty() -> None:
    """拆解结果不是合法 JSON 时，应返回空列表交给上层回退。"""

    client = _ScriptedClient(decompose_json="抱歉我拆不了")
    state = AgentState(location=Location(lat=35.6, lng=139.7))
    tasks = _run(_decompose(client, MULTI_PLAN_DECOMPOSE_PROMPT, state, "随便"))
    assert tasks == [], tasks


def test_format_multi_plan_lists_all_blocks() -> None:
    """多方案聚合应把每套方案带小标题并排列出。"""

    state = AgentState()
    results = [
        SubResult(label="雨天方案", reply="去咖啡馆", state=state),
        SubResult(label="晴天方案", reply="去公园", state=state),
    ]
    text = _format_multi_plan(results)
    assert "【雨天方案】" in text and "【晴天方案】" in text, text
    assert "2 套方案" in text, text


def test_merge_candidates_dedups_by_place_id() -> None:
    """合并候选应按 place_id 去重，并把子任务天气补回主 state。"""

    base = AgentState(location=Location(lat=35.6, lng=139.7))
    s1 = base.model_copy(deep=True)
    s1.candidates.append(Candidate(place_id="a", name="A", category="cafe", distance_meters=1, brief=""))
    s2 = base.model_copy(deep=True)
    s2.candidates.append(Candidate(place_id="a", name="A", category="cafe", distance_meters=1, brief=""))
    s2.candidates.append(Candidate(place_id="b", name="B", category="park", distance_meters=2, brief=""))

    _merge_candidates(base, [SubResult("一", "", s1), SubResult("二", "", s2)])
    ids = sorted(c.place_id for c in base.candidates)
    assert ids == ["a", "b"], ids


def test_run_multi_plan_end_to_end() -> None:
    """多方案端到端：拆 3 套 → 并发执行 → 并排呈现 + 候选并回主 state。"""

    client = _ScriptedClient(decompose_json=_decompose_json(3))
    state = AgentState(location=Location(lat=35.6, lng=139.7))

    original = parallel._run_subtask
    parallel._run_subtask = _stub_subtask()
    try:
        reply = _run(run_multi_plan(client, state, "给我三套方案"))
    finally:
        parallel._run_subtask = original

    assert "【方案1】" in reply and "【方案3】" in reply, reply
    assert len(state.candidates) == 3, state.candidates


def test_run_complex_task_end_to_end() -> None:
    """复杂请求端到端：拆子需求 → 并发检索 → LLM 聚合（eval 关闭）+ 候选并回。"""

    client = _ScriptedClient(decompose_json=_decompose_json(2), aggregate_text="兼顾老人和小孩的合并推荐")
    state = AgentState(location=Location(lat=35.6, lng=139.7))

    original = parallel._run_subtask
    parallel._run_subtask = _stub_subtask()
    try:
        # eval_client=None：质量自检短路，聚合文本即最终回复
        reply = _run(run_complex_task(client, state, "找个适合老人和小孩的地方", eval_client=None))
    finally:
        parallel._run_subtask = original

    assert reply == "兼顾老人和小孩的合并推荐", reply
    assert len(state.candidates) == 2, state.candidates


def test_run_multi_plan_falls_back_when_single_subtask() -> None:
    """只拆出 1 个子任务时，应回退到单点推荐路径。"""

    client = _ScriptedClient(decompose_json=_decompose_json(1))
    state = AgentState(location=Location(lat=35.6, lng=139.7))

    called: list[str] = []

    async def fake_fallback(c, s, msg):
        called.append(msg)
        return "回退单点推荐的回复"

    original = parallel._fallback_single
    parallel._fallback_single = fake_fallback
    try:
        reply = _run(run_multi_plan(client, state, "只给我一个就行"))
    finally:
        parallel._fallback_single = original

    assert reply == "回退单点推荐的回复", reply
    assert called == ["只给我一个就行"], called


def main() -> None:
    tests = [value for name, value in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n全部 {len(tests)} 个用例通过。")


if __name__ == "__main__":
    main()
