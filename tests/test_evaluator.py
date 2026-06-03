"""Evaluator-Optimizer 的端到端测试（阶段四）。

覆盖六块：
1. evaluate：能从模型 JSON 解析出通过/不通过；维度 ok=false 时即使顶层 passed 漏标也收紧为不通过；
2. evaluate fail-open：模型调用失败 / 返回非 JSON 时，按通过处理，绝不阻断回复；
3. run_with_evaluation：首次通过则不重做；首次不过、重做后通过则只重做一次；
4. 重做上限：始终不过时命中 eval_max_retries 后放行最后一版，不死循环；
5. 短路：评估关闭 / 无评估客户端 / 空回复时直接返回原文，零模型调用；
6. format_eval_feedback_block：未通过维度能拼成可操作的反馈文本。

模型调用通过 _FakeClient 打桩，不真的联网。
本文件不依赖 pytest，直接 `python3 tests/test_evaluator.py` 即可运行。
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import evaluator as evaluator_module  # noqa: E402
from agent.evaluator import EvalResult, evaluate, run_with_evaluation  # noqa: E402
from agent.prompts import format_eval_feedback_block  # noqa: E402
from agent.router import Intent  # noqa: E402
from agent.state import AgentState, Candidate, Location  # noqa: E402
from config import config  # noqa: E402


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
    """按预设顺序返回文本；可设 raises 模拟调用失败。记 call_count 验证调用次数。"""

    model = "fake-eval-model"

    def __init__(self, replies: list[str], *, raises: bool = False) -> None:
        self._replies = replies
        self._raises = raises
        self.call_count = 0

    async def chat(self, messages, tools=None, temperature=None):
        self.call_count += 1
        if self._raises:
            raise RuntimeError("模拟评估调用失败")
        idx = min(self.call_count - 1, len(self._replies) - 1)
        return _FakeResponse(self._replies[idx])


def _run(coro):
    return asyncio.run(coro)


def _sample_state() -> AgentState:
    state = AgentState(location=Location(lat=35.0, lng=139.0))
    state.candidates = [
        Candidate(
            place_id="p1",
            name="某书店",
            category="shopping",
            distance_meters=200.0,
            brief="安静的独立书店",
        )
    ]
    return state


# —— 1. evaluate 解析 ——

def test_evaluate_parses_passed() -> None:
    client = _FakeClient(['{"passed": true, "issues": []}'])
    result = _run(
        evaluate(client, Intent.SINGLE_SPOT, "附近有什么", "推荐如下……", _sample_state())
    )
    assert result.passed is True
    assert client.call_count == 1


def test_evaluate_tightens_when_dimension_failed() -> None:
    """模型把某维度标 ok=false 却忘了把顶层 passed 设 false，应被收紧为不通过。"""

    reply = '{"passed": true, "issues": [{"dimension": "weather_fit", "ok": false, "detail": "雨天推荐露天公园"}]}'
    client = _FakeClient([reply])
    result = _run(
        evaluate(client, Intent.SINGLE_SPOT, "去哪玩", "去公园", _sample_state())
    )
    assert result.passed is False
    assert result.failed_issues()[0]["dimension"] == "weather_fit"


# —— 2. fail-open ——

def test_evaluate_fail_open_on_exception() -> None:
    client = _FakeClient([], raises=True)
    result = _run(
        evaluate(client, Intent.SINGLE_SPOT, "去哪", "回复", _sample_state())
    )
    assert result.passed is True


def test_evaluate_fail_open_on_non_json() -> None:
    client = _FakeClient(["这不是 JSON"])
    result = _run(
        evaluate(client, Intent.SINGLE_SPOT, "去哪", "回复", _sample_state())
    )
    assert result.passed is True


# —— 3. run_with_evaluation：通过 / 重做一次 ——

def test_run_passes_first_try_no_regenerate() -> None:
    client = _FakeClient(['{"passed": true, "issues": []}'])
    regenerate_calls = {"n": 0}

    async def regenerate(_feedback: str) -> str:
        regenerate_calls["n"] += 1
        return "不应被调用"

    out = _run(
        run_with_evaluation(
            client, Intent.SINGLE_SPOT, "去哪", "原始回复", _sample_state(), regenerate
        )
    )
    assert out == "原始回复"
    assert regenerate_calls["n"] == 0


def test_run_regenerates_then_passes() -> None:
    # 第一次评估不过 → 重做 → 第二次评估通过
    client = _FakeClient(
        [
            '{"passed": false, "issues": [{"dimension": "reason_quality", "ok": false, "detail": "理由太空"}]}',
            '{"passed": true, "issues": []}',
        ]
    )

    async def regenerate(feedback: str) -> str:
        assert "理由太空" in feedback  # 反馈应被传入重做
        return "改进后的回复"

    out = _run(
        run_with_evaluation(
            client, Intent.SINGLE_SPOT, "去哪", "原始回复", _sample_state(), regenerate
        )
    )
    assert out == "改进后的回复"
    assert client.call_count == 2


# —— 4. 重做上限 ——

def test_run_exhausts_retries_returns_last() -> None:
    """始终不过：用尽重做次数后放行最后一版，不抛异常、不死循环。"""

    client = _FakeClient(['{"passed": false, "issues": [{"dimension": "x", "ok": false, "detail": "d"}]}'])

    async def regenerate(_feedback: str) -> str:
        return "最后一版"

    out = _run(
        run_with_evaluation(
            client, Intent.SINGLE_SPOT, "去哪", "原始回复", _sample_state(), regenerate
        )
    )
    # eval_max_retries 默认 1：评估 2 次、重做 1 次，最终放行重做后的版本
    assert out == "最后一版"


# —— 5. 短路 ——

def test_run_short_circuits_when_no_client() -> None:
    async def regenerate(_feedback: str) -> str:
        raise AssertionError("无评估客户端时不应进入评估")

    out = _run(
        run_with_evaluation(
            None, Intent.SINGLE_SPOT, "去哪", "原始回复", _sample_state(), regenerate
        )
    )
    assert out == "原始回复"


def test_run_short_circuits_when_disabled() -> None:
    client = _FakeClient(['{"passed": false, "issues": []}'])

    async def regenerate(_feedback: str) -> str:
        raise AssertionError("评估关闭时不应调用")

    original = config.eval_enabled
    try:
        object.__setattr__(config, "eval_enabled", False)
        out = _run(
            run_with_evaluation(
                client, Intent.SINGLE_SPOT, "去哪", "原始回复", _sample_state(), regenerate
            )
        )
        assert out == "原始回复"
        assert client.call_count == 0
    finally:
        object.__setattr__(config, "eval_enabled", original)


def test_run_short_circuits_on_empty_reply() -> None:
    client = _FakeClient(['{"passed": false, "issues": []}'])

    async def regenerate(_feedback: str) -> str:
        raise AssertionError("空回复不应进入评估")

    out = _run(
        run_with_evaluation(
            client, Intent.SINGLE_SPOT, "去哪", "   ", _sample_state(), regenerate
        )
    )
    assert out == "   "
    assert client.call_count == 0


# —— 6. 反馈拼装 ——

def test_format_eval_feedback_block() -> None:
    issues = [
        {"dimension": "weather_fit", "ok": False, "detail": "雨天推荐露天公园"},
        {"dimension": "intent_match", "ok": True, "detail": ""},
    ]
    block = format_eval_feedback_block(issues)
    assert "weather_fit" in block
    assert "雨天推荐露天公园" in block
    assert "intent_match" not in block  # 通过的维度不应出现


def test_format_eval_feedback_block_empty_falls_back() -> None:
    block = format_eval_feedback_block([])
    assert block  # 没有可用意见时也应给一句通用改进要求


def main() -> None:
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n全部 {len(tests)} 个用例通过。")


if __name__ == "__main__":
    main()
