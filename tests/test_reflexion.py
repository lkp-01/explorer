"""Reflexion + 长期记忆的端到端测试（阶段三）。

覆盖五块：
1. detect_veto：能从模型 JSON 里解析出否决；非否决/调用失败时返回 None；
2. distill_preferences：能提炼新偏好，且冲突时近期反馈覆盖旧偏好（时效性）；无反馈时不调模型；
3. format_preference_block / build_system_prompt：偏好与否决能被正确注入 prompt，闲聊不注入；
4. 持久化分层：PreferenceStore 能独立读写；SessionStore 存盘**不含 preferences**——
   这是阶段三最关键的不变量（删/重开会话不会丢长期偏好）；
5. _merge_preferences：同 tag 覆盖。

工具/模型调用通过 _FakeClient 打桩，不真的联网。
本文件不依赖 pytest，直接 `python3 tests/test_reflexion.py` 即可运行。
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.prompts import build_system_prompt, format_preference_block  # noqa: E402
from agent.reflexion import detect_veto, distill_preferences  # noqa: E402
from agent.router import Intent  # noqa: E402
from agent.state import (  # noqa: E402
    AgentState,
    Candidate,
    Location,
    PreferenceItem,
    UserPreferences,
    VetoRecord,
)
from memory.preference_store import PreferenceStore  # noqa: E402
from memory.storage import SessionStore  # noqa: E402


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
    """按预设顺序返回文本；可设置 raises 模拟调用失败。"""

    model = "fake-model"

    def __init__(self, replies: list[str], *, raises: bool = False) -> None:
        self._replies = list(replies)
        self._raises = raises
        self.call_count = 0

    async def chat(self, messages, tools=None):
        self.call_count += 1
        if self._raises:
            raise RuntimeError("模拟模型调用失败")
        content = self._replies.pop(0) if self._replies else ""
        return _FakeResponse(content)


def _run(coro):
    return asyncio.run(coro)


def _state_with_candidate() -> AgentState:
    state = AgentState(location=Location(lat=35.6, lng=139.7))
    state.candidates = [
        Candidate(
            place_id="p1",
            name="某某酒吧",
            category="bar",
            distance_meters=1200.0,
            brief="示例",
        )
    ]
    return state


def test_detect_veto_parses_record() -> None:
    """模型判定为否决时，应解析出 target / reason 并返回 VetoRecord。"""

    client = _FakeClient(['{"is_veto": true, "target": "某某酒吧", "reason": "太远"}'])
    record = _run(detect_veto(client, _state_with_candidate(), "这个酒吧太远了，换一个"))
    assert record is not None
    assert record.target == "某某酒吧"
    assert record.reason == "太远"


def test_detect_veto_non_veto_returns_none() -> None:
    """模型判定不是否决（新需求/追问）时，应返回 None。"""

    client = _FakeClient(['{"is_veto": false, "target": "", "reason": ""}'])
    record = _run(detect_veto(client, _state_with_candidate(), "附近还有别的咖啡馆吗"))
    assert record is None


def test_detect_veto_swallows_errors() -> None:
    """模型调用抛异常时，detect_veto 必须吞掉并返回 None，不能让整轮崩溃。"""

    client = _FakeClient([], raises=True)
    record = _run(detect_veto(client, _state_with_candidate(), "随便说点什么"))
    assert record is None


def test_distill_no_feedback_skips_model() -> None:
    """没有任何会话反馈时，应直接返回现有偏好，且完全不调用模型。"""

    client = _FakeClient(['{"items": []}'])
    state = AgentState(preferences=UserPreferences(items=[PreferenceItem(tag="爱书店", polarity="like")]))
    out = _run(distill_preferences(client, state))
    assert client.call_count == 0
    assert [i.tag for i in out.items] == ["爱书店"]


def test_distill_recency_overrides_conflict() -> None:
    """冲突时近期优先：旧偏好 dislike 日料，本次提炼出 like 日料，应以新值覆盖。"""

    client = _FakeClient(['{"items": [{"tag": "日料", "polarity": "like"}]}'])
    state = AgentState(
        preferences=UserPreferences(items=[PreferenceItem(tag="日料", polarity="dislike")]),
    )
    state.session_feedback = [VetoRecord(target="某拉面店", reason="这次就想吃日料")]
    out = _run(distill_preferences(client, state))
    by_tag = {i.tag: i.polarity for i in out.items}
    assert by_tag["日料"] == "like", by_tag


def test_format_preference_block_contents() -> None:
    """偏好块应同时体现长期喜欢/不喜欢与本会话否决；全空时返回空串。"""

    assert format_preference_block(None, None) == ""

    prefs = UserPreferences(
        items=[
            PreferenceItem(tag="爱书店", polarity="like"),
            PreferenceItem(tag="不爱酒吧", polarity="dislike"),
        ]
    )
    feedback = [VetoRecord(target="某某酒吧", reason="太远")]
    block = format_preference_block(prefs, feedback)
    assert "爱书店" in block
    assert "不爱酒吧" in block
    assert "某某酒吧" in block and "太远" in block


def test_build_system_prompt_injects_and_skips_chitchat() -> None:
    """single_spot 应注入偏好块；chitchat 不注入。"""

    prefs = UserPreferences(items=[PreferenceItem(tag="爱书店", polarity="like")])
    single = build_system_prompt(Intent.SINGLE_SPOT, preferences=prefs)
    assert "爱书店" in single

    chit = build_system_prompt(Intent.CHITCHAT, preferences=prefs)
    assert "爱书店" not in chit


def test_preference_store_roundtrip() -> None:
    """PreferenceStore 应能按用户独立读写偏好。"""

    with tempfile.TemporaryDirectory() as tmp:
        store = PreferenceStore(tmp)
        assert store.load("alice").items == []  # 不存在时返回空偏好而非 None
        store.save("alice", UserPreferences(items=[PreferenceItem(tag="爱书店", polarity="like")]))
        again = store.load("alice")
        assert [i.tag for i in again.items] == ["爱书店"]


def test_session_store_excludes_preferences() -> None:
    """关键不变量：会话存盘不含 preferences，删/重开会话不会带走长期偏好。"""

    with tempfile.TemporaryDirectory() as tmp:
        store = SessionStore(tmp)
        state = AgentState(location=Location(lat=35.6, lng=139.7))
        state.preferences = UserPreferences(items=[PreferenceItem(tag="爱书店", polarity="like")])
        state.session_feedback = [VetoRecord(target="某某酒吧", reason="太远")]
        store.save("s1", state)

        # 原始 JSON 里不应出现 preferences，但应保留会话级反馈
        raw = (store._path("s1")).read_text(encoding="utf-8")
        assert "preferences" not in raw
        assert "session_feedback" in raw

        reloaded = store.load("s1")
        assert reloaded is not None
        assert reloaded.preferences.items == []          # 偏好不随会话回来
        assert len(reloaded.session_feedback) == 1       # 会话反馈随会话回来


def main() -> None:
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n全部 {len(tests)} 个用例通过。")


if __name__ == "__main__":
    main()
