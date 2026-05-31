"""对话历史与候选地点的压缩，防止状态无限膨胀、token 越用越多（第 7 步）。

这正是前一阶段标记的隐患：candidates 每次搜索都往里堆，conversation_history 一直变长，
而 messages.py 又把整个 state 快照塞进每一轮请求。这里给两者设上限，做最朴素的
"滑动窗口"——只保留最近的一部分。

注意：生产级方案通常会做"摘要式压缩"（让模型把更早的对话总结成一段话再丢弃原文），
这里先用截断打底，理解了截断的必要性，再去理解摘要压缩会更自然。
"""

from __future__ import annotations

from agent.state import AgentState

MAX_CANDIDATES = 20
MAX_HISTORY_MESSAGES = 40


def compact(
    state: AgentState,
    max_candidates: int = MAX_CANDIDATES,
    max_history: int = MAX_HISTORY_MESSAGES,
) -> AgentState:
    """就地裁剪过长的候选列表与历史，返回同一个 state 方便链式调用。"""

    if len(state.candidates) > max_candidates:
        state.candidates = state.candidates[-max_candidates:]
    if len(state.conversation_history) > max_history:
        state.conversation_history = state.conversation_history[-max_history:]
    return state
