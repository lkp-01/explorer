"""输出护栏：模型回复发给用户之前的最后一道关（第 9 步）。

主要防两件事：一是别把内部状态快照或系统提示词原样泄露给用户；二是别把超长回复直接甩
出去。它和输入护栏对称——一进一出，把 agent 和外部世界的边界守住。
"""

from __future__ import annotations

MAX_OUTPUT_CHARS = 4000

# 一旦回复里出现这些标记，说明很可能把内部上下文泄露出来了
LEAK_MARKERS = (
    "当前状态快照",
    "system prompt",
    "SYSTEM_PROMPT",
)

_EMPTY_FALLBACK = "我现在还没能整理出明确建议，请你补充想去的地点类型、位置或时间。"
_LEAK_FALLBACK = "（我整理推荐时出了点状况，请你换种说法再问我一次。）"


def clean_output(text: str) -> str:
    """清洗模型回复：兜底空回复、拦截泄露、截断超长。"""

    cleaned = (text or "").strip()
    if not cleaned:
        return _EMPTY_FALLBACK

    if any(marker in cleaned for marker in LEAK_MARKERS):
        return _LEAK_FALLBACK

    if len(cleaned) > MAX_OUTPUT_CHARS:
        cleaned = cleaned[:MAX_OUTPUT_CHARS].rstrip() + "……"

    return cleaned
