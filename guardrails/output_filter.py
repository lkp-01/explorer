"""输出护栏：模型回复发给用户之前的最后一道关（第 9 步）。

主要防两件事：一是别把内部状态快照或系统提示词原样泄露给用户；二是别把超长回复直接甩
出去。它和输入护栏对称——一进一出，把 agent 和外部世界的边界守住。

与阶段四 Evaluator 的分工：Evaluator 是"智能版输出护栏"，负责语义质量（是否答对、雨天
是否室内、理由是否具体等），跑在本函数**之前**；本函数只管机械安全（空/泄露/超长），始终
是发给用户前的最后一道关。两者职责不重叠，新的语义检查请加到 Evaluator，不要塞进这里。
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
