"""输入护栏：在把用户消息交给模型之前，挡掉明显异常或有害的输入（第 9 步）。

护栏不是要做得多复杂，而是给 agent 划一条边界：哪些输入根本不该进入昂贵的模型调用。
这里处理三类最常见的情况——空输入、超长输入、以及试图覆盖系统设定的提示词注入。
"""

from __future__ import annotations

from dataclasses import dataclass

MAX_INPUT_CHARS = 2000

# 常见的"提示词注入"措辞：试图让模型忘记/无视既定的系统设定
INJECTION_PATTERNS = (
    "ignore previous",
    "ignore all previous",
    "disregard the above",
    "you are now",
    "system prompt",
    "忽略以上",
    "忽略之前",
    "忽略上面",
    "无视以上",
)


@dataclass(frozen=True)
class FilterResult:
    """护栏判定结果。ok 为真时 message 是清洗后的输入；为假时是给用户的解释。"""

    ok: bool
    message: str


def check_input(text: str) -> FilterResult:
    """校验并清洗用户输入，返回是否放行。"""

    cleaned = (text or "").strip()
    if not cleaned:
        return FilterResult(False, "请输入你想去的地方或想做的事。")
    if len(cleaned) > MAX_INPUT_CHARS:
        return FilterResult(
            False, f"消息有点长（超过 {MAX_INPUT_CHARS} 字），麻烦精简一下再发给我。"
        )

    lowered = cleaned.lower()
    if any(pattern in lowered for pattern in INJECTION_PATTERNS):
        return FilterResult(False, "我只能帮你规划城市漫步，没法执行改变我设定的指令哦。")

    return FilterResult(True, cleaned)
