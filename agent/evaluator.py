"""Evaluator-Optimizer：输出质量自检（阶段四）。

在最终回复发给用户之前，用一次**独立**的 LLM 调用按检查清单审核质量；不通过就把
评估意见作为 feedback 注入，让生成端重做（最多 config.eval_max_retries 次），仍不
通过则放行当前最优版本——评估绝不能成为新的卡死点。

两个对外函数：
- evaluate：跑一次评估，返回结构化 EvalResult。任何异常都 fail-open（判 passed=True）。
- run_with_evaluation：评估-重做编排环。各生成路径只需传入一个 regenerate(feedback) 回调，
  本模块负责"评 → 不过就重做 → 再评"的循环与上限控制，对三条路径一视同仁。

设计要点：
- 评估器用独立客户端（models.build_evaluator_client），与生成端隔离，避免自评盲点。
- 评估结果是瞬态的，不写进 AgentState，避免无谓的序列化开销（与阶段二/三按需扩展 state 一致）。
- 重做不重跑工具：regenerate 由调用方实现，复用已攒下的工具结果/已回填的路线，只重合成文本。
"""

from __future__ import annotations

import json
import logging
from typing import Awaitable, Callable

from pydantic import BaseModel, Field

from agent.messages import extract_text
from agent.prompts import build_evaluator_prompt, format_eval_feedback_block
from agent.router import Intent
from agent.state import AgentState
from config import config
from models.client import LLMClient
from utils.parser import safe_json_loads, strip_json_fence

logger = logging.getLogger(__name__)

# 喂给评估器的"上一轮推荐对象"最多带几条，控制 prompt 长度
_MAX_CONTEXT_NAMES = 12

# 重做回调签名：吃一段反馈文本，产出新的一版回复
Regenerate = Callable[[str], Awaitable[str]]


class EvalIssue(BaseModel):
    """单个检查维度的评估结论。"""

    dimension: str = ""
    ok: bool = True
    detail: str = ""


class EvalResult(BaseModel):
    """一次质量自检的结构化结果（瞬态，不持久化）。"""

    passed: bool = True
    issues: list[EvalIssue] = Field(default_factory=list)

    def failed_issues(self) -> list[dict]:
        """返回未通过的维度（dict 形式），供 format_eval_feedback_block 拼反馈。"""

        return [issue.model_dump() for issue in self.issues if not issue.ok]


def _recommendation_names(state: AgentState) -> list[str]:
    """收集本轮回复涉及的地点名（路线各站 + 候选），给评估器一个核对锚点。"""

    names: list[str] = []
    if state.route_plan is not None:
        names.extend(step.name for step in state.route_plan.steps if step.name)
    names.extend(candidate.name for candidate in state.candidates if candidate.name)

    seen: set[str] = set()
    unique: list[str] = []
    for name in names:
        if name and name not in seen:
            seen.add(name)
            unique.append(name)
    return unique[:_MAX_CONTEXT_NAMES]


def _build_eval_context(
    intent: Intent,
    user_message: str,
    reply: str,
    state: AgentState,
) -> str:
    """组装评估所需的上下文：意图、用户原话、天气、已知偏好/否决、候选锚点、待评回复。"""

    weather = state.weather.model_dump() if state.weather else None
    preferences = [item.model_dump() for item in state.preferences.items]
    session_feedback = [record.model_dump() for record in state.session_feedback]

    context = {
        "intent": intent.value,
        "user_message": user_message,
        "weather": weather,
        "known_preferences": preferences,
        "session_vetoes": session_feedback,
        "recommended_place_names": _recommendation_names(state),
    }
    return (
        "审核所需上下文：\n"
        f"{json.dumps(context, ensure_ascii=False, default=str)}\n\n"
        f"待审核的回复：\n{reply}"
    )


async def evaluate(
    eval_client: LLMClient,
    intent: Intent,
    user_message: str,
    reply: str,
    state: AgentState,
    *,
    level: str = "full",
) -> EvalResult:
    """跑一次质量自检，返回结构化结果。

    任何异常或无法解析的输出都 fail-open（判 passed=True）：评估只是质量增益，
    绝不能因为评估自身出问题而阻断一版本可用的回复。
    """

    messages = [
        {"role": "system", "content": build_evaluator_prompt(intent, level)},
        {"role": "user", "content": _build_eval_context(intent, user_message, reply, state)},
    ]

    try:
        response = await eval_client.chat(messages, temperature=config.eval_temperature)
        payload = safe_json_loads(strip_json_fence(extract_text(response.choices[0].message)))
    except Exception:
        logger.exception("评估调用失败，按通过处理（fail-open）")
        return EvalResult(passed=True)

    if not isinstance(payload, dict):
        logger.warning("评估结果不是 JSON 对象，按通过处理：%s", payload)
        return EvalResult(passed=True)

    try:
        result = EvalResult.model_validate(payload)
    except Exception:
        logger.warning("评估结果字段不合法，按通过处理：%s", payload)
        return EvalResult(passed=True)

    # 容错：模型可能给了 ok=false 的维度却忘了把顶层 passed 设为 false，这里以维度为准收紧。
    if result.passed and any(not issue.ok for issue in result.issues):
        result.passed = False

    logger.info(
        "evaluator passed=%s failed_dims=%s",
        result.passed,
        [issue.dimension for issue in result.issues if not issue.ok],
    )
    return result


async def run_with_evaluation(
    eval_client: LLMClient | None,
    intent: Intent,
    user_message: str,
    reply: str,
    state: AgentState,
    regenerate: Regenerate,
    *,
    level: str = "full",
) -> str:
    """评估-重做编排环：评不过就带反馈重做，最多 config.eval_max_retries 次。

    短路条件（任一满足直接返回原 reply，零额外成本）：评估关闭、无评估客户端、回复为空。
    用尽重做次数仍不通过时，返回**最后一版** reply——保证用户总能拿到一个可读的回复。
    """

    if not config.eval_enabled or eval_client is None or not reply.strip():
        return reply

    current = reply
    for attempt in range(config.eval_max_retries + 1):
        result = await evaluate(
            eval_client, intent, user_message, current, state, level=level
        )
        if result.passed:
            return current

        if attempt >= config.eval_max_retries:
            logger.info("evaluator_exhausted attempts=%s，放行最后一版", attempt + 1)
            return current

        feedback = format_eval_feedback_block(result.failed_issues())
        try:
            revised = await regenerate(feedback)
        except Exception:
            logger.exception("评估后重做失败，放行当前版本")
            return current

        # 重做产出空内容时不替换，避免越改越差
        if revised and revised.strip():
            current = revised

    return current
