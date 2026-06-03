"""并行执行框架（阶段五）：把一个请求拆成多个独立子任务，并发执行后聚合。

合并了原计划 #3（多方案）与 #6（任务分解）——二者底层机制相同（拆子任务 → 并发跑 →
合并结果），只是上游语义不同：

- 多方案（MULTI_PLAN）：同一需求 × 不同约束，产出几套变体方案，平行呈现给用户自己挑。
  聚合是机械拼接（带小标题并排列出），不走 LLM。
- 复杂请求（COMPLEX_TASK）：一个需求拆成多个子需求，并发检索后由一次 LLM 调用交叉合并成
  一份统一结果（照顾"能同时满足多个子需求"的地点）。

关键设计：
- 并发只发生在子任务之间；每个子任务内部仍是顺序的单点 ReAct（react.react_generate）。
- 每个子任务跑在 state 的深拷贝上，彼此互不写脏；跑完后把候选并回主 state，保证持久化
  与后续轮次仍能看到这一轮检索到的地点。
- 子任务数量设上限（MAX_SUBTASKS），并对拆解失败/拆不出多个时回退到单次单点推荐——
  并行只是增益，绝不能因为拆解出问题而让整轮挂掉。
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

from agent.evaluator import run_with_evaluation
from agent.messages import build_messages, extract_text
from agent.prompts import (
    COMPLEX_TASK_AGGREGATE_PROMPT,
    COMPLEX_TASK_DECOMPOSE_PROMPT,
    MULTI_PLAN_DECOMPOSE_PROMPT,
    build_system_prompt,
)
from agent.react import react_generate
from agent.router import Intent
from agent.state import AgentState
from models.client import LLMClient
from tools.registry import get_openai_tools
from utils.parser import safe_json_loads, strip_json_fence

logger = logging.getLogger(__name__)

# 子任务数量上限：避免 LLM 过度拆分导致 LLM/外部 API 调用成本失控（计划明确要求 2-4）。
MAX_SUBTASKS = 4


@dataclass
class SubTask:
    """一个可独立执行的子任务：一句完整、自洽的需求描述 + 一个用于展示的简短名称。"""

    label: str
    instruction: str


@dataclass
class SubResult:
    """一个子任务的执行结果：名称、给用户的回复文本、以及跑完后的子状态（含候选）。"""

    label: str
    reply: str
    state: AgentState


def _decompose_context(state: AgentState) -> str:
    """拆解时给 LLM 的一点上下文：当前位置与天气，便于拆出贴合实际的子任务。"""

    snapshot = state.model_dump(mode="json", include={"location", "weather"})
    return (
        "当前位置与天气（供你拆解时参考，不必复述）：\n"
        f"{json.dumps(snapshot, ensure_ascii=False, default=str)}"
    )


async def _decompose(
    client: LLMClient,
    system_prompt: str,
    state: AgentState,
    user_message: str,
) -> list[SubTask]:
    """调用一次 LLM 把需求拆成子任务；拆不出或解析失败时返回空列表，由调用方回退。

    截断到 MAX_SUBTASKS：即便模型拆得过多，也只取前若干个，守住成本上限。
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"{_decompose_context(state)}\n\n用户需求：{user_message}",
        },
    ]

    try:
        response = await client.chat(messages)
        payload = safe_json_loads(strip_json_fence(extract_text(response.choices[0].message)))
    except Exception:
        logger.exception("子任务拆解调用失败，回退单次推荐")
        return []

    if not isinstance(payload, dict):
        logger.warning("子任务拆解结果不是 JSON 对象，回退单次推荐：%s", payload)
        return []

    subtasks: list[SubTask] = []
    for item in payload.get("subtasks") or []:
        if not isinstance(item, dict):
            continue
        instruction = str(item.get("instruction") or "").strip()
        if not instruction:
            continue
        label = str(item.get("label") or "").strip() or f"方案{len(subtasks) + 1}"
        subtasks.append(SubTask(label=label, instruction=instruction))
        if len(subtasks) >= MAX_SUBTASKS:
            logger.info("子任务数达到上限 %s，截断其余", MAX_SUBTASKS)
            break

    logger.info("decompose subtasks=%s", [task.label for task in subtasks])
    return subtasks


async def _run_subtask(
    client: LLMClient,
    base_state: AgentState,
    subtask: SubTask,
) -> SubResult:
    """在 base_state 的深拷贝上跑一遍单点 ReAct，返回该子任务的结果与子状态。

    深拷贝是并发安全的关键：每个子任务只写自己的副本，互不干扰。
    系统提示词沿用单点推荐那套，并注入长期偏好/本会话否决，保证各子任务都遵守用户偏好。
    """

    sub_state = base_state.model_copy(deep=True)
    system_prompt = build_system_prompt(
        Intent.SINGLE_SPOT,
        preferences=sub_state.preferences,
        session_feedback=sub_state.session_feedback,
    )
    messages = build_messages(sub_state, subtask.instruction, system_prompt)
    reply, _ = await react_generate(client, sub_state, messages, get_openai_tools())
    return SubResult(label=subtask.label, reply=reply, state=sub_state)


async def _run_subtasks(
    client: LLMClient,
    base_state: AgentState,
    subtasks: list[SubTask],
) -> list[SubResult]:
    """并发执行所有子任务。单个子任务抛异常不拖垮整体——它的结果被丢弃，其余照常返回。"""

    results = await asyncio.gather(
        *(_run_subtask(client, base_state, task) for task in subtasks),
        return_exceptions=True,
    )

    ok: list[SubResult] = []
    for task, result in zip(subtasks, results):
        if isinstance(result, Exception):
            logger.warning("子任务执行失败，已跳过：%s（%s）", task.label, result)
            continue
        ok.append(result)
    return ok


def _merge_candidates(base_state: AgentState, results: list[SubResult]) -> None:
    """把各子任务检索到的候选并回主 state（按 place_id 去重），并补齐天气。

    这样本轮并行检索到的地点不会随子状态一起丢掉：持久化后、后续轮次仍能引用。
    """

    by_place_id = {candidate.place_id: candidate for candidate in base_state.candidates}
    for result in results:
        for candidate in result.state.candidates:
            by_place_id[candidate.place_id] = candidate
        # 任一子任务取到天气、而主状态还没有时，顺手补上
        if base_state.weather is None and result.state.weather is not None:
            base_state.weather = result.state.weather
    base_state.candidates = list(by_place_id.values())


def _format_multi_plan(results: list[SubResult]) -> str:
    """多方案聚合：机械拼接，给每套方案加一个小标题并排列出，让用户自己挑。"""

    blocks = [f"【{result.label}】\n{result.reply.strip()}" for result in results if result.reply.strip()]
    if not blocks:
        return "我暂时没能排出多套方案，要不要把需求说得更具体一些？"

    header = f"我给你准备了 {len(blocks)} 套方案，供你对照挑选：\n"
    return header + "\n\n".join(blocks)


def _aggregate_context(user_message: str, results: list[SubResult]) -> str:
    """复杂请求聚合时喂给 LLM 的上下文：原始需求 + 各子需求的结果与候选锚点。"""

    parts = []
    for result in results:
        names = [c.name for c in result.state.candidates if c.name][:8]
        parts.append(
            {
                "label": result.label,
                "reply": result.reply,
                "candidate_names": names,
            }
        )
    payload = {"user_message": user_message, "subresults": parts}
    return (
        "用户的原始复杂需求与各子需求的推荐结果如下，请交叉合并：\n"
        f"{json.dumps(payload, ensure_ascii=False, default=str)}"
    )


async def _aggregate_complex(
    client: LLMClient,
    user_message: str,
    results: list[SubResult],
    feedback: str | None = None,
) -> str:
    """复杂请求聚合：用一次 LLM 调用把各子结果交叉过滤、合并成一份统一回复。

    feedback 非空时（阶段四质量自检不通过后的重做）一并交给模型，要求据此改进，
    但仍只基于已有子结果，不重跑子任务、不新增地点。
    """

    user_content = _aggregate_context(user_message, results)
    if feedback:
        user_content = (
            f"{user_content}\n\n{feedback}\n"
            "请只基于上面已有的子结果改写，不要新增未出现的地点。"
        )
    messages = [
        {"role": "system", "content": COMPLEX_TASK_AGGREGATE_PROMPT},
        {"role": "user", "content": user_content},
    ]
    response = await client.chat(messages)
    return extract_text(response.choices[0].message)


async def _fallback_single(
    client: LLMClient,
    state: AgentState,
    user_message: str,
) -> str:
    """拆不出多个子任务时的回退：就当普通单点推荐跑一次，直接作用在主 state 上。

    这样即便并行框架"没派上用场"，用户依然能得到一个正常的推荐，而不是空手而归。
    """

    logger.info("并行拆解未得到多个子任务，回退单点推荐")
    system_prompt = build_system_prompt(
        Intent.SINGLE_SPOT,
        preferences=state.preferences,
        session_feedback=state.session_feedback,
    )
    messages = build_messages(state, user_message, system_prompt)
    reply, _ = await react_generate(client, state, messages, get_openai_tools())
    return reply


async def run_multi_plan(
    client: LLMClient,
    state: AgentState,
    user_message: str,
) -> str:
    """多方案入口（原 #3）：拆成多套变体 → 并发执行 → 机械并排呈现。

    供 loop 在意图判定为 MULTI_PLAN 时调用。state 会被就地更新（合并各方案的候选）。
    多方案天然就是"多个"，不走质量自检（评估器的 intent_match 维度会与之冲突）。
    """

    subtasks = await _decompose(client, MULTI_PLAN_DECOMPOSE_PROMPT, state, user_message)
    if len(subtasks) < 2:
        return await _fallback_single(client, state, user_message)

    results = await _run_subtasks(client, state, subtasks)
    if not results:
        return await _fallback_single(client, state, user_message)

    _merge_candidates(state, results)
    return _format_multi_plan(results)


async def run_complex_task(
    client: LLMClient,
    state: AgentState,
    user_message: str,
    eval_client: LLMClient | None = None,
) -> str:
    """复杂请求入口（原 #6）：拆成多个子需求 → 并发检索 → LLM 交叉合并 →（可选）质量自检。

    供 loop 在意图判定为 COMPLEX_TASK 时调用。state 会被就地更新（合并各子需求的候选）。
    合并产出一份统一回复，故走完整质量自检；不通过则基于已有子结果重做聚合（不重跑子任务）。
    """

    subtasks = await _decompose(client, COMPLEX_TASK_DECOMPOSE_PROMPT, state, user_message)
    if len(subtasks) < 2:
        return await _fallback_single(client, state, user_message)

    results = await _run_subtasks(client, state, subtasks)
    if not results:
        return await _fallback_single(client, state, user_message)

    _merge_candidates(state, results)
    final_reply = await _aggregate_complex(client, user_message, results)

    # —— 阶段四：合并结果走一次质量自检，不通过则基于已有子结果重新聚合 ——
    async def _regenerate(feedback: str) -> str:
        return await _aggregate_complex(client, user_message, results, feedback)

    return await run_with_evaluation(
        eval_client,
        Intent.COMPLEX_TASK,
        user_message,
        final_reply,
        state,
        _regenerate,
    )
