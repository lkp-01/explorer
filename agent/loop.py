"""城市漫步 Agent 的主编排循环：意图分流 → 各决策路径 → 质量自检 → 护栏。

本模块只负责"编排"：把一轮用户消息按意图分到对应路径，并统一套上输入/输出护栏、
质量自检与历史记录。具体的决策模式各有归属——
- 单点推荐 / 子任务的 ReAct 循环：agent.react
- 路线规划（Plan-and-Solve）：agent.planner
- 多方案 / 复杂请求（并行执行框架）：agent.parallel
"""

from __future__ import annotations

import logging

from agent.evaluator import evaluate, run_with_evaluation
from agent.messages import build_messages, extract_text
from agent.parallel import run_complex_task, run_multi_plan
from agent.planner import resynthesize_route, run_route_plan
from agent.prompts import build_system_prompt
from agent.react import react_generate
from agent.reflexion import detect_veto
from agent.rewoo import run_fast_mode
from agent.router import Intent, classify
from agent.state import AgentState
from guardrails.input_filter import check_input
from guardrails.output_filter import clean_output
from models.client import LLMClient
from tools.registry import get_openai_tools

logger = logging.getLogger(__name__)

# 推荐类意图：会捕获否决反馈、并在生成时注入偏好/否决约束（阶段三、五共用）。
_RECOMMENDATION_INTENTS = (
    Intent.SINGLE_SPOT,
    Intent.ROUTE_PLAN,
    Intent.MULTI_PLAN,
    Intent.COMPLEX_TASK,
    Intent.FAST_MODE,
)


async def run_turn(
    client: LLMClient,
    state: AgentState,
    user_message: str,
    eval_client: LLMClient | None = None,
) -> tuple[str, AgentState]:
    """执行一轮用户消息处理，返回助手回复和更新后的状态。

    client 由调用方（main）构造一次后注入，循环本身不关心 provider 细节。
    输入先过护栏，输出再过护栏——一进一出守住 agent 与外界的边界。

    eval_client（阶段四）是评估专用客户端：推荐类路径产出回复后、机械护栏前走一次质量自检，
    不通过则带反馈重做。为 None 或评估关闭时自动跳过，行为退回阶段三。
    """

    updated_state = state.model_copy(deep=True)

    # —— 输入护栏：异常输入直接拦下，不进入昂贵的模型调用 ——
    checked = check_input(user_message)
    if not checked.ok:
        updated_state.add_message("user", user_message)
        updated_state.add_message("assistant", checked.message)
        return checked.message, updated_state
    user_message = checked.message

    # —— 意图分流（阶段一）：先判断这句话该走哪条路径，再决定用哪套 prompt、是否进工具循环 ——
    intent = await classify(client, user_message)

    # —— Reflexion 反馈捕获（阶段三）：在有上一轮推荐、且本轮是推荐类意图时，
    #    先看用户是不是在否决某个推荐。是就记一条会话级反馈，下面生成时由 prompt 注入规避。——
    if intent in _RECOMMENDATION_INTENTS and (
        updated_state.candidates or updated_state.route_plan is not None
    ):
        veto = await detect_veto(client, updated_state, user_message)
        if veto is not None:
            updated_state.session_feedback.append(veto)

    # —— 路线规划（阶段二）：走独立的 Plan-and-Solve 路径，不进通用 ReAct 循环 ——
    if intent is Intent.ROUTE_PLAN:
        try:
            final_reply = await run_route_plan(client, updated_state, user_message)
        except Exception:
            logger.exception("路线规划执行失败")
            # 排路线失败是错误兜底，不进质量自检（评估/重做对错误提示没有意义）
            final_reply = "我这边暂时排不出这条路线，请稍后再试或换个区域。"
        else:
            # —— 阶段四：路线成形后做一次质量自检，不通过则基于已回填路线重做 ——
            async def _regenerate_route(feedback: str) -> str:
                return await resynthesize_route(client, updated_state, feedback)

            final_reply = await run_with_evaluation(
                eval_client,
                intent,
                user_message,
                final_reply,
                updated_state,
                _regenerate_route,
            )
        return _finish_turn(updated_state, user_message, final_reply)

    # —— 多方案（阶段五 #3）：拆成多套变体并发执行，机械并排呈现，不走质量自检 ——
    if intent is Intent.MULTI_PLAN:
        try:
            final_reply = await run_multi_plan(client, updated_state, user_message)
        except Exception:
            logger.exception("多方案执行失败")
            final_reply = "我这边暂时排不出多套方案，请稍后再试或把需求说得更具体些。"
        return _finish_turn(updated_state, user_message, final_reply)

    # —— 复杂请求（阶段五 #6）：拆成多个子需求并发检索，LLM 交叉合并，内部已含质量自检 ——
    if intent is Intent.COMPLEX_TASK:
        try:
            final_reply = await run_complex_task(
                client, updated_state, user_message, eval_client=eval_client
            )
        except Exception:
            logger.exception("复杂请求执行失败")
            final_reply = "我这边暂时没能兼顾你的多个要求，请稍后再试或拆开来分别告诉我。"
        return _finish_turn(updated_state, user_message, final_reply)

    # —— 快速模式（阶段六 REWOO）：一次规划全部工具调用 → 批量执行 → 单次合成 ——
    # 与阶段四协调：只做轻量评估（格式+完整性）。不通过不在 REWOO 内部重做，而是降级为
    # 单点意图、贯穿到下方标准 ReAct 路径作为回退——REWOO 缺中间纠错，靠 ReAct 兜底。
    if intent is Intent.FAST_MODE:
        try:
            fast_reply = await run_fast_mode(client, updated_state, user_message)
        except Exception:
            logger.exception("REWOO 快速模式执行失败，回退标准 ReAct")
            fast_reply = ""

        if fast_reply.strip():
            result = await evaluate(
                eval_client,
                intent,
                user_message,
                fast_reply,
                updated_state,
                level="lightweight",
            )
            # 无评估客户端时不拦（评估只是增益）；通过则直接返回这版快回复
            if eval_client is None or result.passed:
                return _finish_turn(updated_state, user_message, fast_reply)
            logger.info("REWOO 轻量评估未通过，降级为单点推荐回退")

        # 计划为空 / 回复为空 / 评估未过：降级为单点意图，落到下方标准 ReAct 路径
        intent = Intent.SINGLE_SPOT

    # —— 闲聊路径：只回一句，不进工具循环。调用 chat 时不传 tools，模型就不会去调天气/地点工具 ——
    if intent is Intent.CHITCHAT:
        try:
            system_prompt = build_system_prompt(intent)
            messages = build_messages(state, user_message, system_prompt)
            response = await client.chat(messages)
            final_reply = extract_text(response.choices[0].message)
        except Exception:
            logger.exception("闲聊路径执行失败")
            final_reply = "我在的，想去哪儿逛逛、想做点什么，告诉我就行。"
        return _finish_turn(updated_state, user_message, final_reply)

    # —— 单点推荐（默认路径）：注入偏好/否决后跑一遍单点 ReAct，再做质量自检 ——
    # 阶段三：把长期偏好 + 本会话否决注入系统提示词，单点推荐生成时遵守
    system_prompt = build_system_prompt(
        intent,
        preferences=updated_state.preferences,
        session_feedback=updated_state.session_feedback,
    )
    messages = build_messages(state, user_message, system_prompt)
    final_reply, is_normal_reply = await react_generate(
        client, updated_state, messages, get_openai_tools()
    )

    # —— 阶段四：单点推荐的质量自检。只评估正常推荐回复；不通过则带反馈重做。 ——
    # 重做不重跑工具：复用循环里已攒下工具结果的 messages，仅让模型基于已有候选重新合成。
    if is_normal_reply:

        async def _regenerate_single_spot(feedback: str) -> str:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"{feedback}\n"
                        "请基于上面已有的候选地点重新给出推荐，不要新增未出现的地点。"
                    ),
                }
            )
            # 重做阶段不传 tools：只重新合成文本，避免再次触发工具调用
            response = await client.chat(messages)
            revised = extract_text(response.choices[0].message)
            messages.append({"role": "assistant", "content": revised})
            return revised

        final_reply = await run_with_evaluation(
            eval_client,
            intent,
            user_message,
            final_reply,
            updated_state,
            _regenerate_single_spot,
        )

    return _finish_turn(updated_state, user_message, final_reply)


def _finish_turn(
    state: AgentState,
    user_message: str,
    final_reply: str,
) -> tuple[str, AgentState]:
    """各路径收尾：过一遍输出护栏、把这轮对话记入历史，返回 (回复, 状态)。

    所有决策路径都经由这里收口，保证输出护栏与历史记录逻辑只有一份、行为一致。
    """

    final_reply = clean_output(final_reply)
    state.add_message("user", user_message)
    state.add_message("assistant", final_reply)
    return final_reply, state
