"""Reflexion + 长期记忆（阶段三）：从否决里学习，把偏好沉淀下来。

两件事，对应两个时机：
- detect_veto：**每轮生成前**判断用户是不是在否决上一轮的某个推荐。是就记一条会话级反馈，
  下一次生成时由 prompt 注入规避（注入逻辑在 prompts.format_preference_block）。
- distill_preferences：**会话结束时**把本会话的零散否决，提炼成可跨会话复用的抽象偏好，
  与已有偏好合并（冲突时近期优先），交给 PreferenceStore 落盘。

为什么分两层：一次"第三个太远了"是具体反馈（会话级），而"这个人总嫌远、走不动远路"才是
值得长期记住的偏好（持久级）。前者立刻生效、随会话消失；后者要提炼、要跨会话活下去。
"""

from __future__ import annotations

import json
import logging

from agent.messages import extract_text
from agent.prompts import PREFERENCE_DISTILL_PROMPT, VETO_DETECTION_PROMPT
from agent.state import AgentState, PreferenceItem, UserPreferences, VetoRecord
from models.client import LLMClient
from utils.parser import safe_json_loads, strip_json_fence

logger = logging.getLogger(__name__)

# 喂给否决识别的"上一轮推荐"最多带几条，控制 prompt 长度
_MAX_RECENT_NAMES = 8
# 提炼出的长期偏好条数上限，和 prompt 里的约束保持一致
_MAX_PREFERENCE_ITEMS = 6


def _recent_recommendation_names(state: AgentState) -> list[str]:
    """收集上一轮可能被否决的对象名：路线各站 + 候选地点。

    给否决识别一个"锚点"，让模型能把"第三个""那家酒吧"对上具体地点。
    """

    names: list[str] = []
    if state.route_plan is not None:
        names.extend(step.name for step in state.route_plan.steps if step.name)
    names.extend(candidate.name for candidate in state.candidates if candidate.name)

    # 去重并截断，保持顺序
    seen: set[str] = set()
    unique: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique[:_MAX_RECENT_NAMES]


async def detect_veto(
    client: LLMClient,
    state: AgentState,
    user_message: str,
) -> VetoRecord | None:
    """判断用户这句是否在否决上一轮推荐；是则返回一条 VetoRecord，否则 None。

    调用方（loop）只在"有上一轮推荐可被否决"时才调用本函数，省掉无谓的模型调用。
    任何异常都吞掉返回 None：识别不出否决只是少学一条反馈，绝不能让整轮挂掉。
    """

    recent = _recent_recommendation_names(state)
    context = "、".join(recent) if recent else "（无）"
    messages = [
        {"role": "system", "content": VETO_DETECTION_PROMPT},
        {
            "role": "user",
            "content": f"上一轮推荐过的地点：{context}\n用户最新消息：{user_message}",
        },
    ]

    try:
        response = await client.chat(messages)
        payload = safe_json_loads(strip_json_fence(extract_text(response.choices[0].message)))
    except Exception:
        logger.exception("否决识别调用失败，跳过本轮反馈记录")
        return None

    if not isinstance(payload, dict) or not payload.get("is_veto"):
        return None

    # target 缺失时退回用户原话，至少留下"用户否决过"这一事实
    target = str(payload.get("target") or "").strip() or user_message.strip()
    reason = str(payload.get("reason") or "").strip()
    record = VetoRecord(target=target, reason=reason, turn=state.get_turn_count())
    logger.info("reflexion_veto target=%s reason=%s", record.target, record.reason)
    return record


def _merge_preferences(
    existing: UserPreferences,
    new_items: list[PreferenceItem],
) -> UserPreferences:
    """把新提炼的偏好合并进已有偏好：同 tag 用新的覆盖（新的 updated_at 更近）。

    冲突处理就落在这里——"以前说不爱日料，这次专门要日料"会以同 tag 的新条目覆盖旧条目，
    实现"近期反馈优先"。不同措辞的同义偏好无法自动归并，留给 distill 的 prompt 尽量收敛。
    """

    by_tag: dict[str, PreferenceItem] = {item.tag: item for item in existing.items}
    for item in new_items:
        by_tag[item.tag] = item  # 新条目带更近的 updated_at，直接覆盖同 tag 旧值
    return UserPreferences(items=list(by_tag.values()))


async def distill_preferences(
    client: LLMClient,
    state: AgentState,
) -> UserPreferences:
    """会话结束时调用：把本会话否决提炼成长期偏好，合并已有偏好后返回。

    没有任何新反馈时直接返回现有偏好，省掉一次模型调用。异常时也返回现有偏好，
    保证"提炼失败"最多是没学到新东西，绝不破坏已攒下的偏好。
    """

    existing = state.preferences or UserPreferences()
    if not state.session_feedback:
        return existing

    payload_in = {
        "existing_preferences": [item.model_dump() for item in existing.items],
        "session_feedback": [record.model_dump() for record in state.session_feedback],
    }
    messages = [
        {"role": "system", "content": PREFERENCE_DISTILL_PROMPT},
        {"role": "user", "content": json.dumps(payload_in, ensure_ascii=False, default=str)},
    ]

    try:
        response = await client.chat(messages)
        payload = safe_json_loads(strip_json_fence(extract_text(response.choices[0].message)))
    except Exception:
        logger.exception("偏好提炼调用失败，保留现有偏好")
        return existing

    if not isinstance(payload, dict):
        logger.warning("偏好提炼结果不是 JSON 对象，保留现有偏好：%s", payload)
        return existing

    new_items: list[PreferenceItem] = []
    for raw in payload.get("items", [])[:_MAX_PREFERENCE_ITEMS]:
        if not isinstance(raw, dict):
            continue
        tag = str(raw.get("tag") or "").strip()
        if not tag:
            continue
        polarity = "like" if str(raw.get("polarity")).strip().lower() == "like" else "dislike"
        # 新提炼的条目用默认 updated_at（当下），保证冲突时比旧条目"更近"
        new_items.append(PreferenceItem(tag=tag, polarity=polarity))

    merged = _merge_preferences(existing, new_items)
    logger.info("preferences_distilled count=%s", len(merged.items))
    return merged
