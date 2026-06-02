"""意图分流（阶段一）：进入工具循环前，先判断用户这句话到底想干什么。

为什么要单独抽出这一层：
当前所有请求都走同一条 ReAct 工具循环，连"你好"这种闲聊也要白白进一遍循环、浪费模型调用。
分流之后，闲聊可以只回一句，单点推荐和路线规划各走各的 prompt。
这一层也是后续所有决策模式（Plan-and-Solve / REWOO 等）的入口——以后每加一种模式，
只需在这里多注册一个 Intent 和一条匹配规则，下游按枚举分支即可。
"""

from __future__ import annotations

import logging
from enum import Enum

from agent.messages import extract_text
from agent.prompts import ROUTER_PROMPT
from models.client import LLMClient

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    """用户意图枚举。

    继承 str 是个常见技巧：这样枚举成员既能当枚举用（Intent.CHITCHAT），
    又能直接和字符串比较、直接写进日志和 JSON，省去到处 .value 的麻烦。
    下游一律按枚举分支处理，不要在 loop 里散落字符串判断。
    后续阶段会在这里追加成员，例如 MULTI_PLAN（多方案）、FAST_MODE（赶时间快速出方案）。
    """

    CHITCHAT = "chitchat"        # 闲聊、问候、与漫步无关的对话
    SINGLE_SPOT = "single_spot"  # 单点推荐：找几个地方逛逛（升级前的默认行为）
    ROUTE_PLAN = "route_plan"    # 路线规划：把多个点串成一条有先后顺序的路线


# —— 规则兜底层的关键词表 ——
# 思路：先用关键词快速命中，命中不了再花钱调模型。每类配一组词，命中即返回。
# 路线类的词比单点类更"强"，所以在 _rule_classify 里优先判断路线，避免"规划一条路线"被当成单点。
_ROUTE_KEYWORDS = (
    "路线",
    "怎么走",
    "顺路",
    "串起来",
    "一条线",
    "逛一天",
    "逛一整天",
    "玩一天",
    "行程",
    "安排一天",
    "先去",
    "动线",
)

# 闲聊关键词。判定时还会额外要求句子很短（见下），避免"附近有好玩的吗"被误判成闲聊。
_CHITCHAT_KEYWORDS = (
    "你好",
    "您好",
    "嗨",
    "hi",
    "hello",
    "在吗",
    "你是谁",
    "谢谢",
    "再见",
    "拜拜",
)


def _rule_classify(text: str) -> Intent | None:
    """规则匹配层：只对"非常确定"的情况下判定，拿不准就返回 None 交给 LLM。

    返回 None 不代表"分类失败"，而是"规则层不敢下结论"，把决定权让给更聪明（但更贵）的模型。
    这样能在保证准确率的前提下，省掉相当一部分 LLM 调用。
    """

    lowered = text.lower()

    # 路线类关键词最"强"：一旦出现，基本可以确定是排路线，所以优先判断。
    if any(keyword in lowered for keyword in _ROUTE_KEYWORDS):
        return Intent.ROUTE_PLAN

    # 闲聊判定从严：必须同时满足"句子很短"和"命中问候词"，
    # 否则像"你好，附近有什么好吃的"这种其实是带需求的，不该被当成纯闲聊。
    if len(text) <= 10 and any(keyword in lowered for keyword in _CHITCHAT_KEYWORDS):
        return Intent.CHITCHAT

    # 其余情况规则层不表态，交给下面的 LLM 兜底。
    return None


def _parse_intent(raw: str) -> Intent:
    """把模型返回的文本解析成 Intent；无法识别时退回最安全的 SINGLE_SPOT。

    退回 SINGLE_SPOT 是刻意的选择：它正是项目升级前的默认行为。即使分类出错，
    用户也只是回到"老样子"的单点推荐，而不会得到一个坏掉的体验。
    """

    normalized = raw.strip().lower()

    # 模型可能多带引号、句号或解释，所以用 in 做容错匹配，而不是要求完全相等。
    for intent in Intent:
        if intent.value in normalized:
            return intent

    logger.warning("无法识别路由结果，退回 SINGLE_SPOT: %s", raw)
    return Intent.SINGLE_SPOT


async def classify(client: LLMClient, user_message: str) -> Intent:
    """判断单条用户消息的意图：先走规则兜底，命中不了再走一次轻量 LLM 调用。

    只看当前这一句话、不看历史——分流要快要便宜，复杂的上下文判断留给后面的工具循环。
    client 复用主循环那一个，分类不需要工具，所以调用时不传 tools。
    """

    # 第一层：规则。命中就直接返回，省下一次模型调用。
    ruled = _rule_classify(user_message)
    if ruled is not None:
        logger.info("router intent=%s source=rule", ruled.value)
        return ruled

    # 第二层：LLM。给一个极短的分类 prompt，要求模型只回一个枚举词。
    messages = [
        {"role": "system", "content": ROUTER_PROMPT},
        {"role": "user", "content": user_message},
    ]
    try:
        response = await client.chat(messages)
        intent = _parse_intent(extract_text(response.choices[0].message))
    except Exception:
        # 分类本身出错绝不能让整轮挂掉，退回默认意图，保证 agent 始终可用。
        logger.exception("路由分类调用失败，退回 SINGLE_SPOT")
        intent = Intent.SINGLE_SPOT

    logger.info("router intent=%s source=llm", intent.value)
    return intent
