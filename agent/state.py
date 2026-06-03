"""Structured state used across agent turns."""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _now_iso() -> str:
    """当前 UTC 时间的 ISO 字符串，用作偏好/反馈的时效性标记（阶段三）。

    用字符串而非 datetime 对象，是为了和项目里其他时间字段（如 start_time）保持一致，
    也让 JSON 持久化天然可读、无需自定义序列化。
    """

    return datetime.now(timezone.utc).isoformat()


class Location(BaseModel):
    """A geographic coordinate."""

    lat: float
    lng: float


class WeatherInfo(BaseModel):
    """Current weather at the active location."""

    condition: str
    temp: float
    humidity: int | None = None


class Candidate(BaseModel):
    """A place returned by a search tool."""

    place_id: str
    name: str
    category: str
    distance_meters: float
    rating: float | None = None
    brief: str
    source_query: str | None = None


class TimeConstraints(BaseModel):
    """路线规划用到的时间约束（阶段二）。

    所有字段都可选：用户没说就留空，由 LLM 在计划阶段给出合理默认值，不要自行脑补硬填。
    """

    start_time: str | None = None  # 出发时间，形如 10:00
    end_time: str | None = None    # 期望结束时间
    pace: str | None = None        # 节奏偏好，如 轻松 / 紧凑


class RouteStep(BaseModel):
    """路线中的一站（阶段二）。

    分两部分：前半段是计划阶段由 LLM 给出的意图（去哪类地方、停多久、为什么排这），
    后半段（place_id/name 等）是执行阶段调用搜索工具后回填的真实地点信息。
    """

    # —— 计划阶段：LLM 产出 ——
    order: int                          # 第几站，从 1 开始
    category: str                       # 对应 search_places 的类别枚举
    keyword: str | None = None          # 细化关键词，如 拉面、书店
    stay_minutes: int | None = None     # 建议停留时长（分钟）
    transport_hint: str | None = None   # 到下一站的交通方式提示，如 步行 8 分钟
    reason: str = ""                    # 为什么把这一站排在这个位置

    # —— 执行阶段：搜索工具回填 ——
    place_id: str | None = None
    name: str | None = None
    brief: str | None = None
    distance_meters: float | None = None


class RoutePlan(BaseModel):
    """一条完整的漫步路线（阶段二）。

    这是 Plan-and-Solve 的计划产物：先整体成形，再逐站执行。独立成对象是为了能被
    序列化给用户看、被用户否决修改（为阶段三的 Reflexion 反馈学习埋下接口）。
    """

    summary: str = ""                   # 一句话概括这条路线
    start_time: str | None = None       # 路线出发时间
    total_minutes: int | None = None    # 预计总时长
    steps: list[RouteStep] = Field(default_factory=list)


class VetoRecord(BaseModel):
    """一条会话级否决反馈（阶段三 Reflexion）。

    当用户拒绝某个推荐（"第三个太远了"）时记下来：否决了什么、推断的原因。
    用于本会话内后续生成时规避，会随会话一起持久化（同一会话重启仍生效），
    但不沉淀为跨会话的长期偏好——那是 distill 阶段的事。
    """

    target: str                          # 被否决的对象：地点名或用户的描述
    reason: str = ""                    # 推断出的否决原因，如 太远 / 不喜欢这类
    turn: int = 0                        # 发生在第几轮，便于排查
    created_at: str = Field(default_factory=_now_iso)  # 时效性标记


class PreferenceItem(BaseModel):
    """一条跨会话沉淀的抽象偏好（阶段三长期记忆）。

    与 VetoRecord 的区别：VetoRecord 是"这次否决了哪家具体的店"，PreferenceItem 是
    从多次反馈里提炼出的稳定倾向（"不爱酒吧""走不动远路"）。updated_at 用于冲突时取近期。
    """

    tag: str                             # 抽象偏好描述，如 不爱酒吧 / 爱书店 / 走不动远路
    polarity: str = "dislike"           # like（喜欢）或 dislike（不喜欢）
    updated_at: str = Field(default_factory=_now_iso)  # 近期反馈优先靠它判断


class UserPreferences(BaseModel):
    """某个用户的长期偏好集合（阶段三）。

    刻意独立成对象：它按"用户"而非"会话"持久化，生命周期比会话长，绝不随会话清除。
    """

    items: list[PreferenceItem] = Field(default_factory=list)


class AgentState(BaseModel):
    """Minimal state kept between turns.

    阶段二起，state 不再只服务单点推荐：route_plan / time_constraints 仅在路线规划
    意图下才会被写入，单点推荐与闲聊场景保持为空，避免无谓的序列化开销。

    阶段三新增两类记忆：session_feedback 随会话持久化；preferences 在运行期由
    PreferenceStore 注入、按用户单独持久化，因此不随会话状态一起落盘（见 SessionStore）。
    """

    location: Location | None = None
    weather: WeatherInfo | None = None
    candidates: list[Candidate] = Field(default_factory=list)
    conversation_history: list[dict] = Field(default_factory=list)

    # —— 路线规划扩展字段（阶段二）：非路线场景下保持 None ——
    route_plan: RoutePlan | None = None
    time_constraints: TimeConstraints | None = None

    # —— 反馈与偏好（阶段三）——
    # session_feedback 属于本会话，随会话存盘；preferences 属于用户，单独存盘、运行期注入。
    session_feedback: list[VetoRecord] = Field(default_factory=list)
    preferences: UserPreferences = Field(default_factory=UserPreferences)

    def add_message(self, role: str, content: str) -> None:
        """Append one conversation message to history."""

        self.conversation_history.append({"role": role, "content": content})

    def get_turn_count(self) -> int:
        """Return the number of user turns in history."""

        return sum(
            1
            for message in self.conversation_history
            if message.get("role") == "user"
        )
