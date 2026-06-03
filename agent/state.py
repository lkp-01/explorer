"""Structured state used across agent turns."""

from pydantic import BaseModel, Field


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


class AgentState(BaseModel):
    """Minimal state kept between turns.

    阶段二起，state 不再只服务单点推荐：route_plan / time_constraints 仅在路线规划
    意图下才会被写入，单点推荐与闲聊场景保持为空，避免无谓的序列化开销。
    """

    location: Location | None = None
    weather: WeatherInfo | None = None
    candidates: list[Candidate] = Field(default_factory=list)
    conversation_history: list[dict] = Field(default_factory=list)

    # —— 路线规划扩展字段（阶段二）：非路线场景下保持 None ——
    route_plan: RoutePlan | None = None
    time_constraints: TimeConstraints | None = None

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
