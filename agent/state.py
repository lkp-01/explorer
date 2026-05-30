"""定义城市漫步 Agent 在多轮对话中的结构化状态。"""

from pydantic import BaseModel, Field


class Location(BaseModel):
    """表示一个地理坐标点。"""

    lat: float
    lng: float


class WeatherInfo(BaseModel):
    """表示当前位置的实时天气信息。"""

    condition: str
    temp: float
    humidity: int | None = None


class TimeInfo(BaseModel):
    """表示当前时间的机器时间戳和可读展示文本。"""

    timestamp: float
    display: str


class Preference(BaseModel):
    """表示用户对地点、预算和体验的显式偏好。"""

    likes: list[str] = Field(default_factory=list)
    dislikes: list[str] = Field(default_factory=list)
    budget: str = ""


class Candidate(BaseModel):
    """表示一次工具搜索返回的候选地点。"""

    place_id: str
    name: str
    category: str
    distance_meters: float
    rating: float | None = None
    brief: str
    source_query: str | None = None


class ItineraryItem(BaseModel):
    """表示已经确认加入行程的地点条目。"""

    place_id: str
    name: str
    planned_time: str | None = None
    notes: str | None = None


class AgentState(BaseModel):
    """保存 Agent 推理、推荐和多轮对话所需的全部状态。"""

    user_id: str = "default"
    location: Location | None = None
    current_time: TimeInfo | None = None
    weather: WeatherInfo | None = None
    user_intent: str | None = None
    time_constraints: dict | None = None
    preferences: Preference = Field(default_factory=Preference)
    conversation_history: list[dict] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    rejected_candidates: list[str] = Field(default_factory=list)
    itinerary: list[ItineraryItem] = Field(default_factory=list)

    def reject_candidate(self, place_id: str) -> None:
        """把指定候选地点移出候选列表，并记录到拒绝列表。"""

        remaining_candidates: list[Candidate] = []
        rejected = False

        for candidate in self.candidates:
            if candidate.place_id == place_id:
                rejected = True
                continue
            remaining_candidates.append(candidate)

        self.candidates = remaining_candidates
        if rejected and place_id not in self.rejected_candidates:
            self.rejected_candidates.append(place_id)

    def confirm_candidate(
        self,
        place_id: str,
        planned_time: str | None = None,
        notes: str | None = None,
    ) -> None:
        """把指定候选地点加入行程，并从候选列表中移除。"""

        remaining_candidates: list[Candidate] = []
        confirmed_candidate: Candidate | None = None

        for candidate in self.candidates:
            if candidate.place_id == place_id and confirmed_candidate is None:
                confirmed_candidate = candidate
                continue
            remaining_candidates.append(candidate)

        self.candidates = remaining_candidates
        if confirmed_candidate is not None:
            self.itinerary.append(
                ItineraryItem(
                    place_id=confirmed_candidate.place_id,
                    name=confirmed_candidate.name,
                    planned_time=planned_time,
                    notes=notes,
                )
            )

    def clear_candidates(self) -> None:
        """清空当前候选地点列表。"""

        self.candidates = []

    def add_message(self, role: str, content: str) -> None:
        """把一条对话消息追加到历史记录。"""

        self.conversation_history.append({"role": role, "content": content})

    def get_turn_count(self) -> int:
        """返回历史记录中用户消息的轮次数。"""

        return sum(
            1
            for message in self.conversation_history
            if message.get("role") == "user"
        )
