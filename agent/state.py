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


class AgentState(BaseModel):
    """Minimal state kept between turns."""

    location: Location | None = None
    weather: WeatherInfo | None = None
    candidates: list[Candidate] = Field(default_factory=list)
    conversation_history: list[dict] = Field(default_factory=list)

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
