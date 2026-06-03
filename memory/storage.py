"""会话状态的磁盘持久化：重启程序不丢上下文（第 7 步）。

设计刻意保持朴素：把整个 AgentState 序列化成一个 JSON 文件，按 session_id 命名。
启动时尝试读回，每轮之后写出。pydantic 自带 model_dump_json / model_validate_json，
所以这里几乎不用手写序列化——这正是当初用 pydantic 给 state 建模换来的回报。
"""

from __future__ import annotations

import logging
from pathlib import Path

from agent.state import AgentState

logger = logging.getLogger(__name__)


class SessionStore:
    """以"一个会话一个 JSON 文件"的方式持久化 AgentState。"""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        # 只保留安全字符，避免 session_id 里出现路径分隔符等危险内容
        safe = "".join(c for c in session_id if c.isalnum() or c in "-_") or "default"
        return self.directory / f"{safe}.json"

    def load(self, session_id: str) -> AgentState | None:
        """读回会话；文件不存在或损坏时返回 None，让调用方新开一个会话。"""

        path = self._path(session_id)
        if not path.exists():
            return None
        try:
            return AgentState.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("会话文件损坏，将忽略并重新开始：%s", path)
            return None

    def save(self, session_id: str, state: AgentState) -> None:
        """把当前会话写盘；写失败只记日志，不影响主流程继续对话。

        阶段三关键不变量：排除 preferences。长期偏好按用户、由 PreferenceStore 单独存盘，
        绝不混进会话文件——否则删除/重开会话就会连带丢掉用户攒下来的偏好。
        session_feedback 属于本会话，照常一起存。
        """

        path = self._path(session_id)
        try:
            path.write_text(
                state.model_dump_json(indent=2, exclude={"preferences"}),
                encoding="utf-8",
            )
        except Exception:
            logger.exception("会话状态写入失败：%s", path)
