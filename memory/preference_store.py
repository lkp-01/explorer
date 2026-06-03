"""用户长期偏好的磁盘持久化（阶段三）。

和 SessionStore 是一对"姊妹"，但刻意分开：
- SessionStore 按 session_id 存"会话状态"，可随时删除、重开。
- PreferenceStore 按 user_id 存"用户偏好"，生命周期比会话长，删会话不影响它。

这就是阶段三"持久化分层"的落地：同一种朴素的 JSON 文件存储，按**不同的键和目录**
拆成两套，靠生命周期把它们隔开。实现保持和 SessionStore 一致的极简风格。
"""

from __future__ import annotations

import logging
from pathlib import Path

from agent.state import UserPreferences

logger = logging.getLogger(__name__)


class PreferenceStore:
    """以"一个用户一个 JSON 文件"的方式持久化 UserPreferences。"""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, user_id: str) -> Path:
        # 与 SessionStore 一致：只保留安全字符，避免 user_id 里出现路径分隔符
        safe = "".join(c for c in user_id if c.isalnum() or c in "-_") or "default"
        return self.directory / f"{safe}.json"

    def load(self, user_id: str) -> UserPreferences:
        """读回用户偏好；文件不存在或损坏时返回空偏好（而非 None），方便调用方直接用。"""

        path = self._path(user_id)
        if not path.exists():
            return UserPreferences()
        try:
            return UserPreferences.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("偏好文件损坏，将忽略并重置：%s", path)
            return UserPreferences()

    def save(self, user_id: str, preferences: UserPreferences) -> None:
        """把用户偏好写盘；写失败只记日志，不影响主流程。"""

        path = self._path(user_id)
        try:
            path.write_text(preferences.model_dump_json(indent=2), encoding="utf-8")
        except Exception:
            logger.exception("用户偏好写入失败：%s", path)
