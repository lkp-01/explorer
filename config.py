"""集中管理运行配置：所有环境变量只在这里读一次。

其他模块统一 `from config import config`，再也不直接碰 os.getenv。
换 provider、改默认值、加新配置项，都只动这一个文件。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL_NAME = "deepseek-v4-pro"
DEFAULT_QWEATHER_API_HOST = "devapi.qweather.com"


@dataclass(frozen=True)
class Config:
    """一份只读的运行配置快照。frozen=True 保证启动后不会被意外改写。"""

    # —— LLM（DeepSeek / 任意 OpenAI 兼容服务）——
    api_key: str | None
    base_url: str
    model: str
    max_tool_turns: int
    max_tokens: int

    # —— 输出质量自检 Evaluator-Optimizer（阶段四）——
    # 生成回复后、机械护栏前插入一次独立评估；不通过则带反馈重做，最多 eval_max_retries 次。
    eval_enabled: bool          # 总开关，延迟敏感场景可关
    eval_max_retries: int       # 不通过时最多重做几次（建议 1-2）
    eval_model: str             # 评估用模型，默认回退到主模型，可指向更便宜的
    eval_temperature: float     # 评估调用的 temperature，低温更稳定

    # —— 和风天气 ——
    qweather_api_key: str | None
    qweather_api_host: str

    # —— 腾讯位置服务 ——
    tencent_map_key: str | None

    # —— 记忆持久化（第 7 步）——
    session_dir: str
    session_id: str

    # —— 长期偏好持久化（阶段三）：按用户存，独立于会话 ——
    preference_dir: str
    user_id: str

    # —— 日志（第 8 步）——
    log_level: str
    log_file: str | None

    @property
    def has_llm_key(self) -> bool:
        """是否配置了可用的 LLM Key，main 据此决定要不要进入对话循环。"""

        return bool(self.api_key)


def load_config() -> Config:
    """读取 .env 与环境变量，组装成 Config。整个进程只应调用一次。"""

    load_dotenv()  # 唯一一处 load_dotenv

    return Config(
        api_key=os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=(
            os.getenv("DEEPSEEK_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or DEFAULT_BASE_URL
        ),
        model=(
            os.getenv("DEEPSEEK_MODEL")
            or os.getenv("OPENAI_MODEL")
            or DEFAULT_MODEL_NAME
        ),
        max_tool_turns=int(os.getenv("MAX_TOOL_TURNS", "8")),
        max_tokens=int(os.getenv("MAX_TOKENS", "1200")),
        eval_enabled=os.getenv("EVAL_ENABLED", "true").strip().lower() != "false",
        eval_max_retries=int(os.getenv("EVAL_MAX_RETRIES", "1")),
        # 没配 EVAL_MODEL 就复用主模型（DEEPSEEK_MODEL/OPENAI_MODEL），保证开箱即用
        eval_model=(
            os.getenv("EVAL_MODEL")
            or os.getenv("DEEPSEEK_MODEL")
            or os.getenv("OPENAI_MODEL")
            or DEFAULT_MODEL_NAME
        ),
        eval_temperature=float(os.getenv("EVAL_TEMPERATURE", "0.0")),
        qweather_api_key=os.getenv("QWEATHER_API_KEY"),
        qweather_api_host=os.getenv("QWEATHER_API_HOST") or DEFAULT_QWEATHER_API_HOST,
        tencent_map_key=os.getenv("TENCENT_MAP_KEY"),
        session_dir=os.getenv("SESSION_DIR", ".sessions"),
        session_id=os.getenv("SESSION_ID", "default"),
        preference_dir=os.getenv("PREFERENCE_DIR", ".preferences"),
        user_id=os.getenv("USER_ID", "default"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.getenv("LOG_FILE") or None,
    )


# 模块级单例：任何模块导入 config 时，配置已经就绪。
config = load_config()
