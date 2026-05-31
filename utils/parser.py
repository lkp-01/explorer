"""通用解析与类型转换工具（第 10 步）。

这些函数原本零散、重复地出现在 places.py 和 state_sync.py 里——比如 _float_or_zero
被抄了两遍、_text 和 _optional_float 各写一份。第 10 步的本意正是：当你发现某段解析
逻辑到处复制粘贴，就把它提取成一个公共模块。这里集中放"把外部接口返回的脏数据安全
转成干净 Python 值"的小工具，全项目共用一份实现。
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# 外部接口里"等价于空"的几种取值：None、空串、腾讯接口常见的空数组字面量 "[]"
_EMPTY_VALUES = (None, "", "[]")


def float_or_zero(value: object) -> float:
    """把数值安全转成 float，失败时返回 0.0（用于距离等必有默认值的字段）。"""

    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def optional_float(value: object) -> float | None:
    """把可选数值转成 float；空值或非法值返回 None（用于评分等可缺省字段）。"""

    if value in _EMPTY_VALUES:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def clean_text(value: object) -> str:
    """把可能是空数组/空值的字段统一转成字符串（腾讯接口常把空字段返回成 []）。"""

    if value in _EMPTY_VALUES or isinstance(value, list):
        return ""
    return str(value)


def safe_json_loads(raw: str, default: Any = None) -> Any:
    """容错地解析 JSON 字符串，失败时记日志并返回 default。"""

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("无法解析 JSON: %s", raw)
        return default
