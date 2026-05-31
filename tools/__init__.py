"""工具包：导入本包即完成所有工具的注册。

工具通过 @register_tool 装饰器在"模块被导入时"写进全局注册表。集中到这里之后，
外部只需 `import tools`，注册就一定发生。新增工具只改这一处 import。
"""

from tools import geocode, places, weather  # noqa: F401  导入触发 register_tool 注册

__all__ = ["geocode", "places", "weather"]
