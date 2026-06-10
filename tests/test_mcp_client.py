"""MCP 客户端层的最小单测（阶段七）。

为什么这样测：真实的 MCP 接入要拉起子进程（uvx/npx）、走 stdio 协议，慢且依赖外部环境，
不适合放进单测。但 MCP 桥接的"核心契约"其实和子进程无关——它是：
  list_tools() 报上来的工具，能不能被正确登记进 TOOL_REGISTRY，
  并且 dispatch() 调用时，能不能通过闭包把参数发给 session、把结果取回来。

所以这里用一个**假的 ClientSession**（不联网、不起进程，按预设返回），直接验证这条契约。
覆盖六件事：
1. load_mcp_servers 能读配置、并过滤掉 _ 开头的说明字段、文件缺失时返回空；
2. _register_tools 能把远程工具以 mcp__<server>__<tool> 的名字登记进 TOOL_REGISTRY；
3. dispatch 调用 MCP 工具时，正常结果与错误结果都能正确桥接；
4. allowTools / denyTools 能按配置过滤远程工具（阶段七）；
5. list_tools 带 nextCursor 分页时能翻页取全（阶段七）；
6. 单次调用超时返回结构化错误，不挂起、不抛异常（阶段七）。

本文件不依赖 pytest，直接 `python tests/test_mcp_client.py` 即可运行。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile

# 让测试无论从哪个目录运行都能 import 到项目包（与 test_router.py 同款处理）。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.mcp_client import (  # noqa: E402
    MCPManager,
    _extract_text,
    load_mcp_servers,
)
from tools.registry import TOOL_REGISTRY, dispatch  # noqa: E402


# —— 一组假对象：模拟 mcp SDK 里 list_tools()/call_tool() 返回的结构 ——
class _FakeBlock:
    """模拟一个文本内容块：MCP 的 call_tool 结果由若干这样的块组成。"""

    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeCallResult:
    """模拟 call_tool 的返回（CallToolResult）：content 是内容块列表，isError 标记是否出错。"""

    def __init__(self, text: str, *, is_error: bool = False) -> None:
        self.content = [_FakeBlock(text)]
        self.isError = is_error


class _FakeTool:
    """模拟 list_tools() 里的一个工具描述。"""

    def __init__(self, name: str, description: str, input_schema: dict) -> None:
        self.name = name
        self.description = description
        self.inputSchema = input_schema


class _FakeListResult:
    def __init__(self, tools: list) -> None:
        self.tools = tools


class _FakeSession:
    """假的 ClientSession：不联网、不起进程，按预设返回，用来测桥接契约。

    call_tool 会把收到的参数回显进结果文本，方便断言"参数确实传到了远程"。
    raise_on_call=True 时模拟一次调用异常；error_result=True 时模拟 isError 的返回。
    """

    def __init__(self, *, raise_on_call: bool = False, error_result: bool = False) -> None:
        self._raise = raise_on_call
        self._error = error_result
        self.calls: list[tuple[str, dict]] = []  # 记录被调过哪些工具，验证锁/转发

    async def list_tools(self) -> _FakeListResult:
        return _FakeListResult(
            [
                _FakeTool(
                    name="echo",
                    description="把输入原样返回（测试用）",
                    input_schema={
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                )
            ]
        )

    async def call_tool(self, name: str, arguments: dict) -> _FakeCallResult:
        self.calls.append((name, arguments))
        if self._raise:
            raise RuntimeError("模拟远程调用异常")
        if self._error:
            return _FakeCallResult("远程报错了", is_error=True)
        return _FakeCallResult(f"echo: {json.dumps(arguments, ensure_ascii=False)}")


def _run(coro):
    return asyncio.run(coro)


def test_load_servers_filters_comment_keys_and_missing_file() -> None:
    """配置能读出 server、过滤掉 _ 开头的说明键；文件不存在时安全返回空。"""

    # 文件不存在 -> 空 dict（等价于"这次不接 MCP"，不报错）
    assert load_mcp_servers("／不存在的路径／mcp_servers.json") == {}

    # 写一个临时配置，含一个说明字段 _note 和一个真实 server
    payload = {
        "_note": "这是注释，应被过滤",
        "mcpServers": {
            "_说明": "这也是注释",
            "demo": {"command": "echo", "args": ["hi"]},
        },
    }
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    ) as handle:
        json.dump(payload, handle)
        temp_path = handle.name

    try:
        servers = load_mcp_servers(temp_path)
        assert set(servers.keys()) == {"demo"}, servers
        assert servers["demo"]["command"] == "echo"
    finally:
        os.remove(temp_path)


def test_register_tools_into_registry() -> None:
    """假 session 报上来的 echo 工具，应以 mcp__demo__echo 之名进入 TOOL_REGISTRY。"""

    manager = MCPManager()
    _run(manager._register_tools("demo", _FakeSession()))

    assert "mcp__demo__echo" in TOOL_REGISTRY, list(TOOL_REGISTRY.keys())
    spec = TOOL_REGISTRY["mcp__demo__echo"]
    # 描述与参数 schema 应原样带过来（参数 schema 直接用于喂给模型）
    assert spec.description == "把输入原样返回（测试用）"
    assert spec.parameters["required"] == ["text"]
    assert manager.tool_count == 1


def test_dispatch_bridges_to_remote() -> None:
    """dispatch 调用 MCP 工具时，应通过闭包把参数转发给 session，并取回桥接后的结果。"""

    manager = MCPManager()
    session = _FakeSession()
    _run(manager._register_tools("bridge", session))

    result_json = _run(dispatch("mcp__bridge__echo", {"text": "你好"}))
    payload = json.loads(result_json)

    # 参数确实传到了"远程"
    assert session.calls == [("echo", {"text": "你好"})], session.calls
    # 结果被桥接回来（content 里能看到回显），并打上 provider=mcp 标记
    assert payload["provider"] == "mcp"
    assert "你好" in payload["content"]


def test_dispatch_handles_error_result() -> None:
    """远程返回 isError 时，应桥接成本项目统一的结构化错误（error + fallback）。"""

    manager = MCPManager()
    _run(manager._register_tools("err", _FakeSession(error_result=True)))

    result_json = _run(dispatch("mcp__err__echo", {"text": "x"}))
    payload = json.loads(result_json)
    assert payload.get("fallback") is True
    assert payload.get("provider") == "mcp"
    assert "error" in payload


def test_extract_text_joins_text_blocks() -> None:
    """_extract_text 只取文本块并拼接；非文本/空内容安全返回空串。"""

    assert _extract_text(_FakeCallResult("一段文本")) == "一段文本"

    class _Empty:
        content = []

    assert _extract_text(_Empty()) == ""


class _FakeMultiToolSession(_FakeSession):
    """报多个工具的假 session，测 allowTools / denyTools 过滤。"""

    def __init__(self, names: list[str]) -> None:
        super().__init__()
        self._names = names

    async def list_tools(self) -> _FakeListResult:
        return _FakeListResult(
            [
                _FakeTool(name=name, description="", input_schema={"type": "object"})
                for name in self._names
            ]
        )


def test_allow_and_deny_tools_filtering() -> None:
    """allowTools 只放行名单内的；denyTools 一律拦下；名单针对远程原始名。"""

    # 白名单：三个工具只放行 read / write
    manager = MCPManager()
    session = _FakeMultiToolSession(["read", "write", "delete"])
    _run(
        manager._register_tools(
            "fs_allow", session, {"allowTools": ["read", "write"]}
        )
    )
    assert "mcp__fs_allow__read" in TOOL_REGISTRY
    assert "mcp__fs_allow__write" in TOOL_REGISTRY
    assert "mcp__fs_allow__delete" not in TOOL_REGISTRY
    assert manager.tool_count == 2

    # 黑名单：只拦 delete
    manager = MCPManager()
    session = _FakeMultiToolSession(["read", "delete"])
    _run(manager._register_tools("fs_deny", session, {"denyTools": ["delete"]}))
    assert "mcp__fs_deny__read" in TOOL_REGISTRY
    assert "mcp__fs_deny__delete" not in TOOL_REGISTRY
    assert manager.tool_count == 1

    # 不配名单：全部放行（与旧行为一致，老调用方不传 cfg 也安全）
    manager = MCPManager()
    _run(manager._register_tools("fs_open", _FakeMultiToolSession(["read", "delete"])))
    assert "mcp__fs_open__delete" in TOOL_REGISTRY
    assert manager.tool_count == 2


class _FakePagedSession(_FakeSession):
    """list_tools 分两页返回的假 session，测 nextCursor 翻页。"""

    async def list_tools(self, cursor: str | None = None) -> _FakeListResult:
        if cursor is None:
            page = _FakeListResult(
                [_FakeTool(name="page1_tool", description="", input_schema={})]
            )
            page.nextCursor = "page2"
            return page
        assert cursor == "page2", cursor
        return _FakeListResult(
            [_FakeTool(name="page2_tool", description="", input_schema={})]
        )


def test_list_tools_pagination() -> None:
    """server 分页报工具时，应跟着 nextCursor 翻页，两页工具都登记进来。"""

    manager = MCPManager()
    _run(manager._register_tools("paged", _FakePagedSession()))
    assert "mcp__paged__page1_tool" in TOOL_REGISTRY
    assert "mcp__paged__page2_tool" in TOOL_REGISTRY
    assert manager.tool_count == 2


class _FakeSlowSession(_FakeSession):
    """call_tool 故意拖时间的假 session，测单次调用超时。"""

    async def call_tool(self, name: str, arguments: dict) -> _FakeCallResult:
        await asyncio.sleep(5)
        return _FakeCallResult("不该走到这里")


def test_call_timeout_returns_structured_error() -> None:
    """远程调用超过 mcp_call_timeout 时，应返回 error+fallback 的结构化结果，而非挂起/抛异常。"""

    from config import config

    manager = MCPManager()
    _run(manager._register_tools("slow", _FakeSlowSession()))

    original = config.mcp_call_timeout
    object.__setattr__(config, "mcp_call_timeout", 0.05)  # frozen dataclass 的测试专用后门
    try:
        result_json = _run(dispatch("mcp__slow__echo", {"text": "x"}))
    finally:
        object.__setattr__(config, "mcp_call_timeout", original)

    payload = json.loads(result_json)
    assert payload.get("fallback") is True
    assert payload.get("provider") == "mcp"
    assert "超时" in payload.get("error", ""), payload


def main() -> None:
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n全部 {len(tests)} 个用例通过。")


if __name__ == "__main__":
    main()
