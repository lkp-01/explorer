"""MCP 客户端层（阶段七）：把外部 MCP server 的工具桥接进本地工具表。

—— 这个文件到底在干嘛（一句话）——
它就是 MCP 三角色里缺的那个 **Client**：负责"连上别人写好的 MCP server（Server 角色）→
问它有哪些工具 → 把这些工具包装成本项目的 ToolSpec，登记进 TOOL_REGISTRY"。
登记完之后，远程工具和本地的 get_weather / search_places 长得一模一样，
react 循环里的 get_openai_tools() / dispatch() 会一视同仁地带上它们——所以 agent 的
决策逻辑（ReAct / Plan-Solve / 并行 / REWOO）一行都不用改。

—— 它怎么把"调远程进程"伪装成"调本地函数"——
关键在 _make_caller 返回的那个闭包：dispatch() 里那句 spec.function(**arguments)，
对本地工具是直接执行函数，对 MCP 工具则是通过这个闭包把参数发给远程 server、再把结果取回来。
dispatch() 本来就 await 得了协程（见 registry.dispatch 里的 inspect.isawaitable），所以零改动。

—— 设计原则：MCP 是"增益"，绝不能成为新的卡死点 ——
没装 mcp 包 / 配置文件不存在 / 某个 server 连不上 / 某次调用抛错——任何一环出问题，
都只是"少几个工具"，agent 必须照常用本地工具工作。所以本文件到处是 fail-open 兜底，
且把 `import mcp` 放到真正要连接时才做（延迟导入），让没装包的人也能正常跑整个项目。

—— 阶段七补上的四道严谨性边界（外部 server 是"别人的进程"，必须设防）——
1. 超时：连接+握手有 connect 超时，单次调用有 call 超时（config 可调）。没有超时，
   一个卡死的 server 能把启动或一轮对话永远挂起——串行锁还会把后续调用全堵死。
2. 最小环境：server 子进程只拿到 SDK 的默认安全环境（PATH 等）+ 配置里显式给它的变量，
   不再继承完整 os.environ——否则本项目的全部 API Key 都泄给了第三方 server。
3. 工具白/黑名单：mcp_servers.json 里可按 server 配 allowTools / denyTools，
   不再盲信远程报上来的每一个工具（如 filesystem 的删改类工具）。
4. 翻页：list_tools 按协议带 cursor 翻页取全，不再只拿第一页。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import AsyncExitStack
from typing import Any

from config import config
from tools.registry import ToolSpec, register_tool_spec

logger = logging.getLogger(__name__)


def load_mcp_servers(config_path: str) -> dict[str, dict[str, Any]]:
    """读取 mcp_servers.json，返回 {server 名: {command, args, env?}}。

    文件不存在、不是合法 JSON、或缺少 mcpServers 字段时，都返回空 dict（而不是抛异常）——
    "没有可连的 server"是完全正常的状态，等价于"这次不接 MCP"。
    """

    if not os.path.exists(config_path):
        logger.info("未找到 MCP 配置文件 %s，跳过 MCP 接入", config_path)
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception:
        logger.exception("MCP 配置文件解析失败，跳过 MCP 接入：%s", config_path)
        return {}

    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        logger.warning("MCP 配置缺少 mcpServers 字段，跳过：%s", config_path)
        return {}

    # 过滤掉以 _ 开头的说明性键（mcp_servers.json 里写了 _说明 之类的注释字段）
    return {name: cfg for name, cfg in servers.items() if not name.startswith("_")}


def _extract_text(result: Any) -> str:
    """把 MCP call_tool 的返回（CallToolResult）抽取成纯文本。

    MCP 工具返回的是一组"内容块"（content blocks），可能是文本，也可能是图片等。
    这里只取文本块拼起来——对一个以文字推荐为输出的 agent 来说，文本就够用了。
    """

    content = getattr(result, "content", None) or []
    parts: list[str] = []
    for block in content:
        # 标准文本块：type == "text"，文本在 .text 里
        text = getattr(block, "text", None)
        if text:
            parts.append(str(text))
    return "\n".join(parts).strip()


class MCPManager:
    """管理所有 MCP server 连接的生命周期，并把它们的工具登记进 TOOL_REGISTRY。

    生命周期由 main 持有：进对话循环前 connect_all()，退出时 close()。
    注意：connect_all 与 close 必须在同一个 asyncio 任务里调用（main 协程里），
    因为 stdio 连接基于 anyio 的 cancel scope，跨任务关闭会报错——这是 MCP 客户端的常见坑。
    """

    def __init__(self) -> None:
        # AsyncExitStack 统一"按住"所有连接的上下文管理器，最后一次性、按相反顺序全部关闭。
        # 这是 MCP 官方 quickstart 推荐的写法：stdio 连接是必须保持打开的嵌套异步上下文，
        # 手动逐个 enter/exit 既啰嗦又容易漏关，交给 ExitStack 最稳。
        self._stack = AsyncExitStack()
        self.sessions: dict[str, Any] = {}          # server 名 -> ClientSession
        self._locks: dict[str, asyncio.Lock] = {}   # server 名 -> 串行锁（见下）
        self.tool_count = 0                          # 成功登记的远程工具总数，给启动提示用

    async def connect_all(self, servers: dict[str, dict[str, Any]]) -> None:
        """连接配置里的所有 server。单个 server 失败只跳过它，不影响其余 server 和本地工具。"""

        if not servers:
            return

        # —— 延迟导入：没装 mcp 包也不能让 import 本模块就崩。缺包时静默跳过整个 MCP。——
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            logger.warning("未安装 mcp 包（pip install mcp），本次跳过 MCP 接入。")
            return

        for name, cfg in servers.items():
            try:
                # 连接超时（阶段七）：拉子进程 + 握手 + 登记工具，整段限时。
                # 用 asyncio.timeout 而不是 wait_for——wait_for 会把协程挪进新任务执行，
                # 而 stdio 连接的 cancel scope 要求"进入"和"关闭"在同一个任务里（见类注释），
                # asyncio.timeout 在当前任务内原地取消，不破坏这个约束。
                async with asyncio.timeout(config.mcp_connect_timeout):
                    await self._connect_one(
                        name, cfg, ClientSession, StdioServerParameters, stdio_client
                    )
            except TimeoutError:
                logger.warning(
                    "连接 MCP server 超时（>%.0fs），已跳过：%s",
                    config.mcp_connect_timeout,
                    name,
                )
            except Exception:
                # 单个 server 连不上（命令不存在、初始化失败等）只记录并跳过，保证其余照常。
                logger.exception("连接 MCP server 失败，已跳过：%s", name)

    async def _connect_one(
        self,
        name: str,
        cfg: dict[str, Any],
        ClientSession: Any,
        StdioServerParameters: Any,
        stdio_client: Any,
    ) -> None:
        """连接单个 server：拉起子进程 → 建会话 → 初始化 → 登记它的工具。"""

        command = cfg.get("command")
        if not command:
            logger.warning("MCP server %s 缺少 command，跳过", name)
            return

        # env（阶段七收紧）：server 子进程只该拿到"能跑起来的最小环境 + 配置里显式给它的变量"。
        # 之前这里合并了完整 os.environ，等于把本项目所有 API Key 都交给第三方 server——
        # SDK 的 get_default_environment() 只含 PATH/TEMP 等安全变量（uvx/npx 靠它们就够了），
        # env=None 时 SDK 默认就用它；只有配置里给了额外变量时才需要手动合并。
        extra_env = cfg.get("env") or {}
        env: dict[str, str] | None = None
        if extra_env:
            try:
                from mcp.client.stdio import get_default_environment

                env = {**get_default_environment(), **extra_env}
            except ImportError:
                # 极老的 SDK 没有这个函数：退回旧行为并明确警告，而不是悄悄泄露。
                logger.warning(
                    "当前 mcp SDK 过旧，server %s 将继承完整环境变量（建议升级 mcp 包）",
                    name,
                )
                env = {**os.environ, **extra_env}
        params = StdioServerParameters(
            command=command,
            args=cfg.get("args", []),
            env=env,
        )

        # 进入并"按住"两层异步上下文：stdio 传输 + 会话。交给 _stack 在 close() 时统一关闭。
        read, write = await self._stack.enter_async_context(stdio_client(params))
        session = await self._stack.enter_async_context(ClientSession(read, write))

        # MCP 握手：必须先 initialize 才能 list/call。
        await session.initialize()

        self.sessions[name] = session
        self._locks.setdefault(name, asyncio.Lock())
        await self._register_tools(name, session, cfg)

    async def _register_tools(
        self,
        server_name: str,
        session: Any,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        """问 server 要工具清单，过滤后逐个桥接成 ToolSpec 登记进 TOOL_REGISTRY。

        过滤（阶段七）：cfg 里可配 allowTools（白名单，配了就只放行名单内的）和
        denyTools（黑名单，名单内的一律不要）。两者都针对远程原始工具名（不带前缀）。
        白名单是更严谨的默认姿势——名单写错顶多少工具，黑名单写漏则多放行一个。
        被过滤掉的工具逐个记日志，保证"少了工具"可见、可排查，而不是悄悄消失。
        """

        self._locks.setdefault(server_name, asyncio.Lock())

        cfg = cfg or {}
        allow = cfg.get("allowTools")
        deny = set(cfg.get("denyTools") or [])

        # 翻页（阶段七）：list_tools 按 MCP 协议可能分页（nextCursor），循环取全。
        # 假如 server 不分页（绝大多数），第一轮 nextCursor 为空，循环只走一次。
        tools: list[Any] = []
        cursor: str | None = None
        while True:
            listed = (
                await session.list_tools(cursor=cursor)
                if cursor
                else await session.list_tools()
            )
            tools.extend(listed.tools)
            cursor = getattr(listed, "nextCursor", None)
            if not cursor:
                break

        for tool in tools:
            if allow is not None and tool.name not in allow:
                logger.info("MCP 工具不在 allowTools 白名单，跳过：%s/%s", server_name, tool.name)
                continue
            if tool.name in deny:
                logger.info("MCP 工具命中 denyTools 黑名单，跳过：%s/%s", server_name, tool.name)
                continue

            # 命名空间防撞名：统一加 mcp__<server>__<tool> 前缀（学 Claude 的做法）。
            # 这样即便某个 server 有个叫 search 的工具，也不会和本地工具冲突，来源还一眼可辨。
            full_name = f"mcp__{server_name}__{tool.name}"

            # MCP 工具的 inputSchema 本身就是 JSON Schema，正好等于 OpenAI tools 要的 parameters。
            parameters = getattr(tool, "inputSchema", None) or {
                "type": "object",
                "properties": {},
            }

            registered = register_tool_spec(
                ToolSpec(
                    name=full_name,
                    description=getattr(tool, "description", "") or "",
                    parameters=parameters,
                    function=self._make_caller(server_name, session, tool.name),
                )
            )
            if registered:
                self.tool_count += 1
                logger.info("已登记 MCP 工具：%s", full_name)

    def _make_caller(self, server_name: str, session: Any, remote_name: str):
        """造一个"看起来像本地函数、实际调远程 server"的异步闭包。

        这就是 MCP 桥接的核心：dispatch() 调 spec.function(**arguments) 时，
        实际走到这里，把参数发给远程 server 的 remote_name 工具，再把结果取回、抽成文本。
        """

        async def _call(**kwargs: Any) -> dict[str, Any]:
            lock = self._locks.setdefault(server_name, asyncio.Lock())
            # 串行锁：一个 ClientSession 不保证并发安全，而阶段五的并行框架会让多个子任务
            # 同时 dispatch、可能并发打到同一个 server。用锁把同一 session 的调用串起来，
            # 思路和 places.py 里的 _TENCENT_GATE 信号量一致——保护下游、避免串包。
            try:
                async with lock:
                    # 调用超时（阶段七）：只给"真正在等远程"的时间设限，排队等锁不算——
                    # 否则前一个慢调用会让排队者无辜超时。超时放在锁内还有个关键作用：
                    # 把卡死的调用掐掉、释放锁，该 server 的后续调用才不会被堵死。
                    async with asyncio.timeout(config.mcp_call_timeout):
                        result = await session.call_tool(remote_name, kwargs)
            except TimeoutError:
                return {
                    "error": (
                        f"MCP 工具调用超时（>{config.mcp_call_timeout:.0f}s）：{remote_name}"
                    ),
                    "fallback": True,
                    "provider": "mcp",
                }

            text = _extract_text(result)

            # 与本地工具统一的"结构化错误"约定：MCP 报错时也返回带 error/fallback 的 dict，
            # 这样上层（react 循环、日志）对所有工具的失败处理是一致的。
            if getattr(result, "isError", False):
                return {
                    "error": text or "MCP 工具调用返回错误",
                    "fallback": True,
                    "provider": "mcp",
                }
            return {"content": text, "provider": "mcp"}

        return _call

    async def close(self) -> None:
        """干净地断开所有 MCP 连接（子进程随之退出）。退出路径必调，避免残留僵尸进程。"""

        try:
            await self._stack.aclose()
        except Exception:
            # 关闭异常无所谓（进程都要退了），记一笔即可，绝不能让它盖过正常退出。
            logger.warning("关闭 MCP 连接时出现异常（已忽略）")
        finally:
            self.sessions.clear()
            self._locks.clear()
