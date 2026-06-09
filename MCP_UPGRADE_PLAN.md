# 城市漫步 Agent —— MCP 升级计划

> 目标：让 explorer 的工具来源从「只有本地手写工具」扩展到「本地工具 + 远程 MCP 工具」，
> 且**不改动 agent 的任何决策逻辑**（ReAct / Plan-Solve / 并行 / REWOO 一行不动）。
> 本文先把 MCP 用本项目的代码讲清楚，再给出分支、逐文件改动清单、代码草图与验证方式。

分支：`feature/mcp`

---

## 0. 先把 MCP 讲明白（用你自己的代码讲）

你一直搞不清 MCP，是因为大多数资料把它讲成一个很玄的「协议」。但落到你这个项目，它其实只有一句话：

> **MCP = 一个让你的 `TOOL_REGISTRY` 能从"别人写好的进程"里动态拿到工具的标准。**

### 0.1 你现在的工具是怎么来的

看 [tools/weather.py](tools/weather.py)、[tools/places.py](tools/places.py)：每个工具都是你**亲手写的函数** + 一段 `@register_tool` 装饰器，在 `import tools` 时被塞进 [tools/registry.py](tools/registry.py) 的全局表 `TOOL_REGISTRY`。

```
你写代码 → @register_tool → TOOL_REGISTRY["search_places"] = ToolSpec(..., function=search_places)
```

工具要变多，**只能你自己写更多函数**。这就是「本地工具」。

### 0.2 MCP 改变了什么

MCP 定义了一套标准，让一个**独立的进程**（叫 MCP Server）能对外宣布："我有这些工具，它们的参数长这样，你按这个格式调我就行。"

于是你的 agent 多了一条**不用自己写代码**的工具来源：

```
别人写好的 MCP Server（比如"网页抓取""文件读写""地图"）
        ↓  你的 agent 启动时连上它，问一句 list_tools()
        ↓  把它报上来的工具，注册进同一个 TOOL_REGISTRY
TOOL_REGISTRY["mcp__fetch__fetch"] = ToolSpec(..., function=<去调那个进程>)
```

关键点：**注册进去之后，它和你的本地工具长得一模一样。** 你的 `get_openai_tools()` 会把它一起报给模型，`dispatch()` 会把它一起分发——ReAct 循环根本不知道这个工具是你写的还是 MCP 来的。

### 0.3 三个角色（对号入座）

| MCP 术语 | 在你项目里是谁 |
|---|---|
| **Host**（拥有 LLM、做决策的应用） | 你的 explorer agent（[agent/loop.py](agent/loop.py)） |
| **Client**（负责连接、收发协议消息的连接器） | **本次要新增的** `tools/mcp_client.py` |
| **Server**（提供工具/数据的独立进程） | 别人写好的，比如官方 `mcp-server-fetch`、`server-filesystem` |

你要做的，就是补上中间那个 **Client**，把 Server 的工具桥接进你已有的 registry。

### 0.4 Server 跑在哪？（Transport）

MCP Server 有两种连法：

- **stdio**：Server 是你本机启动的一个子进程（如 `uvx mcp-server-fetch`），通过标准输入输出通信。**本地开发首选，本计划用它。**
- **HTTP / SSE**：Server 是个远程服务，通过网络连。原理一样，只是换个连接方式。

---

## 1. 现状：一次工具调用的完整数据流

先看清楚工具在你代码里是怎么流动的，才能知道 MCP 该插在哪。以单点推荐为例（[agent/react.py](agent/react.py)）：

```
用户消息
  → router.classify()                         判定意图
  → build_messages() + get_openai_tools()      把"可用工具清单"报给模型   ← 接入点 ①
  → client.chat(messages, tools)               模型决定调用哪个工具
  → extract_tool_calls()                       拿到 {name, arguments}
  → dispatch(name, arguments)                  按名字找到函数并执行       ← 接入点 ②
  → sync_state_from_tool_result()              把结果回写 state（仅认识 3 个内置工具）
  → 结果塞回 messages，循环继续
```

**两个接入点 ① 和 ② 背后都是同一个 `TOOL_REGISTRY`。** 这意味着：

> 只要我能在启动时把 MCP 工具**注册进 `TOOL_REGISTRY`**，①②自动就把它们带上了，
> `react.py` / `loop.py` / `parallel.py` / `rewoo.py` **完全不用改**。

这就是为什么说你的架构对 MCP「天生友好」——你早就把工具抽象成了 registry。

---

## 2. 升级目标（一句话）

新增一个 MCP 客户端层：启动时连接配置好的 MCP Server，把它们的工具桥接成 `ToolSpec`
注册进 `TOOL_REGISTRY`；退出时干净地断开。**工具来源变两个，调用路径仍只有一条。**

---

## 3. 改动清单总览

| 文件 | 类型 | 改什么 |
|---|---|---|
| `requirements.txt` | 改 | 加 `mcp`（官方 Python SDK） |
| `config.py` | 改 | 加一段 MCP 配置：要连哪些 server |
| `mcp_servers.json`（新） | 加 | MCP server 清单（命令 + 参数），独立于代码 |
| `tools/mcp_client.py`（新） | **核心** | 连接 server、拉取工具、桥接成 ToolSpec 注册进 registry、管理生命周期 |
| `tools/registry.py` | 微改 | `register_tool` 之外，新增一个「直接注册一个 ToolSpec」的入口（给 MCP 用） |
| `main.py` | 改 | 启动时建立 MCP 连接、退出时断开（用 AsyncExitStack 管理） |
| `.env.example` | 改 | 增加开关 `MCP_ENABLED` 等说明 |
| **不动** | — | `react.py` / `loop.py` / `parallel.py` / `rewoo.py` / `router.py` / `evaluator.py` |

> 「不动」那一行，就是这次升级最值得在面试里讲的东西：**好的抽象让新能力变成"加一块"而不是"改一片"。**

---

## 4. 逐块说明：加什么、为什么、有什么用

### 4.1 `requirements.txt` —— 引入官方 SDK

```diff
 openai
 httpx
 pydantic>=2
 python-dotenv
+mcp            # MCP 官方 Python SDK：提供 ClientSession 和 stdio 连接
```

**有什么用**：`mcp` 包提供两样东西——`stdio_client`（启动并连上一个 server 子进程）和
`ClientSession`（在连接上做 `list_tools()` / `call_tool()`）。这是 Client 角色的轮子，不用自己造。

---

### 4.2 `mcp_servers.json`（新）—— 要连哪些 server

把「连哪些 server」做成配置而非硬编码，照搬 Claude Desktop / Cursor 的惯例：

```json
{
  "mcpServers": {
    "fetch": {
      "command": "uvx",
      "args": ["mcp-server-fetch"]
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "./itineraries"]
    }
  }
}
```

**为什么挑这两个 server（贴合"城市漫步"场景）**：

- **fetch**（官方）：能抓取网页内容。你的 agent 就能去拉**实时攻略/景点介绍**，而不只靠腾讯
  地点接口的简介——这正好呼应「RAG / 让 agent 接触实时外部知识」的方向。
- **filesystem**（官方）：能读写本地文件。你的 agent 就能把排好的路线**存成行程文件**。
  它还是个会产生「实际动作」的工具，是将来接 Human-in-the-Loop（执行前要用户确认）的天然示范。

**有什么用**：换 server、加 server 只动这个 JSON，不碰代码——和你 `config.py`「配置集中」的理念一致。

---

### 4.3 `config.py` —— 读取 MCP 配置

在 `Config` 里加几个字段（保持你「环境变量只读一次」的风格）：

```python
# —— MCP（feature/mcp）——
mcp_enabled: bool        # 总开关，无环境/网络时可一键关闭，行为退回纯本地工具
mcp_config_path: str     # mcp_servers.json 路径
```

`load_config()` 里补：

```python
mcp_enabled=os.getenv("MCP_ENABLED", "true").strip().lower() != "false",
mcp_config_path=os.getenv("MCP_CONFIG_PATH", "mcp_servers.json"),
```

**有什么用**：MCP 是「增益」，不能成为新的卡死点。`MCP_ENABLED=false` 或连不上时，
agent 必须照常用本地工具工作——这和你项目里 evaluator / 工具降级一脉相承的「fail-open」原则一致。

---

### 4.4 `tools/registry.py` —— 开一个给 MCP 用的注册入口

现在注册工具只有 `@register_tool` 装饰器一条路（适合「写函数时就知道工具长什么样」）。
MCP 工具是**运行时**才从 server 拿到的，没有函数可装饰，所以加一个直接登记 `ToolSpec` 的入口：

```python
def register_tool_spec(spec: ToolSpec) -> None:
    """直接注册一个已构造好的 ToolSpec（供 MCP 客户端在运行时登记远程工具）。"""
    if spec.name in TOOL_REGISTRY:
        logger.warning("工具名冲突，跳过注册：%s", spec.name)
        return
    TOOL_REGISTRY[spec.name] = spec
```

**有什么用**：让 registry 同时支持「编译期装饰器注册」和「运行期动态注册」两种来源，
而下游的 `get_openai_tools()` / `dispatch()` **完全无感**——它们只认 `TOOL_REGISTRY`。

---

### 4.5 `tools/mcp_client.py`（新）—— 本次升级的核心

这个文件干四件事，逐一看：

#### (a) 读配置、启动并连上每个 server

```python
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPManager:
    def __init__(self) -> None:
        self._stack = AsyncExitStack()       # 统一管理所有连接的生命周期
        self.sessions: dict[str, ClientSession] = {}

    async def connect_all(self, servers: dict) -> None:
        for name, cfg in servers.items():
            params = StdioServerParameters(command=cfg["command"], args=cfg.get("args", []))
            # 进入并"按住"这两个上下文，让连接在整个会话期间保持打开
            read, write = await self._stack.enter_async_context(stdio_client(params))
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.sessions[name] = session
            await self._register_tools(name, session)
```

> **难点解释（也是面试考点）**：stdio 连接是「嵌套的异步上下文管理器」，必须**保持打开**
> 直到会话结束。用 `AsyncExitStack` 把它们都「按住」，最后一次性 `aclose()` 全部关掉，
> 这是 MCP 客户端的标准写法，避免连接泄漏 / 子进程变僵尸。

#### (b) 把 server 报上来的工具桥接成 ToolSpec

```python
    async def _register_tools(self, server_name: str, session: ClientSession) -> None:
        tools = (await session.list_tools()).tools
        for tool in tools:
            full_name = f"mcp__{server_name}__{tool.name}"   # 命名空间，防撞名
            register_tool_spec(ToolSpec(
                name=full_name,
                description=tool.description or "",
                parameters=tool.inputSchema or {"type": "object", "properties": {}},
                function=self._make_caller(session, tool.name),   # 关键：闭包
            ))
```

#### (c) 闭包：把「调用本地函数」变成「调用远程进程」

```python
    def _make_caller(self, session: ClientSession, remote_name: str):
        async def _call(**kwargs):
            result = await session.call_tool(remote_name, kwargs)
            # MCP 返回的是内容块（文本/图片），抽出文本拼起来返回
            return "".join(
                block.text for block in result.content
                if getattr(block, "type", None) == "text"
            )
        return _call
```

> 这就是 MCP 的「魔法」拆穿后的样子：你 `dispatch()` 里那句 `spec.function(**arguments)`，
> 对本地工具是直接执行函数，对 MCP 工具则是**通过这个闭包把参数发给远程进程、把结果取回来**。
> `dispatch()` 一个字都不用改，因为它本来就支持 `await`（见 `inspect.isawaitable`）。

#### (d) 退出时断开

```python
    async def close(self) -> None:
        await self._stack.aclose()
```

**整个文件的用处一句话**：它就是第 0.3 节缺的那个 **Client 角色**，把外部 Server 的工具
无缝并进你已有的工具体系。

---

### 4.6 `main.py` —— 接上生命周期

在进入对话循环前连接、退出时断开（只加几行，不改对话逻辑）：

```python
mcp_manager = MCPManager()
if config.mcp_enabled:
    try:
        servers = load_mcp_servers(config.mcp_config_path)
        await mcp_manager.connect_all(servers)
        print(f"已接入 {len(mcp_manager.sessions)} 个 MCP 服务。")
    except Exception:
        print("（MCP 接入失败，本次仅使用本地工具，不影响使用。）")

try:
    ...  # 你现有的对话循环，原封不动
finally:
    await mcp_manager.close()
```

**有什么用**：把「连接/断开」收口在最外层，和你 `SessionStore` / `PreferenceStore` 的
生命周期管理放在一起，职责清楚；失败有兜底，绝不阻断 agent 启动。

---

## 5. 设计注意（这些细节就是面试加分点）

1. **命名空间防撞名**：MCP 工具一律前缀 `mcp__<server>__<tool>`（学 Claude 的做法）。
   万一某个 server 也有个叫 `search` 的工具，不会和你的本地工具冲突，来源也一眼可辨。

2. **state_sync 怎么办？** [agent/state_sync.py](agent/state_sync.py) 只认识 `get_weather` /
   `resolve_location` / `search_places` 三个内置工具，会把它们的结果回写进结构化 state。
   **MCP 工具的结果不会自动回写 state，这是有意为之**：它们是「即时上下文型」工具——结果进入
   消息历史供模型这一轮使用即可，不污染你精心设计的 `AgentState`。
   （将来若希望某个 MCP 工具也回写 state，只需在 `state_sync` 里加一条对应分支。）

3. **fail-open 贯穿到底**：MCP 没开 / 连不上 / 某次 `call_tool` 抛错，都不能让整轮挂掉。
   连接失败 → 退回纯本地工具；单次调用失败 → `dispatch()` 已有的 try/except 会兜成结构化错误。

4. **安全边界**：MCP server 是会真正执行动作的外部进程（filesystem 能写文件、fetch 能访问网络）。
   filesystem 务必限定可访问目录（上面 JSON 里的 `./itineraries`）；这也正是为什么
   「会产生实际后果的工具」未来该配 Human-in-the-Loop 确认。

5. **并发与限流**：你阶段五的并行框架会让多个子任务同时 `dispatch`，可能并发打到同一个
   MCP server。`ClientSession` 不保证并发安全——必要时给每个 session 加一把
   `asyncio.Lock`（思路和 [tools/places.py](tools/places.py) 里的 `_TENCENT_GATE` 信号量一致）。

---

## 6. 怎么验证（demo 脚本）

1. `pip install mcp`，并确保本机有 `uvx`（来自 uv）和 `npx`（来自 Node）。
2. `MCP_ENABLED=true python main.py`，启动时应打印「已接入 N 个 MCP 服务」。
3. 对 agent 说：「帮我查一下涩谷站最近有什么活动」——观察日志里是否出现
   `tool_call name=mcp__fetch__fetch`，证明模型选用了 MCP 工具。
4. 说：「把刚才这条路线存成文件」——观察是否调用 `mcp__filesystem__write_file`，
   并在 `./itineraries` 下生成文件。
5. 设 `MCP_ENABLED=false` 再跑一遍，确认 agent 退回本地工具、行为正常（验证 fail-open）。

最小单测（沿用你 `tests/` 不依赖 pytest 的风格）：用一个假的 `ClientSession`（`list_tools`
返回 1 个工具、`call_tool` 返回固定文本），验证 `MCPManager` 能把它正确注册进 `TOOL_REGISTRY`、
且 `dispatch("mcp__x__y", {...})` 能拿到结果——**全程不启动真实子进程**。

---

## 7. 面试怎么讲（一段话模板）

> "我的工具层做了来源无关的抽象：本地工具用装饰器注册，远程工具通过 MCP 客户端在运行时
> 桥接进同一个 registry，背后是同一个 `dispatch`。所以接入 MCP 时，ReAct/并行/REWOO 这些
> 决策逻辑一行都没改——这是抽象到位的体现。我还处理了 MCP 客户端的几个真问题：stdio
> 连接用 AsyncExitStack 管生命周期、工具名做命名空间防撞、外部 server 全程 fail-open
> 兜底、会写文件的工具限定目录并预留了人工确认的接口。"

这段话同时证明了你**懂 MCP 原理、懂抽象设计、懂生产安全**——远超「我接了个 MCP」。

---

## 8. 实施顺序（建议）

1. `requirements.txt` + `config.py` + `mcp_servers.json`（地基，30 分钟）
2. `registry.py` 加 `register_tool_spec`（5 分钟）
3. `tools/mcp_client.py`（核心，主要时间花这）
4. `main.py` 接生命周期
5. 跑通 demo（第 6 节）
6. 补最小单测 + 更新 README

> 全程不碰 `agent/` 下的决策代码——这既是技术上的安全，也是这次升级最好的「叙事」。
