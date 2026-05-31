import asyncio

import tools  # noqa: F401  导入工具包即完成所有工具注册
from agent.loop import run_turn
from agent.state import AgentState, Location
from config import config
from memory.history import compact
from memory.storage import SessionStore
from models.client import build_client
from utils.logger import configure_logging

DEFAULT_LAT, DEFAULT_LNG = 35.6595, 139.7004


def _read_location() -> Location:
    prompt = "请输入纬度,经度（回车默认东京涩谷 35.6595, 139.7004）："
    raw = input(prompt).strip()
    if not raw:
        return Location(lat=DEFAULT_LAT, lng=DEFAULT_LNG)
    try:
        lat_text, lng_text = raw.replace("，", ",").split(",", 1)
        return Location(lat=float(lat_text), lng=float(lng_text))
    except ValueError:
        print("经纬度格式不正确，已使用默认值：东京涩谷 35.6595, 139.7004")
        return Location(lat=DEFAULT_LAT, lng=DEFAULT_LNG)


async def main() -> None:
    configure_logging(config.log_level, config.log_file)

    client = build_client()
    if client is None:
        print(
            "未检测到 LLM API Key。请在 .env 中设置 DEEPSEEK_API_KEY"
            "（或 OPENAI_API_KEY）后重试。"
        )
        return

    store = SessionStore(config.session_dir)

    try:
        # 优先恢复上次会话；没有就新开一个，并在缺少位置时才询问
        state = store.load(config.session_id)
        if state is not None:
            print("已恢复上次会话，输入 quit 或 exit 退出。")
        else:
            state = AgentState(location=_read_location())
        if state.location is None:
            state.location = _read_location()

        print("城市漫步助手已启动。输入你想去的地方，输入 quit 或 exit 退出。")
        while True:
            try:
                user_message = input("> ").strip()
                if user_message.lower() in {"quit", "exit"}:
                    print("已退出。")
                    break
                if not user_message:
                    continue
                reply, state = await run_turn(client, state, user_message)
                print(reply)
                # 每轮后压缩并落盘：状态不膨胀，重启也能接着聊
                compact(state)
                store.save(config.session_id, state)
            except (KeyboardInterrupt, EOFError):
                print("\n已退出。")
                break
            except Exception:
                print("本轮调用失败，请检查 API Key、网络或模型配置后重试。")
    except Exception:
        print("启动失败，请检查 .env、依赖安装和经纬度输入格式。")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n已退出。")
