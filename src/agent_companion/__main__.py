"""启动离线演示或在本机运行飞书消息适配器。"""

import argparse
import asyncio
from collections.abc import AsyncGenerator, Sequence
from contextlib import closing
import json
import logging
import os
from pathlib import Path

import httpx

from .app import feishu_identity, run_feishu, running_agent
from .config import Settings, load_settings
from .models.deepseek import MODEL_NAME, DeepSeek
from .models.openai_responses import OpenAIResponses
from .protocol import (DeliveryStatus, Message, ModelEvent, ReasoningDelta, Receipt, ReplyStream,
                       TextDelta, ToolCall, ToolReady, ToolSpec)
from .runtime.subtasks import DELEGATE_TOOL_NAME, RESULT_PREFIX
from .speech.minimax import MiniMaxASR
from .speech.minimax_tts import MiniMaxTTS
from .speech.vad import VoiceSetupError
from .storage.sqlite import FeishuBindingError, History


_DEMO_DELEGATION = "离线模拟：委派一个后台任务"


class _DemoModel:
    def __init__(self) -> None:
        self.received_subtask_result = False

    async def stream(self, messages: Sequence[Message], tools: Sequence[ToolSpec], *,
                     thinking: bool) -> AsyncGenerator[ModelEvent, None]:
        if thinking:
            yield ReasoningDelta("模拟协议字段")
        last = messages[-1]
        if tools and last == Message("user", _DEMO_DELEGATION):
            yield ToolReady(ToolCall("demo-delegate", DELEGATE_TOOL_NAME,
                                     json.dumps({"prompt": "给出一个模拟子任务结果"})))
        elif last.role == "tool":
            yield TextDelta("离线模拟：委派已受理。")
        elif isinstance(last.content, str) and last.content.startswith(RESULT_PREFIX):
            payload = json.loads(last.content.removeprefix(RESULT_PREFIX))
            self.received_subtask_result = payload.get("result") == "模拟子任务已完成。"
            yield TextDelta("离线模拟：主会话已收到子任务回投。" if self.received_subtask_result
                            else "离线模拟：子任务失败。")
        elif not tools:
            yield TextDelta("模拟子任务已完成。")
        else:
            yield TextDelta("离线模拟：运行时已收到输入。")


class _Console:
    async def deliver(self, reply: ReplyStream) -> Receipt:
        confirmed = "".join([part async for part in reply])
        print(confirmed)
        return Receipt(reply.info.reply_id, DeliveryStatus.FULL, confirmed)


async def run_demo(settings: Settings, *, delegate: bool = False) -> bool:
    model = _DemoModel()
    with closing(History(":memory:")) as history:
        async with running_agent(settings, history, model, _Console()) as (runtime, subtasks):
            runtime.submit_batch([Message("user", _DEMO_DELEGATION if delegate else "离线模拟输入")])
            await runtime.wait_idle()
            await subtasks.wait_idle()
            await runtime.wait_idle()
            return runtime.last_failure is None and (not delegate or model.received_subtask_result)


async def _serve(settings: Settings, model_key: str, minimax_key: str) -> None:
    feishu, identity = feishu_identity(settings)
    llm = settings.llm
    if llm is None:
        raise ValueError("missing llm settings")
    with closing(History(settings.database, feishu_binding=(settings.main_session_id, identity))) as history:
        # 独立客户端，避免任一平台收到另一平台的凭据。不接受网络重定向。
        async with (
            httpx.AsyncClient(headers={"Authorization": f"Bearer {model_key}"}
                              if llm.adapter == "deepseek" else None,
                              follow_redirects=False, trust_env=False, timeout=None) as model_client,
            httpx.AsyncClient(headers={"Authorization": f"Bearer {minimax_key}"},
                              follow_redirects=False, trust_env=False, timeout=None) as speech_client,
        ):
            asr = MiniMaxASR(speech_client, region=settings.minimax.region, timeout=settings.minimax.asr_timeout)
            tts = None
            if feishu.mode == "voice":
                tts = MiniMaxTTS(speech_client, region=settings.minimax.region, model=settings.voice.tts_model,
                                 voice_id=settings.voice.voice_id, timeout=settings.voice.tts_timeout)
            if llm.adapter == "deepseek":
                model = DeepSeek(model_client, model=llm.model or MODEL_NAME,
                                  reasoning_effort=llm.reasoning_effort)
            else:
                assert llm.model is not None
                model = OpenAIResponses(model_client, model=llm.model, api_key=model_key,
                                         base_url=llm.base_url, reasoning_effort=llm.reasoning_effort,
                                         replay_salt=history.replay_salt())
            await run_feishu(settings, history, model, asr, tts=tts)


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 Agent Companion 离线模拟或飞书消息模式")
    parser.add_argument("--config", type=Path, help="可选 TOML 配置；仅 --run 使用配置中的数据库")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--demo-subtasks", action="store_true", help="模拟异步委派、独立执行及结果回投")
    mode.add_argument("--run", action="store_true", help="使用本机飞书 CLI 会话和模型服务运行")
    args = parser.parse_args()
    try:
        settings = load_settings(args.config)
    except (OSError, TypeError, ValueError) as exc:
        # 不输出可能含有访问路径或原始配置内容的异常。
        parser.error(f"无法加载配置（{type(exc).__name__}）")
    if args.run:
        if settings.llm is None:
            parser.error("正式启动需要 --config 中的 [llm] adapter 和 reasoning_effort")
        if settings.feishu is None:
            parser.error("运行飞书模式需要 --config 中的 [feishu] 配置")
        if settings.feishu.cli_timeout >= settings.runtime.delivery_timeout:
            parser.error("feishu.cli_timeout 必须小于 runtime.delivery_timeout")
        if settings.database == Path(":memory:"):
            parser.error("运行飞书模式需要持久化 SQLite 数据库路径")
        key_name = "DEEPSEEK_API_KEY" if settings.llm.adapter == "deepseek" else "OPENAI_API_KEY"
        model_key = os.getenv(key_name)
        minimax_key = os.getenv("MINIMAX_API_KEY")
        if not model_key or not minimax_key:
            parser.error(f"需要设置 {key_name} 与 MINIMAX_API_KEY 环境变量")
        try:
            asyncio.run(_serve(settings, model_key, minimax_key))
        except KeyboardInterrupt:
            return 0
        except FeishuBindingError as exc:
            logging.getLogger(__name__).error("飞书模式无法启动：%s", exc)
            return 1
        except VoiceSetupError as exc:
            logging.getLogger(__name__).error("实时语音无法启动（%s）", exc)
            return 1
        except Exception as exc:
            logging.getLogger(__name__).error("飞书模式已停止（%s）", type(exc).__name__)
            return 1
        return 0
    passed = asyncio.run(run_demo(settings, delegate=args.demo_subtasks))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
