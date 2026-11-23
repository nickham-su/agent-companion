"""启动离线演示或在本机运行飞书消息适配器。"""

import argparse
import asyncio
from collections.abc import AsyncGenerator, Sequence
from contextlib import AsyncExitStack, closing
import json
import logging
from pathlib import Path

import httpx

from .app import feishu_identity, run_feishu, running_agent
from .config import ConfigError, Settings, load_settings
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


async def _serve(settings: Settings) -> None:
    feishu, identity = feishu_identity(settings)
    llm, asr_settings = settings.llm, settings.asr
    tts_settings = settings.tts if feishu.mode == "voice" else None
    if llm is None or asr_settings is None or (feishu.mode == "voice" and tts_settings is None):
        raise ConfigError("正式启动缺少已选中的 Provider 配置")
    with closing(History(settings.database, feishu_binding=(settings.main_session_id, identity))) as history:
        # 独立客户端，避免任一平台收到另一平台的凭据。不接受网络重定向。
        async with AsyncExitStack() as stack:
            async def client(key: str | None) -> httpx.AsyncClient:
                headers = {"Authorization": f"Bearer {key}"} if key is not None else None
                return await stack.enter_async_context(httpx.AsyncClient(
                    headers=headers, follow_redirects=False, trust_env=False, timeout=None))

            model_client = await client(llm.api_key if llm.adapter == "deepseek" else None)
            speech = asr_settings.connection
            asr_client = await client(speech.api_key)
            asr = MiniMaxASR(asr_client, region=speech.region, model=asr_settings.model, timeout=settings.asr_timeout,
                             max_audio_seconds=asr_settings.max_audio_seconds)
            tts = None
            if tts_settings is not None:
                speech_out = tts_settings.connection
                tts_client = asr_client if speech_out is speech else await client(speech_out.api_key)
                tts = MiniMaxTTS(tts_client, region=speech_out.region, model=tts_settings.model,
                                 voice_id=tts_settings.voice_id, timeout=settings.tts_timeout)
            if llm.adapter == "deepseek":
                model = DeepSeek(model_client, model=llm.model or MODEL_NAME,
                                  reasoning_effort=llm.reasoning_effort)
            else:
                assert llm.model is not None
                model = OpenAIResponses(model_client, model=llm.model, api_key=llm.api_key,
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
    config_error = None
    try:
        settings = load_settings(args.config, for_run=args.run)
    except ConfigError as exc:
        config_error = str(exc)
    except Exception:
        # 不输出可能含有访问路径或原始配置内容的异常。
        config_error = "无法加载配置，请检查文件路径和配置格式"
    if config_error is not None:
        parser.error(config_error)
    if args.run:
        try:
            asyncio.run(_serve(settings))
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
