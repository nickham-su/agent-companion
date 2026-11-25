"""启动离线演示或在本机运行飞书消息适配器。"""

import argparse
import asyncio
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import AsyncExitStack, closing
import hmac
import json
import logging
from pathlib import Path
import sys

import httpx

from .app import feishu_settings, prepare_private_feishu, running_agent
from .channels.feishu.cli import CliError, FeishuDiscovery
from .config import ConfigError, Settings, load_settings, valid_config_id
from .logging_setup import stdout_logging
from .models.deepseek import CHAT_COMPLETIONS, DeepSeek
from .models.openai_responses import OpenAIResponses
from .protocol import (AGENT_REPLY_PREFIX, DeliveryStatus, Message, ModelEvent, ReasoningDelta, Receipt, ReplyStream,
                       TextDelta, ToolCall, ToolReady, ToolSpec)
from .runtime.instructions import load_roles
from .runtime.subtasks import AgentRole, DELEGATE_TOOL_NAME, ModelBinding, demo_roles
from .speech.bailian_ws import BailianConnection
from .speech.minimax import MiniMaxASR
from .speech.vad import VoiceSetupError
from .storage.sqlite import DatabaseFormatError, FeishuBindingError, History


# 用固定名称，保证 python -m 启动时也走应用的日志配置。
_LOG = logging.getLogger("agent_companion.__main__")
_DEMO_SESSION_ID = "demo"
_DEMO_DELEGATION = "离线模拟：委派一个后台任务"


class _DemoModel:
    def __init__(self) -> None:
        self.received_subtask_result = False
        self.child_agent_id: str | None = None

    async def stream(self, messages: Sequence[Message], tools: Sequence[ToolSpec], *,
                     thinking: bool) -> AsyncGenerator[ModelEvent, None]:
        if thinking:
            yield ReasoningDelta("模拟协议字段")
        last = messages[-1]
        if tools and last == Message("user", _DEMO_DELEGATION):
            yield ToolReady(ToolCall("demo-delegate", DELEGATE_TOOL_NAME,
                                     json.dumps({"agent": "general", "prompt": "给出一个模拟子任务结果"})))
        elif last.role == "tool":
            yield TextDelta("离线模拟：委派已受理。")
        elif isinstance(last.content, str) and last.content.startswith(AGENT_REPLY_PREFIX):
            payload = json.loads(last.content.removeprefix(AGENT_REPLY_PREFIX))
            if self.child_agent_id is None and payload.get("result") == "模拟子任务已完成。":
                self.child_agent_id = payload["agent_id"]
                yield ToolReady(ToolCall("demo-reuse", DELEGATE_TOOL_NAME, json.dumps({
                    "agent_id": self.child_agent_id, "prompt": "基于同一个子 Agent 继续任务",
                })))
            else:
                self.received_subtask_result = (payload.get("agent_id") == self.child_agent_id
                                                and payload.get("result") == "已续接原子任务历史。")
                yield TextDelta("离线模拟：主会话已收到同一子 Agent 的结果。" if self.received_subtask_result
                                else "离线模拟：子任务失败。")
        elif not tools:
            if last.content == "基于同一个子 Agent 继续任务":
                continued = any(item.role == "assistant" and item.content == "模拟子任务已完成。"
                                for item in messages)
                yield TextDelta("已续接原子任务历史。" if continued else "未找到原子任务历史。")
            else:
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
        async with running_agent(settings, history, model, _Console(), session_id=_DEMO_SESSION_ID,
                                 roles=demo_roles()) as (runtime, tree):
            runtime.submit_batch([Message("user", _DEMO_DELEGATION if delegate else "离线模拟输入")])
            await tree.wait_idle()
            return runtime.last_failure is None and (not delegate or model.received_subtask_result)


def _resolve_role_models(settings: Settings, roles: Mapping[str, AgentRole]) -> tuple[dict[str, str], dict[str, str]]:
    """正式启动先完整解析角色引用和选用凭据，不打开数据库或探测飞书。"""
    if "main" not in roles:
        raise ConfigError("缺少 agents/main.md")
    model_ids, keys = {}, {}
    for role_id, role in roles.items():
        location = f"角色文件 {'agents/' + role_id + '.md'!r} 的 model_id"
        model_id = role.model_id
        if not valid_config_id(model_id):
            raise ConfigError(f"{location} 必须显式设置")
        profile = settings.models.get(model_id)
        if profile is None:
            raise ConfigError(f"{location} 引用的模型配置不存在，请检查 [models.<ID>]")
        provider = settings.llm_providers.get(profile.provider)
        if provider is None:
            raise ConfigError(f"{location} 所引用模型的 provider 不存在或不支持 llm")
        if profile.provider not in keys:
            keys[profile.provider] = provider.require_api_key(f"{location} 所引用模型的 provider")
        model_ids[role_id] = model_id
    return model_ids, keys


def _model_origin(salt: bytes, key: str, adapter: str, endpoint: str, model: str, effort: str) -> bytes:
    # 与 Responses 原生回放作用域独立；角色/档案/Provider 的命名不参与身份。
    target = json.dumps([adapter, endpoint, model, effort], separators=(",", ":")).encode("utf-8")
    return hmac.digest(key.encode("ascii"), b"agent-companion/model-origin\0" + salt + target, "sha256")


class _ServingExitStack(AsyncExitStack):
    """保留退出栈语义，但不把等待方的取消传给模型和 HTTP 客户端的收尾。"""

    async def __aexit__(self, *details):
        exit_stack = super().__aexit__
        active = sys.exception()

        async def unwind():
            try:
                if active is None:
                    return await exit_stack(*details)
                # 新任务不继承异常处理状态。恢复它，让退出栈照常链接收尾异常。
                tb, context = active.__traceback__, active.__context__
                try:
                    raise active
                except BaseException:
                    active.__traceback__, active.__context__ = tb, context
                    return await exit_stack(*details)
            except BaseException as exc:
                return exc  # 保留原异常对象和异常链，交由等待方重抛。

        pending = asyncio.create_task(unwind())
        interrupted = None
        while not pending.done():
            try:
                await asyncio.shield(pending)
            except asyncio.CancelledError as exc:
                interrupted = exc
        outcome = pending.result()
        if interrupted is not None:
            if isinstance(outcome, BaseException):
                raise interrupted from outcome
            raise interrupted
        if isinstance(outcome, BaseException):
            context = outcome.__context__
            try:
                raise outcome
            except BaseException:
                outcome.__context__ = context  # 不让调用方的原异常覆盖栈内已修复的异常链。
                raise
        return outcome


async def _serve(settings: Settings, *, discovery: FeishuDiscovery | None = None,
                 roles: Mapping[str, AgentRole] | None = None) -> None:
    roles = load_roles(Path.cwd()) if roles is None else roles
    model_ids, model_keys = _resolve_role_models(settings, roles)
    feishu = feishu_settings(settings)
    asr_settings = settings.asr
    tts_settings = settings.tts if feishu.mode == "voice" else None
    if asr_settings is None or (feishu.mode == "voice" and tts_settings is None):
        raise ConfigError("正式启动缺少已选中的 Provider 配置")
    async with prepare_private_feishu(settings, discovery=discovery, roles=roles) as (history, run_private):
        # 独立客户端，避免任一平台收到另一平台的凭据。不接受网络重定向。
        async with _ServingExitStack() as stack:
            async def client(key: str | None) -> httpx.AsyncClient:
                headers = {"Authorization": f"Bearer {key}"} if key is not None else None
                return await stack.enter_async_context(httpx.AsyncClient(
                    headers=headers, follow_redirects=False, trust_env=False, timeout=None))

            model_clients = {name: await client(key if settings.llm_providers[name].adapter == "deepseek" else None)
                             for name, key in model_keys.items()}
            speech = asr_settings.connection
            asr_client = None
            if isinstance(speech, BailianConnection):
                from .speech.bailian_asr import BailianASR

                asr = BailianASR(speech, model=asr_settings.model, timeout=settings.asr_timeout,
                                 max_audio_seconds=asr_settings.max_audio_seconds)
            else:
                asr_client = await client(speech.api_key)
                asr = MiniMaxASR(asr_client, region=speech.region, model=asr_settings.model,
                                 timeout=settings.asr_timeout, max_audio_seconds=asr_settings.max_audio_seconds)
            tts = None
            if tts_settings is not None:
                speech_out = tts_settings.connection
                if isinstance(speech_out, BailianConnection):
                    from .speech.bailian_tts import BailianTTS

                    tts = BailianTTS(speech_out, model=tts_settings.model, voice_id=tts_settings.voice_id,
                                     timeout=settings.tts_timeout)
                else:
                    from .speech.minimax_tts import MiniMaxTTS

                    tts_client = asr_client if speech_out is speech and asr_client is not None else await client(
                        speech_out.api_key)
                    tts = MiniMaxTTS(tts_client, region=speech_out.region, model=tts_settings.model,
                                     voice_id=tts_settings.voice_id, timeout=settings.tts_timeout)
            salt = history.replay_salt()
            bindings = {}
            for model_id in dict.fromkeys(model_ids.values()):
                profile = settings.models[model_id]
                provider = settings.llm_providers[profile.provider]
                model_client, key = model_clients[profile.provider], model_keys[profile.provider]
                effort = profile.reasoning_effort
                if provider.adapter == "deepseek":
                    model = DeepSeek(model_client, model=profile.model, reasoning_effort=effort)
                    endpoint, thinking = CHAT_COMPLETIONS, effort != "disabled"
                else:
                    assert provider.base_url is not None  # Responses Provider 在配置阶段已归一化。
                    model = OpenAIResponses(model_client, model=profile.model, api_key=key,
                                            base_url=provider.base_url, reasoning_effort=effort, replay_salt=salt)
                    endpoint, thinking = provider.base_url + "/responses", False
                # Runtime/压缩先关停；所有模型流关闭后，才退出任一 HTTP 客户端。
                stack.push_async_callback(model.aclose)
                origin = _model_origin(salt, key, provider.adapter, endpoint, profile.model, effort)
                bindings[model_id] = ModelBinding(model, profile.runtime, thinking, origin)
            await run_private({role_id: bindings[model_id] for role_id, model_id in model_ids.items()}, asr, tts)


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
    with stdout_logging(settings.log_level):
        _LOG.info("agent companion started; mode=%s", settings.feishu.mode if args.run else "demo")
        try:
            return _run(settings, for_run=args.run, delegate=args.demo_subtasks)
        finally:
            _LOG.info("agent companion stopped")


def _run(settings: Settings, *, for_run: bool, delegate: bool) -> int:
    if for_run:
        try:
            asyncio.run(_serve(settings))
        except KeyboardInterrupt:
            return 0
        except (ConfigError, DatabaseFormatError) as exc:
            print(f"飞书模式无法启动：{exc}", file=sys.stderr)
            return 1
        except (FeishuBindingError, CliError) as exc:
            _LOG.error("飞书模式无法启动：%s", exc)
            return 1
        except VoiceSetupError as exc:
            _LOG.error("实时语音无法启动（%s）", exc)
            return 1
        except Exception as exc:
            _LOG.error("飞书模式已停止（%s）", type(exc).__name__)
            return 1
        return 0
    passed = asyncio.run(run_demo(settings, delegate=delegate))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
