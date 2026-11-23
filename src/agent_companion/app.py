"""组装主运行时与内部委派；模型和用户通道由入口注入。"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
from math import ceil
from time import time

from .channels.feishu.channel import FeishuChannel, FeishuPoller
from .channels.feishu.cli import FeishuCLI, FeishuDiscovery, PrivateChat
from .config import FeishuSettings, Settings
from .feishu_private import run_private_chats
from .protocol import Channel, Message, Model, ReplyInfo
from .runtime.engine import Runtime
from .runtime.subtasks import Subtasks
from .speech.asr import FileASR
from .speech.audio import AudioDevice
from .speech.tts import StreamingTTS
from .speech.vad import SileroVAD, VoiceDetector, VoiceSetupError
from .speech.voice import VoiceChannel
from .storage.sqlite import FeishuBindingError, History


@asynccontextmanager
async def running_agent(
    settings: Settings, history: History, model: Model, channel: Channel, *,
    wait_for_subtask_delivery: bool = True, delivery_idle_timeout: float | None = None,
    on_confirmed_reply: Callable[[ReplyInfo, str], None] | None = None,
    session_id: str | None = None,
    retry_sqlite_locks: bool = False,
) -> AsyncIterator[tuple[Runtime, Subtasks]]:
    def return_result(message: Message) -> None:
        main.submit_subtask_result(message, wait_for_delivery=wait_for_subtask_delivery)

    llm = settings.llm
    thinking = llm is not None and llm.adapter == "deepseek" and llm.reasoning_effort != "disabled"

    session_id = settings.main_session_id if session_id is None else session_id
    subtasks = Subtasks(session_id, history, model, return_result,
                        settings=settings.runtime, thinking=thinking, retry_sqlite_locks=retry_sqlite_locks)
    main = Runtime(session_id, history, model, channel, tools=subtasks.tools,
                   settings=settings.runtime, thinking=thinking, delivery_idle_timeout=delivery_idle_timeout,
                   on_confirmed_reply=on_confirmed_reply, retry_sqlite_locks=retry_sqlite_locks)
    try:
        yield main, subtasks
    finally:
        # 在首次让出事件循环之前关停两端入口，再异步收尾，之后才可关闭 SQLite。
        subtasks.stop_accepting()
        main.stop_accepting()
        for outcome in await asyncio.gather(subtasks.close(), main.close(), return_exceptions=True):
            if isinstance(outcome, BaseException):
                raise outcome


def feishu_settings(settings: Settings) -> FeishuSettings:
    feishu = settings.feishu
    if feishu is None:
        raise ValueError("missing Feishu settings")
    return feishu


def _identity_fingerprint(*identity: str) -> str:
    return sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()


def private_feishu_identity(feishu: FeishuSettings) -> str:
    return _identity_fingerprint("private", feishu.own_open_id)


async def _verified_discovery(feishu: FeishuSettings, discovery: FeishuDiscovery | None) -> FeishuDiscovery:
    discovery = discovery if discovery is not None else FeishuDiscovery(feishu.cli, timeout=feishu.cli_timeout)
    if await discovery.account_open_id() != feishu.own_open_id:
        raise FeishuBindingError("飞书 CLI 当前登录账号与 feishu.own_open_id 不匹配。")
    return discovery


@dataclass(frozen=True, slots=True)
class FixedFeishuRoute:
    chat: PrivateChat
    cli: FeishuCLI
    fingerprint: str

    def checked_fingerprint(self, feishu: FeishuSettings) -> str:
        if feishu.peer_open_id != self.chat.sender_id:
            raise FeishuBindingError("飞书固定路由与配置的对端不匹配。")
        if self.cli.chat_id != self.chat.chat_id:
            raise FeishuBindingError("飞书固定路由的收发地址与已解析私聊不匹配。")
        expected = _identity_fingerprint(self.chat.chat_id, self.chat.sender_id, feishu.own_open_id)
        if self.fingerprint != expected:
            raise FeishuBindingError("飞书固定路由的登录账号或私聊身份不匹配。")
        return expected


async def prepare_fixed_feishu(feishu: FeishuSettings, *, discovery: FeishuDiscovery | None = None) -> FixedFeishuRoute:
    """核验账号并解析一次固定私聊，地址在本次运行中复用。"""
    assert feishu.peer_open_id is not None
    discovery = await _verified_discovery(feishu, discovery)
    chats = await discovery.private_chats(feishu.own_open_id)
    matches = [chat for chat in chats if chat.sender_id == feishu.peer_open_id]
    if not matches:
        raise FeishuBindingError("没有找到 feishu.peer_open_id 对应的真人私聊；请确认该私聊已能被当前账号发现。")
    if len(matches) != 1 or sum(chat.chat_id == matches[0].chat_id for chat in chats) != 1:
        raise FeishuBindingError("无法唯一确定 feishu.peer_open_id 对应的私聊；发现了多个匹配或地址冲突。")
    chat = matches[0]
    # 保持固定库原身份元组的顺序，输出模式不属于身份。
    return FixedFeishuRoute(chat, discovery.for_chat(chat.chat_id),
                            _identity_fingerprint(chat.chat_id, chat.sender_id, feishu.own_open_id))


async def prepare_private_feishu(
    settings: Settings, history: History, *, discovery: FeishuDiscovery | None = None,
) -> Callable[[Model, FileASR], Awaitable[None]]:
    """先确认库的归属和实际登录账号，再提交首次启用基准；不重复查验 CLI。"""
    feishu = feishu_settings(settings)
    assert feishu.peer_open_id is None
    identity = private_feishu_identity(feishu)
    history.check_feishu_private(identity)
    candidate_ms = ceil(time() * 1000)
    discovery = await _verified_discovery(feishu, discovery)
    started_ms = history.bind_feishu_private(identity, started_ms=candidate_ms)

    async def run(model: Model, asr: FileASR) -> None:
        await run_private_chats(
            discovery, history, asr, own_id=feishu.own_open_id, started_ms=started_ms,
            open_agent=lambda session_id, channel: running_agent(
                settings, history, model, channel, session_id=session_id, wait_for_subtask_delivery=False,
                retry_sqlite_locks=True))

    return run


async def run_feishu(settings: Settings, history: History, model: Model, asr: FileASR, *,
                     voice_output: Channel | None = None, tts: StreamingTTS | None = None, vad: VoiceDetector | None = None,
                     audio: AudioDevice | None = None, discovery: FeishuDiscovery | None = None) -> None:
    feishu = feishu_settings(settings)
    if feishu.peer_open_id is None:
        run_private = await prepare_private_feishu(settings, history, discovery=discovery)
        await run_private(model, asr)
        return
    route = await prepare_fixed_feishu(feishu, discovery=discovery)
    await run_fixed_feishu(settings, history, model, asr, route, voice_output=voice_output, tts=tts, vad=vad, audio=audio)


async def run_fixed_feishu(settings: Settings, history: History, model: Model, asr: FileASR, route: FixedFeishuRoute, *,
                           voice_output: Channel | None = None, tts: StreamingTTS | None = None,
                           vad: VoiceDetector | None = None, audio: AudioDevice | None = None) -> None:
    """启动固定路由前，校验本次实际传入的配置、CLI 和历史库归属。"""
    feishu = feishu_settings(settings)
    # 生产入口已原子绑定，仍须防止调用方在最后组装时误传路由或库。
    history.bind_feishu(settings.main_session_id, route.checked_fingerprint(feishu))
    local_voice = None
    if feishu.mode == "voice" and voice_output is None and tts is not None:
        voice = settings.voice
        if settings.asr is None:
            raise VoiceSetupError("voice_asr_settings_required")
        if audio is None:
            if voice.input_device is None or voice.output_device is None:
                raise VoiceSetupError("voice_input_and_output_devices_required")
            from .channels.audio.macos import MacAudioDevice

            audio = MacAudioDevice(voice.input_device, voice.output_device, voice.output_sample_rate)
        if vad is None:
            if voice.vad_model is None:
                raise VoiceSetupError("voice_vad_model_required")
            vad = SileroVAD(voice.vad_model)
        local_voice = VoiceChannel(asr, tts, vad, voice, asr_timeout=settings.asr_timeout,
                                   max_audio_seconds=settings.asr.max_audio_seconds)
        voice_output = local_voice

    cli = route.cli
    channel = FeishuChannel(cli, mode=feishu.mode, voice_output=voice_output)
    wait_for_delivery = feishu.mode == "voice"
    copy_voice_text = wait_for_delivery and feishu.voice_text_copy
    uses_text_queue = not wait_for_delivery or copy_voice_text
    delivery_idle_timeout = settings.voice.idle_timeout if wait_for_delivery else None
    try:
        async with running_agent(settings, history, model, channel, wait_for_subtask_delivery=wait_for_delivery,
                                 delivery_idle_timeout=delivery_idle_timeout,
                                 on_confirmed_reply=channel.enqueue if copy_voice_text else None) as (main, subtasks):
            stopping = False

            def stop_text_delivery() -> None:
                nonlocal stopping
                stopping = True
                subtasks.stop_accepting()
                main.stop_accepting()
                channel.stop_accepting()

            poller = FeishuPoller(cli, asr, channel, chat_id=route.chat.chat_id, sender_id=route.chat.sender_id,
                                  own_id=feishu.own_open_id, session_id=main.session_id, history=history,
                                  submit=lambda batch_id: main.submit_persisted_batch(
                                      batch_id, wait_for_delivery=wait_for_delivery),
                                  before_close=stop_text_delivery if uses_text_queue else None)

            async def run_polling() -> None:
                if not uses_text_queue:
                    await poller.run()
                    return
                # CLI 轮询自身也可能异步清理；应用先关入口及发送，再取消它。
                polling = asyncio.create_task(poller.run())
                try:
                    await asyncio.shield(polling)
                finally:
                    stop_text_delivery()
                    if not polling.done():
                        polling.cancel()
                    await asyncio.gather(polling, return_exceptions=True)

            if local_voice is None:
                await run_polling()
            else:
                assert audio is not None

                def submit_voice(text: str) -> None:
                    # 关闭轮询通知期间，音频转写可能还在收尾。
                    if not stopping:
                        main.submit_batch((Message("user", text),), wait_for_delivery=wait_for_delivery)

                async with local_voice.connected(audio, submit_voice):
                    await run_polling()
    finally:
        await channel.close()
