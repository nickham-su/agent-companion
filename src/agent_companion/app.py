"""组装主运行时与内部委派；模型和用户通道由入口注入。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from hashlib import sha256
import json
from math import ceil
from time import time
from typing import TYPE_CHECKING

from .channels.feishu.cli import FeishuDiscovery
from .config import FeishuSettings, Settings
from .feishu_private import run_private_chats
from .protocol import Channel, Message, Model, ReplyRoute
from .runtime.compaction import Compactions
from .runtime.engine import Runtime
from .runtime.subtasks import Subtasks
from .speech.asr import FileASR
from .speech.tts import StreamingTTS
from .speech.vad import VoiceSetupError
from .storage.sqlite import FeishuBindingError, History

if TYPE_CHECKING:
    from .speech.audio import AudioDevice
    from .speech.voice import VoiceChannel


@asynccontextmanager
async def running_agent(
    settings: Settings, history: History, model: Model, channel: Channel, *,
    session_id: str, wait_for_subtask_delivery: bool = True,
    retry_sqlite_locks: bool = False, compactions: Compactions | None = None,
) -> AsyncIterator[tuple[Runtime, Subtasks]]:
    """组装一组主、子会话；注入的压缩管理器由应用入口负责关闭。"""
    def return_result(message: Message) -> None:
        main.submit_subtask_result(message, wait_for_delivery=wait_for_subtask_delivery)

    def return_routed_result(message: Message, route: ReplyRoute) -> bool:
        if not route.active():
            return False
        main.submit_subtask_result(message, reply_route=route)
        return True

    llm = settings.llm
    thinking = llm is not None and llm.adapter == "deepseek" and llm.reasoning_effort != "disabled"

    owns_compactions = compactions is None
    if compactions is None:
        compactions = Compactions(history, model)
    subtasks = Subtasks(session_id, history, model, return_result,
                        settings=settings.runtime, thinking=thinking, retry_sqlite_locks=retry_sqlite_locks,
                        submit_routed_result=return_routed_result, compactions=compactions)
    main = Runtime(session_id, history, model, channel, tools=subtasks.tools,
                   settings=settings.runtime, thinking=thinking, retry_sqlite_locks=retry_sqlite_locks,
                   compactions=compactions)
    try:
        yield main, subtasks
    finally:
        # 在首次让出事件循环之前关停两端入口，再异步收尾，之后才可关闭 SQLite。
        subtasks.stop_accepting()
        main.stop_accepting()
        cleanups = [subtasks.close(), main.close()]
        if owns_compactions:
            compactions.stop()
            cleanups.append(compactions.close())
        for outcome in await asyncio.gather(*cleanups, return_exceptions=True):
            if isinstance(outcome, BaseException):
                raise outcome


def feishu_settings(settings: Settings) -> FeishuSettings:
    feishu = settings.feishu
    if feishu is None:
        raise ValueError("missing Feishu settings")
    return feishu


def private_feishu_identity(feishu: FeishuSettings) -> str:
    encoded = json.dumps(("private", feishu.own_open_id), separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


async def _verified_discovery(feishu: FeishuSettings, discovery: FeishuDiscovery | None) -> FeishuDiscovery:
    discovery = discovery if discovery is not None else FeishuDiscovery(feishu.cli, timeout=feishu.cli_timeout)
    if await discovery.account_open_id() != feishu.own_open_id:
        raise FeishuBindingError("飞书 CLI 当前登录账号与 feishu.own_open_id 不匹配。")
    return discovery


async def prepare_private_feishu(
    settings: Settings, history: History, *, discovery: FeishuDiscovery | None = None,
) -> Callable[[Model, FileASR, StreamingTTS | None], Awaitable[None]]:
    """先确认库的归属和实际登录账号，再提交首次启用基准；不重复查验 CLI。"""
    feishu = feishu_settings(settings)
    identity = private_feishu_identity(feishu)
    history.check_feishu_private(identity)
    candidate_ms = ceil(time() * 1000)
    discovery = await _verified_discovery(feishu, discovery)
    started_ms = history.bind_feishu_private(identity, started_ms=candidate_ms)

    async def run(model: Model, asr: FileASR, tts: StreamingTTS | None) -> None:
        incoming = None
        if feishu.mode == "voice":
            if tts is None:
                raise VoiceSetupError("voice_tts_settings_required")
            _check_voice_settings(settings)
            from .channels.feishu.macos_calls import MacCallDesktop
            from .feishu_calls import IncomingCalls

            desktop = MacCallDesktop()

            async def prepare_voice() -> tuple[VoiceChannel, AudioDevice]:
                # VAD 的音频状态只属于本通，即使上一通的推理线程仍在退出。
                return await asyncio.to_thread(_make_voice, settings, asr, tts)

            incoming = IncomingCalls(desktop, discovery, prepare_voice, own_id=feishu.own_open_id,
                                     idle_timeout=settings.voice.idle_timeout, copy_confirmed=feishu.voice_text_copy)
        compactions = Compactions(history, model)
        try:
            await run_private_chats(
                discovery, history, asr, own_id=feishu.own_open_id, started_ms=started_ms, incoming=incoming,
                open_agent=lambda session_id, channel: running_agent(
                    settings, history, model, channel, session_id=session_id, wait_for_subtask_delivery=False,
                    retry_sqlite_locks=True, compactions=compactions))
        finally:
            await compactions.close()

    return run


async def run_feishu(settings: Settings, history: History, model: Model, asr: FileASR, *,
                     tts: StreamingTTS | None = None, discovery: FeishuDiscovery | None = None) -> None:
    run_private = await prepare_private_feishu(settings, history, discovery=discovery)
    await run_private(model, asr, tts)


def _check_voice_settings(settings: Settings) -> None:
    voice = settings.voice
    if settings.asr is None:
        raise VoiceSetupError("voice_asr_settings_required")
    if voice.input_device is None or voice.output_device is None:
        raise VoiceSetupError("voice_input_and_output_devices_required")
    if voice.vad_model is None:
        raise VoiceSetupError("voice_vad_model_required")


def _make_voice(settings: Settings, asr: FileASR, tts: StreamingTTS) -> tuple[VoiceChannel, AudioDevice]:
    from .channels.audio.macos import MacAudioDevice
    from .speech.vad import SileroVAD
    from .speech.voice import VoiceChannel

    voice = settings.voice
    assert settings.asr is not None
    assert voice.input_device is not None and voice.output_device is not None and voice.vad_model is not None
    audio = MacAudioDevice(voice.input_device, voice.output_device, voice.output_sample_rate)
    vad = SileroVAD(voice.vad_model)
    return VoiceChannel(asr, tts, vad, voice, asr_timeout=settings.asr_timeout,
                        max_audio_seconds=settings.asr.max_audio_seconds), audio
