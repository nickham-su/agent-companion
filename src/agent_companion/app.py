"""组装主运行时与内部委派；模型和用户通道由入口注入。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager, closing, nullcontext
from hashlib import sha256
import json
from math import ceil
from time import time
from typing import TYPE_CHECKING

from .channels.feishu.cli import FeishuDiscovery
from .config import FeishuSettings, Settings
from .feishu_private import run_private_chats
from .protocol import Channel, Model
from .runtime.compaction import Compactions
from .runtime.engine import Runtime
from .runtime.sqlite_access import SQLiteAccess
from .runtime.subtasks import AgentRole, AgentTree
from .session_scheduler import SessionSchedulers
from .speech.asr import FileASR
from .speech.tts import StreamingTTS
from .speech.vad import VoiceSetupError
from .storage.sqlite import Agent, History

if TYPE_CHECKING:
    from .speech.audio import AudioDevice
    from .speech.voice import VoiceChannel


async def _finish_agent_cleanup(cleanups: list[Awaitable[None]]) -> None:
    pending = asyncio.gather(*cleanups, return_exceptions=True)
    cancelled = None
    while not pending.done():
        try:
            await asyncio.shield(pending)
        except asyncio.CancelledError as exc:
            # 调用者可被反复取消；必须等 shutdown 写入结束才能放开该会话。
            cancelled = exc
    outcomes = pending.result()
    if cancelled is not None:
        raise cancelled
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            raise outcome


@asynccontextmanager
async def running_agent(
    settings: Settings, history: History, model: Model, channel: Channel, *,
    session_id: str, roles: Mapping[str, AgentRole], agent: Agent | None = None,
    retry_sqlite_locks: bool = False, compactions: Compactions | None = None,
    schedulers: SessionSchedulers | None = None,
) -> AsyncIterator[tuple[Runtime, AgentTree]]:
    """组装一条主路由的 Agent 树；注入的压缩管理器由应用入口关闭。"""
    if schedulers is None:
        schedulers = SessionSchedulers()
    scheduler = schedulers.get_or_create(session_id)
    owner = scheduler.reserve_agent()
    tree = None
    owns_compactions = compactions is None
    try:
        llm = settings.llm
        thinking = llm is not None and llm.adapter == "deepseek" and llm.reasoning_effort != "disabled"
        sqlite = SQLiteAccess(retry_locks=retry_sqlite_locks, stopped=lambda: False)
        if agent is None:
            agent = await sqlite.write_input(lambda: history.ensure_main_agent(session_id))
        if agent.session_id != session_id or agent.parent_agent_id is not None:
            raise ValueError("main agent does not match route")
        await sqlite.write_input(lambda: history.interrupt_agent_tree(agent.agent_id, reason="restart"))
        if compactions is None:
            compactions = Compactions(history, model)
        tree = AgentTree(agent, history, model, roles=roles,
                         schedulers=schedulers, settings=settings.runtime, thinking=thinking,
                         retry_sqlite_locks=retry_sqlite_locks, compactions=compactions)
        yield tree.bind_root(channel, owner), tree
    finally:
        try:
            # 首次让出循环前关停全部入口；异步收尾完成前保留会话所有权。
            cleanups = []
            if tree is not None:
                tree.stop_accepting()
                cleanups.append(tree.close())
            if owns_compactions and compactions is not None:
                compactions.stop()
                cleanups.append(compactions.close())
            await _finish_agent_cleanup(cleanups)
        finally:
            scheduler.release_agent(owner)


def feishu_settings(settings: Settings) -> FeishuSettings:
    feishu = settings.feishu
    if feishu is None:
        raise ValueError("missing Feishu settings")
    return feishu


def private_feishu_identity(own_id: str) -> str:
    encoded = json.dumps(("private", own_id), separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


@asynccontextmanager
async def prepare_private_feishu(
    settings: Settings, history: History | None = None, *, roles: Mapping[str, AgentRole],
    discovery: FeishuDiscovery | None = None,
) -> AsyncIterator[tuple[History, Callable[[Model, FileASR, StreamingTTS | None], Awaitable[None]]]]:
    """查一次 CLI 登录账号后绑定数据库；仅负责关闭本函数打开的库。"""
    feishu = feishu_settings(settings)
    if discovery is None:
        discovery = FeishuDiscovery(feishu.cli, timeout=feishu.cli_timeout)
    # 账号查询期间到达的新消息也纳入首次扫描。
    candidate_ms = ceil(time() * 1000)
    own_id = await discovery.account_open_id()
    identity = private_feishu_identity(own_id)
    with (closing(History(settings.database, feishu_private_owner=identity))
          if history is None else nullcontext(history)) as db:
        started_ms = db.bind_feishu_private(identity, started_ms=candidate_ms)

        async def run(model: Model, asr: FileASR, tts: StreamingTTS | None) -> None:
            sqlite = SQLiteAccess(retry_locks=True, stopped=lambda: False)
            await sqlite.write_input(lambda: db.interrupt_agent_work(reason="restart"))
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

                incoming = IncomingCalls(desktop, discovery, prepare_voice, own_id=own_id,
                                         idle_timeout=settings.voice.idle_timeout, copy_confirmed=feishu.voice_text_copy)
            compactions = Compactions(db, model)
            schedulers = SessionSchedulers()
            try:
                await run_private_chats(
                    discovery, db, asr, own_id=own_id, started_ms=started_ms, incoming=incoming,
                    schedulers=schedulers,
                    open_agent=lambda agent, channel: running_agent(
                        settings, db, model, channel, session_id=agent.session_id, agent=agent, roles=roles,
                        retry_sqlite_locks=True, compactions=compactions, schedulers=schedulers))
            finally:
                await compactions.close()

        yield db, run


async def run_feishu(settings: Settings, history: History, model: Model, asr: FileASR, *,
                     roles: Mapping[str, AgentRole], tts: StreamingTTS | None = None,
                     discovery: FeishuDiscovery | None = None) -> None:
    async with prepare_private_feishu(settings, history, discovery=discovery, roles=roles) as (_, run_private):
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
