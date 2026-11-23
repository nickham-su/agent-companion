"""组装主运行时与内部委派；模型和用户通道由入口注入。"""

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from hashlib import sha256
import json

from .channels.feishu.channel import FeishuChannel, FeishuPoller
from .channels.feishu.cli import FeishuCLI
from .config import FeishuSettings, Settings
from .protocol import Channel, Message, Model, ReplyInfo
from .runtime.engine import Runtime
from .runtime.subtasks import Subtasks
from .speech.asr import FileASR
from .speech.audio import AudioDevice
from .speech.tts import StreamingTTS
from .speech.vad import SileroVAD, VoiceDetector, VoiceSetupError
from .speech.voice import VoiceChannel
from .storage.sqlite import History


@asynccontextmanager
async def running_agent(
    settings: Settings, history: History, model: Model, channel: Channel, *,
    wait_for_subtask_delivery: bool = True, delivery_idle_timeout: float | None = None,
    on_confirmed_reply: Callable[[ReplyInfo, str], None] | None = None,
) -> AsyncIterator[tuple[Runtime, Subtasks]]:
    def return_result(message: Message) -> None:
        main.submit_subtask_result(message, wait_for_delivery=wait_for_subtask_delivery)

    llm = settings.llm
    thinking = llm is not None and llm.adapter == "deepseek" and llm.reasoning_effort != "disabled"

    subtasks = Subtasks(settings.main_session_id, history, model, return_result,
                        settings=settings.runtime, thinking=thinking)
    main = Runtime(settings.main_session_id, history, model, channel, tools=subtasks.tools,
                   settings=settings.runtime, thinking=thinking, delivery_idle_timeout=delivery_idle_timeout,
                   on_confirmed_reply=on_confirmed_reply)
    try:
        yield main, subtasks
    finally:
        # 在首次让出事件循环之前关停两端入口，再异步收尾，之后才可关闭 SQLite。
        subtasks.stop_accepting()
        main.stop_accepting()
        for outcome in await asyncio.gather(subtasks.close(), main.close(), return_exceptions=True):
            if isinstance(outcome, BaseException):
                raise outcome


def feishu_identity(settings: Settings) -> tuple[FeishuSettings, str]:
    feishu = settings.feishu
    if feishu is None:
        raise ValueError("missing Feishu settings")
    # 校验归属必须早于设备、轮询和子任务恢复；mode 不属于身份。
    identity = (feishu.chat_id, feishu.sender_open_id, feishu.own_open_id)
    return feishu, sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()


async def run_feishu(settings: Settings, history: History, model: Model, asr: FileASR, *,
                     voice_output: Channel | None = None, cli: FeishuCLI | None = None,
                     tts: StreamingTTS | None = None, vad: VoiceDetector | None = None,
                     audio: AudioDevice | None = None) -> None:
    feishu, identity = feishu_identity(settings)
    history.bind_feishu(settings.main_session_id, identity)
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

    cli = cli if cli is not None else FeishuCLI(feishu.cli, feishu.chat_id, timeout=feishu.cli_timeout)
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

            poller = FeishuPoller(cli, asr, channel, chat_id=feishu.chat_id, sender_id=feishu.sender_open_id,
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
