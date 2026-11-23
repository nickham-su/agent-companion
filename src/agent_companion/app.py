"""组装主运行时与内部委派；模型和用户通道由入口注入。"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from hashlib import sha256
import json

from .channels.feishu.channel import FeishuChannel, FeishuPoller
from .channels.feishu.cli import FeishuCLI
from .config import FeishuSettings, Settings, require_delivery_timeout
from .protocol import Channel, Message, Model
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
    settings: Settings, history: History, model: Model, channel: Channel,
) -> AsyncIterator[tuple[Runtime, Subtasks]]:
    def return_result(message: Message) -> None:
        main.submit_subtask_result(message)

    llm = settings.llm
    thinking = llm is not None and llm.adapter == "deepseek" and llm.reasoning_effort != "disabled"

    subtasks = Subtasks(settings.main_session_id, history, model, return_result,
                        settings=settings.runtime, thinking=thinking)
    main = Runtime(settings.main_session_id, history, model, channel, tools=subtasks.tools,
                   settings=settings.runtime, thinking=thinking)
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
    require_delivery_timeout(feishu.cli_timeout, settings.runtime.delivery_timeout, "feishu.cli_timeout")
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
        require_delivery_timeout(voice.delivery_timeout, settings.runtime.delivery_timeout, "voice.delivery_timeout",
                                 error_type=VoiceSetupError)
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
    async with running_agent(settings, history, model, channel) as (main, _):
        poller = FeishuPoller(cli, asr, channel, chat_id=feishu.chat_id, sender_id=feishu.sender_open_id,
                              own_id=feishu.own_open_id, session_id=main.session_id, history=history,
                              submit=main.submit_persisted_batch)
        if local_voice is None:
            await poller.run()
        else:
            assert audio is not None
            async with local_voice.connected(audio, lambda text: main.submit_batch((Message("user", text),))):
                await poller.run()
