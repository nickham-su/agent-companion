"""启动配置；鉴权材料不从 TOML 读取。"""

from dataclasses import dataclass, field, fields
import math
from pathlib import Path
import re
import tomllib
from typing import Any


def require_seconds(value: float, name: str) -> None:
    if type(value) not in (int, float) or not 0 < value or not math.isfinite(value):
        raise ValueError(f"{name} must be a positive number")


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    model_timeout: float = 60
    delivery_timeout: float = 120
    context_budget: int = 64_000
    tail_budget: int = 16_000
    summary_budget: int = 4_000
    max_tool_rounds: int = 8

    def __post_init__(self) -> None:
        require_seconds(self.model_timeout, "model_timeout")
        require_seconds(self.delivery_timeout, "delivery_timeout")
        for name in ("context_budget", "tail_budget", "summary_budget", "max_tool_rounds"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.tail_budget + self.summary_budget >= self.context_budget:
            raise ValueError("tail_budget + summary_budget must be below context_budget")


@dataclass(frozen=True, slots=True)
class LLMSettings:
    adapter: str
    reasoning_effort: str
    model: str | None = None
    replay_profile: str | None = None

    def __post_init__(self) -> None:
        if self.adapter not in ("deepseek", "openai_responses"):
            raise ValueError("llm.adapter must be deepseek or openai_responses")
        if (not isinstance(self.reasoning_effort, str) or not self.reasoning_effort.strip()
                or self.reasoning_effort != self.reasoning_effort.strip() or len(self.reasoning_effort) > 64
                or not self.reasoning_effort.isprintable()):
            raise ValueError("llm.reasoning_effort must be a non-empty setting")
        if self.model is not None and (not isinstance(self.model, str)
                                       or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", self.model)):
            raise ValueError("invalid llm.model")
        if self.adapter == "openai_responses":
            if self.model is None:
                raise ValueError("llm.model is required for openai_responses")
            if not isinstance(self.replay_profile, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", self.replay_profile):
                raise ValueError("llm.replay_profile is required for openai_responses")
        elif self.replay_profile is not None:
            raise ValueError("llm.replay_profile is only used by openai_responses")


@dataclass(frozen=True, slots=True)
class FeishuSettings:
    chat_id: str = field(repr=False)
    sender_open_id: str = field(repr=False)
    own_open_id: str = field(repr=False)
    mode: str = "text"
    cli: str = "lark-cli"
    cli_timeout: float = 20

    def __post_init__(self) -> None:
        for value, prefix, name in ((self.chat_id, "oc", "chat_id"),
                                    (self.sender_open_id, "ou", "sender_open_id"),
                                    (self.own_open_id, "ou", "own_open_id")):
            if not isinstance(value, str) or not re.fullmatch(rf"{prefix}_[\w-]{{1,160}}", value, flags=re.ASCII):
                raise ValueError(f"invalid Feishu {name}")
        if self.sender_open_id == self.own_open_id:
            raise ValueError("Feishu sender and self must be different users")
        if self.mode not in ("text", "voice"):
            raise ValueError("invalid Feishu mode")
        if not isinstance(self.cli, str) or not self.cli.strip() or "\x00" in self.cli:
            raise ValueError("invalid Feishu CLI executable")
        require_seconds(self.cli_timeout, "cli_timeout")


@dataclass(frozen=True, slots=True)
class MiniMaxSettings:
    region: str = "cn"
    asr_timeout: float = 30

    def __post_init__(self) -> None:
        if self.region not in ("cn", "global"):
            raise ValueError("invalid MiniMax region")
        require_seconds(self.asr_timeout, "asr_timeout")


@dataclass(frozen=True, slots=True)
class VoiceSettings:
    input_device: str | int | None = None
    output_device: str | int | None = None
    vad_model: Path | None = None
    vad_threshold: float = 0.5
    speech_frames: int = 3  # 每帧 32 ms
    silence_seconds: float = 1
    max_turn_seconds: float = 60
    playback_rate: int = 44_100
    prefetch_seconds: float = 0.4
    buffer_seconds: float = 2
    delivery_timeout: float = 90
    tts_timeout: float = 25
    tts_model: str = "speech-2.8-turbo"
    voice_id: str = "male-qn-qingse"

    def __post_init__(self) -> None:
        for name in ("input_device", "output_device"):
            value = getattr(self, name)
            if value is not None and not ((type(value) is int and value >= 0) or (
                isinstance(value, str) and value.strip() and "\x00" not in value
            )):
                raise ValueError(f"invalid voice.{name}")
        for name in ("vad_threshold", "silence_seconds", "max_turn_seconds", "prefetch_seconds",
                     "buffer_seconds", "delivery_timeout", "tts_timeout"):
            require_seconds(getattr(self, name), f"voice.{name}")
        if self.vad_threshold >= 1 or type(self.speech_frames) is not int or not 1 <= self.speech_frames <= 15:
            raise ValueError("invalid voice VAD settings")
        if not self.speech_frames * 0.032 <= self.max_turn_seconds <= 120 or self.silence_seconds > 10:
            raise ValueError("invalid voice turn limits")
        if type(self.playback_rate) is not int or self.playback_rate not in {16_000, 24_000, 32_000, 44_100}:
            raise ValueError("unsupported voice.playback_rate")
        if not 0.032 <= self.buffer_seconds <= 5 or not self.prefetch_seconds < self.buffer_seconds:
            raise ValueError("voice.prefetch_seconds must be below voice.buffer_seconds (0.032–5)")
        if not isinstance(self.tts_model, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", self.tts_model):
            raise ValueError("invalid voice.tts_model")
        if not (isinstance(self.voice_id, str) and 0 < len(self.voice_id) <= 128
                and self.voice_id.strip() == self.voice_id and self.voice_id.isprintable()):
            raise ValueError("invalid voice.voice_id")


@dataclass(frozen=True, slots=True)
class Settings:
    database: Path = Path("db.sqlite3")
    main_session_id: str = "main"
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    llm: LLMSettings | None = None
    feishu: FeishuSettings | None = None
    minimax: MiniMaxSettings = field(default_factory=MiniMaxSettings)
    voice: VoiceSettings = field(default_factory=VoiceSettings)


def _section(data: Any, cls: type) -> dict[str, Any]:
    if not isinstance(data, dict) or data.keys() - {item.name for item in fields(cls)}:
        raise ValueError(f"invalid {cls.__name__} configuration fields")
    return data


def load_settings(path: Path | None = None) -> Settings:
    if path is None:
        return Settings()
    with path.open("rb") as file:
        data = _section(tomllib.load(file), Settings)
    database = data.get("database", "db.sqlite3")
    session_id = data.get("main_session_id", "main")
    if not isinstance(database, str) or not database.strip():
        raise ValueError("database must be a non-empty path")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("main_session_id must be a non-empty string")
    if database != ":memory:":
        database = str(path.parent / database)
    feishu_data = data.get("feishu")
    feishu = FeishuSettings(**_section(feishu_data, FeishuSettings)) if feishu_data is not None else None
    llm_data = data.get("llm")
    llm = LLMSettings(**_section(llm_data, LLMSettings)) if llm_data is not None else None
    voice = _section(data.get("voice", {}), VoiceSettings).copy()
    if "vad_model" in voice:
        model = voice["vad_model"]
        if not isinstance(model, str) or not model.strip() or "\x00" in model:
            raise ValueError("invalid voice.vad_model")
        voice["vad_model"] = path.parent / model
    return Settings(
        database=Path(database),
        main_session_id=session_id,
        runtime=RuntimeSettings(**_section(data.get("runtime", {}), RuntimeSettings)),
        llm=llm,
        feishu=feishu,
        minimax=MiniMaxSettings(**_section(data.get("minimax", {}), MiniMaxSettings)),
        voice=VoiceSettings(**voice),
    )
