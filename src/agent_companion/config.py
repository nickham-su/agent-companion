"""启动配置：仅正式启动解析已选中的 Provider。"""

from dataclasses import MISSING, dataclass, field, fields
from ipaddress import IPv6Address
import math
from pathlib import Path
import re
import tomllib
from typing import Any
from urllib.parse import urlsplit

import httpx


OPENAI_BASE_URL = "https://api.openai.com/v1"
MINIMAX_ASR_MODEL = "asr-1.0"
DEFAULT_ASR_TIMEOUT = 30
DEFAULT_MODEL_TIMEOUT = 60
DEFAULT_TTS_TIMEOUT = 25
_DEFAULT_LOG_LEVEL = "WARNING"
_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_LEGACY_CONFIG_ERROR = "旧配置格式不再支持，请按 config.example.toml 迁移到命名 Provider"


class ConfigError(ValueError):
    """可在启动时显示的诊断，仅使用固定文案，不包含配置值。"""


def normalize_openai_base_url(value: str | None) -> str:
    """用于请求和回放范围的同一 HTTPS 基准地址。"""
    if value is None:
        return OPENAI_BASE_URL
    error = "llm.base_url must be an HTTPS API base URL without credentials, query, or fragment"
    if (not isinstance(value, str) or not 0 < len(value) <= 2048
            or any(char.isspace() or not char.isprintable() or char in "\\?#" for char in value)
            or re.search(r"%(?![0-9a-fA-F]{2})", value)):
        raise ValueError(error)
    try:
        split = urlsplit(value)
        hostname, port = split.hostname, split.port
        if (split.scheme != "https" or not hostname or "@" in split.netloc
                or split.netloc.endswith(":") or port == 0):
            raise ValueError
        url = httpx.URL(value)
        if split.netloc.startswith("["):
            # HTTPX 保留 IPv6 原拼写，规范化后等价地址使用同一指纹。
            url = url.copy_with(host=IPv6Address(hostname).compressed)
        origin = url.copy_with(raw_path=b"/", port=None if port == 443 else port)
        normalized = str(origin).removesuffix("/") + url.raw_path.decode("ascii").rstrip("/")
        if len(normalized) > 2048:
            raise ValueError
        return normalized
    except (ValueError, httpx.InvalidURL):
        raise ValueError(error) from None


def require_seconds(value: float, name: str) -> None:
    if type(value) not in (int, float) or not 0 < value or not math.isfinite(value):
        raise ValueError(f"{name} must be a positive number")


def _config_seconds(value: float, name: str) -> None:
    try:
        require_seconds(value, name)
        return
    except (ValueError, OverflowError):
        pass
    raise ConfigError(f"{name} 必须是有限正数")


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    model_timeout: float = DEFAULT_MODEL_TIMEOUT
    context_budget: int = 64_000
    tail_budget: int = 16_000
    max_tool_rounds: int = 8

    def __post_init__(self) -> None:
        _config_seconds(self.model_timeout, "timeouts.llm")
        for name in ("context_budget", "tail_budget", "max_tool_rounds"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ConfigError(f"runtime.{name} 必须是正整数")
        if self.tail_budget >= self.context_budget:
            raise ConfigError("runtime.tail_budget 必须小于 runtime.context_budget")


@dataclass(frozen=True, slots=True)
class LLMSettings:
    adapter: str
    reasoning_effort: str
    model: str | None = None
    base_url: str | None = field(default=None, repr=False)
    api_key: str = field(kw_only=True, repr=False)

    def __post_init__(self) -> None:
        if self.adapter not in ("deepseek", "openai_responses"):
            raise ConfigError("llm_provider 需要 deepseek 或 openai_responses 适配器")
        if (not isinstance(self.reasoning_effort, str) or not self.reasoning_effort.strip()
                or self.reasoning_effort != self.reasoning_effort.strip() or len(self.reasoning_effort) > 64
                or not self.reasoning_effort.isprintable()):
            raise ConfigError("llm.reasoning_effort 必须显式设置为非空字符串")
        if self.model is not None and (not isinstance(self.model, str)
                                       or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", self.model)):
            raise ConfigError("llm.model 格式无效")
        if self.adapter == "openai_responses":
            if self.model is None:
                raise ConfigError("openai_responses 必须设置 llm.model")
            try:
                object.__setattr__(self, "base_url", normalize_openai_base_url(self.base_url))
            except ValueError:
                raise ConfigError("llm_provider 的 base_url 必须是无凭据、查询或片段的 HTTPS 地址") from None
        elif self.base_url is not None:
            raise ConfigError("llm_provider 的 base_url 仅适用于 openai_responses")


@dataclass(frozen=True, slots=True)
class FeishuSettings:
    mode: str = "text"
    cli: str = "lark-cli"
    cli_timeout: float = 20
    voice_text_copy: bool = True

    def __post_init__(self) -> None:
        if self.mode not in ("text", "voice"):
            raise ConfigError("feishu.mode 必须是 text 或 voice")
        if type(self.voice_text_copy) is not bool:
            raise ConfigError("feishu.voice_text_copy 必须是布尔值")
        if not isinstance(self.cli, str) or not self.cli.strip() or "\x00" in self.cli:
            raise ConfigError("feishu.cli 必须是非空且不含 NUL 字符的命令名或路径")
        _config_seconds(self.cli_timeout, "feishu.cli_timeout")


@dataclass(frozen=True, slots=True)
class MiniMaxConnection:
    api_key: str = field(repr=False)
    region: str = "cn"

    def __post_init__(self) -> None:
        if self.region not in ("cn", "global"):
            raise ConfigError("MiniMax Provider 的 region 必须是 cn 或 global")


def require_asr_audio_seconds(value: float) -> None:
    # 整数不必转成 float：很大的有限整数同样是合法配置。
    if not ((type(value) is int and value > 0) or (type(value) is float and 0 < value < math.inf)):
        raise ConfigError("asr.max_audio_seconds 必须是有限正数")


@dataclass(frozen=True, slots=True)
class ASRSettings:
    connection: MiniMaxConnection
    max_audio_seconds: float
    model: str = MINIMAX_ASR_MODEL

    def __post_init__(self) -> None:
        require_asr_audio_seconds(self.max_audio_seconds)
        if not isinstance(self.model, str) or not self.model.strip():
            raise ConfigError("asr.model 必须是非空字符串")


@dataclass(frozen=True, slots=True)
class TTSSettings:
    connection: MiniMaxConnection
    model: str = "speech-2.8-turbo"
    voice_id: str = "male-qn-qingse"

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", self.model):
            raise ConfigError("tts.model 格式无效")
        if not (isinstance(self.voice_id, str) and 0 < len(self.voice_id) <= 128
                and self.voice_id.strip() == self.voice_id and self.voice_id.isprintable()):
            raise ConfigError("tts.voice_id 格式无效")


@dataclass(frozen=True, slots=True)
class VoiceSettings:
    input_device: str | int | None = None
    output_device: str | int | None = None
    vad_model: Path | None = None
    vad_threshold: float = 0.5
    speech_frames: int = 3  # 每帧 32 ms
    silence_seconds: float = 1
    output_sample_rate: int = 44_100
    prefetch_seconds: float = 0.4
    buffer_seconds: float = 2
    idle_timeout: float = 30
    delivery_timeout: float = 600

    def __post_init__(self) -> None:
        for name in ("input_device", "output_device"):
            value = getattr(self, name)
            if value is not None and not ((type(value) is int and value >= 0) or (
                isinstance(value, str) and value.strip() and "\x00" not in value
            )):
                raise ConfigError(f"voice.{name} 必须是有效的设备名称或非负整数索引")
        for name in ("vad_threshold", "silence_seconds", "prefetch_seconds",
                     "buffer_seconds", "idle_timeout", "delivery_timeout"):
            _config_seconds(getattr(self, name), f"voice.{name}")
        if self.vad_threshold >= 1:
            raise ConfigError("voice.vad_threshold 必须小于 1")
        if type(self.speech_frames) is not int or not 1 <= self.speech_frames <= 15:
            raise ConfigError("voice.speech_frames 必须是 1–15 之间的整数")
        if self.silence_seconds > 10:
            raise ConfigError("voice.silence_seconds 不得超过 10 秒")
        if type(self.output_sample_rate) is not int or self.output_sample_rate not in {16_000, 24_000, 32_000, 44_100}:
            raise ConfigError("voice.output_sample_rate 必须是 16000、24000、32000 或 44100")
        if not 0.032 <= self.buffer_seconds <= 5:
            raise ConfigError("voice.buffer_seconds 必须在 0.032–5 秒之间")
        if self.prefetch_seconds >= self.buffer_seconds:
            raise ConfigError("voice.prefetch_seconds 必须小于 voice.buffer_seconds")


@dataclass(frozen=True, slots=True)
class Settings:
    database: Path = Path("db.sqlite3")
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    llm: LLMSettings | None = None
    feishu: FeishuSettings | None = None
    asr: ASRSettings | None = None
    tts: TTSSettings | None = None
    voice: VoiceSettings = field(default_factory=VoiceSettings)
    # LLM 时限仅保存在 runtime.model_timeout，供既有 Runtime 机制使用。
    asr_timeout: float = DEFAULT_ASR_TIMEOUT
    tts_timeout: float = DEFAULT_TTS_TIMEOUT
    log_level: str = _DEFAULT_LOG_LEVEL

    def __post_init__(self) -> None:
        if not isinstance(self.log_level, str) or self.log_level not in _LOG_LEVELS:
            raise ConfigError("log_level 必须是 DEBUG、INFO、WARNING、ERROR 或 CRITICAL")
        _config_seconds(self.asr_timeout, "timeouts.asr")
        _config_seconds(self.tts_timeout, "timeouts.tts")


def _table(data: Any, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ConfigError(f"{name} 必须是 TOML 表")
    if data.keys() - keys:
        raise ConfigError(f"{name} 配置字段无效，请参考 config.example.toml")
    return data


def _section(data: Any, cls: type, name: str, *, exclude: tuple[str, ...] = ()) -> dict[str, Any]:
    section_fields = [item for item in fields(cls) if item.name not in exclude]
    data = _table(data, {item.name for item in section_fields}, name)
    for item in section_fields:
        if item.default is MISSING and item.default_factory is MISSING and item.name not in data:
            raise ConfigError(f"{name}.{item.name} 必须显式设置")
    return data


def _selected_provider(data: dict[str, Any], providers: dict[str, Any], ability: str,
                       adapters: tuple[str, ...]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    selector = f"{ability}_provider"
    name = data.get(selector)
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(f"必须显式设置 {selector}，请参考 config.example.toml")
    if name not in providers:
        raise ConfigError(f"{selector} 引用的 Provider 不存在")
    provider = _table(providers[name], {"adapter", "api_key", "base_url", "region", "llm", "asr", "tts"}, selector)
    if provider.get("adapter") not in adapters:
        raise ConfigError(f"{selector} 需要显式设置支持 {ability} 的 adapter（{' 或 '.join(adapters)}）")
    options = provider.get(ability)
    if not isinstance(options, dict):
        raise ConfigError(f"{selector} 引用的 Provider 缺少 {ability} 能力配置表")
    if "timeout" in options:
        raise ConfigError(f"Provider 能力配置 {ability}.timeout 不再支持，请改用 timeouts.{ability}")
    return name, provider, options


def _api_key(provider: dict[str, Any], selector: str) -> str:
    key = provider.get("api_key")
    if not isinstance(key, str) or not key:
        raise ConfigError(f"{selector} 引用的 Provider 必须设置非空 api_key 字符串")
    if any(not 33 <= ord(char) <= 126 for char in key):
        raise ConfigError(f"{selector} 引用的 Provider 的 api_key 含不允许的字符")
    return key


def _minimax_connection(provider: dict[str, Any], selector: str) -> MiniMaxConnection:
    if "base_url" in provider:
        raise ConfigError("base_url 仅适用于 openai_responses；MiniMax 请设置 region")
    return MiniMaxConnection(api_key=_api_key(provider, selector), region=provider.get("region", "cn"))


def _load_providers(data: dict[str, Any], *, voice: bool) -> tuple[LLMSettings, ASRSettings, TTSSettings | None]:
    providers = data.get("providers", {})
    if not isinstance(providers, dict):
        raise ConfigError("providers 必须是命名 Provider 配置表")
    _, provider, llm_data = _selected_provider(data, providers, "llm", ("deepseek", "openai_responses"))
    if "region" in provider:
        raise ConfigError("region 仅适用于 MiniMax Provider")
    llm_options = _section(llm_data, LLMSettings, "llm", exclude=("adapter", "base_url", "api_key"))
    llm = LLMSettings(adapter=provider["adapter"], base_url=provider.get("base_url"),
                      api_key=_api_key(provider, "llm_provider"), **llm_options)

    asr_name, provider, asr_data = _selected_provider(data, providers, "asr", ("minimax",))
    connection = _minimax_connection(provider, "asr_provider")
    asr = ASRSettings(connection, **_section(asr_data, ASRSettings, "asr", exclude=("connection",)))
    tts = None
    if voice:
        tts_name, provider, tts_data = _selected_provider(data, providers, "tts", ("minimax",))
        tts_connection = connection if tts_name == asr_name else _minimax_connection(provider, "tts_provider")
        tts = TTSSettings(tts_connection, **_section(tts_data, TTSSettings, "tts", exclude=("connection",)))
    return llm, asr, tts


def load_settings(path: Path | None = None, *, for_run: bool = False) -> Settings:
    if path is None:
        if for_run:
            raise ConfigError("正式启动需要 --config，请参考 config.example.toml")
        return Settings()
    data = _read_toml(path)
    return _parse_settings(data, path, for_run=for_run)


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as file:
            return tomllib.load(file)
    except (ValueError, RecursionError):
        pass  # 原始解析异常可能保留整份 TOML，不链接到可显示的诊断。
    raise ConfigError("TOML 无法解析，请检查配置文件格式")


def _load_voice(data: Any, path: Path) -> VoiceSettings:
    if isinstance(data, dict) and data.keys() & {"tts_model", "tts_timeout", "voice_id"}:
        raise ConfigError(_LEGACY_CONFIG_ERROR)
    voice = _section(data, VoiceSettings, "voice").copy()
    if "vad_model" in voice:
        model = voice["vad_model"]
        if not isinstance(model, str) or not model.strip() or "\x00" in model:
            raise ConfigError("voice.vad_model 必须是非空且不含 NUL 字符的路径")
        voice["vad_model"] = path.parent / model
    return VoiceSettings(**voice)


def _parse_settings(data: dict[str, Any], path: Path, *, for_run: bool) -> Settings:
    runtime_data = data.get("runtime", {})
    if (data.keys() & {"deepseek", "llm", "minimax"}
            or isinstance(runtime_data, dict) and "model_timeout" in runtime_data):
        raise ConfigError(_LEGACY_CONFIG_ERROR)
    _table(data, {"database", "log_level", "runtime", "feishu", "voice", "providers", "timeouts",
                  "llm_provider", "asr_provider", "tts_provider"}, "顶层")
    database = data.get("database", "db.sqlite3")
    if not isinstance(database, str) or not database.strip() or "\x00" in database:
        raise ConfigError("database 必须是非空且不含 NUL 字符的路径")
    if for_run and database == ":memory:":
        raise ConfigError("正式启动的 database 必须是持久化 SQLite 文件路径")
    if database != ":memory:":
        database = str(path.parent / database)
    timeouts = _table(data.get("timeouts", {}), {"llm", "asr", "tts"}, "timeouts")
    runtime = RuntimeSettings(model_timeout=timeouts.get("llm", DEFAULT_MODEL_TIMEOUT),
                              **_section(runtime_data, RuntimeSettings, "runtime", exclude=("model_timeout",)))
    feishu_data = data.get("feishu")
    feishu = FeishuSettings(**_section(feishu_data, FeishuSettings, "feishu")) if feishu_data is not None else None
    use_voice = feishu is not None and feishu.mode == "voice"
    # 离线调用仍可用来校验、读取语音配置；正式 text 不读取该段字段。
    voice = _load_voice(data.get("voice", {}), path) if not for_run or use_voice else VoiceSettings()
    llm = asr = tts = None
    if for_run:
        if feishu is None:
            raise ConfigError("正式启动需要 [feishu] 配置，请参考 config.example.toml")
        llm, asr, tts = _load_providers(data, voice=use_voice)
    return Settings(
        database=Path(database),
        runtime=runtime,
        llm=llm,
        feishu=feishu,
        asr=asr,
        tts=tts,
        voice=voice,
        asr_timeout=timeouts.get("asr", DEFAULT_ASR_TIMEOUT),
        tts_timeout=timeouts.get("tts", DEFAULT_TTS_TIMEOUT) if for_run and use_voice else DEFAULT_TTS_TIMEOUT,
        log_level=data.get("log_level", _DEFAULT_LOG_LEVEL),
    )
