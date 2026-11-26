"""启动配置；语言模型档案与 Provider 连接分别定义。"""

from collections.abc import Mapping
from dataclasses import MISSING, InitVar, dataclass, field, fields
from ipaddress import IPv6Address
import math
from pathlib import Path
import re
import tomllib
from typing import Any
from urllib.parse import urlsplit

import httpx

from .speech.bailian_ws import ASR_MODEL as BAILIAN_ASR_MODEL, TTS_MODELS as BAILIAN_TTS_MODELS, BailianConnection
from .speech.vad import silence_frames


OPENAI_BASE_URL = "https://api.openai.com/v1"
_MINIMAX_BASE_URLS = {"cn": "https://api.minimax.cn/v1", "global": "https://api.minimax.io/v1"}
MINIMAX_ASR_MODEL = "asr-1.0"
BAILIAN_TTS_MODEL = "qwen-audio-3.1-tts-flash"
BAILIAN_TTS_VOICE = "longanhuan_v3.1"
DEFAULT_ASR_TIMEOUT = 30
DEFAULT_MODEL_TIMEOUT = 60
DEFAULT_MAX_TOOL_ROUNDS = 8
DEFAULT_TTS_TIMEOUT = 25
_DEFAULT_WORKSPACE_ROOT = "workspaces"
_DEFAULT_LOG_LEVEL = "WARNING"
_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_LEGACY_CONFIG_ERROR = "旧配置格式不再支持，请按 config.example.toml 迁移到命名 Provider"
_MODEL_MIGRATION = "请迁到 [models.<ID>]，并在每个 Agent 角色文件中设置 model_id，参考 config.example.toml"
_CHAT_ADAPTERS = ("minimax", "bailian")
_LLM_ADAPTERS = ("deepseek", "openai_responses", *_CHAT_ADAPTERS)
_DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")


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


def normalize_minimax_base_url(value: str | None, region: str = "cn") -> str:
    if not isinstance(region, str) or region not in _MINIMAX_BASE_URLS:
        raise ConfigError("MiniMax Provider 的 region 必须是 cn 或 global")
    try:
        return normalize_openai_base_url(_MINIMAX_BASE_URLS[region] if value is None else value)
    except ValueError:
        raise ConfigError("base_url 必须是无凭据、查询或片段的 HTTPS 地址") from None


def bailian_llm_base_url(region: str, workspace_id: str | None) -> str:
    if not all(isinstance(value, str) and _DNS_LABEL.fullmatch(value) for value in (region, workspace_id)):
        raise ConfigError("百炼 LLM Provider 须设置有效的 region 和 workspace_id")
    return f"https://{workspace_id}.{region}.maas.aliyuncs.com/compatible-mode/v1"


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
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS

    def __post_init__(self) -> None:
        _config_seconds(self.model_timeout, "timeouts.llm")
        for name in ("context_budget", "tail_budget", "max_tool_rounds"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ConfigError(f"runtime.{name} 必须是正整数")
        if self.tail_budget >= self.context_budget:
            raise ConfigError("runtime.tail_budget 必须小于 runtime.context_budget")


@dataclass(frozen=True, slots=True)
class LLMProvider:
    adapter: str
    base_url: str | None = field(default=None, repr=False)
    api_key: object = field(default=None, kw_only=True, repr=False)
    region: str | None = field(default=None, kw_only=True)
    workspace_id: str | None = field(default=None, kw_only=True, repr=False)

    def __post_init__(self) -> None:
        if self.adapter not in _LLM_ADAPTERS:
            raise ConfigError(f"adapter 须为 {' 或 '.join(_LLM_ADAPTERS)}")
        if self.adapter == "bailian":
            if self.base_url is not None:
                raise ConfigError("百炼 Provider 请设置 region 和 workspace_id，不支持 base_url")
            region = "cn-beijing" if self.region is None else self.region
            object.__setattr__(self, "base_url", bailian_llm_base_url(region, self.workspace_id))
        elif self.adapter == "minimax":
            region = "cn" if self.region is None else self.region
            object.__setattr__(self, "base_url", normalize_minimax_base_url(self.base_url, region))
        elif self.adapter == "openai_responses":
            try:
                object.__setattr__(self, "base_url", normalize_openai_base_url(self.base_url))
            except ValueError:
                raise ConfigError("base_url 必须是无凭据、查询或片段的 HTTPS 地址") from None
        elif self.base_url is not None:
            raise ConfigError("base_url 仅适用于 openai_responses 和 MiniMax LLM")

    def require_api_key(self, location: str) -> str:
        return _require_api_key(self.api_key, location)


def valid_config_id(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip() and value.isprintable()


@dataclass(frozen=True, slots=True)
class ModelProfile:
    provider: str
    model: str
    reasoning_effort: str
    runtime: RuntimeSettings
    preserve_thinking: bool | None = None
    adapter: InitVar[str | None] = field(default=None, kw_only=True)

    def __post_init__(self, adapter: str | None) -> None:
        if not valid_config_id(self.provider):
            raise ConfigError("provider 必须是非空 Provider ID")
        if not isinstance(self.model, str) or not 0 < len(self.model) <= 128:
            raise ConfigError("model 格式无效")
        parts = self.model.split("/") if adapter == "bailian" else (self.model,)
        if any(not re.fullmatch(r"[A-Za-z0-9_.:-]+", part) or (adapter == "bailian" and part in (".", ".."))
               for part in parts):
            raise ConfigError("model 格式无效")
        if not valid_config_id(self.reasoning_effort) or len(self.reasoning_effort) > 64:
            raise ConfigError("reasoning_effort 必须是非空字符串（至多 64 字符）")
        if self.preserve_thinking is not None and type(self.preserve_thinking) is not bool:
            raise ConfigError("preserve_thinking 必须是布尔值")


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


SpeechConnection = MiniMaxConnection | BailianConnection


def require_asr_audio_seconds(value: float) -> None:
    # 整数不必转成 float：很大的有限整数同样是合法配置。
    if not ((type(value) is int and value > 0) or (type(value) is float and 0 < value < math.inf)):
        raise ConfigError("asr.max_audio_seconds 必须是有限正数")


@dataclass(frozen=True, slots=True)
class ASRSettings:
    connection: SpeechConnection
    max_audio_seconds: float
    model: str = MINIMAX_ASR_MODEL

    def __post_init__(self) -> None:
        require_asr_audio_seconds(self.max_audio_seconds)
        if not isinstance(self.model, str) or not self.model.strip():
            raise ConfigError("asr.model 必须是非空字符串")
        if isinstance(self.connection, BailianConnection) and self.model != BAILIAN_ASR_MODEL:
            raise ConfigError("百炼 asr.model 尚未适配文件及实时转写，请参考 config.example.toml")


@dataclass(frozen=True, slots=True)
class TTSSettings:
    connection: SpeechConnection
    model: str = "speech-2.8-turbo"
    voice_id: str = "male-qn-qingse"

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", self.model):
            raise ConfigError("tts.model 格式无效")
        if not (isinstance(self.voice_id, str) and 0 < len(self.voice_id) <= 128
                and self.voice_id.strip() == self.voice_id and self.voice_id.isprintable()):
            raise ConfigError("tts.voice_id 格式无效")
        if isinstance(self.connection, BailianConnection):
            if self.model not in BAILIAN_TTS_MODELS:
                raise ConfigError("百炼 tts.model 尚未适配流式合成，请参考 config.example.toml")
            # 两个已知的示例音色不能混用；其他音色是否有权限、是否匹配由百炼校验。
            example_model = {BAILIAN_TTS_VOICE: BAILIAN_TTS_MODEL, "longanyang": "cosyvoice-v3-flash"}.get(self.voice_id)
            if example_model is not None and self.model != example_model:
                raise ConfigError("百炼 tts.voice_id 与 tts.model 不匹配")


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
    segment_pause_seconds: float | None = None

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
        pause = self.segment_pause_seconds
        if pause is not None:
            _config_seconds(pause, "voice.segment_pause_seconds")
            if (pause >= self.silence_seconds
                    or silence_frames(pause, strict=True) >= silence_frames(self.silence_seconds, strict=True)):
                raise ConfigError("voice.segment_pause_seconds 和 voice.silence_seconds 折算为帧数后，"
                                  "前者必须小于后者")
        if type(self.output_sample_rate) is not int or self.output_sample_rate not in {16_000, 24_000, 32_000, 44_100}:
            raise ConfigError("voice.output_sample_rate 必须是 16000、24000、32000 或 44100")
        if not 0.032 <= self.buffer_seconds <= 5:
            raise ConfigError("voice.buffer_seconds 必须在 0.032–5 秒之间")
        if self.prefetch_seconds >= self.buffer_seconds:
            raise ConfigError("voice.prefetch_seconds 必须小于 voice.buffer_seconds")


@dataclass(frozen=True, slots=True)
class Settings:
    database: Path = Path("db.sqlite3")
    # 离线模型注入使用这套运行设置；正式角色使用所引用档案的 RuntimeSettings。
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    models: Mapping[str, ModelProfile] = field(default_factory=dict, repr=False)
    llm_providers: Mapping[str, LLMProvider] = field(default_factory=dict, repr=False)
    feishu: FeishuSettings | None = None
    asr: ASRSettings | None = None
    tts: TTSSettings | None = None
    voice: VoiceSettings = field(default_factory=VoiceSettings)
    asr_timeout: float = DEFAULT_ASR_TIMEOUT
    tts_timeout: float = DEFAULT_TTS_TIMEOUT
    log_level: str = _DEFAULT_LOG_LEVEL
    workspace_root: Path = Path(_DEFAULT_WORKSPACE_ROOT)

    def __post_init__(self) -> None:
        if not isinstance(self.log_level, str) or self.log_level not in _LOG_LEVELS:
            raise ConfigError("log_level 必须是 DEBUG、INFO、WARNING、ERROR 或 CRITICAL")
        _config_seconds(self.asr_timeout, "timeouts.asr")
        _config_seconds(self.tts_timeout, "timeouts.tts")
        # 32000 Hz 是本机支持但两种已适配百炼语音模型都不支持的采样率。
        if (self.tts is not None and isinstance(self.tts.connection, BailianConnection)
                and self.voice.output_sample_rate == 32_000):
            raise ConfigError("百炼 TTS 的 voice.output_sample_rate 须为 16000、24000 或 44100")


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
    provider = _table(providers[name], {"adapter", "api_key", "base_url", "region", "workspace_id",
                                        "asr", "tts"}, selector)
    if "workspace_id" in provider and provider.get("adapter") != "bailian":
        raise ConfigError("workspace_id 仅适用于百炼 Provider")
    if provider.get("adapter") not in adapters:
        raise ConfigError(f"{selector} 需要显式设置支持 {ability} 的 adapter（{' 或 '.join(adapters)}）")
    options = provider.get(ability)
    if not isinstance(options, dict):
        raise ConfigError(f"{selector} 引用的 Provider 缺少 {ability} 能力配置表")
    if "timeout" in options:
        raise ConfigError(f"Provider 能力配置 {ability}.timeout 不再支持，请改用 timeouts.{ability}")
    return name, provider, options


def _api_key(provider: dict[str, Any], selector: str) -> str:
    return _require_api_key(provider.get("api_key"), selector)


def _require_api_key(key: object, selector: str) -> str:
    if not isinstance(key, str) or not key:
        raise ConfigError(f"{selector} 引用的 Provider 必须设置非空 api_key 字符串")
    if any(not 33 <= ord(char) <= 126 for char in key):
        raise ConfigError(f"{selector} 引用的 Provider 的 api_key 含不允许的字符")
    return key


def _speech_connection(provider: dict[str, Any], selector: str, llm: LLMProvider | None) -> SpeechConnection:
    if "base_url" in provider and (llm is None or llm.adapter != "minimax"):
        raise ConfigError("base_url 仅适用于 LLM；语音 Provider 请设置 region")
    key = _api_key(provider, selector)
    if provider["adapter"] == "minimax":
        return MiniMaxConnection(api_key=key, region=provider.get("region", "cn"))
    try:
        return BailianConnection(api_key=key, workspace_id=provider.get("workspace_id"),
                                 region=provider.get("region", "cn-beijing"))
    except ValueError:
        raise ConfigError("百炼 Provider 须设置有效的 workspace_id，region 为 cn-beijing 或 ap-southeast-1") from None


def _provider_tables(data: dict[str, Any]) -> dict[str, Any]:
    providers = data.get("providers", {})
    if not isinstance(providers, dict):
        raise ConfigError("providers 必须是命名 Provider 配置表")
    for index, provider in enumerate(providers.values(), 1):
        location = f"providers 第 {index} 项"
        if isinstance(provider, dict) and "llm" in provider:
            raise ConfigError(f"{location} 的 llm 表已移除；{_MODEL_MIGRATION}")
    return providers


def _load_model_profiles(data: Any, providers: dict[str, Any], model_timeout: float,
                         ) -> tuple[dict[str, ModelProfile], dict[str, LLMProvider]]:
    if not isinstance(data, dict):
        raise ConfigError("models 必须是命名模型配置表")
    references = {options["provider"] for options in data.values()
                  if isinstance(options, dict) and isinstance(options.get("provider"), str)}
    llm_providers = {}
    for index, (name, provider) in enumerate(providers.items(), 1):
        if not isinstance(provider, dict) or provider.get("adapter") not in _LLM_ADAPTERS:
            continue  # 非 LLM Provider 仍按语音能力的选用规则解析。
        location = f"providers 第 {index} 项"
        adapter = provider["adapter"]
        keys = {"adapter", "api_key", "base_url"}
        if adapter in _CHAT_ADAPTERS:
            if name not in references:
                continue  # 既有纯语音表不因新增同名 LLM 适配器而被解析为 LLM。
            keys.add("region")
            if adapter == "bailian":
                keys.add("workspace_id")
        _table(provider, keys | ({"asr", "tts"} if adapter in _CHAT_ADAPTERS else set()), location)
        try:
            llm_providers[name] = LLMProvider(**{key: provider[key] for key in keys if key in provider})
        except ConfigError as exc:
            raise ConfigError(f"{location}：{exc}") from None

    models = {}
    required = ("provider", "model", "reasoning_effort", "context_budget", "tail_budget")
    for index, (name, options) in enumerate(data.items(), 1):
        location = f"models 第 {index} 项"
        if not valid_config_id(name):
            raise ConfigError(f"{location} 的 ID 必须是非空字符串且不能有首尾空白")
        options = _table(options, set(required) | {"max_tool_rounds", "preserve_thinking"}, location)
        for key in required:
            if key not in options:
                raise ConfigError(f"{location} 的 {key} 必须显式设置")
        try:
            runtime = RuntimeSettings(model_timeout=model_timeout, context_budget=options["context_budget"],
                                      tail_budget=options["tail_budget"],
                                      max_tool_rounds=options.get("max_tool_rounds", DEFAULT_MAX_TOOL_ROUNDS))
            provider_id = options["provider"]
            llm = llm_providers.get(provider_id) if isinstance(provider_id, str) else None
            profile = ModelProfile(provider_id, options["model"], options["reasoning_effort"], runtime,
                                   options.get("preserve_thinking"), adapter=None if llm is None else llm.adapter)
        except ConfigError as exc:
            raise ConfigError(f"{location}：{str(exc).replace('runtime.', '')}") from None
        if profile.provider not in providers:
            raise ConfigError(f"{location} 的 provider 引用的 Provider 不存在")
        if profile.provider not in llm_providers:
            raise ConfigError(f"{location} 的 provider 需要显式设置支持 llm 的 adapter（{' 或 '.join(_LLM_ADAPTERS)}）")
        if profile.preserve_thinking is not None and llm_providers[profile.provider].adapter != "bailian":
            raise ConfigError(f"{location} 的 preserve_thinking 仅适用于百炼模型")
        models[name] = profile
    return models, llm_providers


def _load_speech(data: dict[str, Any], providers: dict[str, Any], llm_providers: dict[str, LLMProvider], *, voice: bool
                 ) -> tuple[ASRSettings, TTSSettings | None]:
    asr_name, provider, asr_data = _selected_provider(data, providers, "asr", ("minimax", "bailian"))
    connection = _speech_connection(provider, "asr_provider", llm_providers.get(asr_name))
    asr_options = _section(asr_data, ASRSettings, "asr", exclude=("connection",)).copy()
    if isinstance(connection, BailianConnection):
        asr_options.setdefault("model", BAILIAN_ASR_MODEL)
    asr = ASRSettings(connection, **asr_options)
    tts = None
    if voice:
        tts_name, provider, tts_data = _selected_provider(data, providers, "tts", ("minimax", "bailian"))
        tts_connection = connection if tts_name == asr_name else _speech_connection(
            provider, "tts_provider", llm_providers.get(tts_name))
        tts_options = _section(tts_data, TTSSettings, "tts", exclude=("connection",)).copy()
        if isinstance(tts_connection, BailianConnection):
            tts_model = tts_options.setdefault("model", BAILIAN_TTS_MODEL)
            if tts_model == BAILIAN_TTS_MODEL:
                tts_options.setdefault("voice_id", BAILIAN_TTS_VOICE)
            elif "voice_id" not in tts_options:
                raise ConfigError("切换百炼 tts.model 时必须同时设置匹配的 tts.voice_id")
        tts = TTSSettings(tts_connection, **tts_options)
    return asr, tts


def load_settings(path: Path | None = None, *, for_run: bool = False) -> Settings:
    if path is None:
        if for_run:
            raise ConfigError("正式启动需要 --config，请参考 config.example.toml")
        return Settings()
    data = _read_toml(path)
    try:
        return _parse_settings(data, path, for_run=for_run)
    except ConfigError as exc:
        raise ConfigError(f"配置文件 {path.name!r}：{exc}") from None


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as file:
            return tomllib.load(file)
    except (ValueError, RecursionError):
        pass  # 原始解析异常可能保留整份 TOML，不链接到可显示的诊断。
    raise ConfigError(f"配置文件 {path.name!r}：TOML 无法解析，请检查配置文件格式")


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
    if "llm_provider" in data:
        raise ConfigError(f"顶层 llm_provider 已移除；{_MODEL_MIGRATION}")
    if isinstance(runtime_data, dict):
        for key in ("context_budget", "tail_budget", "max_tool_rounds"):
            if key in runtime_data:
                raise ConfigError(f"runtime.{key} 已移除；{_MODEL_MIGRATION}")
    if (data.keys() & {"deepseek", "llm", "minimax"}
            or isinstance(runtime_data, dict) and "model_timeout" in runtime_data):
        raise ConfigError(_LEGACY_CONFIG_ERROR)
    _table(data, {"database", "log_level", "runtime", "feishu", "voice", "providers", "models", "timeouts",
                  "asr_provider", "tts_provider", "workspace_root"}, "顶层")
    _table(runtime_data, set(), "runtime")
    database = data.get("database", "db.sqlite3")
    if not isinstance(database, str) or not database.strip() or "\x00" in database:
        raise ConfigError("database 必须是非空且不含 NUL 字符的路径")
    if for_run and database == ":memory:":
        raise ConfigError("正式启动的 database 必须是持久化 SQLite 文件路径")
    if database != ":memory:":
        database = str(path.parent / database)
    workspace_root = data.get("workspace_root", _DEFAULT_WORKSPACE_ROOT)
    if not isinstance(workspace_root, str) or not workspace_root.strip() or "\x00" in workspace_root:
        raise ConfigError("workspace_root 必须是非空且不含 NUL 字符的路径")
    timeouts = _table(data.get("timeouts", {}), {"llm", "asr", "tts"}, "timeouts")
    runtime = RuntimeSettings(model_timeout=timeouts.get("llm", DEFAULT_MODEL_TIMEOUT))
    models, llm_providers = {}, {}
    feishu_data = data.get("feishu")
    feishu = FeishuSettings(**_section(feishu_data, FeishuSettings, "feishu")) if feishu_data is not None else None
    use_voice = feishu is not None and feishu.mode == "voice"
    # 离线调用仍可用来校验、读取语音配置；正式 text 不读取该段字段。
    voice = _load_voice(data.get("voice", {}), path) if not for_run or use_voice else VoiceSettings()
    asr = tts = None
    if for_run:
        if feishu is None:
            raise ConfigError("正式启动需要 [feishu] 配置，请参考 config.example.toml")
        providers = _provider_tables(data)
        models, llm_providers = _load_model_profiles(data.get("models", {}), providers, runtime.model_timeout)
        asr, tts = _load_speech(data, providers, llm_providers, voice=use_voice)
    return Settings(
        database=Path(database),
        runtime=runtime,
        models=models,
        llm_providers=llm_providers,
        feishu=feishu,
        asr=asr,
        tts=tts,
        voice=voice,
        asr_timeout=timeouts.get("asr", DEFAULT_ASR_TIMEOUT),
        tts_timeout=timeouts.get("tts", DEFAULT_TTS_TIMEOUT) if for_run and use_voice else DEFAULT_TTS_TIMEOUT,
        log_level=data.get("log_level", _DEFAULT_LOG_LEVEL),
        workspace_root=path.parent / workspace_root,
    )
