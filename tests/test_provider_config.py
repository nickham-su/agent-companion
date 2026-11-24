import asyncio
from contextlib import asynccontextmanager
from io import BytesIO, StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import traceback
import unittest
from unittest.mock import AsyncMock, patch
import wave

import httpcore
import httpx
from feishu_test_support import private_discovery

from agent_companion.__main__ import _serve, main
from agent_companion.app import prepare_private_feishu, running_agent
from agent_companion.config import ConfigError, VoiceSettings, load_settings
from agent_companion.models.deepseek import CHAT_COMPLETIONS
from agent_companion.protocol import DeliveryStatus, Message, Receipt
from agent_companion.runtime.subtasks import demo_roles
from agent_companion.speech.asr import AudioFile
from agent_companion.speech.minimax_tts import MiniMaxTTS


_KEY = "synthetic-private-config-value"
_FEISHU = '''
[feishu]
mode = "text"
'''
_CONFIG = '''llm_provider = "chat"
asr_provider = "speech"
tts_provider = "speech"
''' + _FEISHU + f'''
[voice]
output_sample_rate = 24000

[providers.chat]
adapter = "deepseek"
api_key = "{_KEY}"
[providers.chat.llm]
reasoning_effort = "disabled"

[providers.speech]
adapter = "minimax"
api_key = "synthetic-speech-value"
[providers.speech.asr]
max_audio_seconds = 500
[providers.speech.tts]
model = "synthetic-tts-model"
voice_id = "synthetic-shared-voice"
'''


def _native_config(adapter):
    if adapter == "deepseek":
        return _CONFIG
    return _CONFIG.replace('adapter = "deepseek"', 'adapter = "openai_responses"').replace(
        'reasoning_effort = "disabled"', 'model = "fixture-model"\nreasoning_effort = "low"')


class ProviderConfigTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "synthetic.toml"

    def assert_startup_error(self, content, message):
        self.path.write_text(content, encoding="utf-8")
        with (patch("sys.argv", ["agent-companion", "--run", "--config", str(self.path)]),
              patch("sys.stderr", new_callable=StringIO) as stderr,
              patch("os.getenv", return_value="unused-synthetic-env-value") as getenv,
              patch("agent_companion.__main__._serve", new_callable=AsyncMock) as serve):
            with self.assertRaises(SystemExit) as caught:
                main()
            self.assertEqual(caught.exception.code, 2)
            self.assertIn(message, stderr.getvalue())
            self.assertNotIn(_KEY, stderr.getvalue())
            self.assertIsNone(caught.exception.__context__)
            serve.assert_not_awaited()
            getenv.assert_not_called()

    def test_example_contains_switchable_llms_and_shared_speech_provider(self):
        example = Path("config.example.toml").read_text(encoding="utf-8")
        example = example.replace('[timeouts]\nllm = 60', '[timeouts]\nllm = 12.5')
        for name, adapter in (("deepseek_default", "deepseek"), ("openai_default", "openai_responses")):
            with self.subTest(adapter=adapter):
                self.path.write_text(example.replace('llm_provider = "deepseek_default"', f'llm_provider = "{name}"')
                                     + _FEISHU.replace('"text"', '"voice"'), encoding="utf-8")
                with (patch("sys.argv", ["agent-companion", "--run", "--config", str(self.path)]),
                      patch("agent_companion.__main__._serve", new_callable=AsyncMock) as serve,
                      patch("os.getenv") as getenv):
                    self.assertEqual(main(), 0)
                getenv.assert_not_called()
                settings, = serve.await_args.args
                self.assertEqual(settings.llm.adapter, adapter)
                self.assertEqual(settings.runtime.model_timeout, 12.5)
                self.assertEqual(settings.asr_timeout, 30)
                self.assertEqual(settings.tts_timeout, 25)
                self.assertEqual(settings.asr.model, "asr-1.0")
                self.assertEqual(settings.asr.max_audio_seconds, 500)
                self.assertEqual(settings.tts.model, "speech-2.8-turbo")
                self.assertIs(settings.asr.connection, settings.tts.connection)

    def test_selected_references_must_exist_and_support_the_capability(self):
        for before, after, error in (
            ('llm_provider = "chat"', '', '必须显式设置 llm_provider'),
            ('asr_provider = "speech"', '', '必须显式设置 asr_provider'),
            ('llm_provider = "chat"', f'llm_provider = "{_KEY}_absent"', 'llm_provider 引用的 Provider 不存在'),
            ('asr_provider = "speech"', 'asr_provider = "absent"', 'asr_provider 引用的 Provider 不存在'),
            ('asr_provider = "speech"', 'asr_provider = "chat"', '支持 asr 的 adapter'),
            ('llm_provider = "chat"', 'llm_provider = "speech"', '支持 llm 的 adapter'),
            ('[providers.speech.asr]\nmax_audio_seconds = 500\n', '', '缺少 asr 能力配置表'),
        ):
            with self.subTest(error=error):
                self.assert_startup_error(_CONFIG.replace(before, after), error)
        voice = _CONFIG.replace('mode = "text"', 'mode = "voice"')
        for target, error in (('', '必须显式设置 tts_provider'),
                              ('tts_provider = "absent"', 'tts_provider 引用的 Provider 不存在'),
                              ('tts_provider = "chat"', '支持 tts 的 adapter')):
            with self.subTest(error=error):
                self.assert_startup_error(voice.replace('tts_provider = "speech"', target), error)
        self.assert_startup_error(voice.split('[providers.speech.tts]')[0], '缺少 tts 能力配置表')

    def test_selected_credentials_are_checked_without_environment_fallback_or_repr(self):
        for value in ('', 'api_key = 123', 'api_key = ""', f'api_key = "{_KEY}\\ninvalid"'):
            self.assert_startup_error(_CONFIG.replace(f'api_key = "{_KEY}"', value), 'api_key')
        self.assert_startup_error(_CONFIG.replace('api_key = "synthetic-speech-value"', ''), 'asr_provider')
        self.assert_startup_error(_CONFIG.replace('mode = "text"', 'mode = "voice"')
                                  .replace('tts_provider = "speech"', 'tts_provider = "speaker"')
                                  + '\n[providers.speaker]\nadapter = "minimax"\n[providers.speaker.tts]\n',
                                  'tts_provider 引用的 Provider 必须设置非空 api_key')
        self.path.write_text(_CONFIG, encoding="utf-8")
        rendered = repr(load_settings(self.path, for_run=True))
        self.assertNotIn(_KEY, rendered)
        self.assertNotIn("synthetic-speech-value", rendered)

    def test_text_ignores_inactive_tts_and_other_providers(self):
        for selector in ('', 'tts_provider = false', 'tts_provider = "missing"', 'tts_provider = "unselected"',
                         'tts_provider = "speech"'):
            with self.subTest(selector=selector):
                config = _CONFIG.replace('tts_provider = "speech"', selector).split('[providers.speech.tts]')[0]
                config = config.replace('[providers.speech.asr]', 'tts = false\n[providers.speech.asr]')
                config += '''
[providers.unselected]
adapter = "minimax"
[providers.unselected.tts]
model = false
[providers.bad_key]
api_key = 123
'''
                self.path.write_text(config, encoding="utf-8")
                settings = load_settings(self.path, for_run=True)
                self.assertEqual(settings.asr.connection.region, "cn")
                self.assertIsNone(settings.tts)

    def test_formal_text_ignores_invalid_voice_values_and_table_shape(self):
        voice_table = '[voice]\noutput_sample_rate = 24000'
        variants = (
            (_CONFIG.replace(voice_table, '[voice]\ninput_device = false\nvad_model = false'), 'voice.vad_model'),
            (_CONFIG.replace(voice_table, '[voice]\ninput_device = false'), 'voice.input_device'),
            ('voice = false\n' + _CONFIG.replace(voice_table, ''), 'voice 必须是 TOML 表'),
            (_CONFIG.replace(voice_table, f'[voice]\n{_KEY} = true'), 'voice 配置字段无效'),
            (_CONFIG.replace(voice_table, '[voice]\ntts_model = "legacy"'), '旧配置格式不再支持'),
        )
        for config, diagnostic in variants:
            with self.subTest(diagnostic=diagnostic):
                self.path.write_text(config, encoding="utf-8")
                with (patch("sys.argv", ["agent-companion", "--run", "--config", str(self.path)]),
                      patch("agent_companion.__main__._serve", new_callable=AsyncMock) as serve):
                    self.assertEqual(main(), 0)
                serve.assert_awaited_once()
                settings, = serve.await_args.args
                self.assertEqual(settings.voice, VoiceSettings())
                self.assertIsNone(settings.tts)
                self.assertIsNone(settings.voice.vad_model)
                self.assert_startup_error(config.replace('mode = "text"', 'mode = "voice"'), diagnostic)

    def test_non_run_config_still_resolves_and_validates_voice_settings(self):
        self.path.write_text(_CONFIG.replace('[voice]', '[voice]\nvad_model = "models/synthetic.onnx"\ninput_device = 2'),
                             encoding="utf-8")
        settings = load_settings(self.path)
        self.assertEqual(settings.voice.input_device, 2)
        self.assertEqual(settings.voice.output_sample_rate, 24000)
        self.assertEqual(settings.voice.vad_model, self.path.parent / 'models/synthetic.onnx')
        self.path.write_text(_CONFIG.replace('[voice]', '[voice]\nvad_model = false'), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, r"voice\.vad_model"):
            load_settings(self.path)

    def test_known_common_config_errors_name_safe_fields(self):
        for config, diagnostic in (
            (_CONFIG.replace('[feishu]', f'[feishu]\nown_open_id = "{_KEY}"'), 'feishu 配置字段无效'),
            (_CONFIG.replace('[feishu]', '[feishu]\nchat_id = "oc_fixture"'), 'feishu 配置字段无效'),
            (_CONFIG.replace('[feishu]', '[feishu]\nsender_open_id = "ou_peer"'), 'feishu 配置字段无效'),
            (_CONFIG.replace('mode = "text"', f'mode = "{_KEY}"'), 'feishu.mode 必须是 text 或 voice'),
            (_CONFIG.replace('mode = "text"', 'mode = "text"\ncli_timeout = false'), 'feishu.cli_timeout 必须是有限正数'),
            (_CONFIG + '\n[runtime]\ndelivery_timeout = 120\n', 'runtime 配置字段无效'),
            (_CONFIG + '\n[runtime]\ncontext_budget = 42\n',
             'runtime.tail_budget 必须小于 runtime.context_budget'),
            (_CONFIG + '\n[runtime]\nsummary_budget = 4000\n', 'runtime 配置字段无效'),
            (_CONFIG + f'\n[runtime]\n{_KEY} = 42\n', 'runtime 配置字段无效'),
            ('database = false\n' + _CONFIG, 'database 必须是非空且不含 NUL 字符的路径'),
            ('database = ":memory:"\n' + _CONFIG, '正式启动的 database 必须是持久化 SQLite 文件路径'),
        ):
            with self.subTest(diagnostic=diagnostic):
                self.assert_startup_error(config, diagnostic)

    def test_removed_keys_are_rejected_in_text_and_voice(self):
        for mode in ("text", "voice"):
            config = _CONFIG.replace('mode = "text"', f'mode = "{mode}"')
            for key, old, diagnostic in (
                ("main_session_id", f'main_session_id = "{_KEY}"\n' + config, "顶层 配置字段无效"),
                ("peer_open_id", config.replace('[feishu]', f'[feishu]\npeer_open_id = "{_KEY}"'),
                 "feishu 配置字段无效"),
                ("auto_answer", config.replace('[feishu]', '[feishu]\nauto_answer = false'),
                 "feishu 配置字段无效"),
            ):
                with self.subTest(mode=mode, key=key):
                    self.assert_startup_error(old, diagnostic)

    def test_known_voice_config_errors_report_the_field_and_constraint(self):
        voice = _CONFIG.replace('mode = "text"', 'mode = "voice"')
        for setting, diagnostic in (
            ('vad_threshold = 1', 'voice.vad_threshold 必须小于 1'),
            ('speech_frames = false', 'voice.speech_frames 必须是 1–15 之间的整数'),
            ('max_turn_seconds = 60', 'voice 配置字段无效'),
            ('silence_seconds = 11', 'voice.silence_seconds 不得超过 10 秒'),
            ('buffer_seconds = 8', 'voice.buffer_seconds 必须在 0.032–5 秒之间'),
             ('prefetch_seconds = 3', 'voice.prefetch_seconds 必须小于 voice.buffer_seconds'),
             ('idle_timeout = 0', 'voice.idle_timeout 必须是有限正数'),
             ('idle_timeout = inf', 'voice.idle_timeout 必须是有限正数'),
             ('delivery_timeout = nan', 'voice.delivery_timeout 必须是有限正数'),
        ):
            with self.subTest(diagnostic=diagnostic):
                self.assert_startup_error(voice.replace('[voice]', f'[voice]\n{setting}'), diagnostic)
        self.assert_startup_error(voice.replace('output_sample_rate = 24000', f'output_sample_rate = "{_KEY}"'),
                                  'voice.output_sample_rate 必须是 16000、24000、32000 或 44100')
        self.assert_startup_error(voice.replace('output_sample_rate', 'playback_rate'), 'voice 配置字段无效')

    def test_delivery_timeouts_are_independent_and_have_defaults(self):
        self.path.write_text(_CONFIG.replace('mode = "text"', 'mode = "voice"'), encoding="utf-8")
        settings = load_settings(self.path, for_run=True)
        self.assertEqual((settings.feishu.cli_timeout, settings.voice.idle_timeout, settings.voice.delivery_timeout),
                         (20, 30, 600))
        common = _CONFIG.replace('[feishu]', '[feishu]\ncli_timeout = 300').replace(
            '[voice]', '[voice]\ndelivery_timeout = 700\nidle_timeout = 45')
        for mode in ('text', 'voice'):
            with self.subTest(mode=mode):
                self.path.write_text(common.replace('mode = "text"', f'mode = "{mode}"'), encoding="utf-8")
                settings = load_settings(self.path, for_run=True)
                self.assertEqual(settings.feishu.cli_timeout, 300)
                if mode == 'voice':
                    self.assertEqual((settings.voice.idle_timeout, settings.voice.delivery_timeout), (45, 700))
                else:
                    self.assertEqual(settings.voice, VoiceSettings())

    def test_voice_text_copy_defaults_to_true_and_requires_a_boolean(self):
        voice = _CONFIG.replace('mode = "text"', 'mode = "voice"')
        self.path.write_text(voice, encoding="utf-8")
        self.assertIs(load_settings(self.path, for_run=True).feishu.voice_text_copy, True)
        self.path.write_text(voice.replace('[feishu]', '[feishu]\nvoice_text_copy = false'), encoding="utf-8")
        self.assertIs(load_settings(self.path, for_run=True).feishu.voice_text_copy, False)
        for value in ('"false"', '1', '[]'):
            with self.subTest(value=value):
                self.assert_startup_error(voice.replace('[feishu]', f'[feishu]\nvoice_text_copy = {value}'),
                                          'feishu.voice_text_copy 必须是布尔值')

    def test_explicit_llm_settings_and_adapter_specific_model_and_address(self):
        for before, after, message in (
            ('adapter = "deepseek"', '', 'adapter'),
            ('reasoning_effort = "disabled"', '', 'reasoning_effort'),
            ('reasoning_effort = "disabled"', 'reasoning_effort = " "', 'reasoning_effort'),
            ('reasoning_effort = "disabled"', 'reasoning_effort = "high"\nreplay_profile = "legacy"', '配置字段无效'),
            ('adapter = "deepseek"', 'adapter = "openai_responses"', '必须设置 llm.model'),
            ('adapter = "deepseek"', 'adapter = "deepseek"\nbase_url = "https://example.invalid"', 'base_url'),
            ('adapter = "minimax"', 'adapter = "minimax"\nbase_url = "https://example.invalid"', 'base_url'),
            ('[providers.speech.asr]', '[providers.speech.asr]\nmodel = " "', 'asr.model 必须是非空字符串'),
        ):
            with self.subTest(message=message):
                self.assert_startup_error(_CONFIG.replace(before, after), message)

    def test_selected_asr_audio_seconds_must_be_explicit_and_positive_in_both_modes(self):
        for mode in ('text', 'voice'):
            text = _CONFIG.replace('mode = "text"', f'mode = "{mode}"')
            for value in (None, '0', '-0.01', 'inf', 'nan', 'true', '"500"'):
                with self.subTest(mode=mode, value=value):
                    setting = '' if value is None else f'max_audio_seconds = {value}'
                    diagnostic = ('asr.max_audio_seconds 必须显式设置' if value is None
                                  else 'asr.max_audio_seconds 必须是有限正数')
                    self.assert_startup_error(text.replace('max_audio_seconds = 500', setting), diagnostic)

    def test_shared_timeouts_default_when_no_table_is_configured(self):
        for mode in ('text', 'voice'):
            with self.subTest(mode=mode):
                self.path.write_text(_CONFIG.replace('mode = "text"', f'mode = "{mode}"'), encoding="utf-8")
                settings = load_settings(self.path, for_run=True)
                self.assertEqual((settings.runtime.model_timeout, settings.asr_timeout, settings.tts_timeout),
                                 (60, 30, 25))

    def test_shared_timeouts_validate_table_and_active_values_without_leaking_input(self):
        self.assert_startup_error('timeouts = false\n' + _CONFIG, 'timeouts 必须是 TOML 表')
        self.assert_startup_error(_CONFIG + f'\n[timeouts]\n{_KEY} = true\n', 'timeouts 配置字段无效')
        invalid = {'bool': 'false', 'zero': '0', 'negative': '-1', 'nan': 'nan', 'inf': 'inf', 'string': f'"{_KEY}"'}
        for ability in ('llm', 'asr', 'tts'):
            for kind, value in invalid.items():
                with self.subTest(ability=ability, kind=kind):
                    config = _CONFIG + f'\n[timeouts]\n{ability} = {value}\n'
                    if ability == 'tts':
                        self.path.write_text(config, encoding="utf-8")
                        settings = load_settings(self.path, for_run=True)
                        self.assertIsNone(settings.tts)
                        self.assertEqual(settings.tts_timeout, 25)
                        config = config.replace('mode = "text"', 'mode = "voice"')
                    self.assert_startup_error(config, f'timeouts.{ability} 必须是有限正数')

    def test_provider_timeout_migration_only_checks_active_capabilities(self):
        for name, ability in (('chat', 'llm'), ('speech', 'asr'), ('speech', 'tts')):
            with self.subTest(ability=ability):
                table = f'[providers.{name}.{ability}]'
                config = _CONFIG.replace(table, f'{table}\ntimeout = "{_KEY}"')
                if ability == 'tts':
                    self.path.write_text(config, encoding="utf-8")
                    self.assertIsNone(load_settings(self.path, for_run=True).tts)
                    config = config.replace('mode = "text"', 'mode = "voice"')
                self.assert_startup_error(config, f'{ability}.timeout 不再支持，请改用 timeouts.{ability}')
        voice = _CONFIG.replace('mode = "text"', 'mode = "voice"')
        self.path.write_text(voice + '''
[providers.unused]
[providers.unused.llm]
timeout = 3
[providers.unused.tts]
timeout = false
''', encoding="utf-8")
        self.assertEqual(load_settings(self.path, for_run=True).tts_timeout, 25)

    def test_migration_and_toml_parse_errors_have_safe_diagnostics(self):
        for section in ('[deepseek]\nthinking = false', '[llm]\nadapter = "deepseek"', '[minimax]\nasr_timeout = 2',
                        '[runtime]\nmodel_timeout = 1.25'):
            with self.subTest(section=section):
                self.assert_startup_error(section, '旧配置格式不再支持，请按 config.example.toml 迁移')
        invalid = f'[providers.chat]\napi_key = "{_KEY}'.encode()
        self.assert_startup_error(invalid.decode(), 'TOML 无法解析')
        self.path.write_bytes(invalid + b'\xff')
        with self.assertRaises(ConfigError) as caught:
            load_settings(self.path, for_run=True)
        self.assertIsNone(caught.exception.__context__)
        self.assertNotIn(_KEY, ''.join(traceback.format_exception(caught.exception)))

    def test_demo_does_not_resolve_production_fields_and_run_needs_config(self):
        self.path.write_text('llm_provider = false\nproviders = "ignored in demo"\n[timeouts]\ntts = false\n',
                             encoding="utf-8")
        with (patch("sys.argv", ["agent-companion", "--demo-subtasks", "--config", str(self.path)]),
              patch("sys.stdout", new_callable=StringIO) as stdout):
            self.assertEqual(main(), 0)
            self.assertIn("离线模拟：主会话已收到同一子 Agent 的结果。", stdout.getvalue())
        with (patch("sys.argv", ["agent-companion", "--run"]),
              patch("sys.stderr", new_callable=StringIO) as stderr, self.assertRaises(SystemExit)):
            main()
        self.assertIn("正式启动需要 --config", stderr.getvalue())

    def test_unexpected_load_and_startup_errors_do_not_expose_exception_chain(self):
        def error():
            try:
                raise ValueError(_KEY)
            except ValueError as exc:
                raise RuntimeError(_KEY) from exc

        with (patch("sys.argv", ["agent-companion", "--run"]),
              patch("sys.stderr", new_callable=StringIO) as stderr,
              patch("agent_companion.__main__.load_settings", side_effect=error), self.assertRaises(SystemExit) as caught):
            main()
        self.assertIn("无法加载配置，请检查文件路径和配置格式", stderr.getvalue())
        self.assertNotIn(_KEY, stderr.getvalue())
        self.assertIsNone(caught.exception.__context__)

        async def fail_start(settings):
            error()

        self.path.write_text(_CONFIG, encoding="utf-8")
        with (patch("sys.argv", ["agent-companion", "--run", "--config", str(self.path)]),
              patch("agent_companion.__main__._serve", side_effect=fail_start),
              self.assertLogs("agent_companion.__main__", "ERROR") as logs):
            self.assertEqual(main(), 1)
        self.assertIn("飞书模式已停止（RuntimeError）", logs.output[0])
        self.assertNotIn(_KEY, logs.output[0])


class ProviderCompositionTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_transport_internal_close_outlives_runtime_grace(self):
        for adapter in ("deepseek", "openai_responses"):
            for eof_first in (True, False):
                with self.subTest(adapter=adapter, eof_first=eof_first):
                    await self._check_native_transport_close(adapter, eof_first=eof_first)

    async def test_native_transport_internal_close_budget_expires_at_shutdown(self):
        await self._check_native_transport_close("deepseek", eof_first=False, close_mode="timeout")

    async def test_native_transport_internal_close_failure_is_reported(self):
        await self._check_native_transport_close("openai_responses", eof_first=True, close_mode="failure")

    async def _check_native_transport_close(self, adapter, *, eof_first, close_mode="success"):
        trace = []
        read_waiting, wire_closing, release_close, runtime_stopped = (asyncio.Event() for _ in range(4))
        if adapter == "deepseek":
            frame = {"choices": [{"index": 0, "delta": {"content": "fixture text"}}]}
        else:
            frame = {"type": "response.output_text.delta", "output_index": 0, "content_index": 0,
                     "delta": "fixture text"}
        chunk = f"data: {json.dumps(frame)}\n\n".encode()
        packet = (b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
                  + f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")  # 故意缺少终止块。
        loop = asyncio.get_running_loop()
        stop_at = None

        class Wire(httpcore.AsyncNetworkStream):
            sent = False

            async def read(self, max_bytes, timeout=None):
                if not self.sent:
                    self.sent = True
                    return packet
                if eof_first:
                    trace.append("body-eof")
                    return b""
                read_waiting.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    trace.append("body-read-cancelled")
                    raise

            async def write(self, buffer, timeout=None):
                pass

            async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
                return self

            async def aclose(self):
                trace.append("wire-close-start")
                wire_closing.set()
                try:
                    await release_close.wait()
                except asyncio.CancelledError:
                    trace.append("wire-close-cancelled")
                    raise
                if close_mode == "failure":
                    trace.append("wire-close-failed")
                    raise httpcore.ReadError("fixture close failure")
                trace.append("wire-closed")

        class Backend(httpcore.AsyncNetworkBackend):
            async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
                return Wire()

        class Transport(httpx.AsyncHTTPTransport):
            async def __aexit__(self, *args):
                await super().__aexit__(*args)
                trace.append("client-closed")

        class Voice:
            async def deliver(self, reply):
                nonlocal stop_at
                reply.select_voice_delivery()
                await anext(reply)
                await (wire_closing if eof_first else read_waiting).wait()
                trace.append("voice-zero")
                stop_at = loop.time()
                return Receipt(reply.info.reply_id, DeliveryStatus.ZERO)

        pool = httpcore.AsyncConnectionPool
        with patch.object(httpcore, "AsyncConnectionPool", side_effect=lambda **kw: pool(network_backend=Backend(), **kw)):
            transport = Transport(trust_env=False)
        actual_client = httpx.AsyncClient
        clients = []

        def client(**kwargs):
            owned = transport if not clients else httpx.MockTransport(lambda _: httpx.Response(500))
            result = actual_client(transport=owned, **kwargs)
            clients.append(result)
            return result

        @asynccontextmanager
        async def prepare(settings, *, discovery, roles):
            async with prepare_private_feishu(settings, discovery=discovery, roles=roles) as (history, _):
                async def run(model, asr, tts):
                    async with running_agent(settings, history, model, Voice(), session_id="transport",
                                             roles=roles) as (runtime, _):
                        runtime.submit_batch([Message("user", "fixture input")])
                        await asyncio.wait_for(runtime.wait_idle(), 2)
                        outcome, = history.outcomes("transport")
                        self.assertEqual((outcome.model_status, outcome.delivery_status),
                                         ("model_interrupted", DeliveryStatus.ZERO))
                        self.assertNotIn("wire-closed", trace)
                    trace.append("runtime-stopped")
                    runtime_stopped.set()

                yield history, run

        with TemporaryDirectory(dir=Path.cwd()) as folder:
            config = Path(folder) / "native.toml"
            config.write_text('database = "fixture.sqlite"\n' + _native_config(adapter), encoding="utf-8")
            settings = load_settings(config, for_run=True)
            with (patch("agent_companion.__main__.httpx.AsyncClient", side_effect=client),
                  patch("agent_companion.__main__.prepare_private_feishu", side_effect=prepare),
                  self.assertLogs("agent_companion", level="WARNING") as logged):
                discovery = private_discovery("oc_fixture", "ou_peer", "ou_self")
                serving = asyncio.create_task(_serve(settings, discovery=discovery, roles=demo_roles()))
                initial = (["body-eof", "wire-close-start", "voice-zero"] if eof_first else
                           ["voice-zero", "body-read-cancelled", "wire-close-start"])
                try:
                    await asyncio.wait_for(runtime_stopped.wait(), 4)
                    self.assertFalse(serving.done())
                    self.assertEqual(trace, [*initial, "runtime-stopped"])
                    if close_mode == "timeout":
                        await asyncio.wait_for(asyncio.shield(serving), 6)
                        self.assertGreaterEqual(loop.time() - stop_at, 4.9)
                    else:
                        release_close.set()
                finally:
                    release_close.set()
                    await asyncio.wait_for(serving, 4)
            self.assertTrue(any("still running after shutdown grace" in row for row in logged.output))
            if close_mode == "timeout":
                self.assertTrue(any("transport_close_timeout" in row for row in logged.output))
                terminal = "wire-close-cancelled"
            elif close_mode == "failure":
                self.assertTrue(any("model cleanup failed (transport_error)" in row for row in logged.output))
                terminal = "wire-close-failed"
            else:
                terminal = "wire-closed"
            self.assertEqual(trace, [*initial, "runtime-stopped", terminal, "client-closed"])
            self.assertTrue(all(owned.is_closed for owned in clients))

    async def test_native_eof_close_outlives_runtime_grace_and_precedes_client_close(self):
        actual_client = httpx.AsyncClient
        for adapter in ("deepseek", "openai_responses"):
            with self.subTest(adapter=adapter), TemporaryDirectory(dir=Path.cwd()) as folder:
                trace = []
                wire_closing, release_close, runtime_stopped = (asyncio.Event() for _ in range(3))
                if adapter == "deepseek":
                    frame = {"choices": [{"index": 0, "delta": {"content": "confirmed tail"}}]}
                else:
                    frame = {"type": "response.output_text.delta", "output_index": 0, "content_index": 0,
                             "delta": "confirmed tail"}

                class Wire(httpx.AsyncByteStream):
                    async def __aiter__(self):
                        # 正文后直接 EOF：没有 [DONE] 或 response.completed。
                        yield f"data: {json.dumps(frame)}\n\n".encode()

                    async def aclose(self):
                        trace.append("response-close-start")
                        wire_closing.set()
                        try:
                            await release_close.wait()
                        except asyncio.CancelledError:
                            trace.append("response-close-cancelled")
                            raise
                        trace.append("response-closed")

                class Transport(httpx.AsyncBaseTransport):
                    async def handle_async_request(self, request):
                        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=Wire())

                    async def aclose(self):
                        trace.append("model-client-closed")

                class Voice:
                    async def deliver(self, reply):
                        reply.select_voice_delivery()
                        await anext(reply)
                        # 等到 HTTPX 已在 EOF 内置关闭中，才触发语音打断。
                        await wire_closing.wait()
                        return Receipt(reply.info.reply_id, DeliveryStatus.PARTIAL, "confirmed")

                clients = []

                def client(**kwargs):
                    transport = Transport() if not clients else httpx.MockTransport(lambda _: httpx.Response(500))
                    result = actual_client(transport=transport, **kwargs)
                    clients.append(result)
                    return result

                @asynccontextmanager
                async def prepare(settings, *, discovery, roles):
                    async with prepare_private_feishu(settings, discovery=discovery, roles=roles) as (history, _):
                        async def run(model, asr, tts):
                            async with running_agent(settings, history, model, Voice(), session_id="eof",
                                                     roles=roles) as (runtime, _):
                                runtime.submit_batch([Message("user", "prompt")])
                                await asyncio.wait_for(runtime.wait_idle(), 2)
                                outcome, = history.outcomes("eof")
                                self.assertEqual((outcome.model_status, outcome.delivery_status),
                                                 ("model_interrupted", DeliveryStatus.PARTIAL))
                            trace.append("runtime-stopped")
                            runtime_stopped.set()

                        yield history, run

                config = Path(folder) / "native.toml"
                config.write_text('database = "fixture.sqlite"\n' + _native_config(adapter), encoding="utf-8")
                settings = load_settings(config, for_run=True)
                with (patch("agent_companion.__main__.httpx.AsyncClient", side_effect=client),
                      patch("agent_companion.__main__.prepare_private_feishu", side_effect=prepare),
                      self.assertLogs("agent_companion.runtime.engine", level="WARNING") as logged):
                    discovery = private_discovery("oc_fixture", "ou_peer", "ou_self")
                    serving = asyncio.create_task(_serve(settings, discovery=discovery, roles=demo_roles()))
                    try:
                        # 真实的旧 1 秒 Runtime 宽限已耗尽，原生关闭和客户端必须还活着。
                        await asyncio.wait_for(runtime_stopped.wait(), 4)
                        self.assertFalse(serving.done())
                        self.assertEqual(trace, ["response-close-start", "runtime-stopped"])
                    finally:
                        release_close.set()
                        await asyncio.wait_for(serving, 4)
                self.assertTrue(any("still running after shutdown grace" in row for row in logged.output))
                self.assertEqual(trace, ["response-close-start", "runtime-stopped", "response-closed",
                                         "model-client-closed"])
                self.assertTrue(all(owned.is_closed for owned in clients))

    async def test_authentication_region_model_and_sampling_in_text_and_voice(self):
        buf = BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(bytes(320))
        audio = AudioFile(buf.getvalue())
        actual_client = httpx.AsyncClient
        asr_model = "未来/ASR + 预览"

        for mode in ("text", "shared", "separate"):
            with self.subTest(mode=mode), TemporaryDirectory(dir=Path.cwd()) as folder:
                text = _CONFIG if mode == "text" else _CONFIG.replace('mode = "text"', 'mode = "voice"')
                text = text.replace('max_audio_seconds = 500',
                                    f'max_audio_seconds = 600.5\nmodel = "{asr_model}"')
                text += '\n[timeouts]\nllm = 0.75\nasr = 2\ntts = 3\n'
                text = text.replace('[voice]', '[voice]\ninput_device = "mock-in"\n'
                                    'output_device = "mock-out"\nvad_model = "synthetic.onnx"')
                if mode == "separate":
                    text = text.replace('tts_provider = "speech"', 'tts_provider = "speaker"') + '''
[providers.speaker]
adapter = "minimax"
api_key = "synthetic-separate-tts-value"
region = "global"
[providers.speaker.tts]
model = "synthetic-other-model"
voice_id = "synthetic-other-voice"
'''
                config = Path(folder) / "synthetic.toml"
                config.write_text(text, encoding="utf-8")
                settings = load_settings(config, for_run=True)
                requests = []
                clients = []

                async def respond(request):
                    requests.append(request)
                    if str(request.url) == CHAT_COMPLETIONS:
                        return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"OK"},'
                                                        '"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
                    if request.url.path == "/v1/speech_to_text":
                        return httpx.Response(200, json={"text": "合成识别文本"})
                    if request.url.path == "/v1/t2a_v2":
                        return httpx.Response(200, stream=httpx.ByteStream(
                            b'data: {"data":{"status":1,"audio":"0100"}}\n\ndata: {"data":{"status":2}}\n\n'))
                    self.fail("unexpected URL")

                def client(**options):
                    self.assertFalse(options["follow_redirects"])
                    self.assertFalse(options["trust_env"])
                    handle = actual_client(transport=httpx.MockTransport(respond), **options)
                    clients.append(handle)
                    return handle

                async def drive(model, asr, tts):
                    self.assertEqual(settings.runtime.model_timeout, 0.75)
                    self.assertEqual(asr._timeout, 2)
                    self.assertEqual(settings.asr.max_audio_seconds, asr._max_audio_seconds)
                    self.assertEqual(settings.asr.max_audio_seconds, 600.5)
                    _ = [event async for event in model.stream([Message("user", "合成输入")], (), thinking=False)]
                    self.assertEqual(await asr.transcribe(audio), "合成识别文本")
                    if mode == "text":
                        self.assertIsNone(tts)
                    else:
                        chunks = tts.synthesize("合成输出", sample_rate=settings.voice.output_sample_rate)
                        self.assertEqual([chunk async for chunk in chunks], [b"\x01\x00"])

                @asynccontextmanager
                async def prepare(settings, *, discovery, roles):
                    async with prepare_private_feishu(settings, discovery=discovery, roles=roles) as (history, run):
                        async def exercise(model, asr, tts):
                            await drive(model, asr, tts)
                            await run(model, asr, tts)

                        yield history, exercise

                with (patch("agent_companion.__main__.httpx.AsyncClient", side_effect=client),
                      patch("agent_companion.speech.minimax_tts.MiniMaxTTS", wraps=MiniMaxTTS) as open_tts,
                      patch("agent_companion.__main__.prepare_private_feishu", side_effect=prepare),
                      patch("agent_companion.channels.feishu.macos_calls.MacCallDesktop") as desktop,
                      patch("agent_companion.app.run_private_chats", new_callable=AsyncMock) as run_chats):
                    await _serve(settings, discovery=private_discovery("oc_fixture", "ou_peer", "ou_self"),
                                 roles=demo_roles())
                run_chats.assert_awaited_once()
                self.assertEqual(desktop.call_count, int(mode != "text"))
                self.assertEqual(run_chats.await_args.kwargs["incoming"] is not None, mode != "text")
                self.assertEqual(open_tts.call_count, int(mode != "text"))
                self.assertEqual(len(clients), 3 if mode == "separate" else 2)
                self.assertTrue(all(c.is_closed for c in clients))
                self.assertEqual(len(requests), 2 if mode == "text" else 3)
                model, asr, *rest = requests
                self.assertTrue(model.headers["Authorization"] == f"Bearer {_KEY}")
                self.assertEqual(json.loads(model.content)["model"], "deepseek-flash")
                self.assertEqual(str(asr.url), "https://api.minimax.cn/v1/speech_to_text")
                self.assertTrue(asr.headers["Authorization"] == "Bearer synthetic-speech-value")
                self.assertIn(f'name="model"\r\n\r\n{asr_model}\r\n'.encode(), asr.content)
                if rest:
                    tts, = rest
                    body = json.loads(tts.content)
                    self.assertEqual(body["audio_setting"]["sample_rate"], 24000)
                    self.assertEqual(tts.extensions["timeout"]["read"], 3)
                    self.assertEqual(tts.extensions["timeout"]["write"], 3)
                    if mode == "shared":
                        self.assertEqual(str(tts.url), "https://api.minimax.cn/v1/t2a_v2")
                        self.assertTrue(tts.headers["Authorization"] == "Bearer synthetic-speech-value")
                        self.assertEqual(body["model"], "synthetic-tts-model")
                        self.assertEqual(body["voice_setting"]["voice_id"], "synthetic-shared-voice")
                    else:
                        self.assertEqual(str(tts.url), "https://api.minimax.io/v1/t2a_v2")
                        self.assertTrue(tts.headers["Authorization"] == "Bearer synthetic-separate-tts-value")
                        self.assertEqual(body["model"], "synthetic-other-model")
                        self.assertEqual(body["voice_setting"]["voice_id"], "synthetic-other-voice")
