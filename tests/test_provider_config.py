from io import BytesIO, StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import traceback
import unittest
from unittest.mock import AsyncMock, patch
import wave

import httpx
from feishu_test_support import fixed_discovery

from agent_companion.__main__ import _serve, main
from agent_companion.config import ConfigError, VoiceSettings, load_settings
from agent_companion.models.deepseek import CHAT_COMPLETIONS
from agent_companion.protocol import Message
from agent_companion.speech.asr import AudioFile
from agent_companion.speech.minimax_tts import MiniMaxTTS


_KEY = "synthetic-private-config-value"
_FEISHU = '''
[feishu]
peer_open_id = "ou_peer"
own_open_id = "ou_self"
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
            (_CONFIG.replace('peer_open_id = "ou_peer"', f'peer_open_id = "{_KEY}"'),
             'feishu.peer_open_id 必须是以 ou_ 开头的有效飞书 ID'),
            (_CONFIG.replace('[feishu]', '[feishu]\nchat_id = "oc_fixture"'), 'feishu 配置字段无效'),
            (_CONFIG.replace('peer_open_id = "ou_peer"', 'sender_open_id = "ou_peer"'), 'feishu 配置字段无效'),
            (_CONFIG.replace('peer_open_id = "ou_peer"', '').replace('mode = "text"', 'mode = "voice"'),
             '飞书 voice 模式必须设置 feishu.peer_open_id'),
            (_CONFIG.replace('own_open_id = "ou_self"', ''), 'feishu.own_open_id'),
            (_CONFIG.replace('"ou_self"', '"ou_peer"'), 'feishu.peer_open_id 与 feishu.own_open_id 必须不同'),
            (_CONFIG.replace('mode = "text"', f'mode = "{_KEY}"'), 'feishu.mode 必须是 text 或 voice'),
            (_CONFIG.replace('mode = "text"', 'mode = "text"\ncli_timeout = false'), 'feishu.cli_timeout 必须是有限正数'),
            (_CONFIG + '\n[runtime]\ndelivery_timeout = 120\n', 'runtime 配置字段无效'),
            (_CONFIG + '\n[runtime]\ncontext_budget = 42\n',
             'runtime.tail_budget 必须小于 runtime.context_budget'),
            (_CONFIG + '\n[runtime]\nsummary_budget = 4000\n', 'runtime 配置字段无效'),
            (_CONFIG + f'\n[runtime]\n{_KEY} = 42\n', 'runtime 配置字段无效'),
            ('database = false\n' + _CONFIG, 'database 必须是非空且不含 NUL 字符的路径'),
            ('database = ":memory:"\n' + _CONFIG, '正式启动的 database 必须是持久化 SQLite 文件路径'),
            ('main_session_id = false\n' + _CONFIG, 'main_session_id 必须是非空字符串'),
        ):
            with self.subTest(diagnostic=diagnostic):
                self.assert_startup_error(config, diagnostic)

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
            self.assertIn("离线模拟：主会话已收到子任务回投。", stdout.getvalue())
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

                async def drive(settings, history, model, asr, route, *, tts):
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

                with (patch("agent_companion.__main__.httpx.AsyncClient", side_effect=client),
                      patch("agent_companion.__main__.MiniMaxTTS", wraps=MiniMaxTTS) as open_tts,
                      patch("agent_companion.__main__.run_fixed_feishu", side_effect=drive) as run):
                    await _serve(settings, discovery=fixed_discovery("oc_fixture", "ou_peer", "ou_self"))
                run.assert_awaited_once()
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
