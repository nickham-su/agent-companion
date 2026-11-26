from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx

from agent_companion.__main__ import _serve
from agent_companion.config import ConfigError, load_settings
from agent_companion.models.bailian import Bailian
from agent_companion.models.minimax import MiniMax
from agent_companion.runtime.subtasks import AgentRole
from agent_companion.speech.bailian_asr import BailianASR
from agent_companion.speech.bailian_tts import BailianTTS
from agent_companion.speech.minimax import MiniMaxASR
from agent_companion.speech.minimax_tts import MiniMaxTTS


_CONFIG = '''asr_model = "mic"
tts_model = "speaker"
[feishu]
mode = "voice"
[providers.speech]
adapter = "minimax"
api_key = "fixture-speech"
[models.asr.mic]
provider = "speech"
max_audio_seconds = 100
[models.tts.speaker]
provider = "speech"
model = "speech-2.8-turbo"
voice_id = "male-qn-qingse"
[providers.other]
adapter = "bailian"
api_key = "fixture-other"
region = "ap-southeast-1"
workspace_id = "fixture-space"
[models.llm.first]
provider = "speech"
model = "MiniMax-M3.1-Flash"
reasoning_effort = "disabled"
context_budget = 64000
tail_budget = 16000
[models.llm.second]
provider = "other"
model = "qwen4-max"
reasoning_effort = "enabled"
preserve_thinking = true
context_budget = 64000
tail_budget = 16000
'''


class ChatConfigTests(unittest.IsolatedAsyncioTestCase):
    def settings(self, text):
        with TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "fixture.toml"
            path.write_text(text, encoding="utf-8")
            return load_settings(path, for_run=True)

    async def test_role_bindings_keep_chat_clients_separate_from_speech_even_when_provider_is_shared(self):
        class History:
            @staticmethod
            def replay_salt():
                return b"fixture-salt"

        for shared in ("minimax", "bailian"):
            with self.subTest(shared=shared):
                if shared == "minimax":
                    config = _CONFIG.replace('api_key = "fixture-speech"',
                                             'api_key = "fixture-speech"\nbase_url = "https://api.minimax.io/v1/"')
                else:
                    config = (_CONFIG.replace('asr_model = "mic"', 'asr_model = "bailian_mic"')
                              .replace('tts_model = "speaker"', 'tts_model = "bailian_speaker"') + '''
[models.asr.bailian_mic]
provider = "other"
max_audio_seconds = 100
[models.tts.bailian_speaker]
provider = "other"
''')
                settings = self.settings(config)
                self.assertIs(settings.asr.connection, settings.tts.connection)
                roles = {"main": AgentRole(None, model_id="first"), "child": AgentRole(None, model_id="second")}
                actual_client = httpx.AsyncClient
                clients = []

                def unused(_):
                    self.fail("offline binding tests cannot send requests")

                def client(**kwargs):
                    self.assertFalse(kwargs["follow_redirects"])
                    result = actual_client(transport=httpx.MockTransport(unused), **kwargs)
                    clients.append((result, kwargs))
                    return result

                async def run(bindings, asr, tts):
                    first, second = bindings["main"], bindings["child"]
                    self.assertIsInstance(first.model, MiniMax)
                    self.assertFalse(first.thinking)  # 不是本地拒绝，M3.1-Flash 也能配置关闭。
                    self.assertIsInstance(second.model, Bailian)
                    self.assertTrue(second.thinking)
                    self.assertEqual(first.model.endpoint, f"https://api.minimax.{'io' if shared == 'minimax' else 'cn'}"
                                                           "/v1/chat/completions")
                    self.assertEqual(second.model.endpoint, "https://fixture-space.ap-southeast-1.maas.aliyuncs.com"
                                                            "/compatible-mode/v1/chat/completions")
                    self.assertIsNot(first.model._client, second.model._client)
                    for owned, options in clients[:2]:
                        self.assertIsNone(options["headers"])
                        self.assertEqual(options["limits"].keepalive_expiry, 300)
                        self.assertIsNone(owned.headers.get("Authorization"))
                    if shared == "minimax":
                        self.assertIsInstance(asr, MiniMaxASR)
                        self.assertIsInstance(tts, MiniMaxTTS)
                        self.assertIs(asr._client, tts._client)
                        self.assertIsNot(asr._client, first.model._client)
                        self.assertIn("api.minimax.cn", asr._endpoint)  # base_url 只影响 LLM。
                        self.assertEqual(len(clients), 3)
                    else:
                        self.assertIsInstance(asr, BailianASR)
                        self.assertIsInstance(tts, BailianTTS)
                        self.assertEqual(len(clients), 2)  # 百炼语音独立走 WebSocket，而非模型连接池。

                @asynccontextmanager
                async def prepare(*args, **kwargs):
                    yield History(), run

                with (patch("agent_companion.__main__.prepare_private_feishu", prepare),
                      patch("agent_companion.__main__.httpx.AsyncClient", client)):
                    await _serve(settings, roles=roles)
                self.assertTrue(all(owned.is_closed for owned, _ in clients))

    def test_unchosen_speech_providers_are_not_validated_as_llm(self):
        config = _CONFIG.replace('[models.llm.first]\nprovider = "speech"',
                                 '[models.llm.first]\nprovider = "separate"')
        config += '''
[providers.separate]
adapter = "minimax"
api_key = "fixture-separate"
[providers.unused_bailian]
adapter = "bailian"
[models.tts.unused]
provider = "unused_bailian"
model = false
'''
        settings = self.settings(config)
        self.assertEqual(set(settings.llm_providers), {"separate", "other"})
        self.assertEqual(settings.asr.connection.region, "cn")

    def test_model_namespace_is_only_supported_for_bailian_with_valid_segments(self):
        for model in ("kimi/kimi-k3", "owner/collection/model:v2"):
            with self.subTest(model=model):
                config = _CONFIG.replace('model = "qwen4-max"', f'model = "{model}"')
                self.assertEqual(self.settings(config).models["second"].model, model)
        invalid = ("/kimi-k3", "kimi/", "kimi//kimi-k3", "../kimi-k3", "kimi/../other", "kimi/./other", "..",
                   "owner/%2e%2e/model", "owner/model?query", "owner/model ", "owner/" + "x" * 123)
        for model in invalid:
            with self.subTest(model=model), self.assertRaisesRegex(ConfigError, "model 格式无效"):
                self.settings(_CONFIG.replace('model = "qwen4-max"', f'model = "{model}"'))
        for adapter in ("minimax", "deepseek", "openai_responses"):
            config = _CONFIG.replace('model = "MiniMax-M3.1-Flash"', 'model = "kimi/kimi-k3"')
            if adapter != "minimax":
                config = (config.replace('[models.llm.first]\nprovider = "speech"',
                                         '[models.llm.first]\nprovider = "native"')
                          + f'\n[providers.native]\nadapter = "{adapter}"\n')
            with self.subTest(adapter=adapter), self.assertRaisesRegex(ConfigError, "model 格式无效"):
                self.settings(config)

    def test_new_llm_fields_are_checked_without_model_version_rules(self):
        for start, replacement, error in ((
                'workspace_id = "fixture-space"', 'workspace_id = "invalid.example"', 'workspace_id'), (
                'workspace_id = "fixture-space"', '', 'workspace_id'), (
                'region = "ap-southeast-1"', 'region = "region.with.dot"', 'region'), (
                'preserve_thinking = true', 'preserve_thinking = "true"', 'preserve_thinking'), (
                'model = "MiniMax-M3.1-Flash"', 'model = "different-model"\npreserve_thinking = false',
                'preserve_thinking 仅适用'), (
                'workspace_id = "fixture-space"', 'workspace_id = "fixture-space"\nbase_url = "https://example.test"',
                'base_url')):
            with self.subTest(error=error), self.assertRaisesRegex(ConfigError, error):
                self.settings(_CONFIG.replace(start, replacement))
        config = _CONFIG.replace('preserve_thinking = true', 'preserve_thinking = false')
        self.assertIs(self.settings(config).models["second"].preserve_thinking, False)
        self.assertIsNone(self.settings(config.replace('preserve_thinking = false', '')).models["second"].preserve_thinking)
