"""用最小数组和 ONNX Session 替身验证 Silero 适配，不安装音频依赖。"""

from math import prod
from pathlib import Path
import struct
import subprocess
import sys
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent_companion.speech.vad import SileroVAD


class _Array:
    def __init__(self, values, shape, dtype):
        self.values, self.shape, self.dtype = tuple(values), shape, dtype
        assert prod(shape) == len(self.values)

    def astype(self, dtype):
        return _Array(map(float, self.values), self.shape, dtype)

    def __truediv__(self, value):
        return _Array((item / value for item in self.values), self.shape, self.dtype)

    def reshape(self, rows, columns):
        assert (rows, columns) == (1, -1)
        return _Array(self.values, (1, len(self.values)), self.dtype)

    def __getitem__(self, idx):
        rows, columns = idx
        assert rows == slice(None) and self.shape[0] == 1
        values = self.values[columns]
        return _Array(values, (1, len(values)), self.dtype)

    def item(self):
        (value,) = self.values
        return value


def _zeros(shape, dtype):
    return _Array((0,) * prod(shape), shape, dtype)


def _frombuffer(pcm, dtype):
    assert dtype == "<i2"
    values = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    return _Array(values, (len(values),), dtype)


def _concatenate(parts, axis):
    assert axis == 1 and all(part.shape[0] == 1 and part.dtype == "float32" for part in parts)
    values = tuple(value for part in parts for value in part.values)
    return _Array(values, (1, len(values)), "float32")


class SileroVADTest(unittest.TestCase):
    def test_session_contract_context_state_and_reset(self):
        requests, next_states = [], []
        np = SimpleNamespace(float32="float32", int64="int64", zeros=_zeros, frombuffer=_frombuffer,
                             concatenate=_concatenate, array=lambda value, dtype: _Array([value], (), dtype))

        def infer(outputs, inputs):
            self.assertIsNone(outputs)
            self.assertEqual(inputs.keys(), {"input", "state", "sr"})
            self.assertEqual((inputs["input"].shape, inputs["input"].dtype), ((1, 576), "float32"))
            self.assertEqual((inputs["state"].shape, inputs["state"].dtype), ((2, 1, 128), "float32"))
            self.assertEqual((inputs["sr"].shape, inputs["sr"].dtype, inputs["sr"].item()), ((), "int64", 16000))
            requests.append(inputs)
            updated = _Array((len(requests),) * 256, (2, 1, 128), "float32")
            next_states.append(updated)
            return _Array([0.75], (1, 1), "float32"), updated

        def session(path, *, sess_options, providers):
            self.assertEqual(path, "synthetic-model.onnx")
            self.assertEqual(providers, ["CPUExecutionProvider"])
            self.assertEqual((sess_options.inter_op_num_threads, sess_options.intra_op_num_threads), (1, 1))
            return SimpleNamespace(run=infer)

        ort = SimpleNamespace(SessionOptions=SimpleNamespace, InferenceSession=session)
        first = struct.pack("<512h", *([16384] * 448 + [-16384] * 64))
        with patch.dict(sys.modules, {"numpy": np, "onnxruntime": ort}), patch.object(Path, "is_file", return_value=True):
            vad = SileroVAD(Path("synthetic-model.onnx"))
            self.assertEqual(requests[0]["input"].values, (0,) * 576)
            self.assertEqual(vad.probability(first), 0.75)
            self.assertEqual(requests[1]["input"].values, (0,) * 64 + (0.5,) * 448 + (-0.5,) * 64)
            self.assertEqual(requests[1]["state"].values, (0,) * 256)  # 启动探测的状态不会泄漏
            vad.probability(bytes(1024))
            self.assertEqual(requests[2]["input"].values, (-0.5,) * 64 + (0,) * 512)
            self.assertIs(requests[2]["state"], next_states[1])
            vad.reset()
            vad.probability(bytes(1024))
            self.assertEqual(requests[3]["input"].values, (0,) * 576)
            self.assertEqual(requests[3]["state"].values, (0,) * 256)

    def test_text_assembly_can_run_with_audio_imports_disabled(self):
        # 新解释器可检查入口和组装阶段的导入，而不是只检查已经导入模块后的路径。
        code = textwrap.dedent('''
            import asyncio
            from contextlib import closing, suppress
            import sys
            from unittest.mock import patch

            unavailable = ("sounddevice", "numpy", "onnxruntime", "agent_companion.speech.voice",
                           "agent_companion.speech.minimax_tts", "agent_companion.channels.audio.macos",
                           "agent_companion.channels.feishu.macos_calls")
            sys.modules.update(dict.fromkeys(unavailable))
            from agent_companion import __main__
            from agent_companion.app import run_feishu
            from agent_companion.channels.feishu.channel import FeishuPoller
            from agent_companion.config import FeishuSettings, Settings
            from agent_companion.runtime.subtasks import demo_roles
            from agent_companion.storage.sqlite import History

            sys.path.insert(0, "tests")
            from feishu_test_support import private_discovery

            async def run(settings):
                started = asyncio.Event()

                async def poll(poller):
                    started.set()
                    await asyncio.Future()

                discovery = private_discovery("oc_fake", "ou_peer", "ou_self")
                with closing(History(":memory:")) as history, patch.object(FeishuPoller, "run", poll):
                    task = asyncio.create_task(run_feishu(settings, history, object(), object(),
                                                         discovery=discovery, roles=demo_roles()))
                    try:
                        await asyncio.wait_for(started.wait(), 1)
                        assert not task.done()
                    finally:
                        task.cancel()
                        with suppress(asyncio.CancelledError):
                            await task

            for mode in ({}, {"mode": "text"}):
                asyncio.run(run(Settings(feishu=FeishuSettings(**mode))))
        ''')
        subprocess.run([sys.executable, "-c", code], check=True, timeout=5, capture_output=True)
