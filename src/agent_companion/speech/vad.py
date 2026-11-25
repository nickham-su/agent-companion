"""Silero 16 kHz ONNX VAD；依赖和文件只在本地 voice 组装时加载。"""

from math import ceil
from pathlib import Path
from typing import Protocol

INPUT_RATE = 16_000
INPUT_FRAMES = 512
SAMPLE_BYTES = 2
INPUT_BYTES = INPUT_FRAMES * SAMPLE_BYTES


def silence_frames(seconds: float, *, strict: bool = False) -> int:
    """将静默时长向上取整为完整的 VAD 输入帧数。"""
    frames = ceil(seconds / (INPUT_FRAMES / INPUT_RATE))
    if strict:
        # 用整数样本数生成整帧边界，校正浮点商在边界处可能吞掉或多算的一帧。
        if seconds > frames * INPUT_FRAMES / INPUT_RATE:
            return frames + 1
        if frames > 1 and seconds <= (frames - 1) * INPUT_FRAMES / INPUT_RATE:
            return frames - 1
    return frames


class VoiceSetupError(Exception):
    """可安全输出的固定启动诊断；不包含设备名、路径或录音。"""


class VoiceDetector(Protocol):
    def probability(self, pcm: bytes) -> float: ...
    def reset(self) -> None: ...


class SileroVAD:
    """适配 Silero v6 silero_vad.onnx，连续窗口保留模型状态与 64 样本前文。"""

    def __init__(self, path: Path) -> None:
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError:
            raise VoiceSetupError("voice_onnx_dependencies_missing") from None
        if not path.is_file():
            raise VoiceSetupError("voice_vad_model_missing")
        self._np = np
        try:
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            self._session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
            self._rate = np.array(INPUT_RATE, dtype=np.int64)
            self.reset()
            self.probability(bytes(INPUT_BYTES))  # 打开设备前验证模型格式、输入名与形状
            self.reset()
        except Exception:
            raise VoiceSetupError("voice_vad_model_incompatible") from None

    def reset(self) -> None:
        self._state = self._np.zeros((2, 1, 128), dtype=self._np.float32)
        self._context = self._np.zeros((1, 64), dtype=self._np.float32)

    def probability(self, pcm: bytes) -> float:
        if len(pcm) != INPUT_BYTES:
            raise ValueError("invalid VAD window")
        samples = self._np.frombuffer(pcm, dtype="<i2").astype(self._np.float32) / 32768.0
        samples = self._np.concatenate((self._context, samples.reshape(1, -1)), axis=1)
        probability, self._state = self._session.run(None, {"input": samples, "state": self._state, "sr": self._rate})
        self._context = samples[:, -64:]
        return float(probability.item())
