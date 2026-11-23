"""与 IM 无关的双向 PCM 设备契约；格式在启动时声明，不随帧重复传递。"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class AudioFormat:
    sample_rate: int
    channels: int = 1
    sample_format: str = "s16le"


class AudioDevice(Protocol):
    @property
    def input_format(self) -> AudioFormat: ...
    @property
    def output_format(self) -> AudioFormat: ...
    def start(self, capture: Callable[[memoryview, bool], None], render: Callable[[memoryview], None]) -> None: ...
    def check_active(self) -> None: ...
    def close(self) -> None: ...
