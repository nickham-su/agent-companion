"""macOS CoreAudio 设备绑定；使用 PortAudio 的设备格式转换，不重复重采样。"""

from collections.abc import Callable
import sys

from ...speech.audio import AudioFormat
from ...speech.vad import INPUT_FRAMES, INPUT_RATE, VoiceSetupError


class MacAudioDevice:
    def __init__(self, input_device: str | int, output_device: str | int, playback_rate: int) -> None:
        self._input_device = input_device
        self._output_device = output_device
        self._playback_rate = playback_rate
        self._streams = []

    @property
    def input_format(self) -> AudioFormat:
        return AudioFormat(INPUT_RATE)

    @property
    def output_format(self) -> AudioFormat:
        return AudioFormat(self._playback_rate)

    @staticmethod
    def _resolve(sd, selection: str | int, direction: str) -> int:
        devices = sd.query_devices()
        apis = sd.query_hostapis()
        found = [index for index, device in enumerate(devices)
                 if (selection == index or selection == device["name"])
                 and apis[device["hostapi"]]["name"] == "Core Audio"
                 and device[f"max_{direction}_channels"] >= 1]
        if len(found) != 1:
            raise VoiceSetupError(f"voice_{direction}_device_missing_or_ambiguous")
        return found[0]

    def start(self, capture: Callable[[memoryview, bool], None], render: Callable[[memoryview], None]) -> None:
        if sys.platform != "darwin":
            raise VoiceSetupError("voice_requires_macos_coreaudio")
        try:
            import sounddevice as sd
        except (ImportError, OSError):
            raise VoiceSetupError("voice_portaudio_dependency_unavailable") from None
        try:
            input_device = self._resolve(sd, self._input_device, "input")
            output_device = self._resolve(sd, self._output_device, "output")
        except (sd.PortAudioError, ValueError):
            raise VoiceSetupError("voice_coreaudio_device_query_failed") from None
        if input_device == output_device:
            raise VoiceSetupError("voice_requires_two_distinct_devices")

        # 请求应用所需的单声道格式，CoreAudio 在支持时直接转换到设备的物理格式。
        # 不改变设备的全局采样率；既不在 Python 重采样，也不将 TTS 转两次。
        settings = sd.CoreAudioSettings(change_device_parameters=False, fail_if_conversion_required=False)
        input_format = dict(device=input_device, samplerate=self.input_format.sample_rate,
                            channels=self.input_format.channels, dtype="int16",
                            extra_settings=settings)
        output_format = dict(device=output_device, samplerate=self.output_format.sample_rate,
                             channels=self.output_format.channels, dtype="int16",
                             extra_settings=settings)
        try:
            sd.check_input_settings(**input_format)
        except (sd.PortAudioError, ValueError):
            raise VoiceSetupError("voice_input_pcm_16khz_mono_not_supported") from None
        try:
            sd.check_output_settings(**output_format)
        except (sd.PortAudioError, ValueError):
            raise VoiceSetupError("voice_output_pcm_rate_or_channels_not_supported") from None

        def input_callback(indata, frames, time, status) -> None:
            capture(memoryview(indata), bool(status.input_overflow))

        def output_callback(outdata, frames, time, status) -> None:
            render(memoryview(outdata))

        try:
            self._streams.append(sd.RawOutputStream(**output_format, blocksize=round(self._playback_rate / 50),
                                                     callback=output_callback))
            self._streams.append(sd.RawInputStream(**input_format, blocksize=INPUT_FRAMES, callback=input_callback))
            for stream in self._streams:
                stream.start()
        except (sd.PortAudioError, ValueError):
            self.close()
            raise VoiceSetupError("voice_coreaudio_stream_open_failed") from None

    def check_active(self) -> None:
        try:
            if len(self._streams) == 2 and all(stream.active for stream in self._streams):
                return
        except Exception:
            pass
        raise VoiceSetupError("voice_coreaudio_stream_stopped")

    def close(self) -> None:
        while self._streams:
            stream = self._streams.pop()
            try:
                stream.abort()
            except Exception:
                pass
            finally:
                try:
                    stream.close()
                except Exception:
                    pass
