import argparse
import sys
import unittest
from unittest.mock import patch

from agent_companion.channels.audio.macos import MacAudioDevice
from agent_companion.speech.vad import VoiceSetupError


class _PortAudio:
    class PortAudioError(Exception):
        pass

    def __init__(self):
        self.checked = []
        self.opened = []
        self.conflict = None

    def query_devices(self):
        return [dict(name=name, hostapi=0, max_input_channels=2, max_output_channels=2)
                for name in ("Playback from Feishu", "Mic to Feishu")]

    def query_hostapis(self):
        return [{"name": "Core Audio"}]

    CoreAudioSettings = argparse.Namespace

    def check_input_settings(self, **kwargs):
        self.checked.append(("input", kwargs))
        if self.conflict == "input":
            raise self.PortAudioError

    def check_output_settings(self, **kwargs):
        self.checked.append(("output", kwargs))
        if self.conflict == "output":
            raise self.PortAudioError

    def _stream(self, **kwargs):
        stream = argparse.Namespace(active=False, closed=False, **kwargs)
        stream.start = lambda: setattr(stream, "active", True)
        stream.abort = lambda: setattr(stream, "active", False)
        stream.close = lambda: setattr(stream, "closed", True)
        self.opened.append(stream)
        return stream

    RawInputStream = RawOutputStream = _stream


class MacBindingTest(unittest.TestCase):
    def test_preflight_formats_then_forward_pcm_and_cleanup_with_fake_portaudio(self):
        pa = _PortAudio()
        with patch.dict(sys.modules, sounddevice=pa), patch.object(sys, "platform", "darwin"):
            device = MacAudioDevice("Playback from Feishu", 1, 44100)
            captured = []
            device.start(lambda pcm, overflow: captured.append((bytes(pcm), overflow)),
                         lambda pcm: pcm.__setitem__(slice(None), b"\3\0" * (len(pcm) // 2)))
            device.check_active()
            self.assertEqual([(direction, args["device"], args["samplerate"], args["channels"], args["dtype"])
                              for direction, args in pa.checked], [("input", 0, 16000, 1, "int16"),
                                                                 ("output", 1, 44100, 1, "int16")])
            for _, args in pa.checked:
                self.assertFalse(args["extra_settings"].change_device_parameters)
                self.assertFalse(args["extra_settings"].fail_if_conversion_required)
            output, input_ = pa.opened
            input_.callback(b"\2\0" * 512, 512, None, argparse.Namespace(input_overflow=False))
            result = bytearray(6)
            output.callback(result, 3, None, None)
            self.assertEqual(captured, [(b"\2\0" * 512, False)])
            self.assertEqual(result, b"\3\0" * 3)
            device.close()
            self.assertTrue(all(stream.closed and not stream.active for stream in pa.opened))

    def test_conflicting_format_and_duplicate_device_fail_before_open(self):
        pa = _PortAudio()
        with patch.dict(sys.modules, sounddevice=pa), patch.object(sys, "platform", "darwin"):
            with self.assertRaisesRegex(VoiceSetupError, "two_distinct_devices"):
                MacAudioDevice(0, 0, 44100).start(None, None)
            for direction in ("input", "output"):
                pa.conflict = direction
                with self.assertRaisesRegex(VoiceSetupError, f"voice_{direction}_pcm_"):
                    MacAudioDevice(0, 1, 44100).start(None, None)
            self.assertFalse(pa.opened)
