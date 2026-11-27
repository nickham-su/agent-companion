from io import StringIO
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from agent_companion.__main__ import main
from agent_companion.config import ConfigError, load_settings


class LoggingTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "logging.toml"

    def run_cli(self, *args):
        with (patch("sys.argv", ["agent-companion", "--config", str(self.path), *args]),
              patch("sys.stdout", new_callable=StringIO) as stdout,
              patch("sys.stderr", new_callable=StringIO) as stderr):
            self.assertEqual(main(), 0)
            self.assertEqual(stderr.getvalue(), "")
            return stdout.getvalue()

    def test_invalid_level_reports_field_without_echoing_input(self):
        for value in ('"synthetic-invalid-level"', "false", "123"):
            with self.subTest(value=value):
                self.path.write_text(f"log_level = {value}\n", encoding="utf-8")
                with self.assertRaisesRegex(ConfigError, "log_level") as error:
                    load_settings(self.path)
                self.assertNotIn(value, str(error.exception))

    def test_default_level_uses_stdout_and_throttles_warnings_but_not_errors(self):
        self.path.write_text("", encoding="utf-8")
        self.assertEqual(load_settings(self.path).log_level, "WARNING")

        async def demo(settings, *, delegate):
            log = logging.getLogger("agent_companion.test_logging")
            log.debug("synthetic debug")
            log.info("synthetic info")
            for index in range(3):
                log.warning("synthetic warning %d", index)
                log.error("synthetic error %d", index)
            return True

        with (patch("agent_companion.__main__.run_demo", side_effect=demo),
              patch("agent_companion.logging_setup.monotonic", side_effect=(1, 10, 31))):
            output = self.run_cli()
        self.assertNotIn("synthetic debug", output)
        self.assertNotIn("synthetic info", output)
        self.assertIn("synthetic warning 0", output)
        self.assertNotIn("synthetic warning 1", output)
        self.assertIn("synthetic warning 2", output)
        self.assertEqual(output.count("synthetic error"), 3)

    def test_debug_traces_runtime_without_demo_content(self):
        self.path.write_text('log_level = "DEBUG"\n', encoding="utf-8")
        output = self.run_cli("--demo-subtasks")
        debug = "\n".join(line for line in output.splitlines() if " DEBUG " in line)
        for event in ("runtime input recorded", "model request started", "model request finished", "reply committed",
                       "delegation accepted", "agent reply queued"):
            self.assertIn(event, debug)
        submissions = [line for line in debug.splitlines() if "scheduler input submitted" in line]
        self.assertEqual(len(submissions), 5)
        self.assertIn("agent_kind=main position=1/1 kind=user", submissions[0])
        self.assertIn("agent_kind=child position=1/1 kind=user", submissions[1])
        self.assertIn("delegated=True", submissions[1])
        self.assertIn("agent_kind=main position=1/1 kind=async_tool_result", submissions[2])
        self.assertIn("async_kind=agent_reply", submissions[2])
        self.assertIn("INFO agent_companion.__main__ agent companion started", output)
        self.assertIn("output_chars=", debug)
        self.assertIn("elapsed_ms=", debug)
        self.assertNotIn("离线模拟", debug)
        self.assertNotIn("模拟协议字段", debug)
        self.assertNotIn("给出一个模拟子任务结果", debug)
