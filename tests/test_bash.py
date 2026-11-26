import asyncio
from contextlib import nullcontext
from errno import EIO, EPERM, ESRCH
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import signal
from subprocess import DEVNULL, Popen, run
import sys
from tempfile import TemporaryDirectory
from textwrap import dedent
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from agent_companion import workspace as workspace_module
from agent_companion.app import running_agent
from agent_companion.config import ConfigError, Settings, load_settings
from agent_companion.protocol import (ChannelUser, DeliveryStatus, Message,
                                      Receipt, TextDelta, ToolCall, ToolReady)
from agent_companion.runtime import _proc_group_probe
from agent_companion.runtime import bash as bash_runtime
from agent_companion.runtime.bash import BASH_TOOL_NAME, _MAX_OUTPUT_BYTES, Bash
from agent_companion.runtime.instructions import load_roles
from agent_companion.runtime.subtasks import AgentRole, DELEGATE_TOOL_NAME
from agent_companion.runtime.timers import SET_TIMER_TOOL_NAME
from agent_companion.runtime.tools import ToolContext, Tools
from agent_companion.storage.sqlite import History
from agent_companion.workspace import Workspace
from test_message_support import tool_calls


async def _eventually(condition):
    async with asyncio.timeout(4):
        while not condition():
            await asyncio.sleep(0.01)


def _call(call_id, *, name=BASH_TOOL_NAME, **arguments):
    return ToolReady(ToolCall(call_id, name, json.dumps(arguments)))


def _results(messages):
    return {call.id: json.loads(call.result) for call in tool_calls(messages)}


def _record_processes(processes):
    def spawn(*args, **kwargs):
        child = Popen(*args, **kwargs)
        processes.append(child)
        return child
    return patch.object(bash_runtime.subprocess, "Popen", side_effect=spawn)


def _shells(processes):
    return [child for child in processes if str(bash_runtime._BASH_LAUNCHER) in child.args[:5]]


class _Channel:
    async def deliver(self, reply):
        return Receipt(reply.info.reply_id, DeliveryStatus.FULL, "".join([part async for part in reply]))


class _Model:
    def __init__(self, answer):
        self.answer = answer
        self.requests = []

    async def stream(self, messages, tools, *, thinking, instructions=None):
        names = tuple(tool.name for tool in tools)
        self.requests.append((tuple(messages), names))
        for event in self.answer(messages, names):
            yield event


class WorkspaceConfigurationTests(unittest.TestCase):
    def test_workspace_root_relative_to_config_and_not_created_during_loading(self):
        with TemporaryDirectory(dir=Path.cwd()) as folder:
            root = Path(folder)
            config = root / "settings.toml"
            for source, expected in (('', root / 'workspaces'),
                                     ('workspace_root = "fixtures/users"', root / 'fixtures/users'),
                                     (f'workspace_root = {json.dumps(str(root / "absolute"))}', root / 'absolute')):
                with self.subTest(source=source):
                    config.write_text(source, encoding="utf-8")
                    self.assertEqual(load_settings(config).workspace_root, expected)
                    self.assertFalse(expected.exists())
            for bad in ('false', '"  "', '""', '"\\u0000"'):
                with self.subTest(bad=bad):
                    config.write_text(f'workspace_root = {bad}', encoding="utf-8")
                    with self.assertRaisesRegex(ConfigError, 'workspace_root'):
                        load_settings(config)
        self.assertEqual(load_settings(Path("config.example.toml")).workspace_root, Path("workspaces"))

    def test_main_and_child_role_configurations_accept_bash(self):
        with TemporaryDirectory(dir=Path.cwd()) as folder:
            root = Path(folder)
            agents = root / "agents"
            agents.mkdir()
            for filename in ('main.md', 'child.md'):
                (agents / filename).write_text('+++\ndescription = "fixture"\ntools = ["bash"]\nmodel_id = "chat"\n+++\n',
                                               encoding="utf-8")
            self.assertEqual({name: role.tools for name, role in load_roles(root).items()},
                             {"main": (BASH_TOOL_NAME,), "child": (BASH_TOOL_NAME,)})


class BashExecutionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        folder = self.enterContext(TemporaryDirectory(dir=Path.cwd()))
        self.root = Path(folder)
        self.workspace = Workspace(self.root / "workspace")
        self.tools = Tools([Bash().tool])

    def _fixture_launcher(self, hooks):
        launcher = self.root / "launcher.py"
        package_root = str(bash_runtime._BASH_LAUNCHER.parents[2])
        launcher.write_text(
            "import os, sys\nfrom pathlib import Path\n"
            f"sys.path.insert(0, {package_root!r})\n"
            "from agent_companion.runtime._bash_launcher import main\n"
            "from agent_companion.workspace import Workspace\n"
            + hooks + "\nraise SystemExit(main(int(sys.argv[1]), sys.argv[2], sys.argv[3]))\n",
            encoding="utf-8",
        )
        return launcher

    async def _execute(self, *, workspace=True, request_locks=None, **arguments):
        call = _call("call", **arguments).call
        context = ToolContext("test-session", "run", call.id, False, workspace=self.workspace if workspace else None,
                              reply_id="test-reply")
        return json.loads(await self.tools.execute(call, context, request_locks=request_locks))

    async def test_debug_bash_result_reports_only_safe_metadata(self):
        name = "agent_companion.runtime.bash"
        with self.assertLogs(name, level="DEBUG") as logs:
            result = await self._execute(command='printf synthetic-stdout; printf synthetic-stderr >&2; exit 17')
            with patch.object(bash_runtime, "_open_launch_status", side_effect=OSError("synthetic-exception")):
                self.assertEqual(await self._execute(command=":"), {"error": "bash_start_failed"})
        self.assertEqual(result["stdout"], "synthetic-stdout")
        self.assertEqual(result["stderr"], "synthetic-stderr")
        self.assertEqual(len(logs.output), 2)
        self.assertIn("bash call finished; run=run reply=test-reply status=nonzero_exit exit_code=17", logs.output[0])
        self.assertIn("status=bash_start_failed exit_code=-", logs.output[1])
        for secret in ("printf", "synthetic-stdout", "synthetic-stderr", "synthetic-exception",
                       str(self.workspace.directory)):
            self.assertNotIn(secret, "\n".join(logs.output))
        with self.assertNoLogs(name, level="INFO"):
            self.assertEqual((await self._execute(command=":"))["exit_code"], 0)

    async def test_debug_dispatch_rejection_does_not_echo_model_input(self):
        context = ToolContext("test-session", "run", "synthetic-provider-id", False, reply_id="test-reply")
        bad_name = ToolCall(context.call_id, "synthetic-unknown-name", "synthetic-arguments")
        bad_args = ToolCall(context.call_id, "bash", "synthetic-arguments")
        with self.assertLogs("agent_companion.runtime.tools", level="DEBUG") as logs:
            self.assertEqual(json.loads(await self.tools.execute(bad_name, context)), {"error": "unknown_tool"})
            self.assertEqual(json.loads(await self.tools.execute(bad_args, context)), {"error": "invalid_arguments"})
        self.assertIn("reply=test-reply tool=- reason=unknown_tool", logs.output[0])
        self.assertIn("reply=test-reply tool=bash reason=invalid_arguments", logs.output[1])
        self.assertNotIn("synthetic-", "\n".join(logs.output))

    async def test_bash_starts_in_workspace_with_minimal_env_and_reset_cwd(self):
        bash_env = self.root / "synthetic-bash-env"
        bash_env.write_text('printf "unexpected startup file"', encoding="utf-8")
        with patch.dict(os.environ, {"BASH_ENV": str(bash_env), "BASH_FIXTURE_VAR": "synthetic"}):
            result = await self._execute(command='printf "%s\\n" "$PWD" "$HOME" "$PATH" '
                                                 '"${BASH_FIXTURE_VAR-unset}"; '
                                                 'if read -r line; then exit 1; fi; '
                                                 'mkdir inner && cd inner; printf issue >&2; exit 17')
        self.assertEqual(result, {"stdout": f"{self.workspace.directory}\n{self.workspace.directory}\n"
                                         "/usr/bin:/bin:/usr/sbin:/sbin\nunset\n",
                                  "stderr": "issue", "exit_code": 17, "timed_out": False,
                                  "output_limit_exceeded": False})
        self.assertEqual((await self._execute(command='printf %s "$PWD"'))["stdout"], str(self.workspace.directory))
        self.assertTrue((self.workspace.directory / "inner").is_dir())

    def _run_bash_with_closed_stdio(self, closed, *, fail_dup=False):
        # 在独立进程的事件循环已建立后才关闭标准 FD，防止事件循环占用刚释放的编号。
        fixture = dedent("""\
            import asyncio
            from errno import EBADF
            import fcntl
            import json
            import os
            from pathlib import Path
            import sys
            from unittest.mock import patch

            sys.path.insert(0, sys.argv[1])
            from agent_companion.runtime import bash

            root = Path(sys.argv[2])
            fail_dup = sys.argv[3] == 'fail'
            closed = list(map(int, sys.argv[4:]))
            popen = bash.subprocess.Popen
            passed = []

            def record(*args, **kwargs):
                passed.extend((fd, os.get_inheritable(fd)) for fd in kwargs['pass_fds'])
                return popen(*args, **kwargs)

            async def invoke():
                for fd in closed:
                    os.close(fd)
                command = 'if read -r line; then exit 99; fi; printf out; printf err >&2; exit 17'
                with patch.object(bash.subprocess, 'Popen', side_effect=record):
                    result = await bash._run_command(command, root / 'workspace', 3)
                for fd in set(closed) | {fd for fd, _ in passed}:
                    try:
                        os.fstat(fd)
                    except OSError as exc:
                        if exc.errno != EBADF:
                            raise
                    else:
                        raise AssertionError(f'descriptor {fd} was not closed')
                return {'result': result, 'passed': passed}

            if fail_dup:
                with patch.object(fcntl, 'fcntl', side_effect=OSError('synthetic dup failure')) as dup:
                    result = asyncio.run(invoke())
                dup.assert_called_once_with(closed[1], fcntl.F_DUPFD_CLOEXEC, 3)
            else:
                result = asyncio.run(invoke())
            (root / 'result.json').write_text(json.dumps(result), encoding='utf-8')
        """)
        completed = run((sys.executable, "-I", "-S", "-B", "-c", fixture,
                         str(bash_runtime._BASH_LAUNCHER.parents[2]), str(self.root),
                         "fail" if fail_dup else "ok", *map(str, closed)),
                        stdin=DEVNULL, capture_output=True, env={"LC_ALL": "C"}, timeout=8)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads((self.root / "result.json").read_text(encoding="utf-8"))

    def test_closed_standard_fds_do_not_overwrite_launcher_status(self):
        for closed in ((2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)):
            with self.subTest(closed=closed):
                report = self._run_bash_with_closed_stdio(closed)
                self.assertEqual(report["result"], {"exit_code": 17, "stdout": "out", "stderr": "err",
                                                    "timed_out": False, "output_limit_exceeded": False})
                (fd, inheritable), = report["passed"]
                self.assertGreaterEqual(fd, 3)
                self.assertFalse(inheritable)

    def test_launcher_status_dup_failure_closes_original_pipe_without_spawning(self):
        report = self._run_bash_with_closed_stdio((1, 2), fail_dup=True)
        self.assertEqual(report, {"result": {"error": "bash_start_failed"}, "passed": []})
        self.assertFalse(self.workspace.directory.exists())

    async def test_short_output_is_drained_after_child_exit_even_with_partial_reads(self):
        read = os.read
        with patch.object(os, "read", side_effect=lambda fd, size: read(fd, min(2, size))):
            result = await self._execute(command="printf short-stdout; printf short-stderr >&2")
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual((result["stdout"], result["stderr"]), ("short-stdout", "short-stderr"))
        self.assertFalse(result["timed_out"] or result["output_limit_exceeded"])

    def _make_search_only(self, directory):
        if sys.platform == "darwin" and not (hasattr(os, "O_SEARCH") or hasattr(os, "O_EXEC")):
            self.skipTest("Python SDK does not expose Darwin directory search flags")
        if sys.platform == "linux" and not hasattr(os, "O_PATH"):
            self.skipTest("Python does not expose Linux O_PATH")
        original = directory.stat().st_mode & 0o777
        self.addCleanup(directory.chmod, original)
        directory.chmod(0o111)
        try:
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        except PermissionError:
            return
        os.close(fd)
        self.skipTest("effective permissions still allow directory reads")

    async def test_no_rc_with_search_only_ancestor_creates_workspace_and_runs(self):
        ancestor = self.root / "search-only"
        parent = ancestor / "writable"
        parent.mkdir(parents=True)
        self.workspace = Workspace(parent / "workspace")
        self._make_search_only(ancestor)
        result = await self._execute(command='printf "%s|%s" "$PWD" "$HOME"')
        directory = self.workspace.directory.resolve()
        self.assertTrue(directory.is_dir())
        self.assertEqual((result["exit_code"], result["stdout"]), (0, f"{directory}|{directory}"))
        self.assertFalse((directory / ".agent").exists())
        directory.chmod(0o600)
        try:
            self.assertEqual(await self._execute(command=":"), {"error": "workspace_unavailable"})
        finally:
            directory.chmod(0o700)

    async def test_search_only_agent_checks_known_script_without_listing(self):
        self._write_rc("rc_function() { printf rc; }")
        agent = self.workspace.directory / ".agent"
        self._make_search_only(agent)
        result = await self._execute(command="rc_function; printf command")
        self.assertEqual((result["exit_code"], result["stdout"]), (0, "rccommand"))

    async def test_bad_arguments_do_not_create_workspace_or_fall_back_to_service_cwd(self):
        invalid = ({}, {"command": None}, {"command": " \n"}, {"command": "x\x00y"}, {"command": "\ud800"},
                   {"command": "a" * (32 * 1024 + 1)}, {"command": ":", "workdir": "."},
                   *({"command": ":", "timeout": value} for value in (0, 121, True, "1", 1.5, None)))
        for arguments in invalid:
            with self.subTest(fields=tuple(arguments)):
                self.assertEqual(await self._execute(**arguments), {"error": "invalid_arguments"})
        self.assertEqual(await self._execute(workspace=False, command=":"), {"error": "workspace_unavailable"})
        self.assertFalse(self.workspace.directory.exists())

    async def test_workspace_cannot_be_created_inside_running_or_copied_program_package(self):
        package = Path(workspace_module.__file__).resolve().parent
        alias = self.root / "alias"
        alias.symlink_to(package, target_is_directory=True)
        copy = self.root / "source" / "agent_companion"
        copy.mkdir(parents=True)
        for filename in ("__init__.py", "workspace.py"):
            (copy / filename).touch()
        for path in (package, alias, copy):
            with self.subTest(kind=path.name):
                user_dir = path / "not-created-workspace" / "synthetic-user"
                self.workspace = Workspace(user_dir)
                self.assertEqual(await self._execute(command=":"), {"error": "workspace_unavailable"})
                self.assertFalse(user_dir.parent.exists())

    async def test_workspace_link_swapped_during_resolution_is_not_executed(self):
        workspace = self.workspace.directory
        workspace.mkdir()
        outside = self.root / "outside"
        (outside / ".agent").mkdir(parents=True)
        (outside / ".agent" / "bashrc").write_text(": > rc-ran", encoding="utf-8")
        launcher = self._fixture_launcher(
            f"workspace = Path({str(workspace)!r})\n"
            f"outside = Path({str(outside)!r})\n"
            "resolve = Path.resolve\n"
            "def replace_leaf(path, *args, **kwargs):\n"
            "    if path in (workspace, workspace.parent) and not workspace.is_symlink():\n"
            "        workspace.rename(workspace.with_name('moved'))\n"
            "        workspace.symlink_to(outside, target_is_directory=True)\n"
            "    return resolve(path, *args, **kwargs)\n"
            "Path.resolve = replace_leaf\n",
        )
        with patch.object(bash_runtime, "_BASH_LAUNCHER", launcher):
            self.assertEqual(await self._execute(command=": > model-ran"), {"error": "workspace_unavailable"})
        self.assertTrue(workspace.is_symlink())
        self.assertEqual(list(outside.iterdir()), [outside / ".agent"])

    async def test_launcher_changes_directory_using_open_workspace_fd(self):
        workspace = self.workspace.directory
        workspace.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        launcher = self._fixture_launcher(
            f"workspace = Path({str(workspace)!r})\n"
            f"outside = Path({str(outside)!r})\n"
            "fchdir = os.fchdir\n"
            "def replace_then_chdir(fd):\n"
            "    workspace.rename(workspace.with_name('moved'))\n"
            "    workspace.symlink_to(outside, target_is_directory=True)\n"
            "    fchdir(fd)\n"
            "os.fchdir = replace_then_chdir\n",
        )
        with patch.object(bash_runtime, "_BASH_LAUNCHER", launcher):
            command = ': > model-ran; : > "$HOME/home-ran"'
            self.assertEqual((await self._execute(command=command))["exit_code"], 0)
        self.assertTrue((self.root / "moved" / "model-ran").is_file())
        self.assertTrue((self.root / "moved" / "home-ran").is_file())
        self.assertEqual(list(outside.iterdir()), [])

    async def test_chdir_failure_is_fixed_and_shell_exit_is_not_launcher_error(self):
        result = await self._execute(command="printf W >&2; exit 1")
        self.assertEqual((result["exit_code"], result["stderr"]), (1, "W"))
        launcher = self._fixture_launcher(
            "def fail_chdir(path):\n"
            "    raise OSError('synthetic workspace detail')\n"
            "os.fchdir = fail_chdir\n",
        )
        with patch.object(bash_runtime, "_BASH_LAUNCHER", launcher):
            self.assertEqual(await self._execute(command=":"), {"error": "workspace_unavailable"})

    async def test_start_and_workspace_errors_are_fixed(self):
        with patch.object(bash_runtime.subprocess, "Popen", side_effect=OSError("synthetic startup detail")):
            self.assertEqual(await self._execute(command=":"), {"error": "bash_start_failed"})
        self.assertFalse(self.workspace.directory.exists())
        self.workspace.directory.write_text("synthetic file", encoding="utf-8")
        self.assertEqual(await self._execute(command=":"), {"error": "workspace_unavailable"})

    async def test_exec_failure_is_not_mistaken_for_readiness_on_a_partial_read(self):
        launcher = self._fixture_launcher(
            "def fail_exec(*args):\n"
            "    os.write(2, b'synthetic startup detail')\n"
            "    raise OSError('synthetic exec failure')\n"
            "os.execve = fail_exec\n",
        )
        read = os.read
        with (patch.object(bash_runtime, "_BASH_LAUNCHER", launcher),
              patch.object(os, "read", side_effect=lambda fd, size: read(fd, min(1, size)))):
            self.assertEqual(await self._execute(command=":"), {"error": "bash_start_failed"})

    async def test_missing_and_malformed_launch_status_do_not_expose_helper_output(self):
        launcher = self.root / "broken_launcher.py"
        with patch.object(bash_runtime, "_BASH_LAUNCHER", launcher):
            self.assertEqual(await self._execute(command=":"), {"error": "bash_start_failed"})
            launcher.write_text("import os, sys\nos.write(int(sys.argv[1]), b'RRR')\n"
                                "print('synthetic startup detail')\n", encoding="utf-8")
            self.assertEqual(await self._execute(command=":"), {"error": "bash_start_failed"})

    async def test_launcher_preserves_default_sigpipe(self):
        result = await self._execute(command='kill -s PIPE "$$"; printf unexpected')
        self.assertEqual((result["exit_code"], result["stdout"]), (-signal.SIGPIPE, ""))

    async def test_pipe_setup_failure_kills_and_reaps_started_child(self):
        actual_set_blocking = os.set_blocking
        processes, configured = [], []

        def configure(fd, blocking):
            configured.append(fd)
            if len(configured) == 2:
                raise OSError("synthetic pipe setup detail")
            actual_set_blocking(fd, blocking)

        with _record_processes(processes), patch.object(os, "set_blocking", side_effect=configure):
            result = await asyncio.wait_for(self._execute(command="exec /bin/sleep 20"), 2)
        self.assertEqual(result, {"error": "bash_io_failed"})
        child, = _shells(processes)
        self.assertEqual(child.returncode, -signal.SIGKILL)
        self.assertTrue(child.stdout.closed and child.stderr.closed)
        self.assertEqual((await self._execute(command="printf next"))["stdout"], "next")

    async def test_slow_workspace_preparation_does_not_block_other_user_and_times_out(self):
        await self._check_slow_launcher(timeout_in_preparation=True)

    async def test_slow_chdir_does_not_block_other_user_and_can_be_cancelled(self):
        await self._check_slow_launcher(timeout_in_preparation=False)

    async def _check_slow_launcher(self, *, timeout_in_preparation):
        marker, gate = self.root / "started", self.root / "gate"
        os.mkfifo(gate)
        launcher = self._fixture_launcher(
            f"slow = Path({str(self.workspace.directory)!r})\n"
            "def gate(path):\n"
            "    if Path(path) == slow:\n"
            "        import subprocess\n"
            "        descendant = subprocess.Popen(('/bin/sleep', '20'))\n"
            f"        started = Path({str(marker)!r})\n"
            "        started.with_suffix('.tmp').write_text(str(descendant.pid))\n"
            "        started.with_suffix('.tmp').rename(started)\n"
            f"        with open({str(gate)!r}, 'rb') as stream:\n"
            "            stream.read(1)\n"
            "open_workspace = Workspace.open_directory\n"
            "chdir = os.fchdir\n"
            "def slow_open(workspace, *args, **kwargs):\n"
            "    gate(workspace.directory)\n"
            "    return open_workspace(workspace, *args, **kwargs)\n"
            "def slow_chdir(fd):\n"
            "    if os.path.samestat(os.fstat(fd), slow.stat()): gate(slow)\n"
            "    return chdir(fd)\n"
            + ("Workspace.open_directory = slow_open\n" if timeout_in_preparation else "os.fchdir = slow_chdir\n"),
        )
        processes = []
        with patch.object(bash_runtime, "_BASH_LAUNCHER", launcher), _record_processes(processes) as spawn:
            task = asyncio.create_task(self._execute(command="printf wrong", timeout=2 if timeout_in_preparation else 20))
            try:
                await _eventually(marker.exists)
                slow, = _shells(processes)
                self.assertEqual(os.getpgid(int(marker.read_text())), slow.pid)
                fd, = spawn.call_args.kwargs["pass_fds"]
                with self.assertRaises(OSError):
                    os.fstat(fd)  # 启动助手阻塞时，父进程已关闭自己的状态写端。
                other_workspace = Workspace(self.root / "other-workspace")
                call = _call("other", command="printf other").call
                context = ToolContext("other-user", "run", call.id, True, workspace=other_workspace)
                tools = Tools([Bash().tool])
                result = json.loads(await asyncio.wait_for(tools.execute(call, context), 2))
                self.assertEqual(result["stdout"], "other")
                self.assertFalse(task.done())  # 另一用户已完成，而慢目录仍阻塞在真实子进程 I/O 中。
                if timeout_in_preparation:
                    result = await asyncio.wait_for(task, 3)
                    self.assertTrue(result["timed_out"])
                    self.assertEqual((result["stdout"], result["stderr"]), ("", ""))
                    self.assertFalse(self.workspace.directory.exists())
                else:
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(task, 2)
                slow, fast = _shells(processes)
                self.assertEqual((slow.returncode, fast.returncode), (-signal.SIGKILL, 0))
                self.assertTrue(slow.stdout.closed and slow.stderr.closed)
                self.assertTrue(await bash_runtime._confirm_group_exit(slow.pid))
                self.assertTrue(all(call.kwargs.get("cwd") is None for call in spawn.call_args_list))
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    def _write_rc(self, script):
        directory = self.workspace.directory / ".agent"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "bashrc").write_text(script, encoding="utf-8")

    async def test_rc_loaded_once_per_call_in_same_shell(self):
        self._write_rc("export PATH=\"$HOME/bin:$PATH\"\n"
                       "helper() { from_path; }\n"
                       "exec 3<rc-input\n"
                       "printf x >> rc-runs\n")
        (self.workspace.directory / "rc-input").write_text(" from-rc-fd\n", encoding="utf-8")
        bin_dir = self.workspace.directory / "bin"
        bin_dir.mkdir()
        executable = bin_dir / "from_path"
        executable.write_text("#!/bin/bash\nprintf from-path\n", encoding="utf-8")
        executable.chmod(0o700)
        command = 'helper; IFS= read -r line <&3; printf "%s " "$line"; '
        command += " /bin/bash -c 'printf %s \"${BASH_ENV-unset}\"'; exit 31"
        for _ in range(2):
            result = await self._execute(command=command)
            self.assertEqual((result["exit_code"], result["stdout"], result["stderr"]), (31, "from-path from-rc-fd unset", ""))
            self.assertNotIn("error", result)
        self.assertEqual((self.workspace.directory / "rc-runs").read_text(), "xx")

    async def test_rc_failures_do_not_run_command(self):
        for script, reason, code in (
            ("return 23", "source_failed", 23),
            ("if then", "source_failed", 2),
            ("exit 0", "exited_before_command", 0),
            ("exit 19", "exited_before_command", 19),
            ("set -e; false; printf unexpected", "exited_before_command", 1),
            ("printf starting >&2; /bin/sleep 20", "timeout", -signal.SIGKILL),
        ):
            with self.subTest(script=script):
                self._write_rc(script)
                result = await self._execute(command=": > command-ran", timeout=1 if reason == "timeout" else 3)
                self.assertEqual((result["error"], result["reason"], result["exit_code"]),
                                 ("bash_rc_failed", reason, code))
                self.assertEqual(result["timed_out"], reason == "timeout")
                self.assertFalse((self.workspace.directory / "command-ran").exists())
                if reason == "timeout":
                    self.assertEqual(result["stderr"], "starting")
                if script == "if then":
                    self.assertIn("syntax error", result["stderr"])

    async def test_unreadable_or_unsafe_rc_does_not_run_command(self):
        self._write_rc(": > rc-ran")
        launcher = self._fixture_launcher(
            "actual_open = os.open\n"
            "def deny_rc(path, *args, **kwargs):\n"
            "    if path == 'bashrc':\n"
            "        raise PermissionError('synthetic permission detail')\n"
            "    return actual_open(path, *args, **kwargs)\n"
            "os.open = deny_rc\n",
        )
        expected = {"error": "bash_rc_failed", "reason": "unreadable", "exit_code": 1}
        with patch.object(bash_runtime, "_BASH_LAUNCHER", launcher):
            self.assertEqual(await self._execute(command=": > command-ran"), expected)
        script = self.workspace.directory / ".agent" / "bashrc"
        script.unlink()
        os.mkfifo(script)
        self.assertEqual(await self._execute(command=": > command-ran"), expected)
        script.unlink()
        target = self.root / "outside-rc"
        target.write_text(": > rc-ran", encoding="utf-8")
        script.symlink_to(target)
        self.assertEqual(await self._execute(command=": > command-ran"), dict(expected, reason="symlink"))
        for file in ("rc-ran", "command-ran"):
            self.assertFalse((self.workspace.directory / file).exists())

    async def test_regular_agent_file_means_rc_missing_but_agent_links_are_rejected(self):
        self.workspace.directory.mkdir()
        agent = self.workspace.directory / ".agent"
        agent.touch()
        result = await self._execute(command="printf normal")
        self.assertEqual((result["exit_code"], result["stdout"]), (0, "normal"))
        agent.unlink()
        for is_directory in (False, True):
            target = self.root / ("linked-dir" if is_directory else "linked-file")
            if is_directory:
                target.mkdir()
                (target / "bashrc").write_text(": > rc-ran", encoding="utf-8")
            else:
                target.touch()
            agent.symlink_to(target, target_is_directory=is_directory)
            result = await self._execute(command=": > command-ran")
            self.assertEqual((result["error"], result["reason"]), ("bash_rc_failed", "symlink"))
            self.assertFalse((self.workspace.directory / "rc-ran").exists())
            self.assertFalse((self.workspace.directory / "command-ran").exists())
            agent.unlink()

    async def test_fixed_rc_descriptors_survive_loading(self):
        self._write_rc("exec 10>ten 11>eleven 19>nineteen\n")
        result = await self._execute(command="printf x >&10 && printf y >&11 && printf z >&19")
        self.assertNotIn("error", result)
        self.assertEqual((result["exit_code"], result["stderr"]), (0, ""))
        for name, expected in (("ten", "x"), ("eleven", "y"), ("nineteen", "z")):
            self.assertEqual((self.workspace.directory / name).read_text(), expected)

    async def test_rc_handoff_does_not_modify_ifs(self):
        for script, ifs in (("readonly IFS=' :'; set -o posix", " :"), ("readonly IFS=G", "G")):
            with self.subTest(script=script):
                self._write_rc(script)
                result = await self._execute(command='printf "model:%s" "$IFS"')
                self.assertNotIn("error", result)
                self.assertEqual((result["exit_code"], result["stdout"], result["stderr"]), (0, "model:" + ifs, ""))

    async def test_low_fd_limit_preserves_direct_launch_and_reports_rc_setup_failure(self):
        launcher = self._fixture_launcher(
            "import resource\n"
            "soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)\n"
            "resource.setrlimit(resource.RLIMIT_NOFILE, (48 if soft < 0 else min(48, soft), hard))\n",
        )
        command = 'exec 10>direct; printf ok >&10; printf "%s|%s|%s" "$0" "$#" "$BASH_EXECUTION_STRING"'
        with patch.object(bash_runtime, "_BASH_LAUNCHER", launcher):
            result = await self._execute(command=command)
            self.assertEqual((result["exit_code"], result["stdout"]), (0, "/bin/bash|0|" + command))
            self.assertEqual((self.workspace.directory / "direct").read_text(), "ok")
            self._write_rc(": > rc-ran")
            result = await self._execute(command=": > model-ran")
        self.assertEqual(result, {"error": "bash_rc_failed", "reason": "descriptor_unavailable", "exit_code": 1})
        self.assertFalse((self.workspace.directory / "rc-ran").exists())
        self.assertFalse((self.workspace.directory / "model-ran").exists())

    async def test_dynamic_rc_fds_and_xtrace_survive_without_affecting_handoff(self):
        supports = await self._execute(command="(( BASH_VERSINFO[0] * 10 + BASH_VERSINFO[1] >= 41 ))")
        if supports["exit_code"] != 0:
            self.skipTest("Bash 4.1+ is required for dynamic descriptors and BASH_XTRACEFD")
        self._write_rc("for i in {1..64}; do exec {fd}>>dynamic; fds+=(\"$fd\"); done\n"
                       "exec {trace_fd}>trace\nBASH_XTRACEFD=$trace_fd\nset -x\n")
        result = await self._execute(command="for fd in \"${fds[@]}\"; do printf x >&$fd || exit 52; done; printf model-ran")
        self.assertEqual((result["exit_code"], result["stdout"], result["stderr"]), (0, "model-ran", ""))
        self.assertNotIn("error", result)
        self.assertEqual((self.workspace.directory / "dynamic").read_text(), "x" * 64)
        self.assertIn("printf model-ran", (self.workspace.directory / "trace").read_text())

    async def test_completed_handoff_does_not_wait_for_other_copies_of_control_fd(self):
        self._write_rc("printf rc;")
        open_status = bash_runtime._open_launch_status

        def keep_peer_open():
            parent, child = open_status()
            self.addCleanup(os.close, os.dup(child))
            return parent, child

        with patch.object(bash_runtime, "_open_launch_status", side_effect=keep_peer_open):
            result = await self._execute(command="printf model; exit 17")
        self.assertNotIn("error", result)
        self.assertEqual((result["exit_code"], result["stdout"]), (17, "rcmodel"))

    async def test_authorized_but_unread_handoff_preserves_timeout_and_exit(self):
        launcher = self.root / "handoff_fixture.py"
        launcher.write_text(
            "import os, select, sys, time\n"
            "fd = int(sys.argv[1])\n"
            "os.write(fd, b'RBD')\n"
            "if not select.select([fd], [], [], 5)[0]: os._exit(99)\n"
            "os.write(1, b'authorization pending')\n"
            "os.write(2, b'before handoff')\n"
            "if sys.argv[3] == 'wait': time.sleep(10)\n"
            "os._exit(19)  # 关闭套接字时故意不读取已排队的放行字节。\n",
            encoding="utf-8",
        )
        for command, timed_out, code in (("wait", True, -signal.SIGKILL), ("end", False, 19)):
            with self.subTest(command=command), patch.object(bash_runtime, "_BASH_LAUNCHER", launcher):
                result = await self._execute(command=command, timeout=1 if timed_out else 3)
                self.assertEqual(result["error"], "bash_rc_failed")
                self.assertEqual(result["reason"], "handoff_timeout" if timed_out else "exited_before_command")
                self.assertTrue(result["command_authorized"])
                self.assertEqual((result["timed_out"], result["exit_code"]), (timed_out, code))
                self.assertEqual((result["stdout"], result["stderr"]), ("authorization pending", "before handoff"))

    async def test_timeout_after_handoff_is_not_rc_failure(self):
        self._write_rc("printf rc;")
        result = await self._execute(command="printf model; exec /bin/sleep 20", timeout=1)
        self.assertNotIn("error", result)
        self.assertTrue(result["timed_out"])
        self.assertEqual((result["exit_code"], result["stdout"]), (-signal.SIGKILL, "rcmodel"))

    async def test_timeout_and_utf8_serialization(self):
        with self.assertLogs("agent_companion.runtime.bash", level="DEBUG") as logs:
            result = await self._execute(command="printf ready; exec /bin/sleep 20", timeout=1)
        self.assertIn("status=timeout", logs.output[0])
        self.assertTrue(result["timed_out"])
        self.assertEqual(result["stdout"], "ready")
        self.assertEqual(result["exit_code"], -signal.SIGKILL)
        context = ToolContext("test-session", "run", "call", False, workspace=self.workspace)
        output = await self.tools.execute(_call("call", command=r'''printf '汉字\377\n"' ''').call, context)
        self.assertIn("汉字", output)
        self.assertEqual(json.loads(output)["stdout"], '汉字\ufffd\n"')

    async def test_bash_combined_limit_and_public_archive(self):
        context = ToolContext("test-session", "run", "call", False, workspace=self.workspace)
        width = _MAX_OUTPUT_BYTES // 2
        command = f"printf '%0*d' {width} 0; printf '%0*d' {width} 0 >&2; "
        for extra_bytes in (0, 1, 512 * 1024):
            with self.subTest(extra_bytes=extra_bytes):
                extra = bool(extra_bytes)
                more = f"printf '%0*d' {extra_bytes} 0 >&2; " if extra else ""
                call = _call("call", command=command + more + ": > command-finished; exit 37").call
                with self.assertLogs("agent_companion.runtime.bash", level="DEBUG") as logs:
                    output = await self.tools.execute(call, context)
                self.assertIn("status=output_limit" if extra else "status=nonzero_exit", logs.output[0])
                self.assertIn('"output_limit_exceeded": ' + str(extra).lower(), output)
                preview, _, archive_path = output.rpartition("截断前的工具结果：")
                self.assertIn("公共工具层已截断", preview)
                archive = self.workspace.directory / archive_path.removesuffix("]")
                result = json.loads(archive.read_text(encoding="utf-8"))
                self.assertEqual((len(result["stdout"]), len(result["stderr"])), (width, width))
                self.assertEqual(result["output_limit_exceeded"], extra)
                self.assertEqual("output_limit_note" in result, extra)
                self.assertFalse(result["timed_out"])
                self.assertEqual(result["exit_code"], 37)
                (self.workspace.directory / "command-finished").unlink()

    async def test_continuous_output_beyond_limit_still_times_out(self):
        with patch.object(bash_runtime, "_MAX_OUTPUT_BYTES", 128):
            result = await self._execute(command="while :; do printf '%08192d' 0; done", timeout=1)
        self.assertEqual(len(result["stdout"]), 128)
        self.assertTrue(result["output_limit_exceeded"] and result["timed_out"])
        self.assertEqual(result["exit_code"], -signal.SIGKILL)

    async def test_success_stops_same_group_background_writer_even_without_output_pipes(self):
        for redirect in ("", " >/dev/null 2>&1"):
            with self.subTest(redirect=redirect), patch("os.killpg", wraps=os.killpg) as killpg:
                result = await asyncio.wait_for(self._execute(
                    command="(until [[ -e release ]]; do /bin/sleep 0.01; done; : > late)" + redirect +
                            " & printf %s $$"), 2)
                self.assertEqual(result["exit_code"], 0)
                self.assertFalse(result["timed_out"])
                group = int(result["stdout"])
                self.assertEqual([call.args for call in killpg.call_args_list if call.args == (group, signal.SIGKILL)],
                                 [(group, signal.SIGKILL)])
                self.assertIn((group, 0), [call.args for call in killpg.call_args_list])
                release = self.workspace.directory / "release"
                release.touch()
                await asyncio.sleep(0.05)
                self.assertFalse((self.workspace.directory / "late").exists())
                release.unlink()

    async def test_group_confirmation_is_bounded_and_distinguishes_active_and_zombie_processes(self):
        with (patch.object(os, "killpg") as probe,
              patch.object(bash_runtime, "_inspect_group", side_effect=[(True, True), (True, False)]) as inspect):
            self.assertTrue(await bash_runtime._confirm_group_exit(99))
            self.assertEqual(inspect.call_count, 2)
            self.assertEqual([call.args for call in probe.call_args_list], [(99, 0), (99, 0)])

        with (patch.object(os, "killpg"), patch.object(bash_runtime, "_inspect_group", return_value=(True, None)),
              patch.object(bash_runtime, "_GROUP_CLOSE_TIMEOUT", 0.05)):
            self.assertFalse(await asyncio.wait_for(bash_runtime._confirm_group_exit(99), 2))

    async def test_darwin_eperm_preserves_exit_code_only_after_group_confirmation(self):
        for platform, signal_errno, check_errno, live, exit_code, error in (
            ("darwin", EPERM, ESRCH, None, 0, None),
            ("darwin", EPERM, ESRCH, None, 17, None),
            ("darwin", EPERM, EPERM, False, 17, None),
            ("darwin", EPERM, EPERM, True, 0, "bash_cleanup_unconfirmed"),
            ("darwin", EPERM, EPERM, None, 0, "bash_cleanup_unconfirmed"),
            ("darwin", EPERM, EIO, None, 0, "bash_cleanup_unconfirmed"),
            ("darwin", EIO, ESRCH, None, 0, "bash_io_failed"),
            ("linux", EPERM, ESRCH, None, 0, "bash_io_failed"),
            ("linux", EPERM, EPERM, False, 0, "bash_cleanup_unconfirmed"),
        ):
            processes = []

            def signal_group(pid, signum):
                child, = processes
                self.assertEqual(pid, child.pid)
                if signum == signal.SIGKILL:
                    self.assertIsNone(child.returncode)  # 尚未回收才允许发终止信号。
                    raise OSError(signal_errno, "synthetic group signal failure")
                self.assertEqual(signum, 0)
                self.assertEqual(child.returncode, exit_code)
                raise OSError(check_errno, "synthetic group check failure")

            platform_info = SimpleNamespace(platform=platform, executable=sys.executable)
            with (self.subTest(platform=platform, signal_errno=signal_errno, check_errno=check_errno,
                               live=live, exit_code=exit_code),
                  _record_processes(processes), patch.object(bash_runtime, "sys", platform_info),
                  patch.object(os, "killpg", side_effect=signal_group) as killpg,
                  patch.object(os, "kill") as kill,
                  patch.object(bash_runtime, "_inspect_group", return_value=(True, live)) as inspect,
                  patch.object(bash_runtime, "_BLOCKED_WORKSPACES", set()) as blocked,
                  patch.object(bash_runtime, "_GROUP_CLOSE_TIMEOUT", 0.1),
                  patch.object(bash_runtime._LOG, "warning")):
                result = await self._execute(command=f"printf out; exit {exit_code}")
                if error is None:
                    self.assertEqual((result.get("exit_code"), result.get("stdout")), (exit_code, "out"))
                else:
                    self.assertEqual(result, {"error": error})
                self.assertEqual(self.workspace.directory in blocked, error == "bash_cleanup_unconfirmed")
                child, = processes
                self.assertEqual([call.args[1] for call in killpg.call_args_list].count(signal.SIGKILL), 1)
                kill.assert_called_once_with(child.pid, signal.SIGKILL)
                self.assertEqual(inspect.called, platform == "darwin" and check_errno == EPERM)

    async def test_darwin_probe_with_eperm_uses_snapshot_only_if_own_group_disappears(self):
        argv = ("/bin/bash", "--noprofile", "--norc", "-c", "printf '99 Z+\\n'")
        for check_errno, expected in ((ESRCH, (True, False)), (EPERM, (False, None)), (None, (False, None))):
            processes = []

            def signal_group(pid, signum):
                child, = processes
                self.assertEqual(pid, child.pid)
                if signum == signal.SIGKILL:
                    self.assertIsNone(child.returncode)
                    raise PermissionError(EPERM, "synthetic probe signal failure")
                self.assertEqual(signum, 0)
                self.assertEqual(child.returncode, 0)
                if check_errno is not None:
                    raise OSError(check_errno, "synthetic probe check failure")

            with (self.subTest(check_errno=check_errno), _record_processes(processes),
                  patch.object(bash_runtime, "sys", SimpleNamespace(platform="darwin")),
                  patch.object(bash_runtime, "_probe_command", return_value=argv) as command,
                  patch.object(os, "killpg", side_effect=signal_group) as killpg, patch.object(os, "kill")):
                result = await bash_runtime._inspect_group(99, asyncio.get_running_loop().time() + 1)
                self.assertEqual(result, expected)
                command.assert_called_once_with(99)  # 不为探测器再启动探测器。
                child, = processes
                self.assertEqual([call.args for call in killpg.call_args_list],
                                 [(child.pid, signal.SIGKILL), (child.pid, 0)])

    def test_macos_process_snapshot_distinguishes_zombies_and_active_members(self):
        self.assertIs(bash_runtime._macos_live_members(99, b"\n99 Z+\n100 R\n"), False)
        self.assertIs(bash_runtime._macos_live_members(99, b"99 Z\n99 S+\n"), True)
        self.assertIsNone(bash_runtime._macos_live_members(99, b"100 R\n"))

    def test_linux_only_reads_status_for_target_group(self):
        entries = [SimpleNamespace(name=str(pid), path=f"/fixture/{pid}") for pid in (100, 99, 101)]

        def read_stat(path, mode):
            if path != "/fixture/99/stat":
                raise PermissionError("non-target PID")
            return BytesIO(b"99 (name ) with parenthesis) Z 10 42 42 0\n")

        with (patch.object(_proc_group_probe.os, "scandir", return_value=nullcontext(entries)),
              patch.object(_proc_group_probe.os, "getpgid", side_effect={100: 41, 99: 42, 101: 43}.__getitem__),
              patch.object(_proc_group_probe, "open", side_effect=read_stat, create=True) as stat):
            self.assertIs(_proc_group_probe.live_members(42), False)
            stat.assert_called_once_with("/fixture/99/stat", "rb")
            stat.side_effect = PermissionError("target PID unreadable")
            with self.assertRaises(PermissionError):
                _proc_group_probe.live_members(42)

    async def test_probe_deadline_reaps_process_and_does_not_start_thread(self):
        processes = []
        loop = asyncio.get_running_loop()
        with (_record_processes(processes), patch.object(bash_runtime, "_probe_command", return_value=("/bin/sleep", "20")),
              patch.object(asyncio, "to_thread", new_callable=AsyncMock) as thread):
            result = await asyncio.wait_for(bash_runtime._inspect_group(99, loop.time() + 0.06), 2)
        self.assertEqual(result, (True, None))
        child, = processes
        self.assertEqual(child.returncode, -signal.SIGKILL)
        self.assertTrue(child.stdout.closed)
        thread.assert_not_called()

    async def test_oversized_probe_output_does_not_confirm_group(self):
        processes = []
        argv = ("/bin/bash", "--noprofile", "--norc", "-c", "printf '%0256d' 0; exec /bin/sleep 20")
        with (_record_processes(processes), patch.object(bash_runtime, "_probe_command", return_value=argv),
              patch.object(bash_runtime, "_PROBE_MAX_OUTPUT", 64)):
            result = await asyncio.wait_for(bash_runtime._inspect_group(99, asyncio.get_running_loop().time() + 1), 2)
        self.assertEqual(result, (True, None))
        child, = processes
        self.assertEqual(child.returncode, -signal.SIGKILL)
        self.assertTrue(child.stdout.closed)

    async def test_queued_bash_waits_for_group_confirmation_and_unconfirmed_group_blocks_rebind(self):
        for confirmed, cancel in ((True, False), (False, False), (False, True)):
            with self.subTest(confirmed=confirmed, cancel=cancel):
                await self._check_group_confirmation_with_queued_command(confirmed, cancel)

    async def _check_group_confirmation_with_queued_command(self, confirmed, cancel):
        confirming, release, queued = (asyncio.Event() for _ in range(3))
        request_locks = self.tools.request_locks()
        actual_confirm = bash_runtime._confirm_group_exit

        async def confirm(group):
            if confirming.is_set():
                return await actual_confirm(group)
            confirming.set()
            await release.wait()
            return confirmed and await actual_confirm(group)

        async def queued_command():
            queued.set()
            return await self._execute(command="printf next", request_locks=request_locks)

        processes = []
        with (patch.object(bash_runtime, "_BLOCKED_WORKSPACES", set()),
              patch.object(bash_runtime._LOG, "warning") as warning,
              patch.object(bash_runtime, "_confirm_group_exit", side_effect=confirm),
              _record_processes(processes)):
            first = asyncio.create_task(self._execute(command="/bin/sleep 20 >/dev/null 2>&1 & printf first",
                                                      request_locks=request_locks))
            second = None
            try:
                await asyncio.wait_for(confirming.wait(), 2)
                if cancel:
                    first.cancel()
                second = asyncio.create_task(queued_command())
                await asyncio.wait_for(queued.wait(), 2)
                self.assertFalse(first.done())
                self.assertFalse(second.done())
                self.assertEqual(len(_shells(processes)), 1)
                release.set()
                if cancel:
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(first, 2)
                else:
                    result = await asyncio.wait_for(first, 2)
                    self.assertEqual(result["stdout"] if confirmed else result["error"],
                                     "first" if confirmed else "bash_cleanup_unconfirmed")
                result = await asyncio.wait_for(second, 2)
                if confirmed:
                    self.assertEqual(result["stdout"], "next")
                    self.assertEqual(len(_shells(processes)), 2)
                    warning.assert_not_called()
                else:
                    self.assertEqual(result, {"error": "bash_cleanup_unconfirmed"})
                    original_tools = self.tools
                    try:
                        self.tools = Tools([Bash().tool])  # 模拟同一工作区再次绑定到新 Agent 树。
                        self.assertEqual(await self._execute(command=":"), result)
                    finally:
                        self.tools = original_tools
                    self.assertEqual(len(_shells(processes)), 1)
                    warning.assert_called_once()
            finally:
                release.set()
                await asyncio.gather(*(task for task in (first, second) if task is not None), return_exceptions=True)

    async def test_startup_cancellation_reaps_before_allowing_next_command(self):
        actual_set_blocking = os.set_blocking
        actual_confirm = bash_runtime._confirm_group_exit
        request_locks = self.tools.request_locks()
        entered, confirming, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        processes = []

        def configure(fd, blocking):
            if not entered.is_set():
                entered.set()
                first.cancel()  # PID 已取得，取消在首个异步轮询点生效。
            actual_set_blocking(fd, blocking)

        async def confirm(group):
            if not confirming.is_set():
                confirming.set()
                await release.wait()
            return await actual_confirm(group)

        with (_record_processes(processes), patch.object(os, "set_blocking", side_effect=configure),
              patch.object(bash_runtime, "_confirm_group_exit", side_effect=confirm)):
            first = asyncio.create_task(self._execute(command="exec /bin/sleep 20", request_locks=request_locks))
            second = None
            try:
                await asyncio.wait_for(confirming.wait(), 2)
                first.cancel()
                second = asyncio.create_task(self._execute(command="printf next", request_locks=request_locks))
                await asyncio.sleep(0)
                self.assertFalse(first.done())
                self.assertFalse(second.done())
                child, = _shells(processes)
                self.assertEqual(child.returncode, -signal.SIGKILL)
                self.assertTrue(child.stdout.closed and child.stderr.closed)
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(first, 3)
                self.assertEqual((await asyncio.wait_for(second, 2))["stdout"], "next")
            finally:
                release.set()
                if second is not None:
                    await asyncio.gather(first, second, return_exceptions=True)
                else:
                    await asyncio.gather(first, return_exceptions=True)


class BashRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        folder = self.enterContext(TemporaryDirectory(dir=Path.cwd()))
        self.settings = Settings(workspace_root=Path(folder) / "workspaces")
        self.history = History(":memory:")
        self.addCleanup(self.history.close)
        self.user = ChannelUser("synthetic/channel:user A")

    def _workspace(self, user):
        return self.settings.workspace_root / sha256(user.key.encode("utf-8")).hexdigest()

    async def test_disabled_role_never_executes_and_userless_demo_has_no_fallback(self):
        for user, role, expected in ((self.user, AgentRole(None), "unknown_tool"),
                                     (None, AgentRole(None, (BASH_TOOL_NAME,)), "workspace_unavailable")):
            with self.subTest(expected=expected):
                model = _Model(lambda messages, names: (_call("last", command=":"),)
                               if messages[-1].kind == "user" else (TextDelta("done"),))
                session_id = f"test-{expected}"
                async with running_agent(self.settings, self.history, model, _Channel(), session_id=session_id,
                                         roles={"main": role}, user=user) as (runtime, _):
                    runtime.submit_batch((Message("user", "test"),))
                    await asyncio.wait_for(runtime.wait_idle(), 3)
                self.assertEqual(BASH_TOOL_NAME in model.requests[0][1], BASH_TOOL_NAME in role.tools)
                self.assertEqual(_results(model.requests[-1][0])["last"], {"error": expected})
                self.assertFalse(self.settings.workspace_root.exists())

    async def test_stable_user_directory_across_sessions_and_persists_after_close(self):
        for session_id, command, output in (("new-agent-a", "printf marker > fixture", ""),
                                            ("new-agent-b", "cat fixture", "marker")):
            model = _Model(lambda messages, names: (_call("last", command=command),)
                           if messages[-1].kind == "user" else (TextDelta("done"),))
            async with running_agent(self.settings, self.history, model, _Channel(), session_id=session_id,
                                     roles={"main": AgentRole(None, (BASH_TOOL_NAME,))},
                                     user=self.user) as (runtime, _):
                if session_id == "new-agent-a":
                    self.assertFalse(self.settings.workspace_root.exists())
                runtime.submit_batch((Message("user", "test"),))
                await asyncio.wait_for(runtime.wait_idle(), 3)
            self.assertEqual(_results(model.requests[-1][0])["last"]["stdout"], output)
            self.assertEqual((self._workspace(self.user) / "fixture").read_text(), "marker")

    async def test_bash_calls_serialize_only_within_one_model_request(self):
        workspace = self._workspace(self.user)
        other_user = ChannelUser("synthetic/channel:user B")
        other_workspace = self._workspace(other_user)

        def answer(messages, names):
            last = messages[-1]
            if last.content == "begin":
                return (_call("main-bash", command="touch started; until [[ -e release ]]; do /bin/sleep 0.01; done; "
                                                   "printf main > shared", timeout=15),
                        _call("main-next", command="touch second-ran; test -f shared && printf second >> shared"),
                        _call("delegate", name=DELEGATE_TOOL_NAME, agent="worker", prompt="child"))
            if last.content == "child":
                return (_call("child-bash", command="until [[ -e started ]]; do /bin/sleep 0.01; done; "
                                                    "test ! -e release && printf child > child-ran"),)
            if last.content == "other user":
                return (_call("other-bash", command="printf other > shared"),)
            if last.async_result is not None and last.async_result.kind == "agent_reply":
                return (_call("after-reply", command="cat shared"),)
            return (TextDelta("done"),)

        model = _Model(answer)
        roles = {"main": AgentRole(None, (DELEGATE_TOOL_NAME, BASH_TOOL_NAME)),
                 "worker": AgentRole(None, (BASH_TOOL_NAME,), "fixture worker")}
        async with (running_agent(self.settings, self.history, model, _Channel(), session_id="main", roles=roles,
                                  user=self.user) as (runtime, _),
                    running_agent(self.settings, self.history, model, _Channel(), session_id="other", roles=roles,
                                  user=other_user) as (other, _)):
            runtime.submit_batch((Message("user", "begin"),))
            await _eventually(lambda: (workspace / "started").exists())
            await _eventually(lambda: any("child-bash" in _results(messages) for messages, _ in model.requests))
            self.assertEqual((workspace / "child-ran").read_text(), "child")
            other.submit_batch((Message("user", "other user"),))
            await asyncio.wait_for(other.wait_idle(), 3)
            self.assertEqual((other_workspace / "shared").read_text(), "other")
            self.assertFalse((workspace / "second-ran").exists())
            main_requests = [messages for messages, _ in model.requests
                             if any(message.kind == "user" and message.content == "begin" for message in messages)]
            self.assertEqual(len(main_requests), 1)  # 主 Agent 仍在等本请求的两个 Bash 结果。
            (workspace / "release").touch()
            await _eventually(lambda: any("after-reply" in _results(messages) for messages, _ in model.requests))
            await asyncio.wait_for(runtime.wait_idle(), 3)
        self.assertTrue((workspace / "second-ran").exists())
        self.assertEqual((workspace / "shared").read_text(), "mainsecond")
        self.assertEqual(_results(model.requests[-1][0])["after-reply"]["stdout"], "mainsecond")

    async def test_timer_wakes_runtime_and_can_start_bash(self):
        def answer(messages, names):
            if messages[-1].content == "schedule":
                return (_call("register", name=SET_TIMER_TOOL_NAME, interval_seconds=0.03),)
            if messages[-1].async_result is not None and messages[-1].async_result.kind == "timer_fired":
                return (_call("timer-bash", command="printf fired"),)
            return (TextDelta("done"),)

        model = _Model(answer)
        roles = {"main": AgentRole(None, (SET_TIMER_TOOL_NAME, BASH_TOOL_NAME))}
        async with running_agent(self.settings, self.history, model, _Channel(), session_id="main", roles=roles,
                                 user=self.user) as (runtime, _):
            runtime.submit_batch((Message("user", "schedule"),))
            await _eventually(lambda: any("timer-bash" in _results(messages) for messages, _ in model.requests))
            await asyncio.wait_for(runtime.wait_idle(), 3)
        self.assertEqual(_results(model.requests[-1][0])["timer-bash"]["stdout"], "fired")

    async def test_tree_synchronous_stop_rejects_queued_bash_without_cancelling_active_command(self):
        workspace = self._workspace(self.user)
        model = _Model(lambda messages, names: ())
        queued = asyncio.Event()
        processes = []
        with _record_processes(processes):
            async with running_agent(self.settings, self.history, model, _Channel(), session_id="stop-tree",
                                     roles={"main": AgentRole(None, (BASH_TOOL_NAME,))},
                                     user=self.user) as (runtime, tree):
                request_locks = runtime._tools.request_locks()

                async def execute(call_id, command):
                    call = _call(call_id, command=command, timeout=5).call
                    context = ToolContext(runtime.session_id, "run", call.id, True, workspace=runtime._workspace)
                    if call_id == "queued":
                        queued.set()
                    return json.loads(await runtime._tools.execute(call, context, request_locks=request_locks))

                active = asyncio.create_task(execute("active", ": > started; until [[ -e release ]]; do "
                                                    "/bin/sleep 0.01; done; printf active"))
                tasks = [active]
                try:
                    await _eventually((workspace / "started").exists)
                    waiting = asyncio.create_task(execute("queued", ": > queued-ran"))
                    tasks.append(waiting)
                    await asyncio.wait_for(queued.wait(), 2)
                    self.assertFalse(waiting.done())
                    tree.stop_accepting()  # 不进入异步 close，主动暴露锁释放后的窗口。
                    self.assertFalse(active.done())
                    (workspace / "release").touch()
                    result, rejected = await asyncio.wait_for(asyncio.gather(*tasks), 2)
                    self.assertEqual((result["exit_code"], result["stdout"]), (0, "active"))
                    self.assertEqual(rejected, {"error": "bash_closed"})
                    self.assertFalse((workspace / "queued-ran").exists())
                    self.assertEqual(len(_shells(processes)), 1)
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

    async def test_runtime_shutdown_cancels_and_reaps_active_bash(self):
        model = _Model(lambda messages, names: (_call("run-long", command="touch started; exec /bin/sleep 20"),
                                                 _call("queued", command="touch queued-ran")))
        processes = []
        with _record_processes(processes), patch("os.killpg", wraps=os.killpg) as killpg:
            async with running_agent(self.settings, self.history, model, _Channel(), session_id="main", user=self.user,
                                     roles={"main": AgentRole(None, (BASH_TOOL_NAME,))}) as (runtime, _):
                runtime.submit_batch((Message("user", "test"),))
                await _eventually(lambda: (self._workspace(self.user) / "started").exists())
            child, = _shells(processes)
            self.assertEqual(child.returncode, -signal.SIGKILL)
            self.assertEqual([call.args[1] for call in killpg.call_args_list if call.args[0] == child.pid and call.args[1]],
                             [signal.SIGKILL])
        self.assertFalse((self._workspace(self.user) / "queued-ran").exists())
        self.assertEqual(len(model.requests), 1)
