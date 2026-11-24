import asyncio
import ctypes
from errno import ECHILD, EIO, ENOSYS, ESRCH
import json
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agent_companion.runtime import _child_exit
from agent_companion.runtime import bash


class ChildExitTests(unittest.TestCase):
    def test_native_waitid_uses_nonreaping_flags(self):
        waitid = Mock(side_effect=[None, object()])
        native = SimpleNamespace(waitid=waitid, P_PID=1, WEXITED=2, WNOHANG=4, WNOWAIT=8)
        with patch.object(_child_exit, "os", native):
            self.assertFalse(_child_exit.child_exited(42))
            self.assertTrue(_child_exit.child_exited(42))
        self.assertEqual(waitid.call_count, 2)
        waitid.assert_called_with(1, 42, 14)

    def test_darwin_libsystem_waitid_when_python_does_not_expose_it(self):
        states = iter((0, 42, ECHILD))

        def waitid(idtype, pid, info, flags):
            self.assertEqual((idtype, pid, flags), (1, 42, 0x25))
            # Darwin siginfo_t：si_pid 位于前三个 int 之后，由内核填写。
            fields = ctypes.cast(info, ctypes.POINTER(ctypes.c_int))
            self.assertEqual(fields[3], 0)
            state = next(states)
            if state == ECHILD:
                ctypes.set_errno(ECHILD)
                return -1
            fields[3] = state
            return 0

        library = SimpleNamespace(waitid=Mock(side_effect=waitid))
        with (patch.object(_child_exit, "os", SimpleNamespace()),
              patch.object(_child_exit, "sys", SimpleNamespace(platform="darwin")),
              patch.object(_child_exit, "_darwin_waitid", _child_exit._darwin_waitid.__wrapped__),
              patch.object(ctypes, "CDLL", return_value=library) as load):
            self.assertFalse(_child_exit.child_exited(42))
            self.assertTrue(_child_exit.child_exited(42))
            with self.assertRaises(ChildProcessError) as error:
                _child_exit.child_exited(42)
        self.assertEqual(error.exception.errno, ECHILD)
        self.assertEqual(library.waitid.call_count, 3)
        load.assert_called_with(None, use_errno=True)

    def test_darwin_runtime_unavailable_python_waitid_uses_libsystem(self):
        native = SimpleNamespace(waitid=Mock(side_effect=NotImplementedError), P_PID=1,
                                 WEXITED=2, WNOHANG=4, WNOWAIT=8)
        darwin_waitid = Mock(return_value=0)
        with (patch.object(_child_exit, "os", native),
              patch.object(_child_exit, "sys", SimpleNamespace(platform="darwin")),
              patch.object(_child_exit, "_darwin_waitid", return_value=darwin_waitid)):
            self.assertFalse(_child_exit.child_exited(42))
        darwin_waitid.assert_called_once()


def _process(*, stdout=None):
    return SimpleNamespace(pid=42, stdout=stdout, stderr=None, returncode=None,
                           poll=Mock(side_effect=AssertionError("cannot poll the current child")))


class BashChildLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_observation_and_reaping_failures_preserve_signal_order(self):
        for observations, wait_errno, error, order in (
            ([EIO, False], None, "bash_io_failed", ["observe", "observe", "kill", "reap", "confirm"]),
            ([ENOSYS, ENOSYS], None, "bash_cleanup_unconfirmed", ["observe", "observe", "reap"]),
            ([ECHILD], None, "bash_cleanup_unconfirmed", ["observe"]),
            ([ESRCH], None, "bash_cleanup_unconfirmed", ["observe"]),
            ([True], ECHILD, "bash_cleanup_unconfirmed", ["observe", "kill", "reap"]),
        ):
            with self.subTest(observations=observations, wait_errno=wait_errno):
                actual = await self._check_failure_order(observations, wait_errno)
                self.assertEqual(actual, ({"error": error}, order, error == "bash_cleanup_unconfirmed"))

    async def _check_failure_order(self, observations, wait_errno):
        events = []
        process = _process()
        errors = iter(observations)

        def observe(pid):
            self.assertEqual(pid, process.pid)
            events.append("observe")
            result = next(errors)
            if isinstance(result, bool):
                return result
            raise OSError(result, "synthetic observation error")

        def signal_group(pid, sig):
            self.assertEqual(pid, process.pid)
            if sig == 0:
                events.append("confirm")
                raise ProcessLookupError
            self.assertEqual(sig, signal.SIGKILL)
            events.append("kill")

        def waitpid(pid, options):
            self.assertEqual((pid, options), (process.pid, bash.os.WNOHANG))
            events.append("reap")
            if wait_errno is not None:
                raise OSError(wait_errno, "synthetic reaping error")
            return pid, signal.SIGKILL

        with (TemporaryDirectory(dir=Path.cwd()) as folder,
              patch.object(bash.subprocess, "Popen", return_value=process),
              patch.object(bash, "child_exited", side_effect=observe),
              patch.object(bash, "_BLOCKED_WORKSPACES", set()) as blocked,
              patch.object(bash, "_UNREAPED_CHILDREN", set()),
              patch.object(bash._LOG, "warning"),
              patch.object(bash.os, "killpg", side_effect=signal_group),
              patch.object(bash.os, "kill") as kill,
              patch.object(bash.os, "waitpid", side_effect=waitpid)):
            workspace = Path(folder) / "workspace"
            result = await bash._run_command(":", workspace, 1)
            kill.assert_not_called()
            process.poll.assert_not_called()
            return json.loads(result), events, workspace in blocked

    async def test_probe_losing_ownership_neither_signals_nor_reports_a_snapshot(self):
        process = _process()
        with (patch.object(bash, "_probe_command", return_value=("fixture",)),
              patch.object(bash, "_UNREAPED_CHILDREN", set()),
              patch.object(bash.subprocess, "Popen", return_value=process),
              patch.object(bash, "child_exited", side_effect=ChildProcessError(ECHILD, "synthetic")) as observe,
              patch.object(bash.os, "killpg") as killpg,
              patch.object(bash.os, "waitpid") as waitpid):
            self.assertEqual(await bash._inspect_group(99, asyncio.get_running_loop().time() + 1), (False, None))
        observe.assert_called_once_with(process.pid)
        killpg.assert_not_called()
        waitpid.assert_not_called()
        process.poll.assert_not_called()

    async def _collect_with_clock(self, *, deadline, exits, reads, advances, observation_cost=0):
        now = 0
        responses, wakes = iter(exits), iter(advances)

        def observed(pid):
            nonlocal now
            now += observation_cost
            return next(responses)

        async def sleep(delay):
            nonlocal now
            now += next(wakes)

        loop = SimpleNamespace(time=lambda: now)
        clock = SimpleNamespace(get_running_loop=lambda: loop, sleep=sleep)
        process = _process(stdout=SimpleNamespace(fileno=lambda: 7))
        with (patch.object(bash, "asyncio", clock),
              patch.object(bash, "_UNREAPED_CHILDREN", set()),
              patch.object(bash.subprocess, "Popen", return_value=process),
              patch.object(bash, "child_exited", side_effect=observed) as observe,
              patch.object(bash.os, "set_blocking"),
              patch.object(bash.os, "read", side_effect=reads) as read,
              patch.object(bash.os, "killpg") as killpg):
            child = bash._Child(("fixture",), env={}, capture_stderr=True, output_limit=10)
            await child.collect(deadline)
        return child, observe, read, killpg

    async def test_delayed_wake_prefers_timeout_over_late_exit_and_output(self):
        child, observe, read, killpg = await self._collect_with_clock(
            deadline=0.015, exits=[False, True], reads=[BlockingIOError(), b"late"], advances=[0.03],
        )
        self.assertTrue(child.timed_out)
        observe.assert_called_once()
        read.assert_called_once()
        killpg.assert_not_called()  # collect 未确认退出；实际终止由收尾阶段负责。

    async def test_exit_observation_completing_at_deadline_is_too_late(self):
        child, _, read, killpg = await self._collect_with_clock(
            deadline=0.01, exits=[True], reads=[], advances=[], observation_cost=0.01,
        )
        self.assertTrue(child.timed_out)
        read.assert_not_called()
        killpg.assert_not_called()

    async def test_early_exit_allows_independent_but_bounded_pipe_drain(self):
        for reads, advances, drained in (([BlockingIOError(), b"tail", b""], [0.01, 0.01], True),
                                         ([BlockingIOError()], [0.3], False)):
            with self.subTest(drained=drained):
                child, observe, _, killpg = await self._collect_with_clock(
                    deadline=0.005, exits=[True], reads=reads, advances=advances,
                )
                self.assertFalse(child.timed_out)
                self.assertEqual(child.pipes_drained, drained)
                self.assertEqual(bytes(child.stdout), b"tail" if drained else b"")
                observe.assert_called_once()
                killpg.assert_called_once_with(child.process.pid, signal.SIGKILL)
