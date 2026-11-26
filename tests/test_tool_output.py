import asyncio
import os
from pathlib import Path
import stat
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

from agent_companion.protocol import ToolCall, ToolSpec
from agent_companion.runtime import _tool_output
from agent_companion.runtime.tools import Tool, ToolContext, Tools
from agent_companion.workspace import Workspace


_DAY = 24 * 60 * 60
_LARGE = "汉" * 3000 + "\n末尾"
_PREFIX = "汉" * 1365


class ToolOutputTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory(dir=Path.cwd())))
        self.workspace = Workspace(self.root / "workspace")
        self.cache = self.workspace.directory / ".agent" / "tool-output"
        self.enterContext(patch.object(_tool_output, "_LAST_CHECK", {}))

    async def _execute(self, result=_LARGE, *, workspace=True):
        async def handler(args, context):
            return result

        tool = Tool(ToolSpec("example", "", {}), handler)
        context = ToolContext("session", "run", "call", False, workspace=self.workspace if workspace else None)
        return await Tools([tool]).execute(ToolCall("call", "example", "{}"), context)

    def _archive_path(self, output):
        relative = output.partition("截断前的工具结果：")[2].removesuffix("]")
        self.assertTrue(relative.startswith(".agent/tool-output/"))
        self.assertFalse(Path(relative).is_absolute())
        return self.workspace.directory / relative

    async def test_short_and_exact_threshold_are_unchanged_without_archive(self):
        for output in ("汉字", "汉" * 2730 + "ab", "a" * 8192):
            self.assertEqual(await self._execute(output), output)
        self.assertFalse(self.workspace.directory.exists())
        at_threshold = "汉" * 2730 + "abc"
        self.assertEqual(self._archive_path(await self._execute(at_threshold)).read_text(), at_threshold)

    async def test_over_limit_returns_utf8_prefix_and_complete_archive(self):
        output = await self._execute()
        prefix, _, note = output.partition("\n\n[")
        self.assertEqual(prefix, _PREFIX)
        self.assertEqual(len(prefix.encode("utf-8")), 4095)
        self.assertLessEqual(len(output.encode("utf-8")), 8192)
        self.assertIn("公共工具层已截断", note)
        path = self._archive_path(output)
        self.assertEqual(path.read_bytes(), _LARGE.encode("utf-8"))
        self.assertEqual(path.stat().st_mode & 0o077, 0)
        chars = "x" * 8193
        ascii_output = await self._execute(chars)
        self.assertEqual(ascii_output.partition("\n\n")[0], "x" * 4096)
        self.assertEqual(self._archive_path(ascii_output).read_text(), chars)

    async def test_no_workspace_or_write_failure_returns_preview_without_path(self):
        for options, context, note in (({"workspace": False}, None, "没有工作区，未归档"),
                                       ({}, OSError("synthetic detail"), "无法写入归档")):
            with patch.object(_tool_output, "_archive", side_effect=context) as archive:
                output = await self._execute(**options)
                self.assertTrue(output.startswith(_PREFIX + "\n\n"))
                self.assertIn(note, output)
                self.assertNotIn("synthetic detail", output)
                self.assertNotIn(".agent/tool-output/", output)
                if context is None:
                    archive.assert_not_called()

    async def test_partial_write_is_not_published_and_does_not_trigger_cleanup(self):
        with (patch.object(_tool_output.os, "fdopen", side_effect=_failing_writer()),
              patch.object(_tool_output, "_clean_expired") as clean):
            self.assertIn("无法写入归档", await self._execute())
            self.assertEqual(list(self.cache.iterdir()), [])
            clean.assert_not_called()

    async def test_links_and_non_directory_fail_without_writing_elsewhere(self):
        outside = self.root / "outside"
        outside.mkdir()
        self.workspace.directory.symlink_to(outside, target_is_directory=True)
        self.assertIn("无法写入归档", await self._execute())
        self.workspace.directory.unlink()
        self.workspace.directory.mkdir()
        agent = self.cache.parent
        agent.symlink_to(outside, target_is_directory=True)
        self.assertIn("无法写入归档", await self._execute())
        agent.unlink()
        agent.mkdir()
        self.cache.symlink_to(outside, target_is_directory=True)
        self.assertIn("无法写入归档", await self._execute())
        self.cache.unlink()
        self.cache.touch()
        self.assertIn("无法写入归档", await self._execute())
        self.assertEqual(list(outside.iterdir()), [])

    async def test_workspace_ancestor_symlink_dotdot_matches_bash_path_resolution(self):
        target = self.root / "target"
        (target / "child").mkdir(parents=True)
        (self.root / "alias").symlink_to(target / "child", target_is_directory=True)
        self.workspace = Workspace(self.root / "alias" / ".." / "work")
        real_workspace = self.workspace.ensure_directory()
        self.assertEqual(real_workspace, target.resolve() / "work")
        archive = self._archive_path(await self._execute())
        self.assertEqual(archive.resolve().parents[2], real_workspace)
        self.assertFalse((self.root / "work").exists())

    async def test_workspace_link_swapped_during_resolution_is_not_used(self):
        workspace = self.workspace.directory
        workspace.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        sentinel = outside / "unchanged"
        sentinel.touch()
        resolve = Path.resolve

        def replace_leaf(path, *args, **kwargs):
            if path in (workspace, workspace.parent) and not workspace.is_symlink():
                workspace.rename(self.root / "moved")
                workspace.symlink_to(outside, target_is_directory=True)
            return resolve(path, *args, **kwargs)

        with (patch.object(Path, "resolve", replace_leaf), patch.object(_tool_output, "_clean_expired") as clean):
            self.assertIn("无法写入归档", await self._execute())
            clean.assert_not_called()
        self.assertTrue(workspace.is_symlink())
        self.assertEqual(list(outside.iterdir()), [sentinel])

    async def test_acl_check_precedes_any_archive_body_write(self):
        checked_dirs = set()
        checked_files = []

        def reject_inherited_file_acl(fd):
            info = os.fstat(fd)
            if stat.S_ISDIR(info.st_mode):
                checked_dirs.add((info.st_dev, info.st_ino))
            else:
                checked_files.append(info.st_size)
                raise OSError("synthetic unsafe file ACL")

        with (patch.object(_tool_output, "require_no_acl_grants", side_effect=reject_inherited_file_acl),
              patch.object(_tool_output, "_clean_expired") as clean):
            output = await self._execute()
            clean.assert_not_called()
        self.assertIn("无法写入归档", output)
        self.assertNotIn(".agent/tool-output/", output)
        self.assertEqual(checked_files, [0])
        for directory in (self.workspace.directory, self.cache.parent, self.cache):
            info = directory.stat()
            self.assertIn((info.st_dev, info.st_ino), checked_dirs)
        self.assertEqual(list(self.cache.iterdir()), [])

    async def test_uninspectable_cache_acl_prevents_writes_and_cleanup(self):
        self.cache.mkdir(parents=True)
        info = self.cache.stat()
        keep = self.cache / "existing"
        keep.touch()

        def deny_cache(fd):
            if os.path.samestat(os.fstat(fd), info):
                raise OSError("synthetic ACL query failure")

        with (patch.object(_tool_output, "require_no_acl_grants", side_effect=deny_cache),
              patch.object(_tool_output, "_clean_expired") as clean):
            self.assertIn("无法写入归档", await self._execute())
            clean.assert_not_called()
        self.assertEqual(list(self.cache.iterdir()), [keep])

    async def test_untrusted_cache_or_replaceable_parent_is_not_used(self):
        self.cache.mkdir(parents=True)
        for directory in (self.cache, self.cache.parent, self.workspace.directory):
            original = directory.stat().st_mode & 0o777
            try:
                directory.chmod(0o775)
                with patch.object(_tool_output, "_clean_expired") as clean:
                    self.assertIn("无法写入归档", await self._execute())
                    self.assertEqual(list(self.cache.iterdir()), [])
                    clean.assert_not_called()
            finally:
                directory.chmod(original)

    async def test_cache_owned_by_another_account_is_not_used(self):
        self.cache.mkdir(parents=True)
        cache_info = self.cache.stat()
        fstat = os.fstat

        def foreign_cache_owner(fd):
            info = fstat(fd)
            if (info.st_dev, info.st_ino) == (cache_info.st_dev, cache_info.st_ino):
                return SimpleNamespace(st_mode=info.st_mode, st_uid=os.geteuid() + 1)
            return info

        with patch.object(_tool_output.os, "fstat", side_effect=foreign_cache_owner):
            self.assertIn("无法写入归档", await self._execute())
        self.assertEqual(list(self.cache.iterdir()), [])

    async def test_replaced_cache_is_not_advertised(self):
        moved = self.root / "moved"

        def replace_cache(*args, **kwargs):
            self.cache.rename(moved)
            self.cache.mkdir()

        with patch.object(_tool_output, "_clean_expired", side_effect=replace_cache):
            output = await self._execute()
        self.assertIn("无法写入归档", output)
        self.assertNotIn(".agent/tool-output/", output)
        self.assertEqual(list(moved.iterdir()), [])
        self.assertEqual(list(self.cache.iterdir()), [])

    async def test_source_record_write_failure_rolls_back_new_archive(self):
        writer = _tool_output._write_private_file

        def fail_record(fd, name, raw):
            if name.endswith(".meta"):
                raise OSError("synthetic record write failure")
            return writer(fd, name, raw)

        with (patch.object(_tool_output, "_write_private_file", side_effect=fail_record),
              patch.object(_tool_output, "_clean_expired") as clean):
            output = await self._execute()
            self.assertIn("无法写入归档", output)
            self.assertNotIn(".agent/tool-output/", output)
            self.assertEqual(list(self.cache.iterdir()), [])
            clean.assert_not_called()

    async def test_persistent_record_removes_only_original_unchanged_archive_after_restart(self):
        original, replaced = [self._archive_path(await self._execute()) for _ in range(2)]
        record = original.with_suffix(".txt.meta")
        self.assertEqual(record.stat().st_mode & 0o077, 0)
        self.assertIn(original.name.encode(), record.read_bytes())
        self.assertNotIn(_LARGE.encode(), record.read_bytes())
        replaced_record = replaced.with_suffix(".txt.meta")
        replacement = self.cache / "new-result"
        replacement.write_bytes(_LARGE.encode())  # 内容相同，身份不同。
        replacement.chmod(0o600)
        replacement.replace(replaced)
        self.assertTrue(replaced_record.exists())
        # 模拟进程重启和归档经过 7 天，不修改原文件身份。
        now = max(original.stat().st_mtime, replaced.stat().st_mtime) + 8 * _DAY
        with patch.object(_tool_output, "_LAST_CHECK", {}), patch.object(_tool_output, "time", return_value=now):
            self.assertTrue(self._archive_path(await self._execute()).exists())
        self.assertFalse(original.exists())
        self.assertFalse(record.exists())
        self.assertTrue(replaced.exists())
        self.assertTrue(replaced_record.exists())

    async def test_cleanup_only_expired_generated_files_and_concurrent_calls_check_once(self):
        self.cache.mkdir(parents=True)
        now = 10 * _DAY

        def aged(file, age):
            os.utime(file, (now - age, now - age), follow_symlinks=False)
            return file

        def generated_name():
            return self.cache / f"result-{uuid4().hex}.txt"

        expired, boundary, unknown, partial, orphan = (generated_name() for _ in range(5))
        unrelated = self.cache / "notes.txt"
        for file, age in ((expired, 7 * _DAY + 1), (boundary, 7 * _DAY), (unknown, 9 * _DAY),
                          (partial, 9 * _DAY), (unrelated, 9 * _DAY)):
            file.write_text("synthetic", encoding="utf-8")
            file.chmod(0o600)
            aged(file, age)
        for file in (expired, boundary, partial):
            record = file.with_suffix(".txt.meta")
            data = _tool_output._record_bytes(file.name, file.stat())
            record.write_bytes(data[:-1] if file == partial else data)
            record.chmod(0o600)
        orphan_record = orphan.with_suffix(".txt.meta")
        orphan_record.write_bytes(b"synthetic orphan")
        orphan_record.chmod(0o600)
        aged(orphan_record, 9 * _DAY)
        link, directory, hard_link = generated_name(), generated_name(), generated_name()
        outside = self.root / "outside"
        outside.write_text("synthetic target", encoding="utf-8")
        link.symlink_to(outside)
        aged(link, 9 * _DAY)
        directory.mkdir()
        aged(directory, 9 * _DAY)
        os.link(outside, hard_link)
        aged(hard_link, 9 * _DAY)
        with (patch.object(_tool_output, "monotonic", return_value=0) as clock,
              patch.object(_tool_output, "time", return_value=now) as wall,
              patch.object(_tool_output, "_clean_expired", wraps=_tool_output._clean_expired) as clean):
            outputs = await asyncio.gather(*(self._execute() for _ in range(8)))
            paths = {self._archive_path(output) for output in outputs}
            self.assertEqual(len(paths), 8)
            self.assertTrue(all(path.read_bytes() == _LARGE.encode() for path in paths))
            clean.assert_called_once()
            self.assertFalse(expired.exists())
            self.assertFalse(expired.with_suffix(".txt.meta").exists())
            for kept in (boundary, unrelated, unknown, partial, orphan_record, link, directory, outside, hard_link):
                self.assertTrue(kept.exists())
            clock.return_value = _DAY - 1
            wall.return_value = now + _DAY
            await self._execute()
            clean.assert_called_once()
            self.assertTrue(boundary.exists())
            clock.return_value = _DAY
            await self._execute()
            self.assertEqual(clean.call_count, 2)
            self.assertFalse(boundary.exists())
            self.workspace = Workspace(self.root / "another-workspace")
            self.assertTrue(self._archive_path(await self._execute()).exists())
            self.assertEqual(clean.call_count, 3)

    async def test_cleanup_failure_does_not_discard_successful_archive(self):
        with patch.object(_tool_output, "_clean_expired", side_effect=OSError("synthetic detail")) as clean:
            self.assertEqual(self._archive_path(await self._execute()).read_text(), _LARGE)
            self.assertEqual(self._archive_path(await self._execute()).read_text(), _LARGE)
            clean.assert_called_once()


def _failing_writer():
    open_fd = os.fdopen

    class Writer:
        def __init__(self, fd, mode):
            self.file = open_fd(fd, mode)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.file.close()

        def write(self, raw):
            self.file.write(raw[:10])
            raise OSError("synthetic disk write failure")

    return Writer
