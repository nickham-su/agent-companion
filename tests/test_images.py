from contextlib import closing
import os
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from agent_companion import images
from agent_companion.images import MAX_IMAGE_BYTES, UNAVAILABLE_IMAGE, image_request_bytes, prepare_images, save_image
from agent_companion.protocol import Image, Message, Text
from agent_companion.storage.sqlite import History
from agent_companion.workspace import Workspace
from test_message_support import PNG_BYTES, PNG_URL, image_workspace


class ImageFilesTest(TestCase):
    def setUp(self):
        self.workspace = image_workspace(self)
        self.saved = save_image(self.workspace, PNG_BYTES)
        self.file = self.workspace.directory / self.saved.url

    def test_only_local_relative_references_are_saved_to_history(self):
        message = Message("user", (Text("前文"), self.saved, Image(PNG_URL), Text("后文")))
        with closing(History(":memory:")) as history:
            history.append("main", [message])
            row, = history.read_after("main")
            self.assertEqual(row.message.parts, (Text("前文"), self.saved, UNAVAILABLE_IMAGE, Text("后文")))
            raw, = history._db.execute("SELECT parts FROM messages").fetchone()
            self.assertIn(self.saved.url, raw)
            self.assertNotIn("base64", raw)

    def test_load_rejects_invalid_paths_and_isolated_users_without_removing_text(self):
        invalid = (Image(str(self.file)), Image("../" + self.saved.url),
                   Image(".agent/images/../" + self.file.name), Image(PNG_URL), Image("https://example.invalid/a"))
        input_message = Message("user", (Text("前文"), *invalid, self.saved, Text("后文")))
        actual, = prepare_images([input_message], self.workspace)
        self.assertEqual(actual.parts, (Text("前文"), *(UNAVAILABLE_IMAGE for _ in invalid), Image(PNG_URL), Text("后文")))
        self.assertEqual(input_message.parts[-2], self.saved)
        other = Workspace(self.workspace.directory.with_name("other"))
        other.directory.mkdir()
        actual, = prepare_images([Message("user", (self.saved,))], other)
        self.assertEqual(actual.parts, (UNAVAILABLE_IMAGE,))
        self.assertFalse((other.directory / ".agent").exists())

    def test_read_and_write_reject_directory_symlinks(self):
        outside = self.workspace.directory.with_name("outside")
        outside.mkdir()
        for prefix in ("", ".agent", ".agent/images"):
            with self.subTest(prefix=prefix):
                owner = Workspace(outside / f"case-{len(prefix)}")
                directory = owner.directory / prefix
                suffix = ".agent/images"[len(prefix):].lstrip("/")
                directory.parent.mkdir(parents=True, exist_ok=True)
                target = outside / f"target-{len(prefix)}"
                (target / suffix).mkdir(parents=True)
                picture = target / suffix / self.file.name
                picture.write_bytes(PNG_BYTES)
                directory.symlink_to(target, target_is_directory=True)
                result, = prepare_images([Message("user", (self.saved,))], owner)
                self.assertEqual(result.parts, (UNAVAILABLE_IMAGE,))
                with self.assertRaises(OSError):
                    save_image(owner, PNG_BYTES)
                self.assertEqual(list((target / suffix).iterdir()), [picture])

    def test_read_rejects_symlink_hardlink_fifo_missing_and_invalid_or_oversize_data(self):
        outside = self.workspace.directory.with_name("outside")
        outside.write_bytes(PNG_BYTES)
        for mode in ("symlink", "hardlink", "fifo", "missing", "invalid", "oversize"):
            with self.subTest(mode=mode):
                self.file.unlink(missing_ok=True)
                if mode == "symlink":
                    self.file.symlink_to(outside)
                elif mode == "hardlink":
                    self.file.hardlink_to(outside)
                elif mode == "fifo":
                    os.mkfifo(self.file)
                elif mode == "invalid":
                    self.file.write_bytes(PNG_BYTES[1:])
                elif mode == "oversize":
                    with self.file.open("wb") as output:
                        output.truncate(MAX_IMAGE_BYTES + 1)
                result, = prepare_images([Message("user", (Text("前"), self.saved, Text("后")))], self.workspace)
                self.assertEqual(result.parts, (Text("前"), UNAVAILABLE_IMAGE, Text("后")))

    def test_save_never_overwrites_or_follows_an_existing_file(self):
        outside = self.workspace.directory.with_name("outside")
        outside.write_bytes(b"untouched")
        self.file.unlink()
        self.file.symlink_to(outside)
        with patch.object(images, "uuid4", return_value=SimpleNamespace(hex=self.file.name)):
            with self.assertRaises(FileExistsError):
                save_image(self.workspace, PNG_BYTES)
        self.assertTrue(self.file.is_symlink())
        self.assertEqual(outside.read_bytes(), b"untouched")

    def test_read_rechecks_regular_file_if_directory_entry_changes_before_open(self):
        original = os.open

        def open_after_replacement(path, flags, *args, **kwargs):
            if path == self.file.name:
                self.assertTrue(flags & os.O_NONBLOCK)
                self.file.unlink()
                os.mkfifo(self.file)
            return original(path, flags, *args, **kwargs)

        with patch.object(images.os, "open", side_effect=open_after_replacement):
            result, = prepare_images([Message("user", (self.saved,))], self.workspace)
        self.assertEqual(result.parts, (UNAVAILABLE_IMAGE,))

    def test_request_budget_uses_data_url_bytes_for_all_selected_messages(self):
        second = save_image(self.workspace, PNG_BYTES)
        self.assertGreater(image_request_bytes(len(PNG_BYTES), "image/png"), len(self.saved.url))
        with patch.object(images, "MAX_MEDIA_BYTES", image_request_bytes(len(PNG_BYTES), "image/png")):
            first, last = prepare_images([Message("user", (self.saved,)),
                                          Message("user", (Text("仍保留"), second))], self.workspace)
        self.assertEqual(first.parts, (Image(PNG_URL),))
        self.assertEqual(last.parts, (Text("仍保留"), UNAVAILABLE_IMAGE))
