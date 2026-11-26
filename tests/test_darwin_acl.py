import ctypes
from errno import EINVAL, EIO, ENOENT, ENOTSUP
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agent_companion.runtime import _darwin_acl


class DarwinACLTests(unittest.TestCase):
    def check(self, *tags, get_error=None, entry_error=EINVAL):
        entry_tags = iter(tags)
        selectors = []

        def get_acl(fd):
            self.assertEqual(fd, 42)
            if get_error is not None:
                ctypes.set_errno(get_error)
                return None
            return 100

        def get_entry(acl, selector, entry):
            self.assertEqual(acl, 100)
            selectors.append(selector)
            tag = next(entry_tags, None)
            if tag is None:
                ctypes.set_errno(entry_error)
                return -1
            ctypes.cast(entry, ctypes.POINTER(ctypes.c_void_p))[0] = tag
            return 0

        def get_tag(entry, tag):
            ctypes.cast(tag, ctypes.POINTER(ctypes.c_int))[0] = entry.value
            return 0

        api = SimpleNamespace(acl_get_fd=get_acl, acl_valid=lambda acl: 0, acl_get_entry=get_entry,
                              acl_get_tag_type=get_tag, acl_free=Mock())
        with (patch.object(_darwin_acl, "sys", SimpleNamespace(platform="darwin")),
              patch.object(_darwin_acl, "_acl_api", return_value=api)):
            try:
                _darwin_acl.require_no_acl_grants(42)
            finally:
                if get_error is None:
                    api.acl_free.assert_called_once_with(100)
                else:
                    api.acl_free.assert_not_called()
        return selectors

    def test_no_acl_or_empty_acl(self):
        self.assertEqual(self.check(get_error=ENOENT), [])
        self.assertEqual(self.check(), [0])

    def test_deny_only_acl_does_not_grant_access(self):
        self.assertEqual(self.check(2, 2), [0, -1, -1])

    def test_allow_or_unknown_tag_is_rejected(self):
        for tag in (1, 7):
            with self.subTest(tag=tag), self.assertRaises(OSError):
                self.check(2, tag)

    def test_query_or_iteration_failure_is_rejected(self):
        for error in (EIO, ENOTSUP, 0):
            with self.subTest(error=error), self.assertRaises(OSError):
                self.check(get_error=error)
        with self.assertRaises(OSError):
            self.check(2, entry_error=EIO)

    def test_other_platforms_do_not_load_darwin_library(self):
        with (patch.object(_darwin_acl, "sys", SimpleNamespace(platform="linux")),
              patch.object(_darwin_acl, "_acl_api") as load):
            _darwin_acl.require_no_acl_grants(42)
        load.assert_not_called()
