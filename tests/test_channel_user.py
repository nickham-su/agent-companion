from dataclasses import FrozenInstanceError
import unittest

from agent_companion.protocol import ChannelUser


class ChannelUserTests(unittest.TestCase):
    def test_channel_user_is_an_immutable_key_without_visible_repr(self):
        key = "fixture/user:用户"
        user = ChannelUser(key)
        self.assertEqual(user.key, key)
        self.assertEqual({user: "route"}[ChannelUser(key)], "route")
        self.assertNotEqual(user, ChannelUser("other"))
        self.assertNotIn(key, repr(user))
        with self.assertRaises(FrozenInstanceError):
            user.key = "other"
