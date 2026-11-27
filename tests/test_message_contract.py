"""领域消息、可见读侧和 Provider 协议之间的边界。"""

from contextlib import closing
import json
import unittest

import httpx

from agent_companion.models.bailian import Bailian
from agent_companion.models.deepseek import DeepSeek
from agent_companion.models.minimax import MiniMax
from agent_companion.models.openai_responses import OpenAIResponses
from agent_companion.protocol import (AsyncToolResult, Message, MessageDelivery, PrivateData, Reasoning,
                                      Text, ToolCall, UserSource)
from agent_companion.storage.sqlite import History, ReplyOutcome


class MessageContractTest(unittest.IsolatedAsyncioTestCase):
    def test_deliveries_cannot_supply_delegation_source_or_private_parts(self):
        private = PrivateData(bytes(16), b"fixture private")
        for forged in (Message("user", (Text("input", private),)),
                       Message("user", (Reasoning(private),)),
                       Message("user", (ToolCall("id", "lookup", "{}"),)),
                       Message("user", "input", source=UserSource("delegation")),
                       Message("assistant", "input"), Message("compaction_summary", "input")):
            with self.subTest(kind=forged.kind), self.assertRaisesRegex(ValueError, "invalid runtime"):
                MessageDelivery(forged, "trigger")
        with self.assertRaises(ValueError):
            MessageDelivery(Message("user", "input"), "invalid")
        self.assertEqual(MessageDelivery(Message("user", "[异步工具通知]"), "record").message.kind, "user")
        self.assertEqual(MessageDelivery(Message("state_event", "event"), "trigger").message.kind, "state_event")

    def test_history_keeps_tool_result_and_order_but_public_read_excludes_private_data(self):
        private = PrivateData(bytes(16), b"fixture-hidden-metadata")
        call = ToolCall("abc", "lookup", "{}", result="", private=private)
        message = Message("assistant", (Text("visible-before ", private), Reasoning(private), Text("visible-after"), call))
        with closing(History(":memory:")) as history:
            message_id, = history.finalize_reply(ReplyOutcome("a", "main", "a", "complete"), [message])
            self.assertEqual(history.context_snapshot("main")[1][0].message, message)
            visible = Message("assistant", (Text("visible-before "), Text("visible-after"), ToolCall("abc", "lookup", "{}", "")))
            self.assertEqual(history.tail("main", 1)[0].message, visible)
            self.assertEqual(history.search("main", keyword="visible-before")[0].message_id, message_id)
            self.assertEqual(history.search("main", keyword="fixture-hidden-metadata"), [])

    async def test_summary_notice_status_and_human_input_use_same_user_carrier_across_adapters(self):
        connected = Message("state_event", "通话已接通 - 2026-01-20 11:12:13+00:00")
        ended = Message("state_event", "通话已结束 - 2026-01-20 11:14:15+00:00")
        messages = (
            Message("compaction_summary", "old context"),
            Message("async_tool_result", async_result=AsyncToolResult("a_new_kind", {"result": "notice"})),
            connected,
            Message("user", "new prompt"),
            ended,
        )
        async with httpx.AsyncClient() as client:
            responses = OpenAIResponses(client, model="fixture", api_key="fixture", reasoning_effort="low")
            response_payload = responses._payload(messages, (), instructions="role instruction")
            self.assertEqual(response_payload["instructions"], "role instruction")
            reference = response_payload["input"]
            self.assertEqual([item["role"] for item in reference], ["user"] * 5)
            self.assertIn("old context", reference[0]["content"])
            self.assertEqual(json.loads(reference[1]["content"].partition("\n")[2]),
                             {"kind": "a_new_kind", "payload": {"result": "notice"}})
            self.assertEqual(reference[2]["content"], "[状态事件]\n" + connected.text)
            self.assertEqual(reference[4]["content"], "[状态事件]\n" + ended.text)
            for model in (DeepSeek(client), MiniMax(client, model="fixture", api_key="fixture"),
                          Bailian(client, model="fixture", api_key="fixture", base_url="https://example.invalid")):
                payload = model._payload(messages, (), thinking=False, instructions="role instruction")
                self.assertEqual(payload["messages"], [{"role": "system", "content": "role instruction"}, *reference])
