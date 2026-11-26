import asyncio
from contextlib import closing
from dataclasses import dataclass
import json
import unittest
from unittest.mock import patch

from agent_companion.app import running_agent
from agent_companion.config import Settings
from agent_companion.protocol import (DeliveryStatus, InternalInput, Message, Receipt, TIMER_EVENT_PREFIX,
                                      TextDelta, ToolCall, ToolReady)
from agent_companion.runtime.subtasks import demo_roles
from agent_companion.runtime.timers import CANCEL_TIMER_TOOL_NAME, SET_TIMER_TOOL_NAME, Timers
from agent_companion.runtime.tools import ToolContext, Tools
from agent_companion.session_scheduler import SessionScheduler, SessionSchedulers
from agent_companion.storage.sqlite import History


@dataclass
class _Handle:
    when: float
    callback: object
    args: tuple
    cancelled: bool = False

    def cancel(self):
        self.cancelled = True


class _Clock:
    def __init__(self):
        self.now = 0.0
        self.handles = []

    def call_later(self, delay, callback, *args):
        handle = _Handle(self.now + delay, callback, args)
        self.handles.append(handle)
        return handle

    def advance(self, seconds):
        self.now += seconds
        due = sorted((h for h in self.handles if h.when <= self.now), key=lambda h: h.when)
        self.handles = [h for h in self.handles if h.when > self.now]
        for handle in due:
            if not handle.cancelled:
                handle.callback(*handle.args)


def _event(item):
    if isinstance(item, InternalInput):
        item = item.to_model_message()
    assert item.role == "user" and item.content.startswith(TIMER_EVENT_PREFIX)
    return json.loads(item.content.removeprefix(TIMER_EVENT_PREFIX))


async def _execute(tools, name, args, *, session="main", has_user_input=True):
    call = ToolCall("call", name, args if isinstance(args, str) else json.dumps(args))
    return json.loads(await tools.execute(call, ToolContext(session, "run", call.id, has_user_input)))


class _Model:
    def __init__(self, answer):
        self.answer = answer
        self.requests = []

    async def stream(self, messages, tools, *, thinking):
        self.requests.append(tuple(messages))
        for event in self.answer(len(self.requests), messages):
            yield event


class _Channel:
    def __init__(self):
        self.replies = []

    async def deliver(self, reply):
        text = "".join([part async for part in reply])
        self.replies.append((reply.info.run_id, text))
        return Receipt(reply.info.reply_id, DeliveryStatus.FULL, text)


def _register_then_finish(step, messages):
    if step == 1:
        return (ToolReady(ToolCall("register", SET_TIMER_TOOL_NAME,
                                   '{"interval_seconds":2,"mode":"repeat","max_fires":3}')),)
    return (TextDelta("synthetic response"),)


class TimerTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.clock = _Clock()
        clock_patch = patch("agent_companion.runtime.timers.get_running_loop", return_value=self.clock)
        clock_patch.start()
        self.addCleanup(clock_patch.stop)

    def make_tools(self, session="main", *, close_on_unbind=True):
        batches = []
        scheduler = SessionScheduler()
        timers = Timers(session, scheduler.submit_internal)
        self.addCleanup(timers.close)
        revoke = scheduler.bind(batches.append, on_revoke=timers.close if close_on_unbind else None)
        return Tools(timers.tools), scheduler, revoke, batches

    async def test_single_and_repeat_share_one_fired_envelope_with_terminal_state(self):
        tools, _, _, batches = self.make_tools()
        single = await _execute(tools, SET_TIMER_TOOL_NAME, {"interval_seconds": 2.5})
        repeat = await _execute(tools, SET_TIMER_TOOL_NAME,
                                {"mode": "repeat", "interval_seconds": 3, "max_fires": 3, "message": ""})
        self.assertEqual([single["status"], repeat["status"]], ["accepted", "accepted"])
        self.assertRegex(single["timer_id"], r"^t-[0-9a-f]{10}-1$")
        self.assertEqual(repeat["timer_id"], single["timer_id"].rsplit("-", 1)[0] + "-2")
        self.clock.advance(2)
        self.assertEqual(batches, [])
        self.clock.advance(0.5)
        self.assertEqual([_event(item) for (item,) in batches], [{
            "event": "timer_fired", "timer_id": single["timer_id"], "fire_count": 1, "max_fires": 1,
            "status": "completed", "description": "单次定时器已到期并完成，不再触发。",
        }])
        self.clock.advance(100)  # 事件循环恢复时不补发积压的触发。
        self.assertEqual(len(batches), 2)
        self.clock.advance(3)
        self.clock.advance(3)
        self.clock.advance(100)
        self.assertEqual([_event(item) for (item,) in batches[1:]], [
            {"event": "timer_fired", "timer_id": repeat["timer_id"], "fire_count": index, "max_fires": 3,
             "message": "", "status": status, "description": description}
            for index, status, description in (
                (1, "active", "定时器已触发（第 1/3 次），还会继续触发。"),
                (2, "active", "定时器已触发（第 2/3 次），还会继续触发。"),
                (3, "completed", "定时器已触发（第 3/3 次），已达到最大次数，不再触发。"),
            )
        ])
        for result in (single, repeat):
            self.assertEqual(await _execute(tools, CANCEL_TIMER_TOOL_NAME, {"timer_id": result["timer_id"]}),
                             {"status": "not_active"})

    async def test_ids_are_not_reused_after_cancellation_expiry_or_rebind(self):
        scheduler = SessionScheduler()
        batches = []
        with patch("agent_companion.runtime.timers.token_hex", side_effect=("a1b2c3d4e5", "0a1b2c3d4e")):
            first = Timers("main", scheduler.submit_internal)
            self.addCleanup(first.close)
            revoke = scheduler.bind(batches.append, on_revoke=first.close)
            tools = Tools(first.tools)
            initial = await _execute(tools, SET_TIMER_TOOL_NAME, {"interval_seconds": 1})
            self.assertEqual(await _execute(tools, CANCEL_TIMER_TOOL_NAME, {"timer_id": initial["timer_id"]}),
                             {"status": "cancelled"})
            expires = await _execute(tools, SET_TIMER_TOOL_NAME, {"interval_seconds": 1})
            self.clock.advance(1)
            self.assertEqual([_event(item)["timer_id"] for (item,) in batches], [expires["timer_id"]])
            old = await _execute(tools, SET_TIMER_TOOL_NAME, {"interval_seconds": 1})
            self.assertEqual([item["timer_id"] for item in (initial, expires, old)],
                             [f"t-a1b2c3d4e5-{count}" for count in range(1, 4)])
            revoke()

            rebound = Timers("main", scheduler.submit_internal)
            self.addCleanup(rebound.close)
            scheduler.bind(batches.append, on_revoke=rebound.close)
            tools = Tools(rebound.tools)
            fresh = await _execute(tools, SET_TIMER_TOOL_NAME, {"interval_seconds": 1})
            self.assertEqual(fresh["timer_id"], "t-0a1b2c3d4e-1")
            for item in (initial, expires, old):
                self.assertEqual(await _execute(tools, CANCEL_TIMER_TOOL_NAME, {"timer_id": item["timer_id"]}),
                                 {"status": "not_active"})
            self.clock.advance(1)
            self.assertEqual([_event(item)["timer_id"] for (item,) in batches],
                             [expires["timer_id"], fresh["timer_id"]])

    async def test_invalid_arguments_do_not_register_timers(self):
        tools, _, _, _ = self.make_tools()
        invalid_sets = [{}, {"interval_seconds": 1, "extra": "synthetic"},
                        *({"interval_seconds": value} for value in (True, "1", 0, -1, None, 10**400)),
                        '{"interval_seconds":1e309}', '{"interval_seconds":NaN}',
                        *({"interval_seconds": 1, "mode": value} for value in (None, [], "Once", "loop")),
                        {"interval_seconds": 1, "mode": "once", "max_fires": 1},
                        {"interval_seconds": 1, "max_fires": None},
                        {"interval_seconds": 1, "mode": "repeat"},
                        {"interval_seconds": 0.999, "mode": "repeat", "max_fires": 1},
                        *({"interval_seconds": 1, "mode": "repeat", "max_fires": value}
                          for value in (True, 0, -1, 1.0, "1", None, 101)),
                        *({"interval_seconds": 1, "message": value} for value in (None, 123, []))]
        for args in invalid_sets:
            with self.subTest(args=args):
                self.assertEqual(await _execute(tools, SET_TIMER_TOOL_NAME, args), {"error": "invalid_arguments"})
        for args in ({}, {"timer_id": True}, {"timer_id": ""}, {"timer_id": " "}, {"timer_id": "x", "extra": 1}):
            self.assertEqual(await _execute(tools, CANCEL_TIMER_TOOL_NAME, args), {"error": "invalid_arguments"})
        self.assertEqual(self.clock.handles, [])

    async def test_repeat_accepts_exact_interval_and_count_bounds(self):
        tools, _, _, batches = self.make_tools()
        for count in (1, 100):
            result = await _execute(tools, SET_TIMER_TOOL_NAME,
                                    {"interval_seconds": 1, "mode": "repeat", "max_fires": count})
            self.assertEqual(result["status"], "accepted")
        self.clock.advance(1)
        events = [_event(item) for (item,) in batches]
        self.assertEqual([(event["max_fires"], event["status"]) for event in events],
                         [(1, "completed"), (100, "active")])

    async def test_capacity_is_shared_by_modes_and_released_on_cancel_or_completion(self):
        tools, _, _, batches = self.make_tools()
        repeat = {"interval_seconds": 1, "mode": "repeat", "max_fires": 2}
        single = {"interval_seconds": 0.25}
        registrations = [await _execute(tools, SET_TIMER_TOOL_NAME, args) for args in (single, repeat) * 4]
        self.assertTrue(all(result["status"] == "accepted" for result in registrations))
        for args in (single, repeat):
            self.assertEqual(await _execute(tools, SET_TIMER_TOOL_NAME, args), {"error": "timer_capacity_reached"})
        self.assertEqual(await _execute(tools, SET_TIMER_TOOL_NAME, {**repeat, "max_fires": 101}),
                         {"error": "invalid_arguments"})

        self.assertEqual(await _execute(tools, CANCEL_TIMER_TOOL_NAME, {"timer_id": registrations[1]["timer_id"]}),
                         {"status": "cancelled"})
        replacement = await _execute(tools, SET_TIMER_TOOL_NAME, repeat)
        self.assertEqual(replacement["status"], "accepted")
        self.assertEqual(await _execute(tools, SET_TIMER_TOOL_NAME, single), {"error": "timer_capacity_reached"})

        self.clock.advance(0.25)
        self.assertEqual([(_event(item)["timer_id"], _event(item)["status"]) for (item,) in batches],
                         [(result["timer_id"], "completed") for result in registrations[::2]])
        for _ in range(4):
            self.assertEqual((await _execute(tools, SET_TIMER_TOOL_NAME, single))["status"], "accepted")
        self.assertEqual(await _execute(tools, SET_TIMER_TOOL_NAME, single), {"error": "timer_capacity_reached"})

    async def test_cancel_only_own_timer_and_keep_already_buffered_event(self):
        tools, scheduler, _, batches = self.make_tools()
        other_tools, _, _, other_batches = self.make_tools("other")
        result = await _execute(tools, SET_TIMER_TOOL_NAME,
                                {"interval_seconds": 1, "mode": "repeat", "max_fires": 3, "message": "合成内容"})
        args = {"timer_id": result["timer_id"]}
        scheduler.session_input(input_active=True)
        self.clock.advance(1)
        self.assertFalse(batches)
        self.assertEqual(await _execute(other_tools, CANCEL_TIMER_TOOL_NAME, args, session="other"),
                         {"status": "not_active"})
        self.assertEqual(await _execute(tools, CANCEL_TIMER_TOOL_NAME, args, session="other"),
                         {"status": "not_active"})
        self.assertEqual(await _execute(tools, CANCEL_TIMER_TOOL_NAME, args, has_user_input=False),
                         {"status": "cancelled"})
        self.assertEqual(await _execute(tools, CANCEL_TIMER_TOOL_NAME, args), {"status": "not_active"})
        self.clock.advance(10)
        scheduler.session_input(input_active=False)
        (item,), = batches
        self.assertEqual((_event(item)["fire_count"], _event(item)["message"]), (1, "合成内容"))
        self.assertFalse(other_batches)

    async def test_scheduler_rejection_stops_repeat_without_retry(self):
        tools, scheduler, revoke, batches = self.make_tools(close_on_unbind=False)
        result = await _execute(tools, SET_TIMER_TOOL_NAME, {"interval_seconds": 2, "mode": "repeat", "max_fires": 3})
        revoke()
        self.clock.advance(2)
        scheduler.bind(batches.append)
        self.clock.advance(10)
        self.assertEqual(self.clock.handles, [])
        self.assertFalse(batches)
        self.assertEqual(await _execute(tools, CANCEL_TIMER_TOOL_NAME, {"timer_id": result["timer_id"]}),
                         {"status": "not_active"})

    async def test_repeat_keeps_counting_when_scheduler_handoff_fails_once(self):
        attempts, delivered = [], []

        def submit(batch):
            attempts.append(batch)
            if len(attempts) == 1:
                raise RuntimeError("synthetic callback body")
            delivered.extend(batch)

        scheduler = SessionScheduler()
        timers = Timers("main", scheduler.submit_internal)
        self.addCleanup(scheduler.bind(submit, on_revoke=timers.close))
        tools = Tools(timers.tools)
        result = await _execute(tools, SET_TIMER_TOOL_NAME, {"interval_seconds": 1, "mode": "repeat", "max_fires": 2})
        with self.assertLogs("agent_companion.session_scheduler", level="WARNING") as logs:
            self.clock.advance(1)
        self.assertIn("error_type=RuntimeError", logs.output[0])
        self.assertNotIn("synthetic callback body", logs.output[0])
        self.assertEqual(len(attempts), 1)
        self.assertFalse(delivered)

        self.clock.advance(1)
        self.clock.advance(10)
        scheduler.session_input()
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0], (delivered[0],))
        events = [_event(item) for item in delivered]
        self.assertEqual([(event["timer_id"], event["fire_count"], event["status"]) for event in events],
                         [(result["timer_id"], 1, "active"), (result["timer_id"], 2, "completed")])

    async def test_timer_is_an_internal_input_and_wakes_an_idle_runtime_after_voice(self):
        def answer(step, messages):
            if step == 3:
                event = _event(messages[-1])
                return (ToolReady(ToolCall("auto-register", SET_TIMER_TOOL_NAME, '{"interval_seconds":2}')),
                        ToolReady(ToolCall("stop", CANCEL_TIMER_TOOL_NAME, json.dumps({"timer_id": event["timer_id"]}))))
            return _register_then_finish(step, messages)

        schedulers = SessionSchedulers()
        scheduler = schedulers.get_or_create("main")
        model, channel = _Model(answer), _Channel()
        with closing(History(":memory:")) as history:
            async with running_agent(Settings(), history, model, channel, session_id="main",
                                     schedulers=schedulers, roles=demo_roles()) as (runtime, _):
                scheduler.session_input((Message("user", "synthetic registration"),))
                await asyncio.wait_for(runtime.wait_idle(), 2)
                self.assertIsNone(runtime._worker)
                self.assertIsNone(runtime.last_failure)
                self.assertEqual(len(model.requests), 2)
                scheduler.session_input(input_active=True)
                self.clock.advance(2)
                await asyncio.wait_for(runtime.wait_idle(), 2)
                self.assertEqual(len(model.requests), 2)
                scheduler.session_input(input_active=False)
                await asyncio.wait_for(runtime.wait_idle(), 2)
                self.assertIsNone(runtime.last_failure)
                self.assertEqual(len(model.requests), 4)
                tool_results = {m.tool_call_id: json.loads(m.content) for m in model.requests[-1] if m.role == "tool"}
                self.assertEqual(tool_results["auto-register"], {"error": "timer_requires_user_input"})
                self.assertEqual(tool_results["stop"], {"status": "cancelled"})
                self.assertEqual(_event(model.requests[2][-1])["timer_id"], tool_results["register"]["timer_id"])
                self.assertNotEqual(channel.replies[0][0], channel.replies[1][0])
                self.clock.advance(10)
                self.assertEqual(len(model.requests), 4)

    async def test_running_runtime_can_receive_timer_in_its_tool_continuation(self):
        def answer(step, messages):
            if step == 3:
                self.clock.advance(2)
                return (TextDelta("before continuation"),
                        ToolReady(ToolCall("stop", CANCEL_TIMER_TOOL_NAME, json.dumps({"timer_id": timer_id}))))
            return _register_then_finish(step, messages)

        schedulers = SessionSchedulers()
        scheduler = schedulers.get_or_create("main")
        model, channel = _Model(answer), _Channel()
        with closing(History(":memory:")) as history:
            async with running_agent(Settings(), history, model, channel, session_id="main",
                                     schedulers=schedulers, roles=demo_roles()) as (runtime, _):
                scheduler.session_input((Message("user", "synthetic registration"),))
                await asyncio.wait_for(runtime.wait_idle(), 2)
                timer_id = json.loads(model.requests[-1][-1].content)["timer_id"]
                scheduler.session_input((Message("user", "synthetic follow-up"),))
                await asyncio.wait_for(runtime.wait_idle(), 2)
                self.assertIsNone(runtime.last_failure)
                self.assertEqual(len(model.requests), 4)
                self.assertEqual(_event(model.requests[-1][-1])["timer_id"], timer_id)
                before, after = channel.replies[-2:]
                self.assertEqual(before[1], "before continuation")
                self.assertEqual(before[0], after[0])

    async def test_close_or_early_revoke_cancels_handles_before_rebind(self):
        schedulers = SessionSchedulers()
        for early_revoke in (False, True):
            with self.subTest(early_revoke=early_revoke), closing(History(":memory:")) as history:
                model = _Model(_register_then_finish)
                scheduler = schedulers.get_or_create("main")
                async with running_agent(Settings(), history, model, _Channel(), session_id="main",
                                         schedulers=schedulers, roles=demo_roles()) as (runtime, _):
                    scheduler.session_input((Message("user", "synthetic registration"),))
                    await asyncio.wait_for(runtime.wait_idle(), 2)
                    handle, = self.clock.handles
                    self.assertFalse(handle.cancelled)
                    if early_revoke:
                        scheduler.input_revoker()()
                        self.assertTrue(handle.cancelled)
                self.assertTrue(handle.cancelled)
                next_model = _Model(_register_then_finish)
                async with running_agent(Settings(), history, next_model, _Channel(), session_id="main",
                                         schedulers=schedulers, roles=demo_roles()) as (runtime, _):
                    self.clock.advance(10)
                    await asyncio.wait_for(runtime.wait_idle(), 2)
                    self.assertFalse(next_model.requests)


class RealTimerTest(unittest.IsolatedAsyncioTestCase):
    async def test_event_loop_timer_reaches_scheduler(self):
        received = asyncio.get_running_loop().create_future()
        scheduler = SessionScheduler()
        timers = Timers("main", scheduler.submit_internal)
        self.addCleanup(scheduler.bind(received.set_result, on_revoke=timers.close))
        result = await _execute(Tools(timers.tools), SET_TIMER_TOOL_NAME, {"interval_seconds": 0.001, "mode": "once"})
        self.assertFalse(received.done())
        item, = await asyncio.wait_for(received, 2)
        self.assertEqual(_event(item)["timer_id"], result["timer_id"])
