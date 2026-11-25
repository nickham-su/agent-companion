"""飞书按已核验用户注册 Runtime，文字和来电共用主 Session。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack
import logging
from sqlite3 import OperationalError
from time import time
from typing import TYPE_CHECKING

from .channels.feishu.channel import FeishuChannel, FeishuInput, FeishuPoller, private_session_id
from .channels.feishu.cli import FeishuDiscovery
from .protocol import ChannelUser, Message, PersistedInput, ReplyDelivery, ReplyStream
from .runtime.engine import Runtime
from .runtime.subtasks import AgentTree
from .session_scheduler import SessionScheduler, SessionSchedulers
from .speech.asr import FileASR
from .storage.sqlite import Agent, History, is_sqlite_lock_error

if TYPE_CHECKING:
    from .feishu_calls import IncomingCalls
    from .speech.voice import VoiceChannel


_LOG = logging.getLogger(__name__)
_INBOUND_SOURCE = "feishu-private"
_OpenAgent = Callable[[Agent, FeishuChannel, ChannelUser], AbstractAsyncContextManager[tuple[Runtime, AgentTree]]]


class _Route:
    def __init__(self, discovery: FeishuDiscovery, user: ChannelUser, *, history: History, open_agent: _OpenAgent,
                 schedulers: SessionSchedulers, inputs: FeishuInput) -> None:
        self._history = history
        self._inputs = inputs
        self._user = user
        self._session_id = private_session_id(user)
        self._schedulers = schedulers
        self._scheduler: SessionScheduler | None = None
        self._revoke_input: Callable[[], None] | None = None
        self._stack = AsyncExitStack()
        self._agent: tuple[Runtime, AgentTree] | None = None
        self._agent_lock = asyncio.Lock()
        self._closed = False
        self._stop_call: Callable[[], None] | None = None
        self._select_call: Callable[[ReplyStream], ReplyDelivery | None] | None = None
        self._open_agent = open_agent
        self._inbound = asyncio.Event()
        self._last_submitted = 0
        self.channel = FeishuChannel(discovery.for_user(user), select_voice=self._select_voice)

    def notify_inbound(self) -> None:
        self._inbound.set()

    async def prepare_agent(self, *, wait_for_lock: bool = True) -> bool:
        if not wait_for_lock and self._agent_lock.locked():
            return False
        async with self._agent_lock:
            if self._closed:
                return False
            if self._agent is None:
                # 文字批次和来电共用同一存储身份和初始化锁。
                root = self._history.ensure_main_agent(self._session_id)
                self._agent = await self._stack.enter_async_context(
                    self._open_agent(root, self.channel, self._user))
                if self._closed:
                    self._agent[1].stop_accepting()
                else:
                    self._scheduler = self._schedulers.get_or_create(self._session_id)
                    self._revoke_input = self._scheduler.input_revoker()
            return not self._closed

    def _dispatch_pending(self) -> None:
        scheduler = self._scheduler
        if scheduler is None:
            return
        try:
            pending = self._history.pending_inbound(self._session_id, _INBOUND_SOURCE)
        except OperationalError as exc:
            if not is_sqlite_lock_error(exc):
                raise
            _LOG.warning("Feishu user inbox storage busy; retrying pending input on next poll")
            return
        for batch_id in pending:
            if batch_id > self._last_submitted:
                scheduler.session_input((PersistedInput(batch_id),))
                self._last_submitted = batch_id

    def watch_call(self, stop: Callable[[], None]) -> bool:
        if self._closed or self._stop_call is not None:
            return False
        self._stop_call = stop
        return True

    def call_started(self, stop: Callable[[], None], voice: VoiceChannel, *, check_window: Callable[[], bool],
                     idle_timeout: float, copy_confirmed: bool) -> bool:
        scheduler = self._scheduler
        if (self._closed or self._stop_call != stop or self._select_call is not None
                or scheduler is None or voice.failed):
            return False

        target = ReplyDelivery(voice.deliver, idle_timeout=idle_timeout,
                               after_commit=self.channel.enqueue if copy_confirmed else None)

        def select(reply: ReplyStream) -> ReplyDelivery | None:
            if voice.failed or not check_window():
                return None
            return scheduler.select_voice(reply, target)

        scheduler.call_started(stop)
        self._select_call = select
        return True

    def call_input(self, stop: Callable[[], None], text: str | None, input_active: bool | None) -> None:
        scheduler = self._scheduler
        if self._closed or self._stop_call != stop or scheduler is None:
            return
        items = (Message("user", text),) if text else ()
        if scheduler.call_input(stop, items, input_active=input_active) and text:
            self.channel.enqueue_transcript(text)

    def call_ended(self, stop: Callable[[], None]) -> None:
        if self._stop_call != stop:
            return
        self._stop_call = self._select_call = None
        if self._scheduler is not None:
            self._scheduler.call_ended(stop)

    def _select_voice(self, reply: ReplyStream) -> ReplyDelivery | None:
        select = self._select_call
        return select(reply) if select is not None else None

    def stop_accepting(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.channel.stop_accepting()
        if self._revoke_input is not None:
            self._revoke_input()
            self._revoke_input = None
        self._scheduler = None
        if self._stop_call is not None:
            self._stop_call()
        if self._agent is not None:
            _, tree = self._agent
            tree.stop_accepting()

    async def run(self) -> None:
        try:
            async with self._stack:
                self._stack.push_async_callback(self.channel.close)
                try:
                    while True:
                        await self._inbound.wait()
                        self._inbound.clear()
                        try:
                            await self._submit_inbound()
                            while await self._inputs.process_next(self._user, self.channel, self._dispatch_pending):
                                if self._scheduler is None:
                                    await self._submit_inbound()
                                await asyncio.sleep(0)
                        except OperationalError as exc:
                            if not is_sqlite_lock_error(exc):
                                raise
                            _LOG.warning("Feishu user session storage busy; retrying pending input on next poll")
                finally:
                    self.stop_accepting()
                    # 关停会取消来电任务；先等共享锁释放再退出栈。
                    async with self._agent_lock:
                        pass
        except OperationalError as exc:
            if not is_sqlite_lock_error(exc):
                raise
            _LOG.warning("Feishu user session cleanup storage busy; deferred agent recovery")


    async def _submit_inbound(self) -> None:
        if self._scheduler is None and self._history.pending_inbound(self._session_id, _INBOUND_SOURCE):
            await self.prepare_agent()
        if self._scheduler is not None:
            self._dispatch_pending()


async def run_private_chats(discovery: FeishuDiscovery, history: History, asr: FileASR, *, account: ChannelUser,
                            started_ms: int, open_agent: _OpenAgent, schedulers: SessionSchedulers,
                            incoming: IncomingCalls | None = None) -> None:
    inputs = FeishuInput(discovery, asr, history)
    routes: dict[ChannelUser, _Route] = {}
    tasks: list[asyncio.Task[None]] = []
    stopping = False

    async with asyncio.TaskGroup() as group:
        def route_for(user: ChannelUser) -> _Route:
            # 不让出事件循环：文字与来电同时到达时也只生成一条用户路由。
            if user == account or stopping:
                raise ValueError("invalid Feishu peer registration")
            route = routes.get(user)
            if route is None:
                route = _Route(discovery, user, history=history, open_agent=open_agent,
                               schedulers=schedulers, inputs=inputs)
                routes[user] = route
                tasks.append(group.create_task(route.run()))
            return route

        def stop() -> None:
            nonlocal stopping
            if not stopping:
                stopping = True
                for route in routes.values():
                    route.stop_accepting()
                for task in tasks:
                    if not task.cancelling():
                        task.cancel()

        poller = FeishuPoller(discovery, account=account, history=history, private_started_ms=started_ms,
                              clock=time,
                              notify_user=lambda user: route_for(user).notify_inbound(), before_close=stop)
        if incoming is not None:
            tasks.append(group.create_task(incoming.run(route_for)))
        try:
            await poller.run()
        finally:
            stop()
