"""飞书真人私聊发现与每个对端的运行时生命周期。"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from hashlib import sha256
import logging
from sqlite3 import OperationalError
from time import time
from typing import TYPE_CHECKING

from .channels.feishu.channel import FeishuChannel, FeishuPoller
from .channels.feishu.cli import CliError, FeishuDiscovery, PrivateChat
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
_DISCOVERY_SECONDS = 10
_OpenAgent = Callable[[Agent, FeishuChannel, ChannelUser], AbstractAsyncContextManager[tuple[Runtime, AgentTree]]]


def _fingerprint(value: str) -> str:
    return sha256(value.encode("ascii")).hexdigest()


class _Route:
    def __init__(self, discovery: FeishuDiscovery, chat: PrivateChat, agent: Agent, *, history: History,
                 asr: FileASR, account: ChannelUser, started_ms: int, open_agent: _OpenAgent,
                 schedulers: SessionSchedulers) -> None:
        self._root_agent = agent
        self._user = chat.user
        self._session_id = agent.session_id
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
        cli = discovery.for_chat(chat.chat_id)
        self._channel = FeishuChannel(cli, select_voice=self._select_voice)
        self._poller = FeishuPoller(cli, asr, self._channel, chat_id=chat.chat_id, user=chat.user,
                                    account=account, session_id=agent.session_id, history=history, clock=time,
                                    private_started_ms=started_ms, submit=self._submit,
                                    before_close=self.stop_accepting)

    async def prepare_agent(self, *, wait_for_lock: bool = True) -> bool:
        if not wait_for_lock and self._agent_lock.locked():
            return False
        async with self._agent_lock:
            if self._closed:
                return False
            if self._agent is None:
                # 文字批次和来电使用同一个入口；发现私聊时尚不创建 Runtime。
                self._agent = await self._stack.enter_async_context(
                    self._open_agent(self._root_agent, self._channel, self._user))
                if self._closed:
                    self._agent[1].stop_accepting()
                else:
                    self._scheduler = self._schedulers.get_or_create(self._session_id)
                    self._revoke_input = self._scheduler.input_revoker()
            return not self._closed

    async def _submit(self, batch_id: int) -> None:
        if await self.prepare_agent():
            assert self._scheduler is not None
            self._scheduler.session_input((PersistedInput(batch_id),))

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
                               after_commit=self._channel.enqueue if copy_confirmed else None)

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
            self._channel.enqueue_transcript(text)

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
        self._channel.stop_accepting()
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
                self._stack.push_async_callback(self._channel.close)
                try:
                    await self._poller.run()
                finally:
                    self.stop_accepting()
                    # 不让最后一个懒建的 Runtime 注册到已退出的栈。
                    async with self._agent_lock:
                        pass
        except OperationalError as exc:
            if not is_sqlite_lock_error(exc):
                raise
            # 轮询内部已处理遇锁；若关停时无法标记运行及委派，下次启动或重新打开该路由时收尾。
            _LOG.warning("Feishu private route cleanup storage busy; deferred agent recovery")


def _conflicts(chats: set[PrivateChat]) -> tuple[set[ChannelUser], set[str]]:
    by_user, by_chat = defaultdict(set), defaultdict(set)
    for chat in chats:
        by_user[chat.user].add(chat.chat_id)
        by_chat[chat.chat_id].add(chat.user)
    return ({user for user, peers in by_user.items() if len(peers) > 1},
            {chat for chat, peers in by_chat.items() if len(peers) > 1})


async def run_private_chats(discovery: FeishuDiscovery, history: History, asr: FileASR, *, account: ChannelUser,
                            started_ms: int, open_agent: _OpenAgent, schedulers: SessionSchedulers,
                            incoming: IncomingCalls | None = None) -> None:
    active: dict[PrivateChat, tuple[_Route, asyncio.Task[None]]] = {}
    by_user: dict[ChannelUser, _Route] = {}
    # 一个通道用户对应一个父会话；退役不会阻塞其他会话的新核验结果。
    retiring: dict[ChannelUser, asyncio.Task[None]] = {}

    async def run_route(chat: PrivateChat, route: _Route) -> None:
        try:
            await route.run()
        except Exception as exc:
            if retiring.get(chat.user) is not asyncio.current_task():
                raise  # 在线路由的异常仍由 TaskGroup 上报。
            _LOG.warning("Feishu private route retirement failed; error_type=%s", type(exc).__name__)

    def retire(chat: PrivateChat, route: _Route, task: asyncio.Task[None]) -> None:
        route.stop_accepting()
        retiring[chat.user] = task
        if not task.cancelling():
            task.cancel()

    async with asyncio.TaskGroup() as group:
        calls = group.create_task(incoming.run(by_user.get)) if incoming is not None else None
        try:
            while True:
                # 只有在发起新一轮发现前已完成退役的会话，才可参加本轮核验。
                for user, task in tuple(retiring.items()):
                    if task.done():
                        del retiring[user]
                try:
                    chats = await discovery.private_chats(account)
                except CliError as exc:
                    _LOG.warning("Feishu private chat discovery failed (%s)", exc)
                    await asyncio.sleep(_DISCOVERY_SECONDS)
                    continue

                blocked_users, blocked_chats = _conflicts(chats)
                verified = {chat for chat in chats
                            if chat.user not in blocked_users and chat.chat_id not in blocked_chats}
                if chats != verified:
                    _LOG.warning("Feishu private chat route conflict; conflicting routes ignored")

                # 只有成功读完的快照会撤销信任；在注册或等待收尾之前同步关闭旧入口。
                for chat in active.keys() - verified:
                    route, task = active.pop(chat)
                    by_user.pop(chat.user)
                    retire(chat, route, task)
                for chat in verified - active.keys():
                    if chat.user in retiring:
                        continue  # 旧 Runtime 可能仍持有父会话；下次发现重新核验。
                    session_id = "feishu-private:" + _fingerprint(chat.user.key)
                    try:
                        agent = history.bind_feishu_private_agent(session_id, _fingerprint(chat.chat_id))
                    except OperationalError as exc:
                        if not is_sqlite_lock_error(exc):
                            _LOG.error("Feishu private route storage failed; sqlite_code=%s",
                                       getattr(exc, "sqlite_errorcode", None))
                            raise
                        _LOG.warning("Feishu private route storage busy; retrying on next discovery")
                        continue
                    if agent is None:
                        _LOG.warning("Feishu private chat route conflict; conflicting route ignored")
                        continue
                    route = _Route(discovery, chat, agent, history=history, asr=asr, account=account,
                                   started_ms=started_ms, open_agent=open_agent, schedulers=schedulers)
                    by_user[chat.user] = route
                    active[chat] = route, group.create_task(run_route(chat, route))
                await asyncio.sleep(_DISCOVERY_SECONDS)
        finally:
            # 在等待任一子任务收尾前，同步关闭全部路由入口与发送队列。
            if calls is not None:
                calls.cancel()
            for chat, (route, task) in active.items():
                retire(chat, route, task)
            if retiring:
                await asyncio.wait(retiring.values())
