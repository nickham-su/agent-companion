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
from .protocol import ReplyInfo
from .runtime.engine import Runtime
from .runtime.subtasks import Subtasks
from .speech.asr import FileASR
from .storage.sqlite import History, is_sqlite_lock_error

if TYPE_CHECKING:
    from .feishu_calls import IncomingCalls


_LOG = logging.getLogger(__name__)
_DISCOVERY_SECONDS = 10
_OpenAgent = Callable[[str, FeishuChannel], AbstractAsyncContextManager[tuple[Runtime, Subtasks]]]


def _fingerprint(value: str) -> str:
    return sha256(value.encode("ascii")).hexdigest()


class _Route:
    def __init__(self, discovery: FeishuDiscovery, chat: PrivateChat, session_id: str, *, history: History,
                 asr: FileASR, own_id: str, started_ms: int, open_agent: _OpenAgent) -> None:
        self._session_id = session_id
        self._stack = AsyncExitStack()
        self._agent: tuple[Runtime, Subtasks] | None = None
        self._agent_lock = asyncio.Lock()
        self._closed = False
        self._stop_call: Callable[[], None] | None = None
        self._open_agent = open_agent
        cli = discovery.for_chat(chat.chat_id)
        self._channel = FeishuChannel(cli)
        self._poller = FeishuPoller(cli, asr, self._channel, chat_id=chat.chat_id, sender_id=chat.sender_id,
                                    own_id=own_id, session_id=session_id, history=history, clock=time,
                                    private_started_ms=started_ms, submit=self._submit,
                                    before_close=self.stop_accepting)

    async def runtime(self) -> Runtime | None:
        async with self._agent_lock:
            if self._closed:
                return None
            if self._agent is None:
                # 文字批次和来电使用同一个入口；发现私聊时尚不创建 Runtime。
                self._agent = await self._stack.enter_async_context(self._open_agent(self._session_id, self._channel))
            return self._agent[0] if not self._closed else None

    async def _submit(self, batch_id: int) -> None:
        main = await self.runtime()
        if main is not None:
            main.submit_persisted_batch(batch_id, wait_for_delivery=False)

    def watch_call(self, stop: Callable[[], None]) -> bool:
        if self._closed or self._stop_call is not None:
            return False
        self._stop_call = stop
        return True

    def unwatch_call(self, stop: Callable[[], None]) -> None:
        if self._stop_call == stop:
            self._stop_call = None

    def copy_confirmed(self, info: ReplyInfo, text: str) -> None:
        self._channel.enqueue(info, text)

    def stop_accepting(self) -> None:
        self._closed = True
        if self._stop_call is not None:
            self._stop_call()
        if self._agent is not None:
            main, subtasks = self._agent
            main.stop_accepting()
            subtasks.stop_accepting()
        self._channel.stop_accepting()

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
            # 轮询内部已处理遇锁；若收尾时无法标记子任务，下次打开该会话再标记中断。
            _LOG.warning("Feishu private route cleanup storage busy; deferred subtask recovery")


def _conflicts(chats: set[PrivateChat]) -> tuple[set[str], set[str]]:
    by_user, by_chat = defaultdict(set), defaultdict(set)
    for chat in chats:
        by_user[chat.sender_id].add(chat.chat_id)
        by_chat[chat.chat_id].add(chat.sender_id)
    return ({sender for sender, peers in by_user.items() if len(peers) > 1},
            {chat for chat, peers in by_chat.items() if len(peers) > 1})


async def run_private_chats(discovery: FeishuDiscovery, history: History, asr: FileASR, *, own_id: str,
                            started_ms: int, open_agent: _OpenAgent, incoming: IncomingCalls | None = None) -> None:
    active: dict[PrivateChat, tuple[_Route, asyncio.Task[None]]] = {}
    by_sender: dict[str, _Route] = {}
    retiring: list[asyncio.Task[None]] = []
    async with asyncio.TaskGroup() as group:
        calls = group.create_task(incoming.run(by_sender.get)) if incoming is not None else None
        try:
            while True:
                try:
                    chats = await discovery.private_chats(own_id)
                except CliError as exc:
                    _LOG.warning("Feishu private chat discovery failed (%s)", exc)
                    await asyncio.sleep(_DISCOVERY_SECONDS)
                    continue

                blocked_users, blocked_chats = _conflicts(chats)
                verified = {chat for chat in chats
                            if chat.sender_id not in blocked_users and chat.chat_id not in blocked_chats}
                if chats != verified:
                    _LOG.warning("Feishu private chat route conflict; conflicting routes ignored")

                # 只有成功读完的快照会撤销信任；在注册或等待收尾之前同步关闭旧入口。
                for chat in active.keys() - verified:
                    route, task = active.pop(chat)
                    by_sender.pop(chat.sender_id)
                    route.stop_accepting()
                    task.cancel()
                    retiring.append(task)
                for chat in verified - active.keys():
                    session_id = "feishu-private:" + _fingerprint(chat.sender_id)
                    try:
                        bound = history.bind_feishu_private_route(session_id, _fingerprint(chat.chat_id))
                    except OperationalError as exc:
                        if not is_sqlite_lock_error(exc):
                            _LOG.error("Feishu private route storage failed; sqlite_code=%s",
                                       getattr(exc, "sqlite_errorcode", None))
                            raise
                        _LOG.warning("Feishu private route storage busy; retrying on next discovery")
                        continue
                    if not bound:
                        _LOG.warning("Feishu private chat route conflict; conflicting route ignored")
                        continue
                    route = _Route(discovery, chat, session_id, history=history, asr=asr, own_id=own_id,
                                   started_ms=started_ms, open_agent=open_agent)
                    by_sender[chat.sender_id] = route
                    active[chat] = route, group.create_task(route.run())
                if retiring:
                    # 父任务退出时不能再次向正在清理的路由传播取消。
                    await asyncio.wait(retiring)
                    retiring.clear()
                await asyncio.sleep(_DISCOVERY_SECONDS)
        finally:
            # 在等待任一子任务收尾前，同步关闭全部路由入口与发送队列。
            if calls is not None:
                calls.cancel()
            for route, task in active.values():
                route.stop_accepting()
                if not task.cancelling():
                    task.cancel()
                retiring.append(task)
            if retiring:
                await asyncio.wait(retiring)
