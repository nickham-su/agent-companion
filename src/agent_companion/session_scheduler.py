"""按会话暂存输入；调度器及其同步提交回调都由同一个 asyncio 事件循环驱动。"""

import asyncio
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
import logging

from .protocol import (DeliveryStatus, InputItem, InternalInput, PersistedInput, Receipt, ReplyDelivery, ReplyStream,
                       validate_input_items)


_LOG = logging.getLogger(__name__)
_Submit = Callable[[tuple[InputItem, ...]], None]
_RunChild = Callable[[str, str], Awaitable[str | None]]


async def _reject_voice(reply: ReplyStream) -> Receipt:
    reply.abandon()
    return Receipt(reply.info.reply_id, DeliveryStatus.ZERO)


class SessionScheduler:
    def __init__(self) -> None:
        self._agent_owner: object | None = None
        self._submit: _Submit | None = None
        self._revoke_input: Callable[[], None] | None = None
        self._run_child: _RunChild | None = None
        self._call: Callable[[], None] | None = None
        self._input_active = False
        self._obsolete_until = -float("inf")
        self._pending: list[InputItem] = []
        self._pending_inbound: set[int] = set()

    def _ensure_unbound(self, owner: object | None = None) -> None:
        if self._agent_owner is not owner or self._submit is not None or self._run_child is not None:
            raise RuntimeError("session scheduler already bound or owned")

    def reserve_agent(self) -> object:
        """在任何启动副作用之前独占父会话，直到该运行者完成收尾。"""
        self._ensure_unbound()
        owner = self._agent_owner = object()
        return owner

    def release_agent(self, owner: object) -> None:
        if self._agent_owner is not None and self._agent_owner is owner:
            if self._revoke_input is not None:
                self._revoke_input()
            self._agent_owner = None

    def bind(self, submit: _Submit, *, owner: object | None = None,
             on_revoke: Callable[[], None] | None = None) -> Callable[[], None]:
        self._ensure_unbound(owner)
        self._submit = submit

        def revoke() -> None:
            if self._revoke_input is revoke:
                # 立即封住输入和回投，但不释放父会话所有权。
                self._submit = None
                self._revoke_input = None
                self._call = None
                self._input_active = False
                self._obsolete_until = -float("inf")
                self._pending.clear()
                self._pending_inbound.clear()
                if on_revoke is not None:
                    on_revoke()

        self._revoke_input = revoke
        return revoke

    def input_revoker(self) -> Callable[[], None]:
        """捕获当前绑定的撤销函数；旧路由不能撤销后继绑定。"""
        if self._revoke_input is None:
            raise RuntimeError("session scheduler is not bound")
        return self._revoke_input

    @contextmanager
    def bind_child(self, run_child: _RunChild) -> Iterator[None]:
        self._ensure_unbound()
        self._run_child = run_child
        try:
            yield
        finally:
            self._run_child = None

    async def run_child(self, run_id: str, prompt: str) -> str | None:
        """直接放行已获准的子任务；准入、名额和串行仍由 dispatcher 负责。"""
        if self._run_child is None:
            raise RuntimeError("child session scheduler is not bound")
        return await self._run_child(run_id, prompt)

    def call_started(self, stop: Callable[[], None]) -> None:
        # 每通电话各自的绑定方法同时是撤销入口和通话身份。
        if self._submit is None or self._call is not None:
            raise RuntimeError("session scheduler cannot bind call")
        self._call = stop

    def call_input(self, stop: Callable[[], None], items: Sequence[InputItem], *,
                   input_active: bool | None) -> None:
        if self._call == stop:
            self.session_input(items, input_active=input_active)

    def call_ended(self, stop: Callable[[], None]) -> None:
        if self._call == stop:
            self._call = None
            self.session_input(input_active=False)

    def session_input(self, items: Sequence[InputItem] = (), *, input_active: bool | None = None) -> None:
        """同步追加内容、更新状态并放行，不在这三步之间让出事件循环。"""
        submit = self._submit
        if submit is None:
            raise RuntimeError("session scheduler is not bound")
        items = tuple(items)
        validate_input_items(items)
        if input_active is not None and not isinstance(input_active, bool):
            raise ValueError("input state must be a bool or None")
        self._append_inputs(items)
        if input_active is not None:
            if input_active or self._input_active:
                self._obsolete_until = asyncio.get_running_loop().time()
            self._input_active = input_active
        self._flush_pending(submit)

    def _append_inputs(self, items: tuple[InputItem, ...]) -> None:
        for item in items:
            if isinstance(item, PersistedInput):
                if item.batch_id in self._pending_inbound:
                    continue
                self._pending_inbound.add(item.batch_id)
            self._pending.append(item)

    def _flush_pending(self, submit: _Submit, *, internal: bool = False) -> None:
        if self._input_active or not self._pending:
            return
        batch = tuple(self._pending)
        try:
            submit(batch)
        except Exception as exc:
            if not internal:
                raise
            _LOG.warning("runtime input handoff failed; batch retained; error_type=%s", type(exc).__name__)
            return
        self._pending.clear()
        self._pending_inbound.clear()

    def select_voice(self, reply: ReplyStream, target: ReplyDelivery) -> ReplyDelivery:
        """语音选路后同步决定是否拦截，不在转写结束后恢复过期回复。"""
        reply.select_voice_delivery()
        if self._input_active or reply.info.started_at <= self._obsolete_until:
            return ReplyDelivery(_reject_voice)
        return target

    def submit_internal(self, item: InternalInput) -> bool:
        """绑定有效且消息已加入调度器即视为接纳，不等待 Runtime 接手。"""
        submit = self._submit
        if submit is None:
            return False
        items = (item,)
        validate_input_items(items)
        self._append_inputs(items)
        self._flush_pending(submit, internal=True)
        return True


class SessionSchedulers:
    """应用持有实例表；路由撤销可解除绑定，但不回收调度器。"""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionScheduler] = {}

    def get_or_create(self, session_id: str) -> SessionScheduler:
        """查找、创建和登记不含 await，只能在应用的事件循环调用。"""
        scheduler = self._sessions.get(session_id)
        if scheduler is None:
            scheduler = self._sessions[session_id] = SessionScheduler()
        return scheduler

    def submit_internal(self, session_id: str, item: InternalInput) -> bool:
        scheduler = self._sessions.get(session_id)
        return scheduler is not None and scheduler.submit_internal(item)
