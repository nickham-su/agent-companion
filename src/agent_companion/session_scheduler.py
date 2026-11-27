"""按会话暂存输入；调度器及其同步提交回调都由同一个 asyncio 事件循环驱动。"""

import asyncio
from collections.abc import Callable, Sequence
import logging
from typing import Literal

from .protocol import (DeliveryStatus, Image, MessageDelivery, Receipt, ReplyDelivery, ReplyStream, Text,
                       validate_deliveries)


_LOG = logging.getLogger(__name__)
_Submit = Callable[[tuple[MessageDelivery, ...]], None]
_Trigger = Callable[[Callable[[], bool]], None]


async def _reject_voice(reply: ReplyStream) -> Receipt:
    reply.abandon()
    return Receipt(reply.info.reply_id, DeliveryStatus.ZERO)


class SessionScheduler:
    def __init__(self) -> None:
        self._agent_owner: object | None = None
        self._submit: _Submit | None = None
        self._agent_id: str | None = None
        self._agent_kind: Literal["main", "child"] | None = None
        self._trigger: _Trigger | None = None
        self._revoke_input: Callable[[], None] | None = None
        self._call: Callable[[], None] | None = None
        self._input_active = False
        self._input_epoch = 0
        self._obsolete_until = -float("inf")
        self._pending: list[MessageDelivery] = []

    @property
    def has_pending_messages(self) -> bool:
        return bool(self._pending)

    def _ensure_unbound(self, owner: object | None = None) -> None:
        if self._agent_owner is not owner or self._submit is not None:
            raise RuntimeError("session scheduler already bound or owned")

    def reserve_agent(self) -> object:
        """在任何启动副作用之前独占父会话，直到该运行者完成收尾。"""
        self._ensure_unbound()
        owner = self._agent_owner = object()
        return owner

    def release_agent(self, owner: object) -> bool:
        if self._agent_owner is not None and self._agent_owner is owner:
            if self._revoke_input is not None:
                self._revoke_input()
            self._agent_owner = None
            return True
        return False

    def bind(self, submit: _Submit, *, trigger: _Trigger | None = None, owner: object | None = None,
             on_revoke: Callable[[], None] | None = None, agent_id: str | None = None,
             agent_kind: Literal["main", "child"] | None = None) -> Callable[[], None]:
        self._ensure_unbound(owner)
        self._submit = submit
        self._agent_id = agent_id
        self._agent_kind = agent_kind
        self._trigger = trigger

        def revoke() -> None:
            if self._revoke_input is revoke:
                # 立即封住输入和回投，但不释放父会话所有权。
                self._submit = None
                self._agent_id = None
                self._agent_kind = None
                self._trigger = None
                self._revoke_input = None
                self._call = None
                self._input_active = False
                self._obsolete_until = -float("inf")
                self._pending.clear()
                if on_revoke is not None:
                    on_revoke()

        self._revoke_input = revoke
        return revoke

    def input_revoker(self) -> Callable[[], None]:
        """捕获当前绑定的撤销函数；旧路由不能撤销后继绑定。"""
        if self._revoke_input is None:
            raise RuntimeError("session scheduler is not bound")
        return self._revoke_input

    def call_started(self, stop: Callable[[], None]) -> None:
        # 每通电话各自的绑定方法同时是撤销入口和通话身份。
        if self._submit is None or self._call is not None:
            raise RuntimeError("session scheduler cannot bind call")
        self._call = stop

    def call_input(self, stop: Callable[[], None], deliveries: Sequence[MessageDelivery], *,
                   input_active: bool | None, empty_transcript: bool = False,
                   can_wake: Callable[[], bool] | None = None) -> bool:
        """返回本通输入是否被接收（也可能暂存在队列中）。"""
        if self._call != stop:
            return False
        may_trigger = empty_transcript and input_active is False and not (deliveries or self._pending)
        self.session_input(deliveries, input_active=input_active)
        if may_trigger and self._trigger is not None:
            epoch = self._input_epoch

            def current() -> bool:
                return self._call == stop and not self._input_active and not self._pending and self._input_epoch == epoch

            # 窗口检查可能同步解绑并放行输入；资格回调返回后仍需核对绑定和队列。
            self._trigger(lambda: current() and (can_wake is None or can_wake()) and current())
        return True

    def call_ended(self, stop: Callable[[], None], deliveries: Sequence[MessageDelivery] = ()) -> None:
        if self._call == stop:
            self._call = None
            self.session_input(deliveries, input_active=False)
        elif deliveries:
            # 身份只约束语音状态；迟到的生命周期消息仍按到达顺序加入队列。
            self.session_input(deliveries)

    def session_input(self, deliveries: Sequence[MessageDelivery] = (), *, input_active: bool | None = None) -> None:
        """同步追加内容、更新状态并放行，不在这三步之间让出事件循环。"""
        submit = self._submit
        if submit is None:
            raise RuntimeError("session scheduler is not bound")
        deliveries = tuple(deliveries)
        validate_deliveries(deliveries)
        if input_active is not None and not isinstance(input_active, bool):
            raise ValueError("input state must be a bool or None")
        self._pending.extend(deliveries)
        if input_active is not None:
            if input_active or self._input_active:
                self._input_epoch += 1
                self._obsolete_until = asyncio.get_running_loop().time()
            self._input_active = input_active
        self._flush_pending(submit)

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
        if _LOG.isEnabledFor(logging.DEBUG):
            for position, item in enumerate(batch, 1):
                message = item.message
                result_kind = message.async_result.kind if message.async_result is not None else None
                if result_kind not in (None, "agent_reply", "timer_fired"):
                    result_kind = "other"
                _LOG.debug("scheduler input submitted; agent=%s agent_kind=%s position=%d/%d kind=%s "
                           "intent=%s source=%s delegated=%s async_kind=%s text_chars=%d images=%d",
                           self._agent_id or "-", self._agent_kind or "-", position, len(batch), message.kind,
                           item.intent, message.source.kind if message.source is not None else "-",
                           item.delegation is not None, result_kind or "-",
                           sum(len(part.text) for part in message.parts if isinstance(part, Text)),
                           sum(isinstance(part, Image) for part in message.parts))

    def select_voice(self, reply: ReplyStream, target: ReplyDelivery) -> ReplyDelivery:
        """语音选路后同步决定是否拦截，不在转写结束后恢复过期回复。"""
        reply.select_voice_delivery()
        if self._input_active or reply.info.started_at <= self._obsolete_until:
            return ReplyDelivery(_reject_voice)
        return target

    def submit_internal(self, delivery: MessageDelivery) -> bool:
        """绑定有效且消息已加入调度器即视为接纳，不等待 Runtime 接手。"""
        submit = self._submit
        if submit is None:
            return False
        validate_deliveries((delivery,))
        self._pending.append(delivery)
        self._flush_pending(submit, internal=True)
        return True


class SessionSchedulers:
    """应用持有实例表；主路由保留调度器，子调度器由所属 Agent 树回收。"""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionScheduler] = {}

    def get_or_create(self, session_id: str) -> SessionScheduler:
        """查找、创建和登记不含 await，只能在应用的事件循环调用。"""
        scheduler = self._sessions.get(session_id)
        if scheduler is None:
            scheduler = self._sessions[session_id] = SessionScheduler()
        return scheduler

    def release_child(self, session_id: str, scheduler: SessionScheduler, owner: object) -> None:
        """只回收仍由该树持有的子调度器；检查和删除之间不让出事件循环。"""
        if self._sessions.get(session_id) is scheduler and scheduler.release_agent(owner):
            del self._sessions[session_id]

    def submit_internal(self, session_id: str, delivery: MessageDelivery) -> bool:
        scheduler = self._sessions.get(session_id)
        return scheduler is not None and scheduler.submit_internal(delivery)
