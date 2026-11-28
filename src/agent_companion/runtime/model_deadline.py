"""模型生成截止：原生适配器超时后先请求协作停止，再由传输层按预算收尾。"""

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import NamedTuple

from ..protocol import ModelError


class ModelDeadline(NamedTuple):
    disarm: Callable[[], None]
    expired: Callable[[], bool]
    reschedule: Callable[[float | None], None]
    owns_cancellation: Callable[[], bool]


@asynccontextmanager
async def model_deadline_at(
    when: float | None, stop_stream: Callable[[asyncio.Task[object]], bool] | None,
) -> AsyncIterator[ModelDeadline]:
    loop = asyncio.get_running_loop()
    reader = asyncio.current_task()
    assert reader is not None
    timed_out = False
    cancelled_reader = False
    active = True
    timer: asyncio.TimerHandle | None = None

    def expire() -> None:
        nonlocal timed_out, cancelled_reader
        timed_out = True
        # 裸取消可能打断 httpcore 的内部关闭；没有实际请求时才走取消兜底。
        if (stop_stream is None or not stop_stream(reader)) and not reader.cancelling():
            cancelled_reader = reader.cancel()

    def reschedule(at: float | None) -> None:
        nonlocal timer, when
        if active and not timed_out:
            if timer is not None:
                timer.cancel()
            when = at
            timer = loop.call_at(at, expire) if at is not None else None

    def expired() -> bool:
        return timed_out or (when is not None and loop.time() >= when)

    def owns_cancellation() -> bool:
        # 同时有外部取消时，即使本截止也取消过任务，也不能按本地超时结算。
        return cancelled_reader and reader.cancelling() == 1

    reschedule(when)
    try:
        yield ModelDeadline(lambda: reschedule(None), expired, reschedule, owns_cancellation)
    except (asyncio.CancelledError, ModelError):
        if timed_out and reader.cancelling() <= cancelled_reader:
            raise TimeoutError from None
        raise
    else:
        if timed_out:
            if reader.cancelling() > cancelled_reader:
                raise asyncio.CancelledError
            raise TimeoutError
    finally:
        active = False
        if timer is not None:
            timer.cancel()
        if cancelled_reader:
            reader.uncancel()
