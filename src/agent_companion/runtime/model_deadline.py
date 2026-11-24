"""模型生成截止：原生适配器超时后先请求协作停止，再由传输层按预算收尾。"""

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import NamedTuple

from ..protocol import ModelError


class ModelDeadline(NamedTuple):
    disarm: Callable[[], None]
    expired: Callable[[], bool]


@asynccontextmanager
async def model_deadline_at(
    when: float, stop_stream: Callable[[asyncio.Task[object]], bool] | None,
) -> AsyncIterator[ModelDeadline]:
    if stop_stream is None:
        async with asyncio.timeout_at(when) as deadline:
            def disarm() -> None:
                if not deadline.expired():
                    deadline.reschedule(None)

            yield ModelDeadline(disarm, deadline.expired)
        return

    loop = asyncio.get_running_loop()
    reader = asyncio.current_task()
    assert reader is not None
    timed_out = False

    def expire() -> None:
        nonlocal timed_out
        timed_out = True
        # 裸取消可能打断 httpcore 的内部关闭；没有实际请求时才走取消兜底。
        if not stop_stream(reader) and not reader.cancelling():
            reader.cancel()

    timer = loop.call_at(when, expire)
    try:
        yield ModelDeadline(timer.cancel, lambda: timed_out)
    except (asyncio.CancelledError, ModelError):
        if timed_out:
            raise TimeoutError from None
        raise
    else:
        if timed_out:
            raise TimeoutError
    finally:
        timer.cancel()
