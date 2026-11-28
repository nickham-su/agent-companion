"""原生模型适配器持有的 HTTP 请求及响应关闭任务。"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
import logging
from typing import TypeVar

from anyio import CancelScope, move_on_after
from anyio.lowlevel import checkpoint
import httpx

from ..model_diagnostics import current_request, failure_category
from ..protocol import ModelError


_LOG = logging.getLogger(__name__)
_RESPONSE_CLOSE_TIMEOUT = 5.0
_RESPONSE_TAIL_TIMEOUT = 0.05
_RESPONSE_TAIL_MAX_BYTES = 64 * 1024
_T = TypeVar("_T")


def _is_close_timeout(error: Exception) -> bool:
    return isinstance(error, ModelError) and error.diagnostic == "transport_close_timeout"


def _report_close_error(error: Exception) -> None:
    # 预算耗尽已由关闭任务记录，避免每个等待方都打印一次。
    if not _is_close_timeout(error):
        _LOG.warning("native HTTP response cleanup failed; category=%s", failure_category(error))


class _RequestOwner:
    def __init__(self, task: asyncio.Task[object]) -> None:
        self.task = task
        self.settled = task.get_loop().create_future()  # 请求退出或关闭预算耗尽。
        self._scope: CancelScope | None = None
        self._timer: asyncio.TimerHandle | None = None
        self._forced_cancel = False

    @property
    def stopping(self) -> bool:
        return self._timer is not None or self.task.cancelling() > 0

    async def read(self, operation: Callable[[], Awaitable[_T]]) -> _T:
        if self._timer is None:
            # 只覆盖 I/O，不跨 yield；调用方及其工具不在此取消域内。
            with CancelScope() as scope:
                self._scope = scope
                try:
                    return await operation()
                finally:
                    self._scope = None
        raise asyncio.CancelledError

    def stop(self) -> None:
        if self._timer is None:
            self._timer = self.task.get_loop().call_later(_RESPONSE_CLOSE_TIMEOUT, self._expire)
            if self._scope is not None:
                # 传输可在读操作内部先关闭连接；协作取消会尊重其关闭保护区。
                self._scope.cancel()

    def _expire(self) -> None:
        _LOG.warning("native HTTP request close deadline reached (transport_close_timeout)")
        if self._scope is not None:
            # 仅在预算耗尽后中断仍在 I/O 中的请求，不取消调用方的其他工作。
            self._forced_cancel = self.task.cancel()
        self.settled.set_result(None)

    def finish(self) -> None:
        if self._forced_cancel:
            # 强制打断读操作可能中断 httpcore 内部的连接关闭，即使没有显式响应关闭异常。
            diagnostics = current_request()
            if diagnostics is not None and diagnostics.close_failure is None:
                diagnostics.close_failure = "close_unconfirmed"
            # 向外传播取消，但只归还本传输发出的那一次；外部取消仍有独立计数。
            self.task.uncancel()
        if self._timer is not None:
            self._timer.cancel()
        if not self.settled.done():
            self.settled.set_result(None)


class _OwnedStream(httpx.AsyncByteStream):
    def __init__(self, inner: httpx.AsyncByteStream, owner: _RequestOwner, closing: set["_OwnedStream"], *,
                 background_tail: bool) -> None:
        self._inner = inner
        self._owner = owner
        self._closing = closing
        self._background_tail = background_tail
        self.allow_tail = False
        self._iterator: AsyncIterator[bytes] | None = None
        self._eof = False
        self._task: asyncio.Task[None] | None = None
        self._timer: asyncio.TimerHandle | None = None
        self._timed_out = False
        self._detached = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self._iterator = self._inner.__aiter__()
        while True:
            try:
                chunk = await self._next_chunk()
            except StopAsyncIteration:
                return
            yield chunk

    async def _next_chunk(self) -> bytes:
        iterator = self._iterator
        assert iterator is not None
        try:
            return await self._owner.read(lambda: anext(iterator))
        except StopAsyncIteration:
            self._eof = True
            raise

    async def _finish(self, drain: bool) -> None:
        try:
            if drain and not self._owner.stopping:
                # 沿用同一个原始迭代器，只认 HTTP 消息边界，不看 SSE 内容。
                remaining = _RESPONSE_TAIL_MAX_BYTES
                with move_on_after(_RESPONSE_TAIL_TIMEOUT):
                    while remaining > 0:
                        remaining -= len(await self._next_chunk())
                        await checkpoint()
        except (StopAsyncIteration, httpx.RequestError):
            # 读尾部只为了回池，没读完则由正常关闭释放这条响应。
            pass
        finally:
            try:
                await self._inner.aclose()
            finally:
                self._iterator = None

    async def aclose(self) -> None:
        loop = asyncio.get_running_loop()
        if self._task is None:
            drain = self.allow_tail and self._iterator is not None and not self._eof and not self._owner.stopping
            self._detached = drain and self._background_tail
            self._task = loop.create_task(self._finish(drain))
            self._closing.add(self)
            self._timer = loop.call_later(_RESPONSE_CLOSE_TIMEOUT, self._expire)
            self._task.add_done_callback(self._finished)
        if not self._detached:
            await self.wait_closed()

    async def wait_closed(self) -> None:
        task = self._task
        assert task is not None
        loop = asyncio.get_running_loop()
        if not task.done() and not self._timed_out:
            assert self._timer is not None
            # wait 不会把 EOF 读取方或适配器等待方的取消传给唯一的底层关闭任务。
            await asyncio.wait((task,), timeout=max(0, self._timer.when() - loop.time()))
        if not task.done():
            self._expire()
        if self._detached:  # 后台关闭任务自行记录异常；退出时只等待，不再次报错。
            return
        if self._timed_out:
            raise ModelError("transport_close_timeout")
        if task.cancelled():
            raise ModelError("transport_error")
        task.result()

    def _expire(self) -> None:
        task = self._task
        if task is not None and not task.done() and not self._timed_out:
            self._timed_out = True
            _LOG.warning("native HTTP response close deadline reached")
            if not task.cancelling():
                task.cancel()

    def _finished(self, task: asyncio.Task[None]) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._closing.discard(self)
        if not task.cancelled():
            error = task.exception()  # 即使读流方已离开，也回收底层关闭异常。
            if self._detached and not self._timed_out and isinstance(error, Exception):
                _report_close_error(error)


class ResponseStreams:
    """跟踪整个请求（包括读操作内部的关闭），以及已启动的显式关闭。"""

    def __init__(self) -> None:
        self._requests: set[_RequestOwner] = set()
        self._closing: set[_OwnedStream] = set()
        self._closed = False

    def stop(self, reader: asyncio.Task[object]) -> bool:
        found = False
        for owner in self._requests:
            if owner.task is reader:
                owner.stop()
                found = True
        return found

    def stopping(self) -> bool:
        """防止吞掉取消的底层读取把迟到的帧交给当前请求。"""
        reader = asyncio.current_task()
        return any(owner.task is reader and owner.stopping for owner in self._requests)

    @asynccontextmanager
    async def open(
        self, context: AbstractAsyncContextManager[httpx.Response], *, background_tail: bool = False,
    ) -> AsyncIterator[tuple[httpx.Response, Callable[[], None]]]:
        if self._closed:
            raise ModelError("transport_error")
        reader = asyncio.current_task()
        assert reader is not None
        owner = _RequestOwner(reader)
        self._requests.add(owner)
        try:
            async with AsyncExitStack() as stack:
                response = await owner.read(lambda: stack.enter_async_context(context))
                if response.is_closed:
                    yield response, lambda: None  # 传输可能直接返回已读取完的内存响应。
                    return
                assert isinstance(response.stream, httpx.AsyncByteStream)
                stream = _OwnedStream(response.stream, owner, self._closing, background_tail=background_tail)
                response.stream = stream

                def allow_tail() -> None:
                    headers = response.headers
                    connection = headers.get("connection", "").lower().split(",")
                    stream.allow_tail = (response.http_version == "HTTP/1.1" and
                                         not any(token.strip() == "close" for token in connection) and
                                         ("content-length" in headers or "transfer-encoding" in headers))

                primary_error: BaseException | None = None
                try:
                    yield response, allow_tail
                except (Exception, asyncio.CancelledError) as error:
                    primary_error = error
                    stream.allow_tail = False
                    raise
                finally:
                    try:
                        # HTTPX 先标记 is_closed 再 await stream.aclose()；EOF 时需继续加入原任务。
                        if response.is_closed:
                            await stream.aclose()
                        else:
                            await response.aclose()
                    except asyncio.CancelledError:
                        # 停止等待不等于底层关闭已确认；外层截止可能将取消转换成超时。
                        diagnostics = current_request()
                        if diagnostics is not None:
                            diagnostics.close_failure = "close_unconfirmed"
                        raise
                    except Exception as error:
                        diagnostics = current_request()
                        if diagnostics is not None:
                            diagnostics.close_failure = failure_category(error)
                        # 保持既有传输错误语义；关闭超时不覆盖原失败，关闭失败不掩盖取消。
                        if isinstance(primary_error, asyncio.CancelledError):
                            _report_close_error(error)
                        elif primary_error is None or not _is_close_timeout(error):
                            if diagnostics is not None:
                                diagnostics.stage = "response_close"
                            raise
        finally:
            self._requests.remove(owner)
            owner.finish()

    async def aclose(self) -> None:
        self._closed = True
        requests = tuple(self._requests)
        for owner in requests:
            owner.stop()
        # 不抢先显式关闭仍在读的响应。已有显式关闭任务可能比请求所有者活得更久。
        closing = asyncio.gather(*(stream.wait_closed() for stream in tuple(self._closing)), return_exceptions=True)
        await asyncio.wait((closing, *(owner.settled for owner in requests)))
        for error in closing.result():
            if isinstance(error, Exception):
                _report_close_error(error)
