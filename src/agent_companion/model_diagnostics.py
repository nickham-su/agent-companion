"""模型请求诊断；厂商报错与固定的失败分类分开。"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import logging
from typing import Literal

import httpx

from .protocol import ModelError


_LOG = logging.getLogger(__name__)
AdapterID = Literal["deepseek", "openai_responses", "minimax", "bailian"]
_HTTP_ERRORS = (
    (httpx.ConnectTimeout, "connect_timeout"),
    (httpx.ReadTimeout, "read_timeout"),
    (httpx.WriteTimeout, "write_timeout"),
    (httpx.PoolTimeout, "pool_timeout"),
    (httpx.TimeoutException, "transport_timeout"),
    (httpx.ConnectError, "connect_error"),
    (httpx.ReadError, "read_error"),
    (httpx.WriteError, "write_error"),
    (httpx.RemoteProtocolError, "remote_protocol_error"),
    (httpx.LocalProtocolError, "local_protocol_error"),
    (httpx.ProxyError, "proxy_error"),
    (httpx.DecodingError, "decoding_error"),
    (httpx.RequestError, "transport_other"),
)


def failure_category(error: BaseException) -> str:
    for kind, category in _HTTP_ERRORS:
        if isinstance(error, kind):
            return category
    if isinstance(error, ModelError):
        reason = error.diagnostic
        if reason.startswith("http_"):
            return "http_status"
        if reason == "transport_close_timeout":
            return "close_timeout"
        if reason == "transport_error":
            return "transport_other"
        return "model_error"
    if isinstance(error, TimeoutError):
        return "runtime_timeout"
    if isinstance(error, UnicodeError):
        return "unicode_error"
    if isinstance(error, ValueError):
        return "value_error"
    if isinstance(error, TypeError):
        return "type_error"
    return "unexpected_error"


class VendorResponseError(ModelError):
    """只有适配器验证过的厂商字段进日志，异常本身仍只含固定分类。"""

    def __init__(self, reason: str, *, adapter: AdapterID, http_status: int, request_id: str | None,
                 code: int | str | None, message: str | None) -> None:
        super().__init__(reason)
        self._adapter = adapter
        self._status = http_status
        self._request_id = request_id
        self._code = code
        self._message = message

    def log_response(self) -> None:
        details = []
        if self._request_id is not None:
            details.append(f" request_id={self._request_id}")
        if self._code is not None:
            details.append(f" code={self._code}")
        if self._message is not None:
            details.append(f" message={self._message!r}")
        _LOG.warning("vendor model response failed; adapter=%s http_status=%s%s",
                     self._adapter, self._status, "".join(details))


@dataclass(slots=True, repr=False)
class ModelRequestDiagnostics:
    purpose: Literal["reply", "summary"]
    timeout: float
    run_id: str | None = None
    reply_id: str | None = None
    adapter: AdapterID | Literal["unknown"] = "unknown"
    stage: Literal["unknown", "request_build", "http_wait", "http_response", "sse_read", "response_close"] = "unknown"
    body_bytes: int | None = None
    http_status: int | None = None
    transport_failure: str | None = None
    close_failure: str | None = None

    @contextmanager
    def activate(self) -> Iterator[None]:
        token = _CURRENT.set(self)
        try:
            yield
        finally:
            _CURRENT.reset(token)

    def request_ready(self, body_bytes: int) -> None:
        self.body_bytes = body_bytes
        self.stage = "http_wait"

    def response_received(self, status: int) -> None:
        self.http_status = status if type(status) is int and 100 <= status <= 599 else None
        self.stage = "http_response"

    def failed(self, error: Exception) -> None:
        # Runtime 接受失败后才打印厂商报错；退役读流不报告迟到的厂商内容。
        if isinstance(error, VendorResponseError):
            error.log_response()
        if isinstance(error, ModelError):
            reason = error.diagnostic
            category = self.transport_failure if reason == "transport_error" else None
        else:
            reason = "model_timeout" if isinstance(error, TimeoutError) else "model_failed"
            category = None
        _LOG.debug("model request failure; purpose=%s run=%s reply=%s adapter=%s stage=%s "
                   "reason=%s category=%s http_status=%s body_bytes=%s timeout_s=%g close_failure=%s",
                   self.purpose, self.run_id or "-", self.reply_id or "-", self.adapter, self.stage,
                   reason, category or failure_category(error), self.http_status or "-",
                   "-" if self.body_bytes is None else self.body_bytes, self.timeout, self.close_failure or "-")


_CURRENT: ContextVar[ModelRequestDiagnostics | None] = ContextVar("model_request_diagnostics", default=None)


def current_request() -> ModelRequestDiagnostics | None:
    return _CURRENT.get()


def adapter_started(adapter: AdapterID) -> ModelRequestDiagnostics | None:
    diagnostics = current_request()
    if diagnostics is not None:
        diagnostics.adapter = adapter
        diagnostics.stage = "request_build"
    return diagnostics
