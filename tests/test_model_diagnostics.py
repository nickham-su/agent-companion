import asyncio
from contextlib import closing
from io import StringIO
import unittest
from unittest.mock import patch
from uuid import uuid4

import httpx

from agent_companion.config import RuntimeSettings
from agent_companion.logging_setup import stdout_logging
from agent_companion.models.deepseek import DeepSeek
from agent_companion.models.openai_responses import OpenAIResponses
from agent_companion.protocol import Message, MessageDelivery, ModelError
from agent_companion.runtime.compaction import Compactions, Prefix
from agent_companion.runtime.engine import Runtime
from agent_companion.storage.sqlite import History


_INPUT = "synthetic-hidden-input-示例"
_SESSION = "synthetic-hidden-session"
_SECRET = "synthetic-hidden-credential"
_MODEL = "synthetic-hidden-model-name"
_HEADER = "synthetic-hidden-header"
_URL_HOST = "synthetic-hidden-url.invalid"
_RAW_ERROR = "synthetic-hidden-raw-error"
_WIRE = "synthetic-hidden-wire-data"
_MARKER = "model request failure; "


def single_attempt_runtime(history, model, **kwargs):
    # 用真实子 Agent 验证单次请求的终态诊断，不关闭主 Agent 的重试。
    parent = history.ensure_main_agent(_SESSION)
    run = uuid4().hex
    history.start_agent_run(parent.agent_id, run, can_start=lambda: True)
    child, _ = history.accept_delegation(parent.agent_id, run, "test-call", role_id="review")
    return Runtime(child.session_id, history, model, None, agent=child, reply_to_parent=lambda _: True, **kwargs)


class Wire(httpx.AsyncByteStream):
    def __init__(self, *, frame=b"", error=None, close_error=None, stall_close=False):
        self.frame = frame
        self.error = error
        self.close_error = close_error
        self.stall_close = stall_close

    async def __aiter__(self):
        if self.error is not None:
            raise self.error
        yield self.frame

    async def aclose(self):
        if self.stall_close:
            await asyncio.Future()
        if self.close_error is not None:
            raise self.close_error


def response(status, *, frame=b"", error=None, close_error=None, stall_close=False):
    return httpx.Response(status, headers={"Content-Type": "text/event-stream", "x-request-id": _HEADER},
                          stream=Wire(frame=frame, error=error, close_error=close_error, stall_close=stall_close))


class ModelDiagnosticTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def model(adapter, client):
        if adapter == "deepseek":
            return DeepSeek(client, model=_MODEL)
        return OpenAIResponses(client, model=_MODEL, api_key=_SECRET,
                               reasoning_effort="low", base_url=f"https://{_URL_HOST}/v1")

    async def submit(self, runtime, history, session=_SESSION):
        try:
            runtime.submit_messages([MessageDelivery(Message("user", _INPUT), "trigger")])
            await asyncio.wait_for(runtime.wait_idle(), 2)
            outcome, = history.outcomes(session)
            return outcome
        finally:
            await runtime.close()

    async def mock_reply(self, adapter, endpoint, *, timeout=1, prepare_error=None):
        async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
            model = self.model(adapter, client)
            with closing(History(":memory:")) as history:
                runtime = single_attempt_runtime(history, model, settings=RuntimeSettings(model_timeout=timeout))
                try:
                    if prepare_error is None:
                        return await self.submit(runtime, history, runtime.session_id)
                    with patch.object(model, "_payload", side_effect=prepare_error):
                        return await self.submit(runtime, history, runtime.session_id)
                finally:
                    await model.aclose()

    def assert_safe_logs(self, logged):
        output = "\n".join(logged.output)
        for sentinel in (_INPUT, _SESSION, _SECRET, _MODEL, _HEADER, _URL_HOST, _RAW_ERROR, _WIRE,
                         "SyntheticHiddenExceptionClass"):
            self.assertNotIn(sentinel, output)
        for record in logged.records:
            self.assertIsNone(record.exc_info)
            self.assertIsNone(record.stack_info)
        return [dict(field.split("=", 1) for field in line.split(_MARKER, 1)[1].split())
                for line in logged.output if _MARKER in line]

    def assert_reply_fields(self, row, outcome, **expected):
        self.assertEqual((row["purpose"], row["run"], row["reply"]), ("reply", outcome.run_id, outcome.reply_id))
        for name, value in expected.items():
            self.assertEqual(row[name], value, name)

    async def test_http_status_and_existing_request_byte_count_for_both_adapters(self):
        for adapter, status in (("deepseek", 429), ("openai_responses", 401)):
            with self.subTest(adapter=adapter):
                sizes = []

                def endpoint(request):
                    self.assertIn(_INPUT.encode(), request.content)
                    sizes.append(str(len(request.content)))
                    return response(status, frame=_WIRE.encode())

                with self.assertLogs("agent_companion", "DEBUG") as logged:
                    outcome = await self.mock_reply(adapter, endpoint)
                row, = self.assert_safe_logs(logged)
                self.assertEqual(outcome.model_error, f"http_{status}")
                self.assert_reply_fields(row, outcome, adapter=adapter, stage="http_response", http_status=str(status),
                                         reason=outcome.model_error, category="http_status", timeout_s="1",
                                         body_bytes=sizes[0], close_failure="-")

    async def test_build_failure_and_unknown_adapter_do_not_log_raw_exception(self):
        async def unused(_):
            self.fail("request must not reach the transport")

        with self.assertLogs("agent_companion", "DEBUG") as logged:
            outcome = await self.mock_reply("openai_responses", unused, prepare_error=UnicodeError(_RAW_ERROR))
        row, = self.assert_safe_logs(logged)
        self.assertEqual((outcome.model_status, outcome.model_error), ("model_failed", None))
        self.assert_reply_fields(row, outcome, adapter="openai_responses", stage="request_build",
                                 reason="model_failed", category="unicode_error", body_bytes="-", http_status="-")

        class SyntheticHiddenExceptionClass(RuntimeError):
            pass

        class UnknownModel:
            def stream(self, *_args, **_kwargs):
                raise SyntheticHiddenExceptionClass(_RAW_ERROR)

        with self.assertLogs("agent_companion", "DEBUG") as logged, closing(History(":memory:")) as history:
            outcome = await self.submit(Runtime(_SESSION, history, UnknownModel(), None), history)
        row, = self.assert_safe_logs(logged)
        self.assert_reply_fields(row, outcome, adapter="unknown", stage="unknown", category="unexpected_error")

    async def test_transport_categories_distinguish_headers_from_sse(self):
        for kind, category, stage in (
            (httpx.ConnectError, "connect_error", "http_wait"),
            (httpx.ConnectTimeout, "connect_timeout", "http_wait"),
            (httpx.WriteTimeout, "write_timeout", "http_wait"),
            (httpx.PoolTimeout, "pool_timeout", "http_wait"),
            (httpx.ReadError, "read_error", "sse_read"),
            (httpx.ReadTimeout, "read_timeout", "sse_read"),
            (httpx.RemoteProtocolError, "remote_protocol_error", "sse_read"),
            (httpx.RequestError, "transport_other", "sse_read"),
        ):
            with self.subTest(category=category):
                def endpoint(request):
                    error = kind(_RAW_ERROR, request=request)
                    if stage == "http_wait":
                        raise error
                    return response(200, error=error)

                with self.assertLogs("agent_companion", "DEBUG") as logged:
                    outcome = await self.mock_reply("deepseek", endpoint)
                row, = self.assert_safe_logs(logged)
                self.assertEqual(outcome.model_error, "transport_error")
                self.assert_reply_fields(row, outcome, stage=stage, reason="transport_error", category=category,
                                         http_status="200" if stage == "sse_read" else "-")

    async def test_sse_and_cleanup_failures_preserve_existing_error_precedence(self):
        invalid = f"data: {_WIRE}\n\n".encode()
        with self.assertLogs("agent_companion", "DEBUG") as logged:
            outcome = await self.mock_reply("openai_responses", lambda _: response(200, frame=invalid))
        row, = self.assert_safe_logs(logged)
        self.assertEqual(outcome.model_error, "invalid_sse_event")
        self.assert_reply_fields(row, outcome, stage="sse_read", reason=outcome.model_error, http_status="200")

        with (patch("agent_companion.models._http_stream._RESPONSE_CLOSE_TIMEOUT", .03),
              self.assertLogs("agent_companion", "DEBUG") as logged):
            outcome = await self.mock_reply("deepseek", lambda _: response(503, stall_close=True))
        row, = self.assert_safe_logs(logged)
        self.assertEqual(outcome.model_error, "http_503")
        self.assert_reply_fields(row, outcome, stage="http_response", reason="http_503", close_failure="close_timeout")

        with self.assertLogs("agent_companion", "DEBUG") as logged:
            outcome = await self.mock_reply("openai_responses", lambda _: response(
                503, close_error=httpx.ReadError(_RAW_ERROR)))
        row, = self.assert_safe_logs(logged)
        self.assertEqual(outcome.model_error, "transport_error")
        self.assert_reply_fields(row, outcome, stage="response_close", reason="transport_error", category="read_error",
                                 http_status="503", close_failure="read_error")

    async def test_runtime_deadline_is_not_an_httpx_timeout(self):
        async def wait_for_headers(_):
            await asyncio.Future()

        with self.assertLogs("agent_companion", "DEBUG") as logged:
            outcome = await self.mock_reply("deepseek", wait_for_headers, timeout=.05)
        row, = self.assert_safe_logs(logged)
        self.assertEqual((outcome.model_status, outcome.model_error), ("model_timeout", None))
        self.assert_reply_fields(row, outcome, stage="http_wait", reason="model_timeout", category="runtime_timeout",
                                 timeout_s="0.05", http_status="-")

    async def test_concurrent_summary_uses_its_own_request_scope(self):
        summary_text = "synthetic-hidden-summary-input"
        summary_run = uuid4().hex
        both_started = asyncio.Event()
        sizes = {}

        async def endpoint(request):
            summary = summary_text.encode() in request.content
            sizes[summary] = str(len(request.content))
            if len(sizes) == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), 1)
            if summary:
                return response(429)
            return response(200, error=httpx.ReadError(_RAW_ERROR, request=request))

        async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
            with closing(History(":memory:")) as history, self.assertLogs("agent_companion", "DEBUG") as logged:
                model = self.model("openai_responses", client)
                compactions = Compactions(history)
                try:
                    compactions.start(_SESSION, Prefix(0, 1, 1), [Message("user", summary_text)], (),
                                      model=model, thinking=False, timeout=2, run_id=summary_run)
                    runtime = single_attempt_runtime(history, model, compactions=compactions)
                    outcome = await self.submit(runtime, history, runtime.session_id)
                    if compactions._generating is not None:
                        await asyncio.wait_for(compactions._generating, 2)
                finally:
                    await compactions.close()
                    await model.aclose()
        self.assertNotIn(summary_text, "\n".join(logged.output))
        rows = {row["purpose"]: row for row in self.assert_safe_logs(logged)}
        self.assertEqual(set(rows), {"reply", "summary"})
        self.assert_reply_fields(rows["reply"], outcome, adapter="openai_responses", stage="sse_read",
                                 reason="transport_error", category="read_error", http_status="200",
                                 body_bytes=sizes[False], timeout_s="60")
        self.assertEqual(rows["summary"], {
            "purpose": "summary", "run": summary_run, "reply": "-", "adapter": "openai_responses",
            "stage": "http_response", "reason": "http_429", "category": "http_status", "http_status": "429",
            "body_bytes": sizes[True], "timeout_s": "2", "close_failure": "-",
        })

    async def test_default_warning_and_cancelled_request_do_not_log_debug_failure(self):
        with patch("sys.stdout", new_callable=StringIO) as stdout, stdout_logging("WARNING"):
            await self.mock_reply("deepseek", lambda _: response(401))
        self.assertNotIn(_MARKER, stdout.getvalue())
        self.assertIn("model response failed (http_401)", stdout.getvalue())

        class CancelModel:
            async def stream(self, *_args, **_kwargs):
                if False:
                    yield None
                raise asyncio.CancelledError

        with self.assertLogs("agent_companion", "DEBUG") as logged, closing(History(":memory:")) as history:
            outcome = await self.submit(Runtime(_SESSION, history, CancelModel(), None), history)
        self.assertEqual(outcome.model_status, "model_cancelled")
        self.assertEqual(self.assert_safe_logs(logged), [])

    async def test_unrecognized_error_diagnostic_is_not_echoed(self):
        with self.assertLogs("agent_companion", "DEBUG") as logged:
            outcome = await self.mock_reply("deepseek", lambda _: response(200), prepare_error=ModelError(_RAW_ERROR))
        row, = self.assert_safe_logs(logged)
        self.assertEqual(outcome.model_error, "model_failed")
        self.assert_reply_fields(row, outcome, stage="request_build", reason="model_failed", category="model_error")
