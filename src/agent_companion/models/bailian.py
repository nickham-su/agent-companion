"""百炼业务空间的 OpenAI 兼容 Chat Completions，只支持明文推理。"""

from collections.abc import Sequence
from typing import Any

import httpx

from ..protocol import Message, ModelError, ToolSpec
from ._chat_completions import ChatCompletions
from ._request import make_replay_key


_ADAPTER_ID = "bailian"
_REPLAY_NAMESPACE = b"agent-companion/bailian-replay"  # 明文不绑定 API Key 或业务空间。


class Bailian(ChatCompletions):
    adapter = _ADAPTER_ID
    include_tool_index = True

    def __init__(self, client: httpx.AsyncClient, *, model: str, api_key: str, base_url: str,
                 preserve_thinking: bool | None = None) -> None:
        super().__init__(client, model=model, api_key=api_key, endpoint=base_url + "/chat/completions",
                         replay_key=make_replay_key(_REPLAY_NAMESPACE, _ADAPTER_ID, model))
        self._preserve_thinking = preserve_thinking

    def _reasoning(self, message: Message) -> str | None:
        if self._preserve_thinking is False:
            return None
        return super()._reasoning(message)

    def _payload(self, messages: Sequence[Message], tools: Sequence[ToolSpec], thinking: bool, *,
                 instructions: str | None = None) -> dict[str, Any]:
        payload = self._request(messages, tools, instructions=instructions)
        payload["enable_thinking"] = thinking
        if self._preserve_thinking is not None:
            payload["preserve_thinking"] = self._preserve_thinking
        return payload

    def _validate_finish(self, finish: str | None, *, has_tools: bool) -> None:
        # 百炼的兼容接口以 [DONE] 收束；length/content_filter 不算完整输出。
        if finish is not None and finish not in ("stop", "tool_calls"):
            raise ModelError("abnormal_stream_finish")
        if finish != ("tool_calls" if has_tools else "stop"):
            raise ModelError("invalid_stream_finish")

    def _error_fields(self, item: dict[str, Any]) -> tuple[str | None, str | None] | None:
        error = item.get("error")
        if not isinstance(error, dict):
            return None
        code, message = error.get("code"), error.get("message")
        return (code if isinstance(code, str) else None, message if isinstance(message, str) else None)

    def _request_ids(self, response: httpx.Response, item: dict[str, Any]) -> tuple[object, ...]:
        return (item.get("request_id"), *super()._request_ids(response, item))
