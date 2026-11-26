"""MiniMax OpenAI 兼容 Chat Completions。"""

from collections.abc import Sequence
from typing import Any

import httpx

from ..config import normalize_minimax_base_url
from ..protocol import Message, ModelError, ToolSpec
from ._chat_completions import ChatCompletions
from ._request import make_replay_key


_ADAPTER_ID = "minimax"
_REPLAY_NAMESPACE = b"agent-companion/minimax-replay"  # 明文回放命名空间，不是凭据。


class MiniMax(ChatCompletions):
    adapter = _ADAPTER_ID

    def __init__(self, client: httpx.AsyncClient, *, model: str, api_key: str, base_url: str | None = None,
                 reasoning_effort: str | None = None) -> None:
        base_url = normalize_minimax_base_url(base_url)
        super().__init__(client, model=model, api_key=api_key, endpoint=base_url + "/chat/completions",
                         replay_key=make_replay_key(_REPLAY_NAMESPACE, _ADAPTER_ID, model, base_url))
        self._effort = reasoning_effort

    def _payload(self, messages: Sequence[Message], tools: Sequence[ToolSpec], thinking: bool) -> dict[str, Any]:
        payload = self._request(messages, tools)
        payload["thinking"] = {"type": "adaptive" if thinking else "disabled"}
        payload["reasoning_split"] = True  # 不把 <think> 混进用户可见正文。
        if thinking and self._effort is not None:
            payload["reasoning_effort"] = self._effort
        return payload

    def _validate_finish(self, finish: str | None, *, has_tools: bool) -> None:
        # 自然结束与工具停止是两种成功终态；达到长度/内容限制不能提交回放。
        if finish is not None and finish not in ("stop", "tool_calls"):
            raise ModelError("abnormal_stream_finish")
        if finish != ("tool_calls" if has_tools else "stop"):
            raise ModelError("invalid_stream_finish")

    def _error_fields(self, item: dict[str, Any]) -> tuple[int, str | None] | None:
        status = item.get("base_resp")
        if isinstance(status, dict):
            code, message = status.get("status_code"), status.get("status_msg")
            if type(code) is int and code != 0:
                return code, message if isinstance(message, str) else None
        return None
