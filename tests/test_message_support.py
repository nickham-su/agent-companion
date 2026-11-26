"""构造带有有序内容片段的测试消息。"""

from agent_companion.protocol import Message, Reasoning, Text


def assistant_message(text="", calls=(), *, reasoning=None):
    parts = ((Reasoning(reasoning),) if reasoning is not None else ()) + ((Text(text),) if text else ())
    return Message("assistant", parts + tuple(calls))


def reasoning_parts(message):
    return tuple(part.private for part in message.parts if isinstance(part, Reasoning))


def tool_calls(messages):
    return [call for message in messages for call in message.tool_calls]
