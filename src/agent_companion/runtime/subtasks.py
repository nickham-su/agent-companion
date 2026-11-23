"""只在本进程中调度内部模型任务；不会连接用户通道。"""

import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
import json
import logging
from typing import Any
from uuid import uuid4

from ..config import RuntimeSettings
from ..protocol import Message, Model, ToolSpec
from ..storage.sqlite import History
from .engine import Runtime
from .tools import Tool, ToolContext, Tools


_LOG = logging.getLogger(__name__)
_FINISH_ATTEMPTS = 3
_MAX_OUTSTANDING = 8  # 含正在执行和已受理排队的任务
_MAX_CONCURRENT = 2
DELEGATE_TOOL_NAME = "delegate_subtask"
RESULT_PREFIX = "[内部子任务结果]\n"
_DELEGATE = ToolSpec(
    DELEGATE_TOOL_NAME,
    "把需要独立分析或撰写的工作提交给后台模型，不执行命令。工具只返回受理信息；"
    "结果稍后作为内部输入回到主会话。只有纳入新用户输入的运行才可委派。",
    {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "为子会话提供任务及必要上下文"},
            "child_session_id": {"type": "string", "description": "复用本主会话已有的子会话，首次委派省略"},
        },
        "required": ["prompt"],
        "additionalProperties": False,
    },
)


@dataclass(frozen=True, slots=True)
class _Job:
    run_id: str
    prompt: str = field(repr=False)


class Subtasks:
    """一个主会话对应一个调度器；同一子会话共享一条临时队列。"""

    def __init__(self, parent_session_id: str, history: History, model: Model,
                 submit_result: Callable[[Message], None], *, settings: RuntimeSettings,
                 thinking: bool) -> None:
        self._parent = parent_session_id
        self._history = history
        self._model = model
        self._submit_result = submit_result
        self._settings = settings
        self._thinking = thinking
        self._queues: dict[str, deque[_Job]] = {}
        self._workers: dict[str, asyncio.Task[None]] = {}
        self._closed = False
        self._outstanding = 0
        self._slots = asyncio.Semaphore(_MAX_CONCURRENT)
        self.tools = Tools([Tool(_DELEGATE, self._delegate)])
        # 数据库只记录执行，不驱动补跑或重放；重启时把孤立任务标记为中断。
        history.interrupt_subtasks(parent_session_id, reason="restart")

    async def _delegate(self, args: dict[str, Any], context: ToolContext) -> str:
        if context.session_id != self._parent:
            return '{"error":"not_parent_session"}'
        if self._closed:
            return '{"error":"delegation_closed"}'
        if not context.has_user_input:
            return '{"error":"delegation_requires_user_input"}'
        prompt = args.get("prompt")
        if (args.keys() - {"prompt", "child_session_id"}
                or not isinstance(prompt, str) or not prompt.strip()):
            return '{"error":"invalid_arguments"}'
        if "child_session_id" in args:
            child = args["child_session_id"]
            if not isinstance(child, str) or not child.strip():
                return '{"error":"invalid_arguments"}'
            if child == self._parent or not self._history.can_resume_subtask(self._parent, child):
                return '{"error":"unknown_child_session"}'
        else:
            child = f"child-{uuid4().hex}"
        if self._outstanding >= _MAX_OUTSTANDING:
            return '{"error":"delegation_capacity_reached"}'
        run_id = uuid4().hex
        loop = asyncio.get_running_loop()
        self._history.accept_subtask(child, run_id, self._parent, context.run_id, context.call_id)
        self._outstanding += 1
        queue = self._queues.setdefault(child, deque())
        queue.append(_Job(run_id, prompt))
        if child not in self._workers:
            self._workers[child] = loop.create_task(self._drain(child, queue))
        return json.dumps({"status": "accepted", "child_session_id": child, "run_id": run_id})

    async def _drain(self, child: str, queue: deque[_Job]) -> None:
        try:
            runtime = Runtime(child, self._history, self._model, None,
                              settings=self._settings, thinking=self._thinking,
                              instructions="你是内部子会话。完成提交的任务并返回结果，不调用工具或联系用户。")
            while queue and not self._closed:
                job = queue.popleft()
                try:
                    async with self._slots:
                        if self._closed:
                            break
                        await self._run_job(child, runtime, job)
                except Exception:
                    _LOG.error("subtask state or result enqueue could not be recorded")
                finally:
                    self._outstanding -= 1
        finally:
            self._outstanding -= len(queue)
            self._workers.pop(child, None)
            self._queues.pop(child, None)

    async def _run_job(self, child: str, runtime: Runtime, job: _Job) -> None:
        self._history.start_subtask(job.run_id)
        try:
            error = await runtime.run_internal(job.run_id, job.prompt)
        except asyncio.CancelledError:
            if self._closed:
                raise
            error = "subtask_cancelled"
        result = self._history.run_text(child, job.run_id) if error is None else None
        if result is None and error is None:
            error = "empty_result"
        if not await self._persist_terminal(job.run_id, result=result, error=error):
            return
        if not self._closed:
            payload = {"child_session_id": child, "run_id": job.run_id}
            if error is not None:
                payload.update(status="failed", error=error)
            else:
                payload.update(status="completed", result=result)
            # 走主队列，而不是在历史中伪造 tool 结果或直接触碰通道。
            self._submit_result(Message("user", RESULT_PREFIX + json.dumps(payload, ensure_ascii=False)))
            self._history.mark_subtask_enqueued(job.run_id)

    async def _persist_terminal(self, run_id: str, *, result: str | None, error: str | None) -> bool:
        # 只重试这个 SQLite 写入，不重新执行子任务或回投。
        for attempt in range(1, _FINISH_ATTEMPTS + 1):
            try:
                self._history.finish_subtask(run_id, result=result, error=error)
            except Exception as exc:
                if attempt == _FINISH_ATTEMPTS:
                    _LOG.error(
                        "subtask terminal write failed: run_id=%s attempts=%d error_type=%s; result not enqueued",
                        run_id, attempt, type(exc).__name__,
                    )
                    return False
                await asyncio.sleep(0.02)
            else:
                return True

    async def wait_idle(self) -> None:
        while active := {task for task in self._workers.values() if not task.done()}:
            await asyncio.wait(active)

    def stop_accepting(self) -> None:
        """同步阻止新委派和新回投。"""
        self._closed = True

    async def close(self) -> None:
        self.stop_accepting()
        workers = tuple(self._workers.values())
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        self._workers.clear()
        self._queues.clear()
        self._outstanding = 0
        self._history.interrupt_subtasks(self._parent, reason="shutdown")
