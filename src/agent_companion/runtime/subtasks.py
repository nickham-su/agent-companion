"""组装同一主 Agent 下的运行时，处理异步委派与直属回投。"""

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
import logging
from typing import Any

from ..config import RuntimeSettings
from ..protocol import AgentReply, Channel, DelegationTask, InternalInput, Model, ToolSpec
from ..session_scheduler import SessionScheduler, SessionSchedulers
from ..storage.sqlite import Agent, History
from .compaction import Compactions
from .engine import Runtime
from .sqlite_access import SQLiteAccess
from .timers import CANCEL_TIMER_TOOL_NAME, SET_TIMER_TOOL_NAME, Timers
from .tools import Tool, ToolContext, Tools


_LOG = logging.getLogger(__name__)
DELEGATE_TOOL_NAME = "delegate_subtask"
AGENT_TOOL_NAMES = frozenset({DELEGATE_TOOL_NAME, SET_TIMER_TOOL_NAME, CANCEL_TIMER_TOOL_NAME})


@dataclass(frozen=True, slots=True)
class AgentRole:
    instructions: str | None = field(repr=False)
    tools: tuple[str, ...] = ()
    description: str = ""


def demo_roles() -> dict[str, AgentRole]:
    return {
        "main": AgentRole(None, (DELEGATE_TOOL_NAME, SET_TIMER_TOOL_NAME, CANCEL_TIMER_TOOL_NAME)),
        "general": AgentRole("你是内部子 Agent。完成提交的任务，并将结果回复给直属父级。",
                             description="执行独立的分析或撰写任务"),
    }


@dataclass(slots=True)
class _Binding:
    runtime: Runtime
    scheduler: SessionScheduler
    owner: object
    revoke: Callable[[], None]


class AgentTree:
    """一个已绑定主路由的 Agent 树；各 Agent 的唯一工作队列由 Runtime 持有。"""

    def __init__(self, root: Agent, history: History, model: Model, *, roles: Mapping[str, AgentRole],
                 schedulers: SessionSchedulers, settings: RuntimeSettings, thinking: bool,
                 retry_sqlite_locks: bool, compactions: Compactions) -> None:
        self._root = root
        self._history = history
        self._model = model
        self._roles = dict(roles)
        self._schedulers = schedulers
        self._settings = settings
        self._thinking = thinking
        self._retry_sqlite_locks = retry_sqlite_locks
        self._compactions = compactions
        self._bindings: dict[str, _Binding] = {}
        self._closed = False
        self._sqlite = SQLiteAccess(retry_locks=retry_sqlite_locks, stopped=lambda: self._closed)
        descriptions = {name: role.description for name, role in sorted(self._roles.items()) if name != "main"}
        # 没有可选角色时不展示工具，但仍为白名单内的显式复用返回准确错误。
        self._delegate_tool = self._make_delegate_tool(descriptions)

    def _make_delegate_tool(self, descriptions: dict[str, str]) -> Tool:
        choices = "；".join(f"{role}：{description}" for role, description in descriptions.items())
        return Tool(ToolSpec(
            DELEGATE_TOOL_NAME,
            "将工作异步委派给直属子 Agent。工具立即返回受理标识，回复随后作为内部输入送达。"
            f"创建请选择角色：{choices}。复用请提供 agent_id。",
            {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "任务与必要上下文"},
                    "agent": {"type": "string", "enum": list(descriptions), "description": "创建指定角色"},
                    "agent_id": {"type": "string", "description": "复用已存在的直属子 Agent"},
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
        ), self._delegate, advertised=bool(descriptions))

    def bind_root(self, channel: Channel, owner: object) -> Runtime:
        return self._bind(self._root, channel=channel, owner=owner).runtime

    def _bind(self, agent: Agent, *, channel: Channel | None = None, owner: object | None = None) -> _Binding:
        if self._closed:
            raise RuntimeError("agent tree is closed")
        scheduler = self._schedulers.get_or_create(agent.session_id)
        reserved_here = owner is None
        if owner is None:
            owner = scheduler.reserve_agent()
        try:
            role = self._roles[agent.role_id]
            timers = None
            available = {DELEGATE_TOOL_NAME: self._delegate_tool}
            if SET_TIMER_TOOL_NAME in role.tools or CANCEL_TIMER_TOOL_NAME in role.tools:
                timers = Timers(agent.session_id, scheduler.submit_internal)
                available.update((tool.spec.name, tool) for tool in timers.tools)
            tools = Tools(available[name] for name in role.tools if name in available)
            parent_id = agent.parent_agent_id

            def reply_to_parent(notice: AgentReply) -> bool:
                parent = self._bindings.get(parent_id) if parent_id is not None else None
                if self._closed or parent is None:
                    return False
                if parent.runtime.input_retry_forbidden:
                    _LOG.warning("agent reply not delivered; reason=parent_input_retry_forbidden "
                                 "agent=%s run=%s parent=%s",
                                 notice.agent_id, notice.run_id, parent_id)
                    return False
                return parent.scheduler.submit_internal(InternalInput(notice))

            runtime = Runtime(agent.session_id, self._history, self._model, channel, tools=tools,
                              settings=self._settings, thinking=self._thinking, instructions=role.instructions,
                              retry_sqlite_locks=self._retry_sqlite_locks, compactions=self._compactions,
                              agent=agent, reply_to_parent=reply_to_parent if parent_id is not None else None)
            revoke = scheduler.bind(runtime.submit_inputs, owner=owner,
                                    on_revoke=timers.close if timers is not None else None)
            binding = self._bindings[agent.agent_id] = _Binding(runtime, scheduler, owner, revoke)
            return binding
        except BaseException:
            if reserved_here:
                self._schedulers.release_child(agent.session_id, scheduler, owner)
            raise

    async def _delegate(self, args: dict[str, Any], context: ToolContext) -> str:
        if self._closed:
            return '{"error":"delegation_closed"}'
        if not (context.has_user_input or context.has_agent_input):
            return '{"error":"delegation_requires_user_or_agent_input"}'
        parent = self._bindings.get(context.agent_id) if context.agent_id is not None else None
        if parent is None or parent.runtime.session_id != context.session_id:
            return '{"error":"unknown_delegating_agent"}'
        prompt = args.get("prompt")
        if (args.keys() - {"prompt", "agent", "agent_id"}
                or not isinstance(prompt, str) or not prompt.strip()
                or ("agent" in args) == ("agent_id" in args)):
            return '{"error":"invalid_arguments"}'
        # SQLite 和 UTF-8 传输都不能写入孤立代理项。
        try:
            prompt.encode("utf-8")
        except UnicodeEncodeError:
            return '{"error":"invalid_arguments"}'
        role_id = target_id = None
        if "agent" in args:
            role_id = args["agent"]
            if not isinstance(role_id, str) or not role_id.strip():
                return '{"error":"invalid_arguments"}'
            if role_id == "main" or role_id not in self._roles:
                return '{"error":"unknown_agent_role"}'
        else:
            target_id = args["agent_id"]
            if not isinstance(target_id, str) or not target_id.strip():
                return '{"error":"invalid_arguments"}'
            try:
                target_id.encode("utf-8")
            except UnicodeEncodeError:
                return '{"error":"invalid_arguments"}'
            child = await self._sqlite.write_input(lambda: self._history.agent(target_id))
            if child is None or child.parent_agent_id != context.agent_id:
                return '{"error":"unknown_child_agent"}'
            if child.role_id not in self._roles:
                return '{"error":"agent_role_unavailable"}'

        accepted = await self._sqlite.write_input(lambda: self._history.accept_delegation(
            context.agent_id, context.run_id, context.call_id, role_id=role_id, agent_id=target_id))
        if accepted is None:
            return '{"error":"unknown_child_agent"}'
        child, delegation = accepted
        try:
            if self._closed:
                raise RuntimeError("agent tree is closed")
            binding = self._bindings.get(child.agent_id) or self._bind(child)
            if binding.runtime.input_retry_forbidden:
                raise RuntimeError("agent input cannot be safely retried")
            binding.scheduler.session_input((InternalInput(DelegationTask(delegation.delegation_id, prompt)),))
        except Exception as exc:
            _LOG.warning("delegation handoff failed; delegation=%s error_type=%s",
                         delegation.delegation_id, type(exc).__name__)
            cleanup = SQLiteAccess(retry_locks=self._retry_sqlite_locks, stopped=lambda: False)
            await cleanup.write_input(lambda: self._history.interrupt_delegation(
                delegation.delegation_id, reason="delegation_handoff_failed"))
            return '{"error":"delegation_handoff_failed"}'
        _LOG.debug("delegation accepted; agent=%s delegation=%s parent_run=%s",
                   child.agent_id, delegation.delegation_id, context.run_id)
        return json.dumps({"status": "accepted", "agent_id": child.agent_id, "delegation_id": delegation.delegation_id})

    async def wait_idle(self) -> None:
        """供离线验收使用：等待包括回投唤起的运行均停止。"""
        while True:
            await asyncio.gather(*(binding.runtime.wait_idle() for binding in self._bindings.values()))
            if all(binding.runtime.idle for binding in self._bindings.values()):
                return

    def stop_accepting(self) -> None:
        self._closed = True
        for binding in self._bindings.values():
            binding.runtime.stop_accepting()
            # 保留所有权，直到异步关停与状态收尾完成。
            binding.revoke()

    async def close(self) -> None:
        self.stop_accepting()
        bindings = tuple(self._bindings.values())
        try:
            outcomes = await asyncio.gather(*(binding.runtime.close() for binding in bindings), return_exceptions=True)
        finally:
            try:
                cleanup = SQLiteAccess(retry_locks=self._retry_sqlite_locks, stopped=lambda: False)
                await cleanup.write_input(lambda: self._history.interrupt_agent_tree(self._root.agent_id,
                                                                                     reason="shutdown"))
            finally:
                for binding in bindings:
                    if binding.runtime.agent != self._root:
                        self._schedulers.release_child(binding.runtime.session_id, binding.scheduler,
                                                       binding.owner)
                self._bindings.clear()
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
