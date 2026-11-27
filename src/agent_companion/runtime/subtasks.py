"""组装同一主 Agent 下的运行时，处理异步委派与直属回投。"""

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
import logging
from typing import Any

from ..config import RuntimeSettings
from ..protocol import AgentReply, Channel, DelegatedInput, Message, MessageDelivery, Model, ToolSpec
from ..session_scheduler import SessionScheduler, SessionSchedulers
from ..storage.sqlite import Agent, History
from ..workspace import Workspace
from .bash import BASH_TOOL_NAME, Bash
from .compaction import Compactions
from .engine import Runtime
from .sqlite_access import SQLiteAccess
from .timers import CANCEL_TIMER_TOOL_NAME, SET_TIMER_TOOL_NAME, Timers
from .tools import Tool, ToolContext, Tools
from .view_image import VIEW_IMAGE_TOOL


_LOG = logging.getLogger(__name__)
DELEGATE_TOOL_NAME = "delegate_subtask"
AGENT_TOOL_NAMES = frozenset({
    DELEGATE_TOOL_NAME, SET_TIMER_TOOL_NAME, CANCEL_TIMER_TOOL_NAME, BASH_TOOL_NAME, VIEW_IMAGE_TOOL.spec.name,
})


@dataclass(frozen=True, slots=True)
class AgentRole:
    instructions: str | None = field(repr=False)
    tools: tuple[str, ...] = ()
    description: str = ""
    model_id: str | None = None  # 内置/替身角色可不引用生产档案。


@dataclass(frozen=True, slots=True)
class ModelBinding:
    model: Model
    settings: RuntimeSettings
    thinking: bool = False
    origin: bytes | None = field(default=None, repr=False)


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

    def __init__(self, root: Agent, history: History, models: Mapping[str, ModelBinding], *, roles: Mapping[str, AgentRole],
                 schedulers: SessionSchedulers,
                 retry_sqlite_locks: bool, compactions: Compactions, workspace: Workspace | None = None) -> None:
        self._root = root
        self._history = history
        self._models = dict(models)
        self._roles = dict(roles)
        self._schedulers = schedulers
        self._retry_sqlite_locks = retry_sqlite_locks
        self._compactions = compactions
        self._bindings: dict[str, _Binding] = {}
        self._closed = False
        self._workspace = workspace
        self._bash = Bash()  # 主、子 Agent 共用关停状态，不共用执行锁。
        self._sqlite = SQLiteAccess(retry_locks=retry_sqlite_locks, stopped=lambda: self._closed)
        descriptions = {name: role.description for name, role in sorted(self._roles.items()) if name != "main"}
        self._delegate_tool = self._make_delegate_tool(descriptions)

    def _make_delegate_tool(self, descriptions: dict[str, str]) -> Tool:
        role_property: dict[str, Any] = {"type": "string", "description": "创建指定角色"}
        if descriptions:
            choices = "；".join(f"{role}：{description}" for role, description in descriptions.items())
            role_property["enum"] = list(descriptions)
            availability = f"创建请选择角色：{choices}。"
        else:
            availability = "当前没有可用于创建子 Agent 的角色。"
        return Tool(ToolSpec(
            DELEGATE_TOOL_NAME,
            "将工作异步委派给直属子 Agent。工具立即返回受理状态和 agent_id，回复随后作为内部输入送达。"
            f"{availability}复用请提供 agent_id。",
            {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "任务与必要上下文"},
                    "agent": role_property,
                    "agent_id": {"type": "string", "description": "复用已存在的直属子 Agent"},
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
        ), self._delegate)

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
            available = {DELEGATE_TOOL_NAME: self._delegate_tool, BASH_TOOL_NAME: self._bash.tool,
                         VIEW_IMAGE_TOOL.spec.name: VIEW_IMAGE_TOOL}
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
                return parent.scheduler.submit_internal(MessageDelivery(notice.to_history_message(), "trigger"))

            model = self._models[agent.role_id]
            runtime = Runtime(agent.session_id, self._history, model.model, channel, tools=tools,
                              settings=model.settings, thinking=model.thinking, model_origin=model.origin,
                              instructions=role.instructions,
                              retry_sqlite_locks=self._retry_sqlite_locks, compactions=self._compactions,
                              workspace=self._workspace, agent=agent,
                              reply_to_parent=reply_to_parent if parent_id is not None else None)
            revoke = scheduler.bind(runtime.submit_messages, owner=owner,
                                    agent_id=agent.agent_id, agent_kind="main" if parent_id is None else "child",
                                    trigger=runtime.trigger_run if parent_id is None else None,
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
            binding.scheduler.session_input((MessageDelivery(
                Message("user", prompt), "trigger", delegation=DelegatedInput(delegation.delegation_id)),))
        except Exception as exc:
            _LOG.warning("delegation handoff failed; delegation=%s error_type=%s",
                         delegation.delegation_id, type(exc).__name__)
            cleanup = SQLiteAccess(retry_locks=self._retry_sqlite_locks, stopped=lambda: False)
            await cleanup.write_input(lambda: self._history.interrupt_delegation(
                delegation.delegation_id, reason="delegation_handoff_failed"))
            return '{"error":"delegation_handoff_failed"}'
        _LOG.debug("delegation accepted; agent=%s delegation=%s parent_run=%s",
                   child.agent_id, delegation.delegation_id, context.run_id)
        return json.dumps({"status": "accepted", "agent_id": child.agent_id})

    async def wait_idle(self) -> None:
        """供离线验收使用：等待包括回投唤起的运行均停止。"""
        while True:
            await asyncio.gather(*(binding.runtime.wait_idle() for binding in self._bindings.values()))
            if all(binding.runtime.idle for binding in self._bindings.values()):
                return

    def stop_accepting(self) -> None:
        self._closed = True
        self._bash.stop_accepting()
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
