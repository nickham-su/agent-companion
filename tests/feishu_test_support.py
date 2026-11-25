from dataclasses import replace
import json

from agent_companion.channels.feishu.cli import CommandResult, FeishuDiscovery
from agent_companion.runtime.subtasks import demo_roles


def private_discovery(_chat_id, sender_open_id, own_open_id, *, cli=None, search_runner=None):
    """账号查询、空收件或指定合成搜索响应，不调用真实飞书。"""
    async def runner(argv, data, timeout):
        assert argv[-4:] == ("--as", "user", "--format", "json")
        if argv[1:4] == ("api", "GET", "/open-apis/authen/v1/user_info"):
            payload = {"open_id": own_open_id}
        elif search_runner is not None:
            return await search_runner(argv, data, timeout)
        elif argv[1:4] == ("api", "POST", "/open-apis/im/v1/messages/search"):
            payload = {"items": [], "has_more": False}
        else:
            raise AssertionError("unexpected Feishu CLI command")
        return CommandResult(0, json.dumps({"ok": True, "identity": "user", "data": payload}).encode(), b"")

    class Discovery(FeishuDiscovery):
        def for_user(self, user):
            if user.key != sender_open_id:
                raise AssertionError("unexpected Feishu peer")
            if cli is not None:
                return cli
            return super().for_user(user)

    return Discovery("lark-cli", timeout=2, runner=runner)


def configured_roles(model_id="chat"):
    return {name: replace(role, model_id=model_id) for name, role in demo_roles().items()}
