import json

from agent_companion.channels.feishu.cli import CommandResult, FeishuDiscovery


def private_discovery(chat_id, sender_open_id, own_open_id, *, cli=None):
    """用真实发现解析器处理合成响应，必要时注入原测试的收发替身。"""
    async def runner(argv, data, timeout):
        assert argv[-4:] == ("--as", "user", "--format", "json")
        if argv[1:4] == ("api", "GET", "/open-apis/authen/v1/user_info"):
            payload = {"open_id": own_open_id}
        elif argv[1:5] == ("im", "+chat-list", "--types", "p2p"):
            payload = {"chats": [{"chat_id": chat_id, "chat_mode": "p2p", "p2p_target_type": "user",
                                  "p2p_target_id": sender_open_id}], "has_more": False}
        else:
            raise AssertionError("unexpected Feishu CLI command")
        return CommandResult(0, json.dumps({"ok": True, "identity": "user", "data": payload}).encode(), b"")

    class Discovery(FeishuDiscovery):
        def for_chat(self, resolved_chat_id):
            if resolved_chat_id != chat_id:
                raise AssertionError("unexpected fixed chat")
            if cli is not None:
                return cli
            return super().for_chat(resolved_chat_id)

    return Discovery("lark-cli", timeout=2, runner=runner)
