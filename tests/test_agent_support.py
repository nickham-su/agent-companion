from agent_companion.storage.sqlite import Delegation, History


def stored_delegation(history: History, parent_session: str, tool_call_id: str) -> Delegation:
    """从内部记录取证；测试中工具调用 ID 在父会话内唯一。"""
    row, = history._db.execute("""
        SELECT d.* FROM delegations AS d JOIN agents AS a ON a.agent_id = d.parent_agent_id
        WHERE a.session_id = ? AND d.tool_call_id = ?
        """, (parent_session, tool_call_id))
    return Delegation(**dict(row))
