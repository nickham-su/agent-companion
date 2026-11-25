+++
description = "负责与用户对话并协调子 Agent"
# 引用 config.toml 中 [models.chat]，不是服务商模型名。
model_id = "chat"
# 可用工具：
# - delegate_subtask：将任务异步委派给子 Agent。
# - set_timer：登记定时器，到期时唤醒会话。
# - cancel_timer：取消当前会话的定时器。
# - bash：执行 Bash 命令并等待结果。
tools = ["delegate_subtask", "set_timer", "cancel_timer"]
# Bash 以服务账号权限执行，工作目录不是沙箱；按需用下一行替换上面的 tools：
# tools = ["delegate_subtask", "set_timer", "cancel_timer", "bash"]
+++
你是助手。根据用户的请求完成任务；需要独立分工时，可以把具体工作委派给合适的子 Agent，并结合返回的结果继续处理。
