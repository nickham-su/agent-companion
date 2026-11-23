# agent-companion

飞书多模态 Agent 首版，提供共享历史、异步委派、飞书文字／图文／语音条与 Mac 实时语音。离线验收不接入真实飞书、模型账号或设备。

## 目录与依赖方向

路径相对于 `src/agent_companion/`：

| 路径 | 职责 | 状态 |
|---|---|---|
| `__main__.py`、`app.py`、`config.py` | 程序组装、启动与配置 | 已实现 |
| `protocol.py` | 模型、通道与消息的共同契约 | 已实现 |
| `runtime/engine.py`、`runtime/context.py`、`runtime/tools.py` | 运行编排、上下文投影与工具注册 | 已实现 |
| `storage/sqlite.py`、`models/` | SQLite 历史存储、DeepSeek 和 OpenAI Responses LLM 适配 | 离线模拟通过，真实调用待验 |
| `runtime/subtasks.py` | 异步委派与结果回投 | 已实现 |
| `channels/feishu/` | CLI、轮询、消息解析与通道路由 | 两种模式保留飞书输入，待实机验收 |
| `speech/` | 与 IM 无关的文件 ASR、VAD、TTS 和实时语音调度 | 离线模拟通过，待实机验收 |
| `channels/audio/` | macOS CoreAudio 双路音频设备 | 离线替身验证，待实机验收 |

模型与通道实现由启动入口按 `protocol.py` 中的契约注入；Runtime 不导入具体模型、飞书或语音实现。SQLite 存储使用共同消息类型，不反向依赖 Runtime。

## 运行离线验收

需要 Python 3.11+ 和启用了 FTS5 的 SQLite（使用 Python 自带 `sqlite3`）。在仓库根目录执行：

```sh
python -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/python -m unittest discover -s tests -v
```

最后一条是统一离线验收入口：不需要模型密钥、飞书登录或音频设备；集成用例使用项目内临时 SQLite，结束后清理。安装依赖若不能访问公开索引，可从准备好的本地 wheel 目录安装：`.venv/bin/python -m pip install --no-index --find-links <wheel目录> .`（还需包含 `pyproject.toml` 的构建依赖）。精简环境若缺 `ensurepip`，需先提供带 pip 的虚拟环境；目标 Mac 建议使用完整的 Python 发行版。

`.venv/bin/python -m agent_companion` 和 `.venv/bin/python -m agent_companion --demo-subtasks` 可单独运行两个简短演示；不需要 `[llm]`，始终使用内存数据库，不等同于全量验收。可传 `--config config.example.toml` 验证其他模拟配置；只有 `--run` 使用配置中的数据库路径。单次模型请求总时限默认 60 秒。

## 选择 LLM

正式启动必须在 `[llm]` 指定 `adapter` 和 `reasoning_effort`，两项均无默认值。删除旧的 `[deepseek]` 段；旧配置不兼容。已有 SQLite 会话历史保留。

| `adapter` | 配置与凭据 |
| --- | --- |
| `deepseek` | `model` 可省略，默认 `deepseek-flash`。`reasoning_effort = "disabled"` 显式关闭思考；其他值由所选模型解释。凭据仅从 `DEEPSEEK_API_KEY` 读取。历史切到思考开启时，如有不兼容旧记录，用摘要承接，原文仍可查。 |
| `openai_responses` | 必填 `model`、`replay_profile`。`reasoning_effort` 原样传给 Responses，例如 `low`，或选用明确支持 `none` 的模型来关闭推理。凭据仅从 `OPENAI_API_KEY` 读取。 |

`replay_profile` 使用稳定、不敏感的本地项目标识，例如 `personal-project`；**换 OpenAI 账户或项目时要换标识**，不要填 API key。模型不支持所填强度时不会自动改用其他值或供应商。

OpenAI 只使用官方 `/v1/responses`，设置 `store: false` 并显式请求加密推理项，不使用 `previous_response_id`，也不主动请求可读推理摘要。完成后原样保存密文，按原生 item 顺序回传；可读摘要若随协议返回，只作为私有回传字段。SQLite 中的回传记录与回复、工具结果和回执原子提交，不进入正文、搜索、日志或会话摘要。不要共享历史数据库或提取并分享私有记录。

旧回传仅用于相同适配器、模型、`replay_profile` 和 `reasoning_effort`，且实际请求前缀匹配的请求。切换配置、压缩或替换图片会让受影响的旧回传失效，仍可使用通用历史。早期不含强度指纹的回传记录也只使用对应的通用历史，不改写原数据库。语音零投递或部分投递只保留改写前的原生顺序，改写后的尾部仅用通用项，不回传未确认正文或受影响的密文。

活动工具续轮在上下文估算和实际 JSON 均可容纳时，优先保留上一请求的投影再追加工具组和安全边界的新输入，不主动压缩或替换原图。若终态明确没有推理项，必要时可以改用稳定 `call_id` 的通用工具组；若必需回传缺失、不兼容或放不下，则结束续轮，不重跑工具、不自动重试。按 [Mac 实机验收清单](docs/acceptance/mac-checklist.md)分别核对密文入库、出站请求携带密文和模型端可观察行为。

## 交付包与验收索引

现有 `dist/agent-companion-mvp.zip` 和下方离线记录对应首版交付，不含后来新增的 OpenAI Responses 适配器。验证当前适配器请使用当前源码。下列命令用于制作新包，会覆盖同名产物；仅校验旧包时无需执行。

```sh
python scripts/build_source_zip.py
```

输出 `dist/agent-companion-mvp.zip` 与 `dist/agent-companion-mvp.zip.sha256`，在 `dist/` 执行 `shasum -a 256 -c agent-companion-mvp.zip.sha256` 可核对。ZIP 内只有源码、测试、说明与配置样例；不会包含本机配置、凭据、真实媒体、数据库、虚拟环境或参考源码。解压到干净目录后按上一节安装及验收。

- [云端离线验证记录](docs/acceptance/offline-results.md)：已验证的场景、替身边界与本地待验收项。
- [目标 Mac 实机验收清单](docs/acceptance/mac-checklist.md)：两种模式的验证步骤、通过判据和记录格式。

## 本机启动飞书文字模式

1. 使用 [lark-cli v1.0.100](https://github.com/larksuite/cli/releases/tag/v1.0.100) 或保留同等协议的版本；在本机完成飞书用户登录，并确认应用有指定聊天的列表、媒体下载和发消息权限。程序不会发起登录或读取 CLI 的凭据文件。参考 [CLI JSON 输出契约](https://github.com/larksuite/cli/blob/v1.0.100/README.md#json-output-contract)和[飞书消息列表 API](https://open.feishu.cn/document/server-docs/im-v1/message/list)。
2. 复制 `config.example.toml` 为 `config.local.toml`，取消 `[feishu]` 注释，填写目标对话 `chat_id`、业务对端 `sender_open_id` 与登录账号 `own_open_id`，使用同一 CLI 应用下的 Open ID；两者必须不同。`mode="text"` 不打开音频设备。`cli_timeout` 必须短于 Runtime 的投递时限。每个数据库只允许一个飞书主 Session 绑定，首次绑定要求全库没有历史或入站状态。日后要更换主 Session ID、对话、对端或账号，或旧数据库已有数据但未绑定，请配置新的数据库路径，不要覆盖旧库。仅切换输出模式不影响绑定。
3. 按[选择 LLM](#选择-llm)配置模型、本机模型环境变量及 `MINIMAX_API_KEY`，不写到配置或日志。MiniMax 区域按账号选择 `cn`（[国内 API](https://platform.minimax.cn/docs/api-reference/speech-to-text)）或 `global`（[国际 API](https://platform.minimax.io/docs/api-reference/speech-to-text)）。工作目录需要允许建立用于资源下载的短期目录。

```sh
.venv/bin/python -m agent_companion --run --config config.local.toml
```

轮询使用用户身份调用原始 `/open-apis/im/v1/messages`，保留 `body.content`、发送者和媒体键；每 10 秒扫描一次，单次最多 200 页，未读完时从下一页继续同一个固定时间窗口，不提前入队或推进水位。取齐后按原时间顺序提交一个批次。累计窗口超过状态上限时尝试缩短时间范围重新读取；即使最小可推进范围也超限则报错，不跳过数据。初次从启动时所在秒开始，不回放更早聊天；后续查询回看 10 秒并以消息 ID 去重，重启从持久化水位重读未完成窗口。水位、去重和待接收输入在同一事务保存，已接收但尚未进入历史的批次重启后继续移交；用户输入进入历史后不自动重跑未完成的模型或外发操作。一个数据库由一个实例运行，不复制 SQLite 文件到交付包。云端只使用模拟 CLI、模型与合成音频，本机真实协议和权限待验收。

每个新窗口默认每页 50 条。列表 CLI 标准输出超过 4 MiB 时，将页大小减半（最低 1 条），丢弃该窗口未提交的消息及游标，下次轮询从同一时间窗口的第一页重读。每页仅 1 条仍超限时报告 `cli_single_result_output_limit`，不提交部分结果、不推进水位，也不跳过消息。

图片按文件头识别（JPEG、PNG、GIF、WebP；单张本机上限 8 MiB）；语音条下载后验证容器，使用 `asr-1.0` 提交一次完整文件（50 MB、500 秒；非 WAV 时移交服务端检查时长）。飞书入口按原处理顺序执行现有单图和同批媒体限额，受限图片用文字占位，不挤掉同批可用文本；不会为了找更新的图片突破下载限额。macOS/Linux 的 CLI 下载子进程在写文件前已设置相应文件大小硬上限。富文本链接保留为普通文字，不发起下载。语音条失败或超时不重试，原对话尝试发送一次「语音接收失败，请重新发送。」，通知不走模型、不阻塞同批成功输入，崩溃后不补发。文字回复发送成功才写入确认正文；只有已知明确未送达的错误才记失败，通用 API 错误或无法确认的结果记未知，不重发。

如果在轮询批次落库前进程中断，下次会重新查询窗口；未落库的内容可能重新下载或转写。此时没有已接受的输入或失败通知需要恢复。命令兼容性还依赖于 [raw API 的 `--data -` stdin 支持](https://github.com/larksuite/cli/blob/v1.0.100/cmd/api/api.go)与[资源下载快捷命令](https://github.com/larksuite/cli/blob/v1.0.100/skills/lark-im/references/lark-im-messages-resources-download.md)。

## 本机启动飞书实时语音

1. 在目标 Mac 上安装本机虚拟音频驱动，准备**两台不同的 CoreAudio 虚拟设备**。飞书扬声器设为「飞书 → Agent」设备，程序读取该设备；程序把播报写到「Agent → 飞书」设备，飞书麦克风从后一设备读取。不要把两条线路接到同一设备；按 macOS 提示授予启动程序的终端/应用音频输入权限。
2. 在目标 Mac 的虚拟环境安装可选依赖，再查询设备名称和索引（命令不启动 Agent）：

   ```sh
   .venv/bin/python -m pip install '.[voice]'
   .venv/bin/python -m sounddevice
   ```

3. 从 [Silero VAD 官方源码](https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/data/silero_vad.onnx)单独取得 `silero_vad.onnx`（使用带 `input/state/sr` 接口的 v6 系列模型，不选只支持其他输入形状的变体），放在本机配置指定路径。程序只加载已有文件，不下载模型，不需要 PyTorch；相对路径以 TOML 文件所在目录为准。音频和 ONNX 依赖不在 `text` 或模拟模式下加载。
4. 沿用已配置的飞书 CLI 和 MiniMax，设置 `[feishu].mode = "voice"` 并填写 `config.example.toml` 中的 `[voice]` 设备与模型路径。用同一个 `--run --config config.local.toml` 命令重启程序，手动用飞书接听和挂断；切回 `text` 同样修改配置并重启。同一数据库和主 Session 历史保留。不判断通话状态：`voice` 模式即使没有通话，也只向虚拟设备播报模型回复，不自动发飞书文字。飞书文本、图文和语音条仍轮询；语音条失败通知始终通过飞书文字发送。

程序在启动时检查两路设备是否支持所需流格式。输入为 16 kHz 单声道有符号 16-bit PCM（512 样本一帧），直接供 Silero 和 WAV 文件转写使用。输出为 MiniMax 相同采样率的单声道 16-bit PCM，默认 44.1 kHz，可配 16/24/32/44.1 kHz。CoreAudio 在需要且支持时转换设备格式，程序不更改设备全局采样率、不再做第二次重采样。若报 `voice_input_pcm_16khz_mono_not_supported` 或 `voice_output_pcm_rate_or_channels_not_supported`，在「音频 MIDI 设置」和 `sounddevice` 列表核对两条线路；输出可选一个双方支持的采样率，输入需选择能提供 16 kHz 单声道流的设备或系统转换。若通话对端无声，核对虚拟设备首个单声道的路由、飞书麦克风/扬声器选项和设备权限。

VAD 默认连续 3 个 32 ms 人声窗口触发打断，默认静默 1 秒结束话轮，随后提交一次 16 kHz WAV；录音长度、转写等待和录音队列有上限。实时通话录音超限、空白、识别失败或超时不入队，也不发送飞书语音条式失败通知。MiniMax TTS 用 [`/v1/t2a_v2` HTTP SSE](https://platform.minimax.cn/docs/api-reference/speech-t2a-http)逐段发送，首块到达即可播放；每段流结束且待播放低于 `prefetch_seconds` 时才提交下一段。默认首次输出起 `voice.delivery_timeout = 90` 秒，低于 Runtime 的投递看门狗，独立于 LLM 请求的 60 秒默认时限。

运行时若设备流断开或 VAD 推理出错，实时语音停用并记日志：丢弃尚未提交的录音，取消在途实时转写，忽略迟到结果，不等待取消清理。当前播报仍保留已交设备的确认前缀。模型、工具、飞书输入轮询和飞书语音条转写不受影响。不会自动切成文字输出或重连设备；排查后需重启。

MiniMax 流式合成也可能以 HTTP 200 返回 JSON 业务错误而不是 SSE。该响应最多读取 64 KiB，受 `tts_timeout` 限制；适配器使用固定错误码分类，不输出响应正文。

实机需验证：两路可独立收发且无自触发；真实 ASR、流式 PCM 和设备格式匹配；说完到开始回答的延迟可接受；用户打断后旧播报在 ASR 成功/失败时都不恢复，历史只记交给本机设备的确认前缀，不代表飞书对端已听到。最后挂断、切回 `text` 并重启，验证历史与 CLI 文字回复仍可用。离线模拟结果不替代这些本机检查。

## 运行时边界

通道通过 `Runtime.submit_batch()` 原子提交输入批次，按 `ReplyInfo` 区分回复、运行和会话。单个 `ReplyStream` 的时间戳是请求开始前的事件循环单调时间，可供后续音频通道判断首个文本片段到达时该回复是否已经过时。通道消费文本流并返回唯一终态回执；纯工具响应不创建通道任务。SQLite 原子提交助手记录和对应的工具结果，不自动重放已中断的模型调用或外部发送。阶段一的待处理输入和工具执行不跨进程恢复。
通道应在 Runtime 投递看门狗之前结束，并随终态回传已确认的正文前缀。看门狗到期则记为结果未知，丢弃迟到回执，不等待取消清理。

`delegate_subtask` 是主会话唯一注册的工具。它只返回本地受理标识，不等待任务完成。同一子会话的各次运行串行，不同子会话可并行；子会话没有工具或用户通道，只将终态结果回投主队列。仅由后台回投触发且尚未纳入新用户输入的主会话运行不能再次委派；已包含用户输入的运行不受此限。

用 `History.subtask_run(child_session_id, run_id)` 可追溯受理、开始、完成和回投入队时间。`enqueued_at` 只表示已入内存队列，不代表已被主会话处理。终态写入最多尝试三次，不重跑模型；持续失败会记录运行 ID、异常类型和未回投，不写假终态，数据库可能保留最后一次状态。退出开始时同步停止接收新委派和回投；可写时将未结束的记录标为中断，异常退出后由下次启动标记。不补跑、不重放；原有子会话 ID 仍可用于新委派。

历史查找可叠加时间范围和 FTS5 关键词筛选，默认使用 SQLite 的分词规则，不承诺中文短词的子串查找。上下文 token 预算为本地估算。DeepSeek 请求另有 40 MiB 正文硬上限，OpenAI Responses 有 32 MiB 本机请求上限（不是供应商的协议承诺）；包括摘要请求。Runtime 只在本轮请求需要裁剪图片时，从批次中位置靠后的图片开始尝试（同一富文本内也一样）；某张放不下就继续尝试更早的图。选定后仍按原顺序发送图文，并保留省略提示、普通文字及完整工具组，不改 SQLite 原文。不根据“第一张”等自然语言改变选择，也不会补下载在飞书入口已被省略的图片。旧历史或摘要可按预算缩减；若本轮文字和完整工具块仍放不下，结束该次运行，后续输入照常处理。实际发送前检查序列化后的字节数。

方案见 [首版架构方案](<docs/design/0001-MVP方案/飞书多模态 Agent 首版架构方案.md>)和[首版实施与验收计划](<docs/design/0001-MVP方案/飞书多模态 Agent 首版实施与验收计划.md>)。
