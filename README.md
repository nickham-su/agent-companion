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

`.venv/bin/python -m agent_companion` 和 `.venv/bin/python -m agent_companion --demo-subtasks` 可单独运行两个简短演示；不需要生产 Provider 或密钥，始终使用内存数据库，不等同于全量验收。可传 `--config config.example.toml` 验证其他模拟配置；只有 `--run` 使用配置中的数据库路径和 Provider。

## 选择 Provider

正式启动通过顶层 `llm_provider`、`asr_provider` 引用 `[providers.<名字>]`；`voice` 还需 `tts_provider`，`text` 不使用它。名字是任意配置 ID，`adapter` 才是实际适配器。每套 Provider 在本机配置中直接填写 `api_key`，不从环境变量读取；未选用的 Provider 不需要密钥。改引用后重启即可切换。

旧的 `[llm]`、`[deepseek]`、`[minimax]` 及旧语音模型字段不再兼容，请按 [config.example.toml](config.example.toml) 迁移。已有 SQLite 会话历史保留。`adapter` 必须显式指定，`.llm.reasoning_effort` 必填。

请求时限统一配置在 `[timeouts]`，单位秒：`llm` 是单次模型请求总时限（默认 60），`asr` 是单次转写总时限（默认 30），`tts` 是单次网络读写及等待响应的时限（默认 25，仅 `voice` 启用）。切换 Provider 不改变这些值。旧的 `runtime.model_timeout` 和 Provider 能力表里的 `timeout` 不再兼容；语音投递与飞书 CLI 的时限独立配置。升级时删除 `[runtime]` 中旧的 `delivery_timeout`。

| `adapter` | 配置 |
| --- | --- |
| `deepseek` | `.llm.model` 可省略，默认 `deepseek-flash`。`reasoning_effort = "disabled"` 显式关闭思考；其他值由所选模型解释。历史切到思考开启时，如有不兼容旧记录，用摘要承接，原文仍可查。 |
| `openai_responses` | `.llm.model` 必填。可选 Provider 层 `base_url`，默认 `https://api.openai.com/v1`，请求时在其后追加 `/responses`。`reasoning_effort` 原样传给 Responses，例如 `low`，或选用明确支持 `none` 的模型来关闭推理。 |
| `minimax` | Provider 层 `region` 选 `cn`（默认）或 `global`。`.asr.model` 默认 `asr-1.0`，也可填写其他兼容同一文件转写接口的模型名；`.asr.max_audio_seconds` 必填，无默认值。`.tts` 配置模型和音色。ASR、TTS 可以共用一套 Provider，也可以分别使用不同区域和 Key。 |

升级旧配置时，删除 `[voice]` 的 `max_turn_seconds`，在 `asr_provider` 选中的 `[providers.<名称>.asr]` 中明确设置 `max_audio_seconds`（有限正数，单位为秒）。正式的 `text` 和 `voice` 模式都要填写；示例中的 `500` 是 `asr-1.0` 的参考值，使用其他模型时请按其接口限制填写。

**只配置可信的 HTTPS 地址**：自定义服务会收到所选 Provider 的 API Key 和发送给模型的输入。地址不要包含凭据、查询参数或片段；客户端不跟随重定向。其他适配器不支持 `base_url`。

Responses 请求设置 `store: false` 并显式请求加密推理项，不使用 `previous_response_id`，也不主动请求可读推理摘要。请求正常完成后原样保存有效密文，之后只在原作用域（同一适配器、端点、模型、凭据与推理档位）按原生项顺序回传；压缩请求也遵守此规则。可读摘要若随协议返回，只作为私有回传字段，不是需要概括的内容。SQLite 中的私有回传记录与回复、工具结果和回执原子提交，客户端不主动把结构化私有协议字段写入正文、搜索、日志或会话摘要。不要共享历史数据库或提取并分享私有记录。

本地用数据库级随机盐、API Key、规范化后的 `base_url` 和 `model` 自动生成指纹，Provider 名字不参与计算。更换 Key（包括同一项目轮换）、地址、模型或 `reasoning_effort` 后，不回传不匹配的密文，但仍可使用通用聊天历史。旧手工标签下的回传记录不再读取，原数据库与通用历史保留。

工具定义和请求前缀变化不阻止回传；续轮仍按通用预算压缩历史或替换图片，已纳入摘要的私有记录不再加载。语音零投递或部分投递不回传未确认正文，保留其他有效推理项及配对的工具调用和结果。私有记录缺失或损坏时可使用通用历史；供应商是否利用密文由其决定。预算不足或服务端拒绝时结束当前请求，不自动重试、切换供应商或重跑工具。按 [Mac 实机验收清单](docs/acceptance/mac-checklist.md)分别核对私有项入库、出站携带和供应商实际利用。

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
2. 复制 `config.example.toml` 为 `config.local.toml`，取消 `[feishu]` 注释，填写目标对话 `chat_id`、业务对端 `sender_open_id` 与登录账号 `own_open_id`，使用同一 CLI 应用下的 Open ID；两者必须不同。`mode="text"` 不打开音频设备。`cli_timeout` 限制单次 CLI 调用（默认 20 秒）。每个数据库只允许一个飞书主 Session 绑定，首次绑定要求全库没有历史或入站状态。日后要更换主 Session ID、对话、对端或账号，或旧数据库已有数据但未绑定，请配置新的数据库路径，不要覆盖旧库。仅切换输出模式不影响绑定。
3. 按[选择 Provider](#选择-provider)填好所选 LLM 和 MiniMax ASR 的 `api_key`；只在本机私有配置填写真实 Key，不要分享或提交该文件。MiniMax 区域按账号选择 `cn`（[国内 API](https://platform.minimax.cn/docs/api-reference/speech-to-text)）或 `global`（[国际 API](https://platform.minimax.io/docs/api-reference/speech-to-text)）。工作目录需要允许建立用于资源下载的短期目录。

```sh
.venv/bin/python -m agent_companion --run --config config.local.toml
```

轮询使用用户身份调用原始 `/open-apis/im/v1/messages`，保留 `body.content`、发送者和媒体键；每 10 秒扫描一次，单次最多 200 页，未读完时从下一页继续同一个固定时间窗口，不提前入队或推进水位。取齐后按原时间顺序提交一个批次。累计窗口超过状态上限时尝试缩短时间范围重新读取；即使最小可推进范围也超限则报错，不跳过数据。初次从启动时所在秒开始，不回放更早聊天；后续查询回看 10 秒并以消息 ID 去重，重启从持久化水位重读未完成窗口。水位、去重和待接收输入在同一事务保存，已接收但尚未进入历史的批次重启后继续移交；用户输入进入历史后不自动重跑未完成的模型或外发操作。一个数据库由一个实例运行，不复制 SQLite 文件到交付包。云端只使用模拟 CLI、模型与合成音频，本机真实协议和权限待验收。

每个新窗口默认每页 50 条。列表 CLI 标准输出超过 4 MiB 时，将页大小减半（最低 1 条），丢弃该窗口未提交的消息及游标，下次轮询从同一时间窗口的第一页重读。每页仅 1 条仍超限时报告 `cli_single_result_output_limit`，不提交部分结果、不推进水位，也不跳过消息。

图片按文件头识别（JPEG、PNG、GIF、WebP；单张本机上限 8 MiB）；语音条下载后验证容器和 50 MB 文件上限，提交一次完整文件，不做截断。WAV 超过 `max_audio_seconds` 时在本机拒绝；非 WAV 时长仍由服务端检查。飞书入口按原处理顺序执行现有单图和同批媒体限额，受限图片用文字占位，不挤掉同批可用文本；不会为了找更新的图片突破下载限额。macOS/Linux 的 CLI 下载子进程在写文件前已设置相应文件大小硬上限。富文本链接保留为普通文字，不发起下载。语音条失败或超时不重试，原对话尝试发送一次「语音接收失败，请重新发送。」，通知不走模型、不阻塞同批成功输入，崩溃后不补发。

文字回复成功生成后，先将完整正文与对应工具结果落库，再进入通道的内存队列，由一个后台任务按顺序尽力发送。Runtime 不等 CLI；历史里的全文只表示已生成，投递状态为空，不代表已送达。队列最多 8 条待发，另有最多 1 条发送中；满时丢弃新回复，发送失败只记日志，不重试。关闭时丢弃待发并取消当前发送，不等待队列排空；重启不会补发。

如果在轮询批次落库前进程中断，下次会重新查询窗口；未落库的内容可能重新下载或转写。此时没有已接受的输入或失败通知需要恢复。命令兼容性还依赖于 [raw API 的 `--data -` stdin 支持](https://github.com/larksuite/cli/blob/v1.0.100/cmd/api/api.go)与[资源下载快捷命令](https://github.com/larksuite/cli/blob/v1.0.100/skills/lark-im/references/lark-im-messages-resources-download.md)。

## 本机启动飞书实时语音

1. 在目标 Mac 上安装本机虚拟音频驱动，准备**两台不同的 CoreAudio 虚拟设备**。飞书扬声器设为「飞书 → Agent」设备，程序读取该设备；程序把播报写到「Agent → 飞书」设备，飞书麦克风从后一设备读取。不要把两条线路接到同一设备；按 macOS 提示授予启动程序的终端/应用音频输入权限。
2. 在目标 Mac 的虚拟环境安装可选依赖，再查询设备名称和索引（命令不启动 Agent）：

   ```sh
   .venv/bin/python -m pip install '.[voice]'
   .venv/bin/python -m sounddevice
   ```

3. 从 [Silero VAD 官方源码](https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/data/silero_vad.onnx)单独取得 `silero_vad.onnx`（使用带 `input/state/sr` 接口的 v6 系列模型，不选只支持其他输入形状的变体），放在本机配置指定路径。程序只加载已有文件，不下载模型，不需要 PyTorch；相对路径以 TOML 文件所在目录为准。音频和 ONNX 依赖不在 `text` 或模拟模式下加载。
4. 设置 `[feishu].mode = "voice"`，检查 `tts_provider` 所选 MiniMax 配置含 `.tts` 配置表和 `api_key`，再按 `config.example.toml` 填写 `[voice]` 的设备与 VAD 模型路径。用同一个 `--run --config config.local.toml` 命令重启程序，手动用飞书接听和挂断；切回 `text` 同样修改配置并重启。同一数据库和主 Session 历史保留。不判断通话状态：`voice` 模式即使没有通话，也只向虚拟设备播报模型回复，不自动发飞书文字。飞书文本、图文和语音条仍轮询；语音条失败通知始终通过飞书文字发送。

程序在启动时检查两路设备是否支持所需流格式。输入为 16 kHz 单声道有符号 16-bit PCM（512 样本一帧），直接供 Silero 和 WAV 文件转写使用。输出为 MiniMax 相同采样率的单声道 16-bit PCM，默认 44.1 kHz，可配 16/24/32/44.1 kHz。CoreAudio 在需要且支持时转换设备格式，程序不更改设备全局采样率、不再做第二次重采样。若报 `voice_input_pcm_16khz_mono_not_supported` 或 `voice_output_pcm_rate_or_channels_not_supported`，在「音频 MIDI 设置」和 `sounddevice` 列表核对两条线路；输出可选一个双方支持的采样率，输入需选择能提供 16 kHz 单声道流的设备或系统转换。若通话对端无声，核对虚拟设备首个单声道的路由、飞书麦克风/扬声器选项和设备权限。

输出采样率由 `[voice]` 中的 `output_sample_rate` 设置，单位为 Hz。升级旧配置时，把 `playback_rate` 改为 `output_sample_rate`，数值不变。

VAD 默认连续 3 个 32 ms 人声窗口触发打断，默认静默 1 秒结束话轮。实时录音按实际样本数计时（含预录和已录停顿），以 `max_audio_seconds` 和含 WAV 文件头的 50 MB 字节保护中先到的上限为准。达到上限时立即将保留的前段送入转写队列，不等用户停下。之后的声音只用于检测静默，不录音、不打断也不阻止回复播报；连续静默达到 `silence_seconds` 后，再次开口才恢复正常录音和打断。ASR 仍串行处理，队列满时仍丢弃新录音；空转写结果、识别失败或超时不会提交给 Runtime，也不发送飞书语音条式失败通知。

若 `max_audio_seconds` 短到放不下一个 16 kHz 采样点，实时语音无法启动，并报告 `voice_asr_max_audio_seconds_below_one_sample`。

MiniMax TTS 用 [`/v1/t2a_v2` HTTP SSE](https://platform.minimax.cn/docs/api-reference/speech-t2a-http)逐段发送，首块到达即可播放；每段流结束且待播放低于 `prefetch_seconds` 时才提交下一段。从语音通道开始处理起，`voice.delivery_timeout`（默认 600 秒）是不可续期的总上限。模型输出结束后，Runtime 启动 `voice.idle_timeout`（默认 30 秒）；只有设备实际取走本条回复的 PCM 才续期，补静音或只在合成不算。任一时限到期，投递记为 `UNKNOWN`，历史只保留已确认的连续完整段落，不估算半句。空闲超时由 Runtime 取消投递任务，不等清理或迟到回执。已确认的正文仍进入后续会话历史，不代表对端一定听到了。

如果本机的 `[voice]` 仍显式使用旧示例的 `delivery_timeout = 90`，可改为 `600` 或删除这行使用新默认值。

运行时若设备流断开或 VAD 推理出错，实时语音停用并记日志：丢弃尚未提交的录音，取消在途实时转写，忽略迟到结果，不等待取消清理。当前播报仍保留已交设备的确认前缀。模型、工具、飞书输入轮询和飞书语音条转写不受影响。不会自动切成文字输出或重连设备；排查后需重启。

MiniMax 流式合成也可能以 HTTP 200 返回 JSON 业务错误而不是 SSE。该响应最多读取 64 KiB，受 `timeouts.tts` 限制；适配器使用固定错误码分类，不输出响应正文。

实机需验证：两路可独立收发且无自触发；真实 ASR、流式 PCM 和设备格式匹配；说完到开始回答的延迟可接受；用户打断后旧播报在 ASR 成功/失败时都不恢复，历史只记交给本机设备的确认前缀，不代表飞书对端已听到。最后挂断、切回 `text` 并重启，验证历史与 CLI 文字回复仍可用。离线模拟结果不替代这些本机检查。

## 本地系统提示词

在启动工作目录下创建 UTF-8 编码的 `AGENTS.md`；Runtime 初始化后固定读取该目录，不向父目录查找。每次组装模型请求时重新读取，包括工具续轮、内部子任务和压缩；文件内容附在原有系统指令之后，修改后下一次请求生效，无需重启。文件缺失或只有空白时维持原行为。不可读、非 UTF-8、超过 32 KiB、符号链接或非普通文件会报固定诊断，停止受影响的请求，不记录文件正文。

仓库根目录的 `AGENTS.md` 已被 Git 忽略，但**正文会作为系统提示词发送给配置的模型服务**。请只在可信工作目录启动，不要在文件中放入密钥、凭据或隐私信息。

## 运行时边界

通道每次提交输入时指定是否等待投递确认，工具续轮沿用本次运行的策略：语音等待回执，飞书文字落库后后台发送。按 `ReplyInfo` 区分回复、运行和会话；其中时间戳是模型请求开始前的事件循环单调时间。纯工具响应不投递。SQLite 原子提交助手记录和对应工具结果，不自动重放已中断的模型调用或外部发送。

`delegate_subtask` 是主会话唯一注册的工具。它只返回本地受理标识，不等待任务完成。同一子会话的各次运行串行，不同子会话可并行；子会话没有工具或用户通道，只将终态结果回投主队列。仅由后台回投触发且尚未纳入新用户输入的主会话运行不能再次委派；已包含用户输入的运行不受此限。

用 `History.subtask_run(child_session_id, run_id)` 可追溯受理、开始、完成和回投入队时间。`enqueued_at` 只表示已入内存队列，不代表已被主会话处理。终态写入最多尝试三次，不重跑模型；持续失败会记录运行 ID、异常类型和未回投，不写假终态，数据库可能保留最后一次状态。退出开始时同步停止接收新委派和回投；可写时将未结束的记录标为中断，异常退出后由下次启动标记。不补跑、不重放；原有子会话 ID 仍可用于新委派。

历史查找可叠加时间范围和 FTS5 关键词筛选，默认使用 SQLite 的分词规则，不承诺中文短词的子串查找。上下文 token 预算为本地估算。DeepSeek 请求另有 40 MiB 正文硬上限，OpenAI Responses 有 32 MiB 本机请求上限（不是供应商的协议承诺）；包括摘要请求。Runtime 只在本轮请求需要裁剪图片时，从批次中位置靠后的图片开始尝试（同一富文本内也一样）；某张放不下就继续尝试更早的图。选定后仍按原顺序发送图文，并保留省略提示、普通文字及完整工具组，不改 SQLite 原文。不根据“第一张”等自然语言改变选择，也不会补下载在飞书入口已被省略的图片。旧历史可按预算缩减；若本轮文字和完整工具块仍放不下，结束该次运行，后续输入照常处理。实际发送前检查序列化后的字节数。

压缩请求独立组装，沿用正常请求的系统提示词、已有摘要格式、工具定义顺序和推理设置，只发送待压缩的旧历史，在末尾追加一条内部压缩指令。保留尾窗和当前轮输入不进入压缩请求；压缩不执行工具，模型若返回工具调用则本次压缩失败。兼容的旧历史按原协议投影，以便复用服务端的输入前缀缓存，不保证实际命中；不兼容的 DeepSeek 旧块，以及带私有回传字段、单独原生发送仍超预算的旧块，可将完整可见内容作为归档材料发送，不伪造或拼入私有推理字段。

压缩指令要求模型不归纳私有协议字段。客户端会拒绝明显原样回显（摘要全文与字段相同，或原样包含较长字段），不保存该摘要、不推进覆盖点；对模型改写、截取或自行生成的文字，不能保证识别。

压缩不提前触发；按需用 `[runtime]` 的 `context_budget` 和 `tail_budget` 给系统提示词、工具定义、已有摘要及压缩指令预留空间。会话摘要完整保存到 SQLite。回复请求放不下完整摘要时，整条省略，不修改已保存摘要；如果完整旧摘要挤占了原本可摘要的历史，压缩不推进覆盖点。升级时请从本机配置的 `[runtime]` 移除旧的 `summary_budget`。

方案见 [首版架构方案](<docs/design/0001-MVP方案/飞书多模态 Agent 首版架构方案.md>)和[首版实施与验收计划](<docs/design/0001-MVP方案/飞书多模态 Agent 首版实施与验收计划.md>)。
