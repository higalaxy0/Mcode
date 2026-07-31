# Mcode
Mcode — Coding Agent

## 目录

- [Mcode — Coding Agent 核心架构文档](#mcode--coding-agent-核心架构文档)
  - [目录](#目录)
  - [项目总览](#项目总览)
  - [项目亮点](#项目亮点)
    - [1. 三级渐进式上下文压缩——拒绝"一刀切"的信息丢失](#1-三级渐进式上下文压缩拒绝一刀切的信息丢失)
    - [2. 全量工具清单与设计亮点](#2-全量工具清单与设计亮点)
      - [文件系统与 Shell 工具（6 个，所有代理共享）](#文件系统与-shell-工具6-个所有代理共享)
      - [任务管理工具（5 个，仅 Lead）](#任务管理工具5-个仅-lead)
      - [代理协作工具（6 个，仅 Lead）](#代理协作工具6-个仅-lead)
      - [知识工具（1 个，仅 Lead）](#知识工具1-个仅-lead)
      - [工具调度与注入机制](#工具调度与注入机制)
    - [3. 文件总线 + 原子重命名——零依赖的跨线程通信](#3-文件总线--原子重命名零依赖的跨线程通信)
    - [4. 多代理协作 + 协议状态机——结构化的"人机/机机"协作](#4-多代理协作--协议状态机结构化的人机机机协作)
    - [5. MCP 异步核心 + 同步门面——让异步 SDK 无缝融入同步代码库](#5-mcp-异步核心--同步门面让异步-sdk-无缝融入同步代码库)
    - [6. 全链路防御性编程——"不崩溃"是底线](#6-全链路防御性编程不崩溃是底线)
    - [7. 路径沙箱 + 命令拦截——双重安全屏障](#7-路径沙箱--命令拦截双重安全屏障)
    - [8. 后台线程记忆系统——"边聊边学"且零阻塞](#8-后台线程记忆系统边聊边学且零阻塞)
    - [9. 流式响应聚合 + SDK 解耦——不绑死 pydantic](#9-流式响应聚合--sdk-解耦不绑死-pydantic)
    - [10. 延迟导入打破循环依赖——包加载顺序无关](#10-延迟导入打破循环依赖包加载顺序无关)
  - [目录结构](#目录结构)
  - [架构总览](#架构总览)
  - [分层依赖关系](#分层依赖关系)
  - [模块详解](#模块详解)
    - [入口层 `mcode.py`](#入口层-mcodepy)
    - [`config.py` — 配置中心](#configpy--配置中心)
    - [`exceptions.py` — 异常定义](#exceptionspy--异常定义)
    - [`context.py` — 全局运行时状态](#contextpy--全局运行时状态)
    - [`utils.py` — 通用工具函数](#utilspy--通用工具函数)
    - [`fsops.py` — 文件系统与 Shell 工具](#fsopspy--文件系统与-shell-工具)
    - [`tasks.py` — 任务看板 CRUD](#taskspy--任务看板-crud)
    - [`bus.py` — 消息总线与协议状态机](#buspy--消息总线与协议状态机)
    - [`hooks.py` — 钩子系统](#hookspy--钩子系统)
    - [`skills.py` — 技能注册表](#skillspy--技能注册表)
    - [`memory.py` — 记忆系统](#memorypy--记忆系统)
    - [`streaming.py` — 流式响应封装](#streamingpy--流式响应封装)
    - [`compact.py` — Token 估算与上下文压缩](#compactpy--token-估算与上下文压缩)
    - [`tools.py` — 工具定义与调度](#toolspy--工具定义与调度)
    - [`mcp.py` — MCP 远程工具集成](#mcppy--mcp-远程工具集成)
    - [`subagent.py` — 同步子代理](#subagentpy--同步子代理)
    - [`teammates.py` — 线程化队友代理](#teammatespy--线程化队友代理)
    - [`agent.py` — Lead Agent 主循环](#agentpy--lead-agent-主循环)
  - [关键设计决策](#关键设计决策)
    - [1. 配置 vs 状态分离](#1-配置-vs-状态分离)
    - [2. 延迟导入打破循环](#2-延迟导入打破循环)
    - [3. 后端兼容性修复](#3-后端兼容性修复)
    - [4. 三级渐进式压缩](#4-三级渐进式压缩)
    - [5. 文件总线而非内存队列](#5-文件总线而非内存队列)
    - [6. 薄 shim 保持兼容](#6-薄-shim-保持兼容)
    - [7. MCP 异步核心 + 同步门面](#7-mcp-异步核心--同步门面)

---

## 项目总览

`Mcode` 是一个基于 OpenAI 兼容 API的命令行编码 Agent。它支持：

- **流式对话 + 工具调用**：实时输出，自动执行工具循环。
- **三级上下文压缩**：snip / micro / persist + LLM 摘要自动压缩 + 反应式压缩。
- **记忆系统**：自动提取、检索、合并用户偏好/项目事实，注入对话。
- **技能系统**：Markdown 技能文件热加载，按需检索。
- **任务看板**：多代理协作的持久化任务依赖图。
- **多代理协作**：同步子代理（subagent）+ 线程化队友代理（teammate）+ 消息总线 + 协议状态机。
- **钩子系统**：UserPromptSubmit / PreToolUse / PostToolUse / Stop 四类生命周期钩子。
- **MCP 集成**：通过 Model Context Protocol（Streamable HTTP 传输）连接外部工具服务器，自动发现并注入远程工具，全程容错。

---

## 项目亮点

以下亮点提炼自项目源码，体现 Mcode 在工程实现上的关键设计思考与技术深度。

### 1. 三级渐进式上下文压缩——拒绝"一刀切"的信息丢失

长对话是编码 Agent 的核心挑战。Mcode 没有采用单一的摘要策略，而是设计了**四级联动的上下文管理体系**，按代价从低到高逐级触发，最大程度保留信息：

| 层级 | 机制 | 触发条件 | 代价 |
|------|------|----------|------|
| **L1 Snip** | 保留头 1 轮 + 尾 N 轮，中间替换为 `[snipped X turns]` 占位 | 对话轮数 > 26 | 极低（纯内存） |
| **L2 Micro** | 旧轮次的 `tool_result` 内容替换为占位，保留最近 25 轮完整 | 有 > 25 个含工具调用的轮次 | 低（纯内存） |
| **L3 Persist** | 超 30KB 的单条工具输出落盘 `.task_outputs/`，返回预览占位 | 单轮工具输出 > 200KB | 中（磁盘 IO） |
| **L4 Auto/Reactive** | 落盘完整 transcript → LLM 摘要 → 重建上下文块 | 估算 token 超 `CONTEXT_LIMIT` 或 API 报 `prompt_too_long` | 高（一次 LLM 调用） |

```python
# agent.py: 每轮循环的压缩管线
api_messages[:] = tool_result_budget(api_messages)   # L3 持久化
api_messages[:] = snip_compact(api_messages)         # L1 截断
api_messages[:] = micro_compact(api_messages)        # L2 微压缩
if estimate_tokens_messages(api_messages) > CONTEXT_LIMIT:
    api_messages[:] = compact_history(api_messages)  # L4 自动摘要
```

更关键的是，L4 压缩后并非丢弃一切只留摘要，而是通过 `_build_post_compact_context()` **重建结构化上下文块**——自动提取最近访问的 15 个文件路径、当前 plan/todo、任务看板活跃项、已派生的队友列表，与 LLM 摘要拼接为新对话起点。这使 Agent 在"失忆"后仍能快速恢复工作上下文。

**反应式压缩**作为最后兜底：当 LLM API 返回 `prompt_too_long` 时自动触发同样的压缩流程并重试（最多 `MAX_REACTIVE_RETRIES=3` 次），保证对话不会因上下文超限而中断。

### 2. 全量工具清单与设计亮点

`tools.py` 汇总了 Lead Agent 的全部内置工具 schema + handler 映射。以下是完整工具清单，逐一说明参数与实现亮点。

#### 文件系统与 Shell 工具（6 个，所有代理共享）

| 工具 | 必选参数 | 可选参数 | 设计亮点 |
|------|----------|----------|----------|
| **bash** | `command: str` | -- | `bg:` 前缀转后台进程（返回 PID + 日志路径）；尾部 `# timeout=N` 自定义超时；独立 reader 线程 + queue 流式读取，超时跨平台杀进程树（Win `taskkill /T/F`、Linux `killpg`）；输出 50KB 截断；响应 `AgentInterrupt` 中断时清理子进程 |
| **read_file** | `path: str` | `offset: int`(1-based)、`limit: int`(默认 2000，上限 5000) | 行号右对齐前缀；二进制文件检测 `\x00` 头直接返回字节数；多编码回退链 `utf-8-sig -> utf-8 -> gbk -> latin-1`；越界 offset 友好提示 |
| **write_file** | `path: str`, `content: str` | -- | 自动 `mkdir parents`；`safe_path` 沙箱校验防逃逸 |
| **edit_file** | `path: str`, `old_text: str`, `new_text: str` | -- | 首次精确匹配替换（`replace(old, new, 1)`）；`old_text` 不存在时返回明确错误而非静默 |
| **glob** | `pattern: str` | -- | 按 mtime 倒序排列（最近修改优先）；限 100 条，超出提示精简 pattern；去重 resolve 后的路径；沙箱校验过滤逃逸匹配 |
| **grep** | `pattern: str` | `path: str`(默认 `.`)、`include: str`(文件名 glob 过滤) | 正则编译，`re.error` 友好提示；限 50 匹配后截断；递归 `rglob`，沙箱校验每条结果路径 |

#### 任务管理工具（5 个，仅 Lead）

| 工具 | 必选参数 | 可选参数 | 设计亮点 |
|------|----------|----------|----------|
| **todo_write** | `todos: [{content, status}]` | -- | 会话级待办列表，status 枚举校验（pending/in_progress/completed）；彩色终端渲染（▸ 进行中 / ✓ 完成）；仅存内存（`ctx.current_todos`），不持久化 |
| **create_task** | `subject: str` | `description: str`、`blockedBy: [str]` | 持久化到 `.tasks/task_*.json`；ID 含时间戳+随机数；`blockedBy` 声明依赖任务 ID 列表 |
| **list_tasks** | `include_completed: bool` | -- | 从磁盘扫描全部 JSON 重建；按状态渲染图标（○ pending / ● in_progress / ✓ completed）；显示 owner 和 blockedBy |
| **get_task** | `task_id: str` | -- | 返回完整 JSON（含 description 全文），FileNotFoundError 友好提示 |
| **claim_task** | `task_id: str` | -- | 三重校验：status 必须为 pending、无 owner、`can_start` 检查所有 blockedBy 依赖已 completed；成功后 status -> in_progress 并记录 owner |
| **complete_task** | `task_id: str` | -- | status 必须为 in_progress 才可完成；完成后自动扫描并报告新解锁的下游任务 |

#### 代理协作工具（6 个，仅 Lead）

| 工具 | 必选参数 | 可选参数 | 设计亮点 |
|------|----------|----------|----------|
| **subagent** | `description: str` | -- | 同步阻塞子代理，最多 50 轮；仅用 6 个基础工具（无协作能力，防递归失控）；返回最终摘要文本；自带超时重试 |
| **spawn_teammate** | `name: str`, `role: str`, `prompt: str` | -- | 后台 daemon 线程异步运行；`name` 唯一性校验（已存在则拒绝）；自动注入记忆索引；全程团队历史日志 `.team_history/<name>.jsonl`；MCP 工具自动注入到队友工具集 |
| **send_message** | `to: str`, `content: str` | -- | 通过文件总线投递到目标 agent 的 `.mailboxes/<name>.jsonl`；实时终端日志 |
| **check_inbox** | `include_read: bool` | -- | 原子 rename 读取后清空收件箱；自动路由 `*_response` 协议消息到状态机 `match_response`；字符串 `"true"`/`"false"` 自动转 bool |
| **request_shutdown** | `teammate: str` | -- | 生成 `request_id`，创建 `ProtocolState(type=shutdown)`，向队友发 `shutdown_request`；队友确认后优雅退出 |
| **request_plan** | `teammate: str`, `task: str` | -- | 要求队友提交计划供审批 |
| **review_plan** | `request_id: str`, `approve: bool` | `feedback: str` | 幂等校验：已处理的 request 返回 "already approved/rejected"；approve/reject 后通过 bus 回传 `plan_approval_response` |

#### 知识工具（1 个，仅 Lead）

| 工具 | 必选参数 | 可选参数 | 设计亮点 |
|------|----------|----------|----------|
| **load_skill** | `name: str` | -- | 从 `ctx.skill_registry`（import 时扫描 `skills/*/SKILL.md`）返回技能全文；未注册时返回 `Skill not found` |

#### 工具调度与注入机制

工具执行遵循统一的 **hook -> parse -> dispatch** 管线：

```python
# agent.py 中每个 tool_call 的处理流程
blocked = trigger_hooks("PreToolUse", tool_call.function)   # 1. 钩子拦截（DENY_LIST 检查）
if blocked:
    messages.append({"role":"tool","tool_call_id":..., "content": str(blocked)})
    continue
handler = TOOL_HANDLERS.get(tool_call.function.name)         # 2. 查表
args = parse_tool_args(tool_call.function.arguments)         # 3. 安全解析（失败返回 {}）
output = handler(**args) if handler else f"Unknown tool: ..." # 4. 执行（异常被 try/except 兜底）
trigger_hooks("PostToolUse", tool_call.function, output)     # 5. 后置钩子
```

**延迟填充**：`TOOL_HANDLERS` 中 `subagent`/`load_skill`/`send_message` 等跨层依赖槽位初始为 `None`，由 `_fill_delayed_handlers()` 在包加载末尾统一导入填充，打破 `tools -> subagent/skills/bus` 的循环依赖。

**MCP 动态注入**：`_inject_mcp_tools()` 在 `agent.main()` 启动阶段调用，将所有已连接 MCP 服务器的远程工具 schema 追加到 `TOOLS`/`SUB_TOOLS`，handler 注册到 `TOOL_HANDLERS`/`SUB_HANDLERS`。重名工具跳过并告警；未连接 MCP 时为 no-op。远程工具名加 `mcp__{server}__{tool}` 前缀防冲突，外部能力对 Lead 和子代理均透明可用。

### 3. 文件总线 + 原子重命名——零依赖的跨线程通信

多代理协作需要消息传递。Mcode 没有引入 Redis/RabbitMQ 等外部中间件，而是用 **JSONL 文件 + 原子 rename** 实现了进程内消息总线：

```python
def read_inbox(self, agent: str) -> list[dict]:
    inbox = MAILBOX_DIR / f"{agent}.jsonl"
    with self._read_lock:
        self._read_counter += 1
        tmp = MAILBOX_DIR / f"{agent}.jsonl.reading_{self._read_counter}"
    inbox.rename(tmp)   # 原子操作：读即清空，杜绝并发重复读
    msgs = [json.loads(line) for line in tmp.read_text().splitlines() if line.strip()]
    tmp.unlink()
    return msgs
```

`rename()` 在同一文件系统上是原子的，配合类级 `_read_counter` + `_read_lock` 保证临时文件名唯一，从根本上避免多个队友线程同时读同一收件箱导致的消息丢失或重复。消息天然持久化到磁盘，可事后审计，重启不丢。

### 4. 多代理协作 + 协议状态机——结构化的"人机/机机"协作

Mcode 构建了 **Lead → Subagent → Teammate** 三级代理体系，辅以**任务看板依赖图**和**请求/响应协议状态机**，实现真正的多代理工程协作：

```
Lead Agent (主线程 REPL)
  ├─ subagent: 同步阻塞，处理独立子任务，返回摘要
  ├─ teammate: 后台线程，自治运行，通过 bus 通信
  │    ├─ 内循环(≤50轮): LLM + 工具执行
  │    └─ idle_poll: 空闲时自动认领看板任务 / 响应 shutdown
  └─ 任务看板: JSON 持久化，支持 blockedBy 依赖图
       create → claim → complete → 自动报告解锁的下游任务
```

协议状态机覆盖两类交互：**计划审批**（teammate 提交 plan → lead approve/reject → 结果回传）和**优雅关停**（lead 发 shutdown_request → teammate 确认后退出）。每步都有 `ProtocolState` 跟踪、类型校验（`match_response` 拒绝不匹配的响应类型）、状态幂等检查（已处理的请求不再重复处理）。

### 5. MCP 异步核心 + 同步门面——让异步 SDK 无缝融入同步代码库

MCP SDK 是异步的，而 Mcode 整体是同步的。`MCPClient` 在**专用 daemon 线程**上运行独立事件循环，通过 `run_coroutine_threadsafe` 桥接，使 `ClientSession` 跨多次 `call_tool` 存活（这是逐次 `asyncio.run()` 无法实现的）：

```python
# 每个 MCPClient 拥有独立事件循环（daemon 线程）
self._loop = asyncio.new_event_loop()
self._thread = threading.Thread(target=_run_loop, daemon=True)

# 同步门面：主线程零 async 样板
def call_tool(self, name, args) -> str:
    future = asyncio.run_coroutine_threadsafe(self._async_call_tool(name, args), self._loop)
    return future.result(timeout=self.CALL_TIMEOUT)
```

工具命名采用 `mcp__{server}__{tool}` 前缀，避免与内置工具冲突且支持无歧义路由。整个子系统**全程容错**——配置缺失、JSON 损坏、连接失败、调用异常均降级为 no-op，绝不影响主程序启动。

### 6. 全链路防御性编程——"不崩溃"是底线

Mcode 在每个可能出错的地方都设置了安全网，保证 Agent 主循环不会因单点异常而崩溃：

| 防御点 | 实现 | 代码位置 |
|--------|------|----------|
| **工具参数解析** | `parse_tool_args` 替代裸 `json.loads`，失败返回 `{}` 而非抛 `JSONDecodeError` | `utils.py` |
| **消息格式** | `sanitize_message` 确保每条消息含 `content` 键（后端必需），缺失补 `""` | `utils.py` |
| **流式响应** | `StreamMessage.model_dump` 保证 assistant 消息含 `content`；content 永远为 `""` 而非 `None` | `streaming.py` |
| **工具执行** | 每个 handler 调用包裹 `try/except`，异常转为 `Error: {e}` 字符串返回给 LLM | `agent.py` |
| **记忆系统** | 所有 LLM 调用 `try/except` 吞异常，记忆失败不影响主流程 | `memory.py` |
| **MCP 集成** | 配置缺失/连接失败/调用异常均降级为 no-op | `mcp.py` |
| **API 重试** | 超时自动重试（≤3 次）；`prompt_too_long` 触发反应式压缩后重试 | `agent.py` |

这种"每个边界都设防"的设计使 Agent 在面对不可控的 LLM 输出（畸形 JSON、空响应、截断流）时依然稳健。

### 7. 路径沙箱 + 命令拦截——双重安全屏障

| 层级 | 机制 | 实现 |
|------|------|------|
| **路径沙箱** | 所有文件操作经 `safe_path()` 校验，resolve 后若逃逸 `WORKDIR` 则抛 `ValueError` | `fsops.py` |
| **命令拦截** | `PreToolUse` 钩子 `permission_hook` 检查 `_DENY_LIST`（sudo/shutdown/reboot/mkfs/dd/REN），命中即返回 `Permission denied` | `hooks.py` |

沙箱覆盖 `read_file`/`write_file`/`edit_file`/`glob`/`grep` 全部文件工具；命令拦截在工具执行**之前**触发，被拦截的命令以 `tool` 角色消息返回给 LLM 而非执行。两者共同防止 Agent 被 LLM 幻觉引导执行危险操作。

### 8. 后台线程记忆系统——"边聊边学"且零阻塞

记忆系统的全部 LLM 操作（检索/提取/合并）都在**后台 daemon 线程**执行，不阻塞主对话流：

```python
# 对话开始时：后台线程异步加载相关记忆
_mem_holder = _load_memories_async(messages)   # 立即返回 ["", thread]
# ... 主循环执行压缩、LLM 调用 ...
memories_content = _await_memories(_mem_holder) # join(timeout=60) 取结果

# 对话结束时：后台线程提取新记忆 + 合并去重
threading.Thread(target=_post_turn_memory, args=(pre_compress,),
                 daemon=True, name="memory-extract").start()
```

记忆检索采用 **LLM 优先 + 关键词兜底** 双策略：先用 LLM 从记忆目录中选出相关条目（返回 JSON 索引数组），LLM 调用失败则回退关键词匹配。记忆文件数达阈值（10）时自动触发 `consolidate_memories` 合并去重，防止记忆膨胀。全部操作加 `memory_lock` 防并发冲突。

### 9. 流式响应聚合 + SDK 解耦——不绑死 pydantic

`streaming.py` 用纯 `dataclass` 模拟 OpenAI SDK 的返回对象（`StreamResponse`/`StreamChoice`/`StreamMessage`/`ToolCall`），使上层代码可统一调用 `.model_dump(exclude_none=True)` / `.choices[0].message` 等，**不依赖 SDK 的 pydantic 模型**。

流式处理实时将 `delta.content` 打印到 stdout（带 `Mcode:` 前缀），同时按 `index` 聚合 `tool_call` 片段（id/name/arguments 分片到达）。流被截断时（`finish_reason=None` 但有部分 tool_calls）标记为 `interrupted`，由上层优雅处理。

### 10. 延迟导入打破循环依赖——包加载顺序无关

`tools.py` 依赖 `subagent`/`skills`/`bus`（上层模块），而这些模块又依赖 `tools` 的 `SUB_HANDLERS`/`SUB_TOOLS`，形成循环。Mcode 用两种模式解决：

- **`_fill_delayed_handlers()`**：在 `tools.py` 模块底部（所有定义之后）统一执行延迟导入，填充 `TOOL_HANDLERS` 中值为 `None` 的槽位。
- **惰性属性**：`context.bus` / `context.mcp` 属性首次访问时才 import 对应类，打破 `context ↔ bus` / `context ↔ mcp` 循环。

这保证了各模块的 import 顺序无关，包加载稳定可靠。

---

## 目录结构

```
src/
├── mcode.py              # 入口：薄再导出 shim，对外保持单文件 API 兼容
│
├── mcodecore/            # ★ 核心包
│   ├── __init__.py       # 包初始化；导入即注册默认钩子
│   ├── config.py         # 配置常量 + OpenAI client + 路径常量
│   ├── exceptions.py     # 自定义异常（AgentInterrupt）
│   ├── context.py        # AppContext 全局可变状态单例
│   ├── calibrator.py     # TokenCalibrator 滑动窗口校准
│   ├── utils.py          # 通用工具（parse_tool_args / sanitize_message 等）
│   ├── fsops.py          # 文件系统 & shell 工具实现
│   ├── tasks.py          # 任务看板 CRUD（JSON 持久化）
│   ├── bus.py            # 消息总线 + 协议状态机
│   ├── hooks.py          # 钩子注册/分发 + 内置钩子
│   ├── skills.py         # 技能注册表（扫描 SKILL.md）
│   ├── memory.py         # 记忆读写/提取/合并/异步加载
│   ├── streaming.py      # 流式响应 dataclass 封装
│   ├── compact.py        # Token 估算 + 三级压缩 + 自动/反应式压缩
│   ├── tools.py          # 工具 schema + handler 映射 + system prompt
│   ├── mcp.py            # MCP 客户端（Streamable HTTP，异步核心+同步门面）
│   ├── subagent.py       # 同步子代理
│   ├── teammates.py      # 线程化队友代理
│   └── agent.py          # Lead Agent 主循环 + REPL 入口
│
│
└── skills/               # 技能文件目录（SKILL.md）
```

---

## 架构总览

整体采用**分层 + 依赖注入**架构，核心原则：

1. **配置与状态分离** — `config.py` 存放启动期固定的不可变配置；`context.py` 的 `AppContext` 单例存放运行期可变状态。
2. **全局单例 `ctx`** — 所有运行时状态（todos、活跃队友、技能注册表、消息总线、钩子、内存锁、校准器）集中在 `ctx`，模块间通过 `from .context import ctx` 共享，避免散落的 module-level 全局变量。
3. **延迟导入打破环** — `bus`、`tools` 中涉及跨层依赖（如 `tools` → `subagent`/`skills`/`bus`）的 handler 通过 `_fill_delayed_handlers()` 在包加载末尾延迟填充，消除循环导入。
4. **薄入口 + 再导出** — `mcode.py` 仅做 `from mcodecore.X import *`，保持与旧单文件完全相同的公开 API，调用方零改动。

```
            ┌──────────────────────────────────────────────────┐
            │                  mcode.py (shim)                 │
            │         re-export 全部 mcodecore 公开符号         │
            └────────────────────────┬─────────────────────────┘
                                     │
            ┌────────────────────────▼─────────────────────────┐
            │                    agent.py                      │
            │   Lead Agent 主循环 / REPL / inbox 轮询           │
            └──┬──────────┬──────────┬──────────┬──────────┬───┘
               │          │          │          │          │
        ┌──────▼──┐ ┌─────▼────┐ ┌───▼───┐ ┌────▼────┐ ┌───▼─────┐
        │streaming│ │ compact  │ │memory │ │  bus    │ │ tools   │
        │ (LLM流) │ │ (3级压缩) │ │(记忆) │ │(消息总线)│ │(工具表) │
        └────┬────┘ └────┬─────┘ └───┬───┘ └────┬────┘ └────┬────┘
             │           │           │          │           │
             │      ┌────▼─────┐ ┌────▼────┐ ┌───▼────┐ ┌────▼────┐
             │      │calibrator│ │ skills  │ │ tasks  │ │ fsops   │
             │      └──────────┘ └─────────┘ └────────┘ └─────────┘
             │
        ┌─────────┐  (MCP 远程工具，init 阶段注入 TOOLS)
        │  mcp    │ ──────────────► tools._inject_mcp_tools()
        └────┬────┘
             │
        ┌────▼────┐  ┌────────────┐
        │ config  │  │  context   │  ◄── 全模块共享 ctx 单例
        │(常量/client)│ │(AppContext) │
        └─────────┘  └─────┬──────┘
                           │
                      ┌────▼────┐
                      │ hooks   │  ◄── PreToolUse/PostToolUse/Stop
                      └─────────┘

        ┌────────────┐         ┌──────────────┐
        │ subagent   │         │  teammates   │  ◄── 线程化，通过 bus 通信
        │ (同步子代理)│         │ (队友代理)    │
        └────────────┘         └──────────────┘
```

---

## 分层依赖关系

包内模块按依赖方向自底向上分层（上层依赖下层，禁止反向）：

| 层级 | 模块 | 职责 |
|------|------|------|
| **L0 基础** | `config`, `exceptions`, `calibrator` | 无内部依赖的纯基础层 |
| **L1 工具** | `utils` | 通用函数，仅依赖 `config`（延迟） |
| **L2 文件/任务** | `fsops`, `tasks`, `skills` | 具体资源操作，依赖 L0/L1 |
| **L3 状态/通信** | `context`, `bus`, `hooks` | 运行时状态与消息总线，依赖 L2 |
| **L4 记忆/流式/压缩** | `memory`, `streaming`, `compact` | 高级能力，依赖 L0-L3 |
| **L5 工具表** | `tools`, `mcp` | schema+handler 汇总；MCP 远程工具发现/调用，依赖 L2-L4 |
| **L6 代理** | `subagent`, `teammates`, `agent` | 顶层循环，依赖全部下层 |

> `context.bus` 属性采用**惰性创建**（首次访问才 import `bus.MessageBus`），避免 `context ↔ bus` 循环。

---

## 模块详解

### 入口层 `mcode.py`

- **角色**：薄再导出 shim，约 60 行，无任何逻辑。
- **作用**：将 `mcodecore` 各模块的公开符号全部 re-export， `from mcode import agent_loop, main, ...` 。
- **入口**：`if __name__ == "__main__": main()` → 调用 `mcodecore.agent.main()`。
- **意义**：重构对调用方零侵入，旧脚本无需改动。

---

### `config.py` — 配置中心

存放**启动期固定、运行期不变**的配置，刻意与 `context.py`（可变状态）分离。

| 类别 | 内容 |
|------|------|
| API 配置 | `API_BASE`、`API_KEY`、`LLM_MODEL`（`glm-5.2`）、`client: OpenAI` |
| 路径常量 | `WORKDIR`、`MEMORY_DIR`、`SKILLS_DIR`、`TRANSCRIPT_DIR`、`TOOL_RESULTS_DIR`、`TASKS_DIR`、`MAILBOX_DIR`、`_BG_OUTPUT_DIR` 等，启动即 `mkdir` |
| 数值阈值 | `BASH_TIMEOUT`、`CONTEXT_LIMIT`（128k×0.9）、`KEEP_RECENT_LOOP_TURN`、`PERSIST_THRESHOLD`、`CONSOLIDATE_THRESHOLD`、`IDLE_*` |
| 平台适配 | `_IS_WINDOWS`、`_POPEN_KWARGS`（Windows 用 `CREATE_NEW_PROCESS_GROUP`）、`_enable_ansi()` |

> **设计要点**：路径常量在 import 时绑定，路径常量在 import 时绑定到各消费模块命名空间。

---

### `exceptions.py` — 异常定义

仅定义 `AgentInterrupt(Exception)`：用户中断 agent 执行时抛出。`agent_loop`、`fsops.run_bash` 等处 `except AgentInterrupt` 后优雅退出（杀进程树）。

---

### `context.py` — 全局运行时状态

`AppContext` 类 —— **所有可变运行时状态的唯一容器**，全局单例 `ctx`。

```python
ctx.current_todos        # 当前会话 todo 列表
ctx._bus                 # MessageBus（惰性创建，见 .bus 属性）
ctx.pending_requests     # 协议状态机：request_id -> ProtocolState
ctx.active_teammates     # name -> threading.Event（队友完成信号）
ctx.teammate_registry    # name -> {role, spawned_at, status}
ctx.skill_registry       # name -> {name, description, content}
ctx.hooks                # 事件 -> [callback] 四类钩子
ctx.memory_lock          # 记忆读写锁（防并发）
ctx.calibrator           # TokenCalibrator 实例
ctx._mcp_manager         # MCPManager（惰性创建，见 .mcp 属性）
```

提供 `register_hook` / `trigger_hooks` 便捷方法。`bus` 属性惰性创建以打破 `context ↔ bus` 循环依赖；`mcp` 属性惰性创建 `MCPManager`，避免 `context ↔ mcp` 循环。


### `utils.py` — 通用工具函数

| 函数 | 作用 |
|------|------|
| `parse_frontmatter(text)` | 解析 `---` YAML 头，返回 `(meta, body)` |
| `parse_bg_command(cmd)` | 解析 `bg:` 前缀 → `(is_bg, log_name, cmd_core)` |
| `parse_explicit_timeout(cmd)` | 解析尾部 `# timeout=N` → `(timeout, stripped_cmd)` |
| `truncate(text, limit)` | 截断并追加 `...` |
| `new_request_id()` | 生成 `req_XXXXXX` 随机 ID |
| **`parse_tool_args(arguments)`** | **安全解析**工具参数 JSON，失败返回 `{}` 而非抛异常 |
| **`sanitize_message(msg)`** | **确保消息含 `content` 键**（后端必需，缺失会被拒） |
| **`sanitize_messages(msgs)`** | 批量 `sanitize_message` 副本 |

> `parse_tool_args` / `sanitize_message` / `sanitize_messages` 是本次重构的关键健壮性修复，被 `agent`、`subagent`、`teammates`、`hooks` 统一使用，替代原始的裸 `json.loads(... or "{}")`。

---

### `fsops.py` — 文件系统与 Shell 工具

实现 6 个基础工具的 handler，供 subagent / teammate / lead 共享：

| 函数 | 说明 |
|------|------|
| `safe_path(p)` | 路径沙箱化：resolve 后校验 `is_relative_to(WORKDIR)`，逃逸则抛 `ValueError` |
| `run_bash(command)` | Shell 执行：支持 `bg:` 后台、`# timeout=N`、超时杀进程树、流式读取、50KB 截断 |
| `run_read(path, offset, limit)` | 读文件：行号前缀、分页、二进制检测、多编码回退（utf-8/gbk/latin-1） |
| `run_write(path, content)` | 写文件 |
| `run_edit(path, old_text, new_text)` | 精确替换（首次匹配） |
| `run_glob(pattern)` | 递归 glob，按 mtime 倒序，限 100 条，沙箱校验 |
| `run_grep(pattern, path, include)` | 正则搜索，限 50 匹配，沙箱校验 |

`run_bash` 关键实现：独立 reader 线程 + queue，主循环轮询超时与进程结束，超时调 `_kill_process_tree`（Windows `taskkill /T /F`，Linux `killpg`），并响应 `AgentInterrupt` 中断。

---

### `tasks.py` — 任务看板 CRUD

基于 JSON 文件的轻量任务系统，支持**依赖图**：

- `Task` dataclass：`id, subject, description, status, owner, blockedBy`。
- `create_task` / `save_task` / `load_task` / `list_tasks` / `get_task`。
- `can_start(task_id)`：检查所有 `blockedBy` 依赖是否 `completed`。
- `claim_task(task_id, owner)`：pending → in_progress，记录 owner。
- `complete_task(task_id)`：in_progress → completed，并报告新解锁的下游任务。
- `scan_unclaimed_tasks()`：扫描所有可认领的 pending 任务（无 owner 且依赖已满足），供 teammate idle 自动认领。

持久化到 `.tasks/task_*.json`，ID 含时间戳+随机数。

---

### `bus.py` — 消息总线与协议状态机

进程内**基于 JSONL 文件**的消息总线 + 请求/响应协议状态机。

**`MessageBus`**：
- 每个 agent 拥有 `.mailboxes/<name>.jsonl` 收件箱。
- `send(from, to, content, type, metadata)`：追加一条 JSON 消息。
- `read_inbox(agent)`：**原子 rename**（`inbox → inbox.reading_N`）后读取并删除，避免并发重复读；类级 `_read_counter` + `_read_lock` 保证唯一性。

**`ProtocolState`** dataclass：单次协议交互的状态（shutdown / plan_approval），存于 `ctx.pending_requests[request_id]`。

**`match_response`**：校验响应类型匹配后更新状态为 approved/rejected。

**`consume_lead_inbox`**：读取 lead 收件箱，自动路由 `*_response` 类型消息到 `match_response`。

**`idle_poll`**：teammate 空闲轮询 —— 检查收件箱（shutdown 请求则优雅关闭）和未认领任务（自动 claim），超时返回 `"timeout"`。

**Lead 侧 handler**：`run_send_message`、`run_check_inbox`、`run_request_shutdown`、`run_request_plan`、`run_review_plan`、`_teammate_submit_plan`。

---

### `hooks.py` — 钩子系统

四类生命周期钩子，"首个非 None 返回值胜出"语义：

| 事件 | 触发时机 | 内置钩子 |
|------|----------|----------|
| `UserPromptSubmit` | 用户输入提交 | `context_inject_hook`（打印 WORKDIR） |
| `PreToolUse` | 工具执行前 | `permission_hook`（DENY_LIST 拦截 sudo/shutdown 等）、`log_hook`（打印工具名） |
| `PostToolUse` | 工具执行后 | （预留） |
| `Stop` | agent 轮结束 | `summary_hook`（统计工具调用次数） |

`install_default_hooks()` 在 `mcodecore/__init__.py` import 时自动调用。`permission_hook` 使用 `parse_tool_args` 安全解析参数。

---

### `skills.py` — 技能注册表

- `_scan_skills()`：扫描 `skills/*/SKILL.md`，解析 frontmatter，填充 `ctx.skill_registry`（import 时执行一次）。
- `list_skills()`：返回 Markdown 目录（注入 system prompt）。
- `load_skill(name)`：返回技能全文。
- `SKILL_REGISTRY`：模块级别名 = `ctx.skill_registry`。

---

### `memory.py` — 记忆系统

基于 `.memory/*.md` 文件 + `MEMORY.md` 索引的长期记忆。

| 函数 | 作用 |
|------|------|
| `write_memory_file` | 写记忆文件 + 重建索引 |
| `_rebuild_index` | 扫描所有 `.md` 生成 `MEMORY.md` |
| `read_memory_index` / `read_memory_file` / `list_memory_files` | 读取 |
| `select_relevant_memories` | **LLM 选择**相关记忆（返回 JSON 索引数组），失败回退关键词匹配 |
| `load_memories` | 加载选中记忆到 `<relevant_memories>` 文本块（加锁） |
| `extract_memories` | 轮后从最近 10 条对话**提取新记忆**（LLM 返回 JSON 数组） |
| `consolidate_memories` | 文件数 ≥ 阈值时**合并去重**（LLM 重写全部记忆） |
| `_post_turn_memory` | 轮后后台执行 extract + consolidate（加锁） |
| `_load_memories_async` | **后台线程**加载记忆，返回 `["", thread]` holder |
| `_await_memories` | 等待异步加载完成（join timeout=60s） |

记忆类型：`user` / `feedback` / `project` / `reference`。所有 LLM 调用均 `try/except` 吞异常，不影响主流程。

---

### `streaming.py` — 流式响应封装

用 **dataclass** 模拟 OpenAI SDK 返回对象，使上层代码可统一调用 `.model_dump(exclude_none=True)` / `.choices[0].message` 等，**不依赖 SDK 的 pydantic 模型**。

| 类 | 说明 |
|----|------|
| `ToolCallFunction` | 工具调用的 function 部分（name, arguments） |
| `ToolCall` | 单个工具调用（id, type, function, index） |
| `StreamMessage` | 聚合后的 assistant 消息；`model_dump` **保证 assistant 消息含 `content` 键** |
| `StreamChoice` | 单个 choice（message + finish_reason） |
| `StreamResponse` | 完整响应（choices + usage） |

**`stream_response(**kwargs)`**：
- 实时打印 content 到 stdout（带 `Mcode:` 前缀）。
- 累积 tool_call 片段（按 index 聚合 id/name/arguments）。
- 流被截断（finish_reason None 但有部分 tool_calls）标记 interrupted。
- **content 永远为 `""` 而非 `None`**（后端要求）。
- 自动设置 `stream_options={"include_usage": True}`。

---

### `compact.py` — Token 估算与上下文压缩

上下文管理的核心，实现**三级压缩 + 自动压缩 + 反应式压缩**：

```
每轮循环执行顺序：
  tool_result_budget → snip_compact → micro_compact → [超限则 compact_history]
```

| 函数 | 层级 | 说明 |
|------|------|------|
| `estimate_tokens_messages` | 估算 | 4 字符≈1 token + overhead，乘校准因子 |
| `tool_result_budget` | **L3 持久化** | 限制最近一轮 tool 输出总字节（200KB），超限的调 `persist_large_output` 落盘 |
| `persist_large_output` | L3 | 超 `PERSIST_THRESHOLD`（30KB）的输出写到 `.task_outputs/`，返回预览占位 |
| `snip_compact` | **L1 截断** | 保留头部 1 轮 + 尾部 N 轮，中间替换为 `[snipped X turns]` 占位 |
| `micro_compact` | **L2 微压缩** | 旧轮次 tool_result 内容替换为占位，保留最近 N 轮完整 |
| `compact_history` | **自动压缩** | 落盘 transcript → LLM 摘要 → `_build_post_compact_context` 重建上下文块 |
| `reactive_compact` | **反应式压缩** | API 报 prompt_too_long 时触发，同自动压缩 |
| `write_transcript` | 持久化 | 写 `.transcripts/transcript_*.jsonl` |
| `summarize_history` | LLM 摘要 | 保留目标/发现/文件/待办/约束 |
| `_build_post_compact_context` | 重建 | 汇总：最近访问文件 + 当前 plan + 任务看板 + 活跃队友 |
| `group_turns` / `ensure_valid_start` / `_strip_orphan_*` | 辅助 | 按工具调用分组、清理孤儿 tool 消息 |

---

### `tools.py` — 工具定义与调度

汇总**工具 schema + handler 映射 + system prompt**，是工具系统的中央注册表。

**System Prompt**：
- `build_system()`：Lead system prompt（含工作目录、技能目录、记忆索引、OS 提示、记忆注入说明）。
- `SUB_SYSTEM`：子代理 system prompt。

**工具集分层**：

| 集合 | 工具 | 使用者 |
|------|------|--------|
| `_BASE_TOOLS` | bash, read_file, write_file, edit_file, glob, grep | 全部 |
| `SUB_TOOLS` | = `_BASE_TOOLS` | subagent |
| `TEAMMATE_TOOLS` | _BASE + send_message, submit_plan, list_tasks, claim_task, complete_task | teammate |
| `TOOLS` | SUB + todo_write, subagent, load_skill, create_task, list_tasks, get_task, claim_task, complete_task, spawn_teammate, send_message, check_inbox, request_shutdown, request_plan, review_plan | **Lead Agent** |

**Handler 映射**：
- `SUB_HANDLERS`：6 个基础 handler（直接映射 `fsops` 函数）。
- `TOOL_HANDLERS`：完整映射，含 `run_todo_write`、`run_create_task` 等。
- **`_fill_delayed_handlers()`**：包加载末尾延迟填充 `subagent`→`spawn_subagent`、`load_skill`、`send_message`/`check_inbox`/`request_*`/`review_plan` 等，**打破循环导入**。
- **`_inject_mcp_tools()`**：MCP 初始化后由 `agent.main` 调用，将所有已连接 MCP 服务器的工具 schema 注入 `TOOLS`/`SUB_TOOLS` 并注册 handler 到 `TOOL_HANDLERS`/`SUB_HANDLERS`。重名工具跳过并告警；未连接 MCP 时为 no-op。

---

### `mcp.py` — MCP 远程工具集成

通过 **Model Context Protocol**（Streamable HTTP 传输）连接外部工具服务器，自动发现远程工具并注入到工具表，全程容错（配置缺失 / 连接失败 / 调用异常均不影响主程序）。

**架构**：同步门面 `MCPManager` 包裹异步核心，主线程零 async 样板。

```
MCPManager  (同步门面，主线程)
 └─ MCPClient × N          (每服务器一个)
     ├─ 专用 asyncio 事件循环，运行在 daemon 线程
     ├─ streamablehttp_client + ClientSession（AsyncExitStack 保活）
     └─ 同步方法经 run_coroutine_threadsafe 桥接到该循环
```

**核心组件**：

| 组件 | 职责 |
|------|------|
| `MCPServerConfig` | 单服务器配置 dataclass（name / url / headers / enabled） |
| `MCPClient` | 单服务器异步客户端 + 同步门面；connect / list_tools / call_tool |
| `MCPManager` | 多服务器注册表；init / shutdown / list_all_tool_schemas / call / build_handlers |

**工具命名约定**：`mcp__{server}__{tool}`（`MCP_PREFIX = "mcp__"`），避免与内置工具冲突并支持无歧义路由。

**关键设计**：
- 每个 `MCPClient` 拥有**专用事件循环**（daemon 线程），保证 `ClientSession` 跨多次 `call_tool` 存活（session 绑定于创建它的 loop，逐次 `asyncio.run()` 不可行）。
- `list_tools()` 首次拉取后缓存，避免重复 RPC。
- `call_tool()` 返回纯文本：`TextContent` 拼接，图片/音频/嵌入资源以占位符替代；`isError` 时返回 `[MCP Error]` 前缀。
- `init()` 逐服务器连接，失败跳过并告警，最终 `is_connected` 反映是否至少连上一台。
- `build_handlers()` 为每个工具生成闭包 handler，`__name__` 设为全限定工具名。

**配置文件** `.mcp.json`（工作目录，路径见 `config.MCP_CONFIG_PATH`）：
```json
{
  "mcpServers": {
    "my-server": {
      "url": "http://localhost:8000/mcp",
      "headers": {"Authorization": "Bearer xxx"},
      "enabled": true
    }
  }
}
```

**模块级便捷函数**：`init_mcp()` / `shutdown_mcp()` 容错包装，供 `agent.main` 调用。

---

### `subagent.py` — 同步子代理

`spawn_subagent(description)`：**同步阻塞**的子代理，处理复杂子任务后返回最终摘要。

- 最多 50 轮循环。
- 仅用 6 个基础工具（`SUB_TOOLS` / `SUB_HANDLERS`）。
- PreToolUse / PostToolUse 钩子。
- `finish_reason != "tool_calls"` 时结束。
- 消息列表 `sanitize_messages` 清洗，assistant 消息 `sanitize_message` 包装。
- 用 `parse_tool_args` 安全解析参数。
- 返回最后一条 assistant content；空则回溯查找。

---

### `teammates.py` — 线程化队友代理

`spawn_teammate_thread(name, role, prompt)`：**后台线程**运行的自治队友。

**生命周期**：
1. 注册到 `ctx.active_teammates[name] = Event()` + `ctx.teammate_registry`。
2. `run()` 线程：内循环（最多 50 轮）→ idle 轮询 → 循环。
3. 每轮先 `read_inbox` 处理协议消息（shutdown/plan_response），非协议消息注入对话。
4. LLM 调用（`sanitize_messages` 清洗），工具执行（`parse_tool_args`）。
5. 内循环结束后 `idle_poll`：检查收件箱 / 自动认领任务，超时则退出。
6. 完成后 `bus.send(name, "lead", result, "result")`，设置 Event 信号。

**特性**：
- 独立工具集（含 send_message / submit_plan / task 操作）。
- 独立 handler 闭包（`_run_send_message` 等捕获 `name`）。
- **团队历史日志**：`.team_history/<name>.jsonl`，记录 spawned/inbox_received/llm_response/tool_called/finished 等事件（带锁）。
- 记忆异步加载注入。
- `finally` 块保证 Event.set + registry 状态更新。

---

### `agent.py` — Lead Agent 主循环

顶层 Agent，包含主循环、REPL、inbox 轮询。

**`agent_loop(messages)`** —— 核心循环：

```
while True:
  1. build_system() 重建 system prompt
  2. 三级压缩: tool_result_budget → snip_compact → micro_compact
  3. 超限则 compact_history 自动压缩
  4. 组装 request_messages（插 system + 注入记忆到最后一条 user）
  5. sanitize_messages 清洗
  6. stream_response 流式调用（temperature=0.7, max_tokens=16384, enable_thinking=False）
  7. 异常处理: timeout 重试 / prompt_too_long 反应式压缩
  8. 记录 token 校准样本
  9. append sanitize_message(assistant)
 10. finish_reason != tool_calls → 后台 _post_turn_memory + Stop hook → return
 11. 遍历 tool_calls: PreToolUse hook → handler(**parse_tool_args) → PostToolUse hook → append tool result
```

**`_run_agent_turn(history)`**：执行一次 `agent_loop`，捕获异常，返回是否应退出 REPL。

**`_drain_inbox(history)`**：清理已完成队友，消费 lead 收件箱（路由协议响应），注入历史并触发一轮。

**`main()`**：REPL 入口 —— `input()` 循环，`q/exit/quit` 退出，空输入时 drain inbox，UserPromptSubmit hook。

**MCP 生命周期**：`main()` 启动时调用 `init_mcp()` 连接所有已配置服务器，随后 `_inject_mcp_tools()` 将远程工具注入工具表；退出时 `shutdown_mcp()` 优雅关闭所有会话。


---

## 关键设计决策

### 1. 配置 vs 状态分离
`config.py`（不可变）与 `context.py`（可变 `ctx` 单例）严格分离。这使配置可独立管理、状态可统一注入。

### 2. 延迟导入打破循环
`tools.py` → `subagent`/`skills`/`bus` 的反向依赖通过 `_fill_delayed_handlers()` 在包加载末尾填充；`context.bus` 属性惰性创建。保证 import 顺序无关。

### 3. 后端兼容性修复
API要求每条消息含 `content` 字段：：
- `sanitize_message` / `sanitize_messages`：在每次 API 调用前确保 `content` 存在。
- `StreamMessage.model_dump`：assistant 消息保证 `content` 键。
- `stream_response`：content 永远为 `""` 而非 `None`。
- `parse_tool_args`：替代裸 `json.loads(... or "{}")`，解析失败返回 `{}` 而非崩溃。

### 4. 三级渐进式压缩
L1 snip（截断中间轮）→ L2 micro（占位旧 tool_result）→ L3 persist（落盘超长输出），自动 + 反应式 LLM 摘要兜底。避免单一策略导致的信息丢失。

### 5. 文件总线而非内存队列
`MessageBus` 用 JSONL 文件 + 原子 rename 实现跨线程通信，天然持久化、可观测、无需额外依赖，适合单机多线程场景。

### 6. 薄 shim 保持兼容
`mcode.py` 仅 re-export，使重构对调用方零侵入，可渐进迁移。

### 7. MCP 异步核心 + 同步门面
MCP SDK 为异步，但整个 codebase 同步。`MCPClient` 在专用 daemon 线程跑独立事件循环，通过 `run_coroutine_threadsafe` 桥接，使 `ClientSession` 跨多次调用存活且主线程零 async 样板。工具名加 `mcp__{server}__{tool}` 前缀防冲突；整个子系统容错——配置缺失、JSON 损坏、连接或调用失败均降级为 no-op，绝不影响主程序启动。
