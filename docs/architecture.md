# NexusAgent 架构设计文档

## 1. 设计目标

NexusAgent 是一个面向学习的 AI Agent 框架，核心目标是让开发者理解 Agent 框架的内部运转机制。设计上融合了 **LangGraph 的图式编排**、**OpenAI Agents SDK 的简洁原语**、**CrewAI 的角色分工**、**Anthropic 的渐进式上下文管理** 四大流派的核心思想。

## 2. 为什么选择 ReAct 模式？

ReAct (Reasoning + Acting) 是 2025 年 AI Agent 领域的事实标准循环模式：

```
Observe → Think → Act → Observe → Think → ... → Respond
```

**行业共识**: OpenAI Agents SDK、LangGraph、CrewAI、Claude Code 的底层都是 ReAct 的变体。

**核心优势**:
- LLM 在"思考该做什么"和"实际去做"之间交替，解决了 LLM 只能生成文本的先天局限
- 灵活性与可控性的最佳平衡点——不像 Plan-then-Execute 那样死板，不像 AutoGPT 那样不可控

## 3. 为什么分 5 层？

每一层回答一个独立问题，对应一个独立的变化轴：

| 层 | 回答的问题 | 变化原因 |
|------|----------|----------|
| LLM Provider | "找哪个模型说话？" | 换模型、换 API |
| Tool 系统 | "能做什么事？" | 新增工具能力 |
| Agent 核心 | "怎么思考和决策？" | ReAct 策略调整 |
| Memory | "记住什么？" | 记忆策略切换 |
| Orchestrator | "多个 Agent 怎么配合？" | 协作模式改变 |

**隔离变化的收益**: 换一个 LLM Provider 不改 Agent 代码，加一个 Tool 不改 Memory 代码。

## 4. 为什么核心全部自研？

**学习目标决定技术选型**。如果底层使用 LangChain，学到的是"怎么调 LangChain API"；自研引擎，学到的是"ReAct 循环怎么写出来的"。

框架只依赖各厂商的官方 SDK（openai、anthropic），因为自己实现 HTTP 调用和流式解析没有学习价值。

生态互通通过适配器层实现（后续扩展），框架核心保持零框架依赖。

## 5. 核心设计决策

### 5.1 依赖注入 > 继承 > 全局单例

```python
# 好：依赖注入
agent = ReActAgent(name="助手", llm=OpenAIProvider(), tools=[...], memory=BufferMemory())
```

- 可测试：测试时注入 mock LLM
- 灵活：同一类配不同 tools 就是不同的 Agent
- 显式：看构造函数就知道能力边界

### 5.2 Tool 从函数签名自动推导 Schema

```python
@Tool.from_function(name="get_weather", description="查询天气")
async def get_weather(city: str, date: str = "today") -> str:
    ...
```

- `city: str` → `{"type": "string"}` 自动映射
- `date: str = "today"` → 带默认值，不加入 required
- DRY 原则：信息只有一个来源（函数签名），不会出现"改了参数忘记改 Schema"

### 5.3 双层记忆架构

参考人类记忆模型：
- **工作记忆** (BufferMemory)：最近 N 轮，精确保留
- **长期记忆** (VectorStoreMemory)：跨会话语义检索
- **摘要压缩** (SummaryMemory)：参考 Anthropic Compaction 策略

### 5.4 渐进式编排

| 模式 | 学习顺序 | 教学目的 |
|------|----------|----------|
| Sequential | 第一个例子 | 理解 Agent 间数据流动 |
| Supervisor | 第二个例子 | 理解 Handoff 机制 (OpenAI Agents SDK 核心概念) |
| Graph | 最终形态 | 理解条件分支和循环 (LangGraph 核心概念) |

## 6. 技术选型理由

| 技术 | 理由 |
|------|------|
| Python 3.11+ | async/await 生态成熟，类型系统够用，行业一致性 |
| Pydantic 2.0+ | JSON Schema 生成的事实标准，类型校验 |
| ChromaDB | `pip install chromadb` 一行安装，零配置，适合学习 |
| structlog | 结构化日志，便于调试 Agent 行为 |
| uv | 2025 年最快的 Python 包管理器 |

## 7. 实现路线

按"每一步都可运行验证"的原则分 7 个阶段：

1. **基础设施** — 类型、异常、配置、日志 ✅
2. **LLM Provider** — OpenAI + Anthropic + Google ✅
3. **Tool 系统** — 函数→JSON Schema 自动推导 ✅
4. **ReAct Agent** — Think→Act→Observe 主循环 ✅
5. **Memory 系统** — 双层记忆（短期+长期向量）✅
6. **多 Agent 编排** — Sequential / Supervisor / Graph ✅
7. **示例与测试** — 147 个测试覆盖全部模块 ✅

## 8. 功能里程碑

### Tier 1: 核心增强 (502d2ce)

**LLM 调用重试机制** (`nexus/llm/base.py`)
- `_retry_call()` 指数退避重试包装器，自动处理网络抖动和速率限制
- `_is_retryable_error()` 可重试错误判断（429/5xx/ConnectionError/Timeout）
- 所有 Provider 共享同一套重试逻辑，通过 `max_retries`/`retry_delay` 配置

**死循环主动干预** (`nexus/agent/react_agent.py:_check_loop()`)
- 检测连续 N 次相同的 tool_call 调用（默认 3 次）
- 检测到循环时向消息历史注入干预提示，引导 Agent 尝试其他方法
- 不再仅打 warning，而是主动改变 LLM 看到的消息上下文

**流式输出完善** (`nexus/agent/react_agent.py:stream()`)
- 重写 stream() 方法的流式循环逻辑
- text chunk 立即 yield 给用户，tool_call 累积后执行，无 tool_call 时流式输出即为最终回复
- 用户在工具调用过程中也能看到模型的"思考"过程

### Tier 2: 生态扩展 (b6e1e91)

**Google Gemini Provider** (`nexus/llm/providers/google.py`)
- 基于 `google-genai` 2.7.0 SDK 实现
- 关键设计：`AutomaticFunctionCallingConfig(disable=True)` 禁用 SDK 自动函数调用
- Gemini 消息格式转换：system→`system_instruction`，assistant→`"model"` 角色，tool→`"user"` 角色 + `function_response`
- 注册到 `LLMRegistry`，通过 `create_llm("google")` 使用

**Prompt 模板系统** (`nexus/prompts/`)
- `PromptTemplate`：零依赖、正则驱动的 `{{ variable }}` 模板引擎
- Jinja2 兼容语法，未来可无缝切换
- `from_file()` 支持外部 .txt 文件加载
- 8 个预设模板：ASSISTANT、CHINESE_ASSISTANT、CODE_REVIEWER、CODE_WRITER 等

**Human-in-the-Loop 安全审批** (`nexus/agent/guard.py`)
- `CLIApproval` 命令行交互式审批器
- Tool 级 `requires_approval` 标记 + Agent 级 `approval_callback` 回调
- 拒绝后返回错误消息给 LLM（不抛异常），Agent 可调整策略继续

### Tier 3: 深度完善 (34e6fcd)

**Embedding Provider** (`nexus/embeddings/`)
- `BaseEmbeddingProvider` 抽象基类：`embed()` + `embed_batch()`
- `OpenAIEmbeddingProvider`：封装 OpenAI `text-embedding-3-small`，支持自定义 base_url
- `VectorStoreMemory` 改造：优先使用 provider 嵌入，未提供时保留 SHA-256 哈希回退
- 遵循与 LLM Provider 相同的抽象基类 + 依赖注入模式

**Eval 评估框架** (`nexus/eval/`)
- `EvalCase`：测试用例定义，支持期望输出（字符串/列表）、工具调用验证
- `EvalResult`：评估结果，包含通过状态、实际输出、调用详情
- `EvalSuite`：批量执行 + 格式化报告（通过率、失败详情）
- Agent 异常不中断评估，记录为失败用例

**测试覆盖补充** (新增 80 个测试)
- `test_config.py`：YAML 加载、环境变量解析、自动发现
- `test_factory.py`：LLMRegistry 注册/创建/覆盖
- `test_handoff.py`：HandoffTool、SequentialOrchestrator、数据流验证
- `test_embeddings.py`：Provider 行为、VectorStore 集成
- `test_eval.py`：用例定义、执行、报告生成
- `test_providers.py`：补充 OpenAI 流式输出测试
- 总测试数：147 个（从初始 68 个增长 116%）
