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

1. **基础设施** — 类型、异常、配置、日志
2. **LLM Provider** — 能调用模型获得回复
3. **Tool 系统** — 能定义工具让 LLM 生成 tool_calls
4. **ReAct Agent** — 配合工具完成多步骤任务 ⭐核心
5. **Memory 系统** — 跨会话记忆
6. **多 Agent 编排** — 多 Agent 协同工作流
7. **示例与测试** — 可交付使用
