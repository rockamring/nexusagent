# NexusAgent

一个模块化、可扩展的 AI Agent 学习框架。融合了 **LangGraph 的图式编排**、**OpenAI Agents SDK 的简洁原语**、**CrewAI 的角色分工**、**Anthropic 的渐进式上下文管理** 四大流派的核心思想。

## 快速开始

```bash
# 安装依赖
pip install -e .

# 设置 API Key
export OPENAI_API_KEY="sk-..."

# 运行示例
python examples/01_hello_world.py
python examples/02_tool_agent.py
python examples/03_memory_agent.py
python examples/04_multi_agent.py
```

## 架构总览

```
用户请求 → [Orchestrator 编排层]
                │
                ├── Agent 1 ──→ LLM ←── Memory
                │      └── Tools
                ├── Agent 2 ──→ LLM ←── Memory
                │      └── Tools
                └── ...
```

**5 层架构**：LLM Provider → Tool 系统 → Agent 核心 → Memory 系统 → Orchestrator 编排

## 核心特性

- **ReAct 循环引擎**：Think → Act → Observe，行业标准 Agent 模式
- **多 LLM Provider**：OpenAI、Anthropic，统一接口轻松切换
- **Tool Calling**：普通 Python 函数 + 类型注解 = 可被 LLM 调用的工具
- **双层记忆**：滑动窗口短期记忆 + ChromaDB 向量长期记忆
- **三种编排模式**：顺序流水线、监督者模式(Handoff)、图式编排

## 基础用法

```python
from nexus.llm.providers.openai import OpenAIProvider
from nexus.agent import ReActAgent

llm = OpenAIProvider(api_key="sk-...", default_model="gpt-4o-mini")

agent = ReActAgent(
    name="助手",
    llm=llm,
    system_prompt="你是一个有帮助的 AI 助手。",
)

result = await agent.run("你好！")
print(result.content)
```

## 定义工具

```python
from nexus.tools.base import Tool
from nexus.tools.registry import ToolRegistry

@Tool.from_function(name="get_weather", description="查询天气")
async def get_weather(city: str) -> str:
    return f"{city}: 晴, 28°C"

tools = ToolRegistry()
tools.register(get_weather)
agent = ReActAgent(name="天气助手", llm=llm, tools=tools)
```

## 多 Agent 编排

```python
from nexus.orchestrator import SequentialOrchestrator

pipeline = SequentialOrchestrator([
    (analyzer_agent, "分析需求：{input}"),
    (writer_agent, "根据分析写方案：{input}"),
])

result = await pipeline.run("设计一个待办事项应用")
```

## 运行测试

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## 项目结构

```
nexus/
├── core/          # 公共类型、异常、配置
├── llm/           # LLM Provider 抽象层
│   └── providers/ # OpenAI, Anthropic
├── agent/         # ReAct Agent 核心引擎
├── tools/         # Tool 系统
│   └── builtin/   # 内置工具（计算器、文件、搜索）
├── memory/        # Memory 系统
├── orchestrator/  # 多 Agent 编排
└── utils/         # 日志、流式工具
```

## 设计文档

详细架构设计文档见 [docs/architecture.md](docs/architecture.md)。
