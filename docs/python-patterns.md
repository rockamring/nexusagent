# Python 关键技术点

本文档记录 NexusAgent 工程中用到的 Python 语言层面的关键技术点和设计模式，面向中大型项目中的模块设计、测试、异步编程等必须掌握的能力。

## 1. 模块设计

### 1.1 显式公开 API：`__init__.py` + `__all__`

每个子包在 `__init__.py` 中从内部模块导入公开符号并汇总到 `__all__`，用户在外部只需一行导入，无需关心内部文件布局：

```python
# nexus/agent/__init__.py
from nexus.agent.base import BaseAgent
from nexus.agent.react_agent import ReActAgent

__all__ = ["BaseAgent", "ReActAgent"]
```

```python
# 用户使用
from nexus.agent import ReActAgent     # 好
from nexus.agent.react_agent import ReActAgent  # 也能用，但不推荐
```

**原则**：
- 每个 `__init__.py` 就是一个公开 API 契约
- 不导出的符号视为内部实现，随时可能变更
- `__all__` 让 `from package import *` 也可控（虽然不推荐用 star import）

### 1.2 抽象基类（ABC）：定义接口，隔离变化

5 层架构每层各有一个抽象基类，定义该层的最小契约：

```python
# nexus/llm/base.py
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @abstractmethod
    async def generate(self, messages, tools, *, model, temperature,
                       max_tokens, stop_sequences) -> LLMResponse: ...
```

**ABC 的关键用法**：
- `@abstractmethod` 强制子类实现，漏写则实例化时报 `TypeError`
- 非抽象方法可以有默认实现（见 `BaseLLM.count_tokens()`），子类可选覆盖
- `stream()` 在 `BaseAgent` 中有默认实现（调 `run()` 后一次性输出），`ReActAgent` 覆盖为真正流式

```python
# nexus/agent/base.py — 默认实现，不强制子类写流式
class BaseAgent(ABC):
    @abstractmethod
    async def run(self, user_input: str, **kwargs) -> AgentResult: ...

    async def stream(self, user_input: str, **kwargs) -> AsyncIterator[str]:
        result = await self.run(user_input, **kwargs)
        yield result.content
```

### 1.3 依赖方向：上层依赖下层（接口），下层不感知上层

```
orchestrator/ → agent/ → tools/ → core/
                         → llm/   → core/
                         → memory/ → core/
```

每层的构造器接收接口类型（如 `BaseLLM`），而非具体实现。换 Provider 不改 Agent，加 Tool 不改 Memory。

## 2. 类型注解

### 2.1 TypedDict：轻量级消息结构

`Message` 和 `ToolCall` 是框架中最频繁传递的数据结构，使用 `TypedDict` 而非 `dataclass` 或 Pydantic：

```python
# nexus/core/types.py
from typing import Literal, TypedDict

class ToolCall(TypedDict):
    id: str
    name: str
    arguments: dict[str, object]

class Message(TypedDict, total=False):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None
    tool_calls: list[ToolCall] | None
    tool_call_id: str | None
    name: str | None
```

**设计要点**：
- `total=False`：所有字段可选，构建消息时只写需要的字段
- `Literal["system", "user", "assistant", "tool"]`：静态检查约束 role 值
- 选择 TypedDict 而非 dataclass 的原因：消息需要 `msg["role"]` 字典式访问，与 OpenAI/Anthropic 原生格式直接对应

### 2.2 dataclass：数据传输对象

```python
from dataclasses import dataclass, field

@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: str = "stop"
    model: str = ""
```

- 每个字段有默认值，构造时按需赋值
- `field(default_factory=TokenUsage)` 避免可变默认值的经典陷阱

### 2.3 `type[BaseLLM]`：泛型类引用

```python
# nexus/llm/registry.py
class LLMRegistry:
    _providers: dict[str, type[BaseLLM]] = {}
    #                      ^^^^^^^^^^^^^^
    #                      存储的是"类对象"而非"类实例"
```

`type[BaseLLM]` 表示"BaseLLM 或其子类的类对象"，运行时通过 `provider_cls(**kwargs)` 实例化。

## 3. 异步编程

### 3.1 全链路异步：async/await 贯穿 5 层

框架中所有 I/O 操作均异步优先。从 LLM API 调用、Tool 执行到 Agent 循环、Orchestrator 编排，全套链路都使用 `async`/`await`：

```python
# nexus/agent/react_agent.py
async def run(self, user_input: str, **kwargs) -> AgentResult:
    messages = await self._build_initial_messages(user_input)
    while self._iteration < self._max_iterations:
        response = await self._llm.generate(messages=messages, tools=tool_schemas)
        if response.tool_calls:
            await self._act(response.tool_calls, messages)
            continue
        else:
            return AgentResult(content=response.content or "", ...)
```

### 3.2 AsyncIterator：流式输出

LLM 流式响应使用 `AsyncIterator`，消费方用 `async for` 逐块接收，内存友好：

```python
# nexus/llm/base.py — 生成端
async def generate_stream(self, messages, tools, ...) -> AsyncIterator[str | ToolCall]:
    ...

# nexus/agent/react_agent.py — 消费端
async for chunk in self._llm.generate_stream(messages=messages, ...):
    if isinstance(chunk, str):
        yield chunk
```

`AsyncIterator[str | ToolCall]` 的联合类型表示一次流中既可能产出文本片段，也可能产出完整的 ToolCall。

### 3.3 兼容同步与异步函数

`Tool.execute()` 自动判断被包装函数是同步还是协程：

```python
# nexus/tools/base.py
async def execute(self, **kwargs: Any) -> str:
    result = self._func(**kwargs)
    if hasattr(result, "__await__"):   # 是协程 → await
        result = await result
    return str(result)
```

这样同一个 `@Tool.from_function` 装饰器可以同时支持 `def` 和 `async def`。

### 3.4 asyncio.Queue 合并多流

当多个 Agent 并行执行需要实时合并输出时，用 `asyncio.Queue`：

```python
# nexus/utils/streaming.py
async def merge_streams(*streams: AsyncIterator) -> AsyncIterator:
    queue: asyncio.Queue = asyncio.Queue()

    async def _feed(stream):
        async for item in stream:
            await queue.put(item)
        await queue.put(None)  # 哨兵标记流结束

    tasks = [asyncio.create_task(_feed(s)) for s in streams]
    # 收集所有流直到全部结束
    ...
```

## 4. 装饰器模式

### 4.1 `@Tool.from_function`：自动推导 JSON Schema

框架中最精巧的设计之一。一个普通 Python 函数加上类型注解即可变成 LLM 可调用的 Tool：

```python
# 用法
@Tool.from_function(name="calculator", description="执行数学运算")
async def calculator(expression: str) -> str:
    return str(eval(expression))
```

实现原理——带参数的装饰器工厂：

```python
# nexus/tools/base.py
class Tool(BaseTool):
    @staticmethod
    def from_function(name: str, description: str) -> Callable:
        def decorator(func: Callable) -> Tool:
            schema = build_schema(func, name, description)
            return Tool(name=name, description=description, func=func, schema=schema)
        return decorator
```

`build_schema` 内部流程：

```python
# nexus/tools/schema.py
def build_schema(func, name, description):
    sig = inspect.signature(func)          # 提取参数名和默认值
    hints = typing.get_type_hints(func)     # 提取类型注解
    properties = {}
    required = []
    for param_name, param in sig.parameters.items():
        py_type = hints.get(param_name, str)
        properties[param_name] = _type_to_json_schema(py_type)
        if param.default is inspect.Parameter.empty:
            required.append(param_name)     # 无默认值 → required
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required}}
```

**信息唯一来源原则**：类型注解 `expression: str` 既用于 Python 类型检查，又用于 JSON Schema 生成——改一处，两处生效，不会出现"改了参数忘记改 Schema"的问题。

## 5. 依赖注入

### 5.1 构造器注入：显式、可测试、灵活

```python
# nexus/agent/react_agent.py
class ReActAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        llm: BaseLLM,                         # 注入接口类型
        tools: ToolRegistry | None = None,
        memory: BaseMemory | None = None,
        *,
        system_prompt: str = "...",
        max_iterations: int = 10,
        pre_model_hook: Callable | None = None,
        post_model_hook: Callable | None = None,
    ):
```

**设计要点**：
- `*` 强制其后的所有参数为 keyword-only，避免 `ReActAgent("助手", llm, tools, "prompt")` 这种容易搞错顺序的调用
- 位置参数只保留必须的 `name` 和 `llm`
- Hook 函数类型精确到入参出参：`Callable[[list[Message]], list[Message]]`
- Agent 不创建任何组件，全部通过构造器注入——这使测试时注入 MockLLM 成为可能

### 5.2 测试中的注入：结构子类型（不强制继承）

`MockLLM` **不继承** `BaseLLM`，只实现相同的 `async def generate(...)` 方法签名：

```python
# tests/test_agent.py
class MockLLM:
    def __init__(self, responses: list[LLMResponse] | None = None):
        self.responses = responses or []
        self.calls: list[dict] = []

    async def generate(self, messages, tools=None, **kwargs):
        self.calls.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        if self.responses:
            return self.responses.pop(0)
        return LLMResponse(content="Mock 回复", finish_reason="stop")
```

这利用了 Python 的**鸭子类型**——只要对象有相同的方法签名，就能在运行时替换。`self.responses.pop(0)` 实现预设响应队列，`self.calls` 记录所有调用供测试断言。

## 6. 注册表模式

### 6.1 LLMRegistry：类级单例

全局只需一套 Provider 注册，所有方法都是 `@classmethod`，操作模块级 `_providers` 字典：

```python
# nexus/llm/registry.py
class LLMRegistry:
    _providers: dict[str, type[BaseLLM]] = {}

    @classmethod
    def register(cls, name: str, provider_cls: type[BaseLLM]) -> None:
        cls._providers[name] = provider_cls

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseLLM:
        return cls._providers[name](**kwargs)

    @classmethod
    def clear(cls) -> None:
        cls._providers.clear()  # 专为测试设计
```

### 6.2 ToolRegistry：实例级注册

每个 Agent 可以有独立工具集，所以用实例方法：

```python
# nexus/tools/registry.py
class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.schema["name"]] = tool

    def __contains__(self, name: str) -> bool:   # 支持 "name" in registry
        return name in self._tools
```

**类级 vs 实例级的选择**：
- 全局唯一 → 类级（LLMRegistry）
- 每个 Agent 不同 → 实例级（ToolRegistry、BufferMemory）
- 这个区分本身就是重要的架构决策

## 7. 工厂模式

### 7.1 `create_llm()`：封装复杂的参数决策逻辑

```python
# nexus/llm/factory.py
def create_llm(
    provider: str | None = None,
    *,
    config: NexusConfig | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    default_model: str | None = None,
    **extra_kwargs,
) -> BaseLLM:
```

**设计要点**：
- **延迟导入**：内置 Provider 类在首次调用时 `import`，避免启动时就加载所有依赖
- **幂等注册**：`_builtin_registered` 标志防止重复注册
- **三层优先级合并**：代码参数 > 配置文件 > 默认值
- **用 `x is None` 而非 `or`**：空字符串 `""`（如 base_url）是合法值，不能短路掉

```python
# 用 is None 检查，保留空字符串
if final_api_key is None:
    final_api_key = provider_cfg.api_key
# 而不是 final_api_key = api_key or provider_cfg.api_key
```

## 8. Pydantic 配置模型

### 8.1 BaseModel + Field + @classmethod

```python
# nexus/core/config.py
class ProviderConfig(BaseModel):
    api_key: str = Field(default="", description="API Key")
    base_url: str = Field(default="", description="Base URL")
    default_model: str = Field(default="", description="默认模型")

class NexusConfig(BaseModel):
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    agent: AgentConfig = Field(default_factory=AgentConfig)
```

- `Field(default_factory=dict)` 而非 `Field(default={})`——Pydantic 禁止可变默认值
- `@classmethod` 实现多路加载：`from_env()` / `from_yaml(path)` / `from_file()`

### 8.2 环境变量模板替换

YAML 配置文件中的 `${VAR}` 被自动替换为环境变量值：

```python
_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")

def _resolve_string_env(value: str) -> str:
    def _replace(m):
        return os.getenv(m.group(1), m.group(0))  # 未匹配则保留原样
    return _ENV_VAR_RE.sub(_replace, value)
```

递归遍历嵌套 dict，支持任意深度的配置引用。

## 9. 测试

### 9.1 pytest-asyncio：自动异步模式

```toml
# pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

`auto` 模式下，任何 `async def test_*` 函数自动按异步执行，无需手动加 `@pytest.mark.asyncio`。

### 9.2 Fixture：共享测试配置

```python
# tests/conftest.py
@pytest.fixture
def config():
    return NexusConfig(
        default_provider="openai",
        providers={
            "openai": ProviderConfig(api_key="test-key", default_model="gpt-4o"),
        },
        agent=AgentConfig(max_iterations=10),
    )
```

函数名即 fixture 名，测试函数参数名匹配即可自动注入。

### 9.3 测试覆盖模式

| 测试类型 | 示例 | 验证内容 |
|----------|------|----------|
| 简单对话 | `test_react_agent_simple_response` | 无工具时直接返回 |
| 工具调用链 | `test_react_agent_with_tool` | Think→Act→Respond 完整流程 |
| 异常路径 | `test_react_agent_max_iterations` | 超时抛出 MaxIterationError |
| 消息顺序 | `test_to_openai_messages_wrong_order_raises` | 错误顺序自动检测 |
| 跨轮记忆 | `test_agent_memory_across_turns` | 第 2 轮能读取第 1 轮信息 |
| 工具记忆 | `test_agent_writes_tool_calls_to_memory` | tool_call 完整写入 |

### 9.4 注册表清理

`LLMRegistry.clear()` 专为测试间隔离设计——测试修改了注册表，下一个测试不受影响。

## 10. 完整技术栈

| 技术 | 用途 |
|------|------|
| Python 3.11+ | 基础语言 |
| Pydantic 2.0+ | 配置模型、JSON Schema 生成 |
| openai SDK | OpenAI / 兼容 API 客户端 |
| anthropic SDK | Anthropic API 客户端 |
| PyYAML | 配置文件解析 |
| ChromaDB | 向量存储（长期记忆） |
| structlog | 结构化日志 |
| pytest + pytest-asyncio | 测试框架 |
| ruff | 代码检查与格式化 |
