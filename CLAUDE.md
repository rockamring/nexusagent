# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 常用命令

```bash
# 安装依赖 (Python 3.11+)
pip install -e ".[dev]"

# 运行全部测试
pytest tests/ -v

# 运行单个测试文件
pytest tests/test_agent.py -v

# 运行单个测试函数
pytest tests/test_tools.py::test_build_schema_simple -v

# 代码检查
ruff check nexus/ tests/

# 类型检查
mypy nexus/
```

## 5 层架构

框架按依赖方向从底到顶分为 5 层，每层通过抽象基类定义接口：

```
core/     ← 公共类型 (Message, LLMResponse, AgentResult, ToolCallRecord)
llm/      ← BaseLLM: 统一 LLM 调用接口，消息格式转换
tools/    ← BaseTool / Tool.from_function: 函数→JSON Schema 自动推导
agent/    ← ReActAgent: Think→Act→Observe 主循环（核心引擎）
memory/   ← BaseMemory: add / get_context / clear
orchestrator/ ← BaseOrchestrator: 多 Agent 编排 (Sequential / Supervisor / Graph)
```

层间通过依赖注入连接：`ReActAgent(llm=..., tools=..., memory=...)`，换组件不影响其他层。

## ReAct 循环数据流

`react_agent.py` 是框架最核心的文件。一次完整的 Agent 执行流程：

1. **构建初始消息**: `[System] → [Memory Context] → [User Input]`
2. **Think**: 调用 `llm.generate(messages, tools=schema)` — LLM 返回 `content` 或 `tool_calls`
3. **检查**: `tool_calls` 非空 → 进入 Act；为空 → 进入 Respond
4. **Act**: `ToolRegistry.execute(name, **args)` → 追加 `{"role": "tool", ...}` 到消息历史 → 回到 Think
5. **Respond**: 返回 `AgentResult(content, iterations, tool_calls_history)`

关键机制：
- `_check_loop()`: 检测连续 N 次相同 tool_call，发现死循环时仅打 warning（不中断，靠 max_iterations 兜底）
- `_trim_messages()`: token_budget 超限时，保留 system + 从后往前保留最近消息
- `pre_model_hook` / `post_model_hook`: 扩展点，用于摘要压缩等中间件

## 关键抽象与扩展方式

### 新增 LLM Provider
继承 `BaseLLM`，实现 `generate()`, `generate_stream()`, `count_tokens()`，然后 `LLMRegistry.register("name", ProviderCls)`。
消息格式转换在 `llm/messages.py` 中统一处理。

### 新增 Tool
```python
@Tool.from_function(name="xxx", description="...")
async def xxx(param: str, count: int = 1) -> str: ...
```
装饰器自动从类型注解推导 JSON Schema。`str/int/float/bool/list/dict` 自动映射，带默认值的参数不加入 `required`。

### 新增 Memory
继承 `BaseMemory`，实现 `add()`, `get_context()`, `clear()`。多个 Memory 用 `CompositeMemory` 组合。

### 新增编排模式
继承 `BaseOrchestrator`，实现 `run()` 和可选的 `stream()`。

## 测试约定

- 测试 LLM 调用使用 `MockLLM`（见 `tests/test_agent.py`），预设 `LLMResponse` 列表来模拟多轮对话
- `pytest-asyncio` 的 `asyncio_mode = "auto"`，所有 `async def test_*` 自动适配
- `LLMRegistry._providers` 是模块级共享状态，必要时用 `clear()` 重置

## 代码风格

- 所有注释和文档字符串使用中文
- 代码标识符使用英文，遵循 PEP 8
- 类型注解全面使用（`list[Message]`, `dict[str, Any]`, `X | None`）
- 行宽上限 120 字符（ruff 配置）
