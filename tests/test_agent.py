"""Agent 核心测试。

注意：这些测试使用 mock LLM，不需要真实的 API Key。
"""

import pytest

from nexus.agent.base import BaseAgent
from nexus.agent.react_agent import ReActAgent
from nexus.core.types import AgentResult, LLMResponse
from nexus.llm.messages import to_openai_messages
from nexus.tools.registry import ToolRegistry
from nexus.tools.base import Tool


# ── Mock LLM ────────────────────────────────

class MockLLM:
    """模拟 LLM Provider，返回预设的回复。"""

    def __init__(self, responses: list[LLMResponse] | None = None):
        self.responses = responses or []
        self.calls: list[dict] = []
        self.provider_name = "mock"
        self.default_model = "mock-model"
        self.supported_models = ["mock-model"]

    async def generate(self, messages, tools=None, **kwargs):
        self.calls.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        if self.responses:
            return self.responses.pop(0)
        return LLMResponse(content="Mock 回复", finish_reason="stop")

    async def generate_stream(self, messages, tools=None, **kwargs):
        response = await self.generate(messages, tools, **kwargs)
        if response.content:
            yield response.content

    async def count_tokens(self, messages):
        return sum(len(m.get("content", "") or "") for m in messages) // 4

    async def supports_tool_calling(self):
        return True


# ── BaseAgent 测试 ──────────────────────────

class SimpleAgent(BaseAgent):
    """最简 Agent 实现，直接返回输入。"""

    async def run(self, user_input: str, **kwargs) -> AgentResult:
        return AgentResult(content=f"收到: {user_input}", iterations=1)


@pytest.mark.asyncio
async def test_base_agent():
    agent = SimpleAgent()
    result = await agent.run("你好")
    assert result.content == "收到: 你好"
    assert result.iterations == 1


# ── ReActAgent 测试 ─────────────────────────

@pytest.mark.asyncio
async def test_react_agent_simple_response():
    """测试 ReActAgent：LLM 不调用工具，直接回复。"""
    llm = MockLLM(responses=[
        LLMResponse(content="你好！有什么可以帮助你的？", finish_reason="stop"),
    ])

    agent = ReActAgent(
        name="测试助手",
        llm=llm,
        system_prompt="你是一个测试助手。",
    )

    result = await agent.run("你好")
    assert "你好" in result.content
    assert result.iterations == 1
    assert len(result.tool_calls_history) == 0


@pytest.mark.asyncio
async def test_react_agent_with_tool():
    """测试 ReActAgent：LLM 调用工具，然后生成最终回复。"""
    @Tool.from_function(name="get_time", description="获取当前时间")
    async def get_time() -> str:
        return "2026-05-29 16:00:00"

    tools = ToolRegistry()
    tools.register(get_time)

    llm = MockLLM(responses=[
        # 第一轮：要求调用工具
        LLMResponse(
            content=None,
            tool_calls=[{"id": "call_1", "name": "get_time", "arguments": {}}],
            finish_reason="tool_calls",
        ),
        # 第二轮：收到工具结果后给出最终回复
        LLMResponse(content="现在是 2026年5月29日 16:00", finish_reason="stop"),
    ])

    agent = ReActAgent(
        name="时间助手",
        llm=llm,
        tools=tools,
        system_prompt="你是一个时间助手。",
    )

    result = await agent.run("现在几点了？")
    assert result.iterations == 2
    assert len(result.tool_calls_history) == 1
    assert result.tool_calls_history[0].tool_name == "get_time"


@pytest.mark.asyncio
async def test_react_agent_max_iterations():
    """测试 ReActAgent：超过最大循环次数应抛出异常。"""
    @Tool.from_function(name="loop", description="循环工具")
    async def loop() -> str:
        return "again"

    tools = ToolRegistry()
    tools.register(loop)

    # LLM 始终返回 tool_calls，永远不会给出最终回复
    llm = MockLLM(responses=[
        LLMResponse(
            content=None,
            tool_calls=[{"id": f"call_{i}", "name": "loop", "arguments": {}}],
            finish_reason="tool_calls",
        )
        for i in range(15)
    ])

    agent = ReActAgent(
        name="循环助手",
        llm=llm,
        tools=tools,
        max_iterations=3,
    )

    with pytest.raises(Exception) as exc_info:
        await agent.run("测试循环")
    assert "最大循环次数" in str(exc_info.value) or "3" in str(exc_info.value)


# ── 消息顺序校验测试 ──────────────────────

def test_to_openai_messages_correct_order():
    """正确顺序: assistant(tool_calls) → tool，应通过。"""
    messages = [
        {"role": "system", "content": "你是助手。"},
        {"role": "user", "content": "帮我查时间。"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "name": "get_time", "arguments": {}}
        ]},
        {"role": "tool", "content": "16:00", "tool_call_id": "call_1"},
    ]
    result = to_openai_messages(messages)
    assert len(result) == 4


def test_to_openai_messages_wrong_order_raises():
    """错误顺序: tool 消息在 assistant(tool_calls) 之前，应抛出 ValueError。"""
    messages = [
        {"role": "system", "content": "你是助手。"},
        {"role": "user", "content": "帮我查时间。"},
        # assistant 消息在 tool 之后 — 这是修复前的 bug
        {"role": "tool", "content": "16:00", "tool_call_id": "call_1"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "name": "get_time", "arguments": {}}
        ]},
    ]
    with pytest.raises(ValueError, match="消息顺序错误"):
        to_openai_messages(messages)


def test_to_openai_messages_multiple_tool_results():
    """一个 assistant 带多个 tool_calls，后面连续多个 tool 结果，应通过。"""
    messages = [
        {"role": "system", "content": "你是助手。"},
        {"role": "user", "content": "帮我查时间和日期。"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "name": "get_time", "arguments": {}},
            {"id": "call_2", "name": "get_date", "arguments": {}},
        ]},
        {"role": "tool", "content": "16:00", "tool_call_id": "call_1"},
        {"role": "tool", "content": "2026-05-29", "tool_call_id": "call_2"},
    ]
    result = to_openai_messages(messages)
    assert len(result) == 5


def test_to_openai_messages_orphan_tool_raises():
    """没有 assistant(tool_calls) 但有 tool 消息，应抛出 ValueError。"""
    messages = [
        {"role": "system", "content": "你是助手。"},
        {"role": "user", "content": "帮我查时间。"},
        {"role": "tool", "content": "16:00", "tool_call_id": "call_1"},
    ]
    with pytest.raises(ValueError, match="消息顺序错误"):
        to_openai_messages(messages)


# ── Memory 写入测试 ──────────────────────

@pytest.mark.asyncio
async def test_agent_writes_conversation_to_memory():
    """测试 Agent 将用户输入和助手回复写入 Memory。"""
    from nexus.memory.buffer import BufferMemory

    memory = BufferMemory(max_messages=20)
    llm = MockLLM(responses=[
        LLMResponse(content="你好小明！我记住了。", finish_reason="stop"),
    ])

    agent = ReActAgent(
        name="记忆测试",
        llm=llm,
        memory=memory,
        system_prompt="你是测试助手。",
    )

    await agent.run("我叫小明")

    ctx = await memory.get_context()
    assert len(ctx) == 2
    assert ctx[0]["role"] == "user"
    assert ctx[0]["content"] == "我叫小明"
    assert ctx[1]["role"] == "assistant"
    assert "小明" in ctx[1]["content"]


@pytest.mark.asyncio
async def test_agent_memory_across_turns():
    """测试 Memory 跨轮对话：第 2 轮能获取第 1 轮的消息。"""
    from nexus.memory.buffer import BufferMemory

    memory = BufferMemory(max_messages=20)
    llm = MockLLM(responses=[
        LLMResponse(content="记住了，你叫小明。", finish_reason="stop"),
        LLMResponse(content="小明，你之前说过喜欢 Python。", finish_reason="stop"),
    ])

    agent = ReActAgent(
        name="记忆测试",
        llm=llm,
        memory=memory,
        system_prompt="你是一个有记忆的助手。",
    )

    # 第 1 轮
    result1 = await agent.run("我叫小明，我喜欢 Python")
    assert "小明" in result1.content

    # 第 2 轮 — memory 应包含上一轮的用户和助手消息
    result2 = await agent.run("我喜欢什么？")
    assert "Python" in result2.content

    # 验证 memory 中有 4 条消息 (user1, assistant1, user2, assistant2)
    ctx = await memory.get_context()
    assert len(ctx) == 4
    assert ctx[0]["role"] == "user"
    assert ctx[0]["content"] == "我叫小明，我喜欢 Python"
    assert ctx[1]["role"] == "assistant"
    assert ctx[2]["role"] == "user"
    assert ctx[2]["content"] == "我喜欢什么？"
    assert ctx[3]["role"] == "assistant"


@pytest.mark.asyncio
async def test_agent_writes_tool_calls_to_memory():
    """测试 Agent 将工具调用和结果写入 Memory。"""
    from nexus.memory.buffer import BufferMemory

    @Tool.from_function(name="get_time", description="获取当前时间")
    async def get_time() -> str:
        return "16:00"

    tools = ToolRegistry()
    tools.register(get_time)

    memory = BufferMemory(max_messages=20)
    llm = MockLLM(responses=[
        LLMResponse(
            content=None,
            tool_calls=[{"id": "call_1", "name": "get_time", "arguments": {}}],
            finish_reason="tool_calls",
        ),
        LLMResponse(content="现在是16:00。", finish_reason="stop"),
    ])

    agent = ReActAgent(
        name="工具记忆测试",
        llm=llm,
        tools=tools,
        memory=memory,
        system_prompt="你是时间助手。",
    )

    await agent.run("现在几点了？")

    ctx = await memory.get_context()
    # 应有 4 条: user, assistant(tool_calls), tool(result), assistant(final)
    assert len(ctx) == 4
    assert ctx[0]["role"] == "user"
    assert ctx[1]["role"] == "assistant"
    assert ctx[1].get("tool_calls") is not None
    assert ctx[2]["role"] == "tool"
    assert ctx[2].get("tool_call_id") == "call_1"
    assert ctx[3]["role"] == "assistant"
    assert "16:00" in ctx[3]["content"]
