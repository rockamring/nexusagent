"""LLM Provider 测试 — 消息转换和 Provider 行为。

使用 mock 避免真实 API 调用。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus.core.types import ToolCall
from nexus.llm.messages import to_gemini_messages, tools_to_gemini_format

# ── 消息转换测试 ──────────────────────────────

class TestToGeminiMessages:
    """to_gemini_messages() 转换逻辑测试。"""

    def test_simple_user_message(self):
        system, contents = to_gemini_messages([
            {"role": "user", "content": "你好"},
        ])
        assert system is None
        assert len(contents) == 1
        assert contents[0]["role"] == "user"
        assert contents[0]["parts"] == [{"text": "你好"}]

    def test_system_prompt_extraction(self):
        system, contents = to_gemini_messages([
            {"role": "system", "content": "你是一个助手。"},
            {"role": "user", "content": "问题"},
        ])
        assert system == "你是一个助手。"
        assert len(contents) == 1
        assert contents[0]["role"] == "user"

    def test_multiple_system_messages_last_wins(self):
        system, contents = to_gemini_messages([
            {"role": "system", "content": "第一条"},
            {"role": "system", "content": "第二条"},
            {"role": "user", "content": "问题"},
        ])
        assert system == "第二条"
        assert len(contents) == 1

    def test_assistant_text_response(self):
        system, contents = to_gemini_messages([
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": "回答"},
        ])
        assert len(contents) == 2
        assert contents[1]["role"] == "model"
        assert contents[1]["parts"] == [{"text": "回答"}]

    def test_assistant_with_tool_calls(self):
        system, contents = to_gemini_messages([
            {"role": "user", "content": "查天气"},
            {"role": "assistant", "content": "让我查一下", "tool_calls": [
                ToolCall(id="call_1", name="get_weather", arguments={"city": "北京"}),
            ]},
        ])
        assert len(contents) == 2
        model = contents[1]
        assert model["role"] == "model"
        assert len(model["parts"]) == 2  # text + function_call
        assert model["parts"][0] == {"text": "让我查一下"}
        assert model["parts"][1] == {
            "function_call": {"id": "call_1", "name": "get_weather", "args": {"city": "北京"}},
        }

    def test_tool_result_to_function_response(self):
        system, contents = to_gemini_messages([
            {"role": "user", "content": "查天气"},
            {"role": "assistant", "content": None, "tool_calls": [
                ToolCall(id="call_1", name="get_weather", arguments={"city": "北京"}),
            ]},
            {"role": "tool", "content": "晴天 25°C", "tool_call_id": "call_1", "name": "get_weather"},
        ])
        assert len(contents) == 3
        func = contents[2]
        assert func["role"] == "user"
        fr = func["parts"][0]["function_response"]
        assert fr["id"] == "call_1"
        assert fr["name"] == "get_weather"
        assert fr["response"] == {"result": "晴天 25°C"}


class TestToolsToGeminiFormat:
    """tools_to_gemini_format() 转换测试。"""

    def test_basic_conversion(self):
        result = tools_to_gemini_format([
            {
                "name": "get_weather",
                "description": "获取天气",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        ])
        assert len(result) == 1
        assert "function_declarations" in result[0]
        assert len(result[0]["function_declarations"]) == 1
        fd = result[0]["function_declarations"][0]
        assert fd["name"] == "get_weather"
        assert fd["description"] == "获取天气"
        assert fd["parameters"]["required"] == ["city"]

    def test_multiple_tools(self):
        result = tools_to_gemini_format([
            {"name": "tool_a", "description": "A", "parameters": {}},
            {"name": "tool_b", "description": "B", "parameters": {}},
        ])
        assert len(result[0]["function_declarations"]) == 2
        assert result[0]["function_declarations"][0]["name"] == "tool_a"
        assert result[0]["function_declarations"][1]["name"] == "tool_b"

    def test_empty_tools(self):
        result = tools_to_gemini_format([])
        assert result == [{"function_declarations": []}]


# ── GeminiProvider 测试 ────────────────────────

class TestGeminiProvider:
    """GeminiProvider 行为测试（mock API 调用）。"""

    @pytest.fixture
    def provider(self):
        from nexus.llm.providers.google import GeminiProvider
        return GeminiProvider(api_key="test-key", default_model="gemini-2.5-flash")

    @pytest.mark.asyncio
    async def test_generate_text_response(self, provider):
        """测试普通文本回复的解析。"""
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content = MagicMock()
        mock_response.candidates[0].finish_reason = MagicMock()
        mock_response.candidates[0].finish_reason.name = "STOP"

        from google.genai import types
        mock_response.candidates[0].content.parts = [
            types.Part(text="你好，我是 Gemini。"),
        ]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 5
        mock_response.usage_metadata.total_token_count = 15

        with patch.object(provider._client.aio.models, "generate_content",
                          AsyncMock(return_value=mock_response)):
            result = await provider.generate([{"role": "user", "content": "Hello"}])

        assert result.content == "你好，我是 Gemini。"
        assert result.tool_calls is None
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_generate_tool_call_response(self, provider):
        """测试工具调用回复的解析。"""
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content = MagicMock()
        mock_response.candidates[0].finish_reason = MagicMock()
        mock_response.candidates[0].finish_reason.name = "STOP"

        from google.genai import types
        mock_response.candidates[0].content.parts = [
            types.Part(function_call=types.FunctionCall(
                id="call_abc123",
                name="get_weather",
                args={"city": "东京"},
            )),
        ]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 5
        mock_response.usage_metadata.total_token_count = 15

        with patch.object(provider._client.aio.models, "generate_content",
                          AsyncMock(return_value=mock_response)):
            result = await provider.generate([{"role": "user", "content": "查东京天气"}])

        assert result.content is None
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["id"] == "call_abc123"
        assert result.tool_calls[0]["name"] == "get_weather"
        assert result.tool_calls[0]["arguments"] == {"city": "东京"}

    @pytest.mark.asyncio
    async def test_generate_empty_response(self, provider):
        """测试空响应（安全拦截等）。"""
        mock_response = MagicMock()
        mock_response.candidates = []
        mock_response.usage_metadata = None

        with patch.object(provider._client.aio.models, "generate_content",
                          AsyncMock(return_value=mock_response)):
            result = await provider.generate([{"role": "user", "content": "Hello"}])

        assert result.content == ""
        assert result.tool_calls is None
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_generate_stream_text(self, provider):
        """测试流式文本输出。"""
        from google.genai import types

        chunk1 = MagicMock()
        chunk1.candidates = [MagicMock()]
        chunk1.candidates[0].content = MagicMock()
        chunk1.candidates[0].content.parts = [types.Part(text="你好")]

        chunk2 = MagicMock()
        chunk2.candidates = [MagicMock()]
        chunk2.candidates[0].content = MagicMock()
        chunk2.candidates[0].content.parts = [types.Part(text="，世界")]

        async def mock_stream(*args, **kwargs):
            yield chunk1
            yield chunk2

        with patch.object(provider._client.aio.models, "generate_content_stream",
                          AsyncMock(return_value=mock_stream())):
            chunks = []
            async for chunk in provider.generate_stream(
                [{"role": "user", "content": "Hello"}]
            ):
                chunks.append(chunk)

        assert chunks == ["你好", "，世界"]

    @pytest.mark.asyncio
    async def test_generate_stream_tool_call(self, provider):
        """测试流式工具调用输出。"""
        from google.genai import types

        chunk = MagicMock()
        chunk.candidates = [MagicMock()]
        chunk.candidates[0].content = MagicMock()
        chunk.candidates[0].content.parts = [
            types.Part(function_call=types.FunctionCall(
                id="call_x",
                name="search",
                args={"query": "Python"},
            )),
        ]

        async def mock_stream(*args, **kwargs):
            yield chunk

        with patch.object(provider._client.aio.models, "generate_content_stream",
                          AsyncMock(return_value=mock_stream())):
            chunks = []
            async for chunk in provider.generate_stream(
                [{"role": "user", "content": "Search"}]
            ):
                chunks.append(chunk)

        assert len(chunks) == 1
        assert isinstance(chunks[0], dict)
        assert chunks[0]["name"] == "search"
        assert chunks[0]["arguments"] == {"query": "Python"}

    def test_is_retryable_error(self, provider):
        """测试 Gemini 特定的重试判断。"""
        assert provider._is_retryable_error(ConnectionError("timeout"))
        assert provider._is_retryable_error(OSError("network"))

        # 按类名匹配
        class ResourceExhaustedError(Exception):
            pass

        assert provider._is_retryable_error(ResourceExhaustedError())
        assert not provider._is_retryable_error(ValueError("bad input"))

    @pytest.mark.asyncio
    async def test_count_tokens(self, provider):
        """测试 token 计数。"""
        mock_response = MagicMock()
        mock_response.total_tokens = 42

        with patch.object(provider._client.aio.models, "count_tokens",
                          AsyncMock(return_value=mock_response)):
            count = await provider.count_tokens([
                {"role": "user", "content": "Hello"},
            ])

        assert count == 42

    def test_provider_metadata(self, provider):
        """测试 Provider 元数据。"""
        assert provider.provider_name == "google"
        assert provider.default_model == "gemini-2.5-flash"
        assert "gemini-2.5-pro" in provider.supported_models
        assert "gemini-2.5-flash" in provider.supported_models
