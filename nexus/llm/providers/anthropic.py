"""Anthropic Provider — 封装 Anthropic Messages API。

Anthropic API 和 OpenAI API 有关键差异：
- system 是独立的 top-level 参数
- tool_use 和 tool_result 使用 content blocks 而非顶层字段
- 不支持流式 tool_use 增量（tool_use 在 stream 中以完整块出现）
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic

from nexus.core.types import LLMResponse, Message, TokenUsage, ToolCall
from nexus.llm.base import BaseLLM
from nexus.llm.messages import to_anthropic_messages, tools_to_anthropic_format


class AnthropicProvider(BaseLLM):
    """Anthropic LLM Provider。

    用法:
        llm = AnthropicProvider(api_key="sk-ant-...", default_model="claude-sonnet-4-6")
        response = await llm.generate(messages, tools=[...])
    """

    def __init__(
        self,
        api_key: str,
        default_model: str = "claude-sonnet-4-6",
    ):
        self._client = AsyncAnthropic(api_key=api_key)
        self._default_model = default_model

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def supported_models(self) -> list[str]:
        return [
            "claude-opus-4-7",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
        ]

    async def generate(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop_sequences: list[str] | None = None,
    ) -> LLMResponse:
        system_prompt, api_messages = to_anthropic_messages(messages)
        api_tools = tools_to_anthropic_format(tools) if tools else None

        kwargs: dict = {
            "model": model or self._default_model,
            "messages": api_messages,
            "max_tokens": max_tokens,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if api_tools:
            kwargs["tools"] = api_tools
        if stop_sequences:
            kwargs["stop_sequences"] = stop_sequences
        # Anthropic 需要至少 1 条消息，传入 temperature 确保格式正确
        if temperature > 0:
            kwargs["temperature"] = temperature

        response = await self._client.messages.create(**kwargs)

        content = None
        tool_calls = None

        # Anthropic 返回 content blocks，需要从中提取 text 和 tool_use
        texts = []
        tc_list = []
        for block in response.content:
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "tool_use":
                tc_list.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))

        if tc_list:
            tool_calls = tc_list
        else:
            content = "\n".join(texts)

        usage = TokenUsage(
            prompt_tokens=response.usage.input_tokens if response.usage else 0,
            completion_tokens=response.usage.output_tokens if response.usage else 0,
            total_tokens=(
                response.usage.input_tokens + response.usage.output_tokens
                if response.usage else 0
            ),
        )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=response.stop_reason or "end_turn",
            model=response.model,
        )

    async def generate_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str | ToolCall]:
        system_prompt, api_messages = to_anthropic_messages(messages)
        api_tools = tools_to_anthropic_format(tools) if tools else None

        kwargs: dict = {
            "model": model or self._default_model,
            "messages": api_messages,
            "max_tokens": max_tokens,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if api_tools:
            kwargs["tools"] = api_tools
        if temperature > 0:
            kwargs["temperature"] = temperature

        async with self._client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "text":
                    yield event.text
                elif event.type == "content_block_stop":
                    # Anthropic stream 中 tool_use 以完整块出现
                    if hasattr(event, "content_block") and event.content_block.type == "tool_use":
                        block = event.content_block
                        yield ToolCall(
                            id=block.id,
                            name=block.name,
                            arguments=block.input if isinstance(block.input, dict) else {},
                        )

    async def count_tokens(self, messages: list[Message]) -> int:
        """使用 Anthropic 官方 tokenizer 精确计数。"""
        try:
            system_prompt, api_messages = to_anthropic_messages(messages)
            count = await self._client.messages.count_tokens(
                model=self._default_model,
                messages=api_messages,
                system=system_prompt or "",
            )
            return count.input_tokens
        except Exception:
            return await super().count_tokens(messages)
