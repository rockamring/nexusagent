"""OpenAI Provider — 封装 OpenAI Chat Completions API。

支持所有兼容 OpenAI API 格式的服务（包括本地 ollama、vLLM 等）。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from nexus.core.types import LLMResponse, Message, TokenUsage, ToolCall
from nexus.llm.base import BaseLLM
from nexus.llm.messages import to_openai_messages, tools_to_openai_format


class OpenAIProvider(BaseLLM):
    """OpenAI LLM Provider。

    用法:
        llm = OpenAIProvider(api_key="sk-...", default_model="gpt-4o")
        response = await llm.generate(messages, tools=[...])

    支持:
    - 所有 OpenAI Chat Completions 模型 (gpt-4o, gpt-4o-mini, o3-mini 等)
    - 任何兼容 OpenAI API 的服务 (通过 base_url 配置)
    - 原生 Tool Calling (Function Calling)
    """

    def __init__(
        self,
        api_key: str,
        default_model: str = "gpt-4o",
        base_url: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        super().__init__(max_retries=max_retries, retry_delay=retry_delay)
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**client_kwargs)
        self._default_model = default_model

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def supported_models(self) -> list[str]:
        return [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "o3",
            "o3-mini",
            "o4-mini",
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
        api_messages = to_openai_messages(messages)
        api_tools = tools_to_openai_format(tools) if tools else None

        kwargs: dict = {
            "model": model or self._default_model,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if api_tools:
            kwargs["tools"] = api_tools
        if stop_sequences:
            kwargs["stop"] = stop_sequences

        response = await self._retry_call(
            lambda: self._client.chat.completions.create(**kwargs),
            "OpenAI API",
        )
        choice = response.choices[0]

        tool_calls = None
        content = None

        if choice.message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=_parse_json(tc.function.arguments),
                )
                for tc in choice.message.tool_calls
            ]
        else:
            content = choice.message.content or ""

        usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
        )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=choice.finish_reason or "stop",
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
        api_messages = to_openai_messages(messages)
        api_tools = tools_to_openai_format(tools) if tools else None

        kwargs: dict = {
            "model": model or self._default_model,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if api_tools:
            kwargs["tools"] = api_tools

        stream = await self._retry_call(
            lambda: self._client.chat.completions.create(**kwargs),
            "OpenAI 流式 API",
        )

        # 流式处理中累积 tool_call 信息
        tool_call_buffers: dict[int, dict] = {}

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            if delta.content:
                yield delta.content

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_call_buffers:
                        tool_call_buffers[idx] = {"id": "", "name": "", "arguments": ""}
                    buf = tool_call_buffers[idx]
                    if tc_delta.id:
                        buf["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            buf["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            buf["arguments"] += tc_delta.function.arguments

        # 流结束后产出完整的 ToolCall
        for buf in tool_call_buffers.values():
            if buf["name"]:
                yield ToolCall(
                    id=buf["id"],
                    name=buf["name"],
                    arguments=_parse_json(buf["arguments"]) if buf["arguments"] else {},
                )


def _parse_json(text: str) -> dict:
    import json
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_raw": text}
