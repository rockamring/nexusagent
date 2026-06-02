"""Gemini Provider — 封装 Google Gemini API (google-genai SDK)。

Gemini API 与 OpenAI/Anthropic 的关键差异：
- system 是 GenerateContentConfig 中的 system_instruction 参数
- assistant 角色映射为 "model"
- tool 角色通过 function_response 映射为 "user"
- Tool Calling 使用 function_declarations 格式
- 需要显式禁用 automatic_function_calling（否则 SDK 会自动执行工具）
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from google import genai
from google.genai import types

from nexus.core.types import LLMResponse, Message, TokenUsage, ToolCall
from nexus.llm.base import BaseLLM
from nexus.llm.messages import to_gemini_messages, tools_to_gemini_format


class GeminiProvider(BaseLLM):
    """Google Gemini LLM Provider。

    用法:
        llm = GeminiProvider(api_key="...", default_model="gemini-2.5-flash")
        response = await llm.generate(messages, tools=[...])

    支持:
    - Gemini 2.5 Pro / Flash, Gemini 2.0 Flash
    - 原生 Tool Calling (Function Calling)
    - 异步生成和流式输出
    """

    def __init__(
        self,
        api_key: str,
        default_model: str = "gemini-2.5-flash",
        base_url: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        super().__init__(max_retries=max_retries, retry_delay=retry_delay)
        client_kwargs: dict = {"api_key": api_key}
        if base_url:
            client_kwargs["http_options"] = types.HttpOptions(base_url=base_url)
        self._client = genai.Client(**client_kwargs)
        self._default_model = default_model

    @property
    def provider_name(self) -> str:
        return "google"

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def supported_models(self) -> list[str]:
        return [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
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
        system_instruction, contents = to_gemini_messages(messages)
        api_tools = tools_to_gemini_format(tools) if tools else None

        config_kwargs: dict = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if api_tools:
            config_kwargs["tools"] = [types.Tool(**t) for t in api_tools]
        if stop_sequences:
            config_kwargs["stop_sequences"] = stop_sequences

        config = types.GenerateContentConfig(**config_kwargs)

        response = await self._retry_call(
            lambda: self._client.aio.models.generate_content(
                model=model or self._default_model,
                contents=contents,
                config=config,
            ),
            "Gemini API",
        )

        return self._parse_response(response, model or self._default_model)

    async def generate_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str | ToolCall]:
        system_instruction, contents = to_gemini_messages(messages)
        api_tools = tools_to_gemini_format(tools) if tools else None

        config_kwargs: dict = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if api_tools:
            config_kwargs["tools"] = [types.Tool(**t) for t in api_tools]

        config = types.GenerateContentConfig(**config_kwargs)

        stream = await self._retry_call(
            lambda: self._client.aio.models.generate_content_stream(
                model=model or self._default_model,
                contents=contents,
                config=config,
            ),
            "Gemini 流式 API",
        )

        async for chunk in stream:
            if chunk.candidates and chunk.candidates[0].content:
                for part in chunk.candidates[0].content.parts:
                    if part.text:
                        yield part.text
                    elif part.function_call:
                        fc = part.function_call
                        yield ToolCall(
                            id=fc.id or "",
                            name=fc.name or "",
                            arguments=fc.args or {},
                        )

    async def count_tokens(self, messages: list[Message]) -> int:
        """使用 Gemini 官方 API 精确计数 token。"""
        try:
            system_instruction, contents = to_gemini_messages(messages)
            response = await self._client.aio.models.count_tokens(
                model=self._default_model,
                contents=contents,
            )
            return response.total_tokens or 0
        except Exception:
            return await super().count_tokens(messages)

    # ── 内部方法 ─────────────────────────────────

    def _is_retryable_error(self, error: Exception) -> bool:
        """Gemini 特定的重试判断，额外匹配 Google API 错误类型。"""
        if super()._is_retryable_error(error):
            return True

        name = type(error).__name__
        gemini_retryable = ("ResourceExhausted", "ServerError", "ServiceUnavailable", "InternalServerError")
        for pattern in gemini_retryable:
            if pattern in name:
                return True

        gemini_non_retryable = ("PermissionDenied", "Unauthenticated", "InvalidArgument", "NotFound")
        for pattern in gemini_non_retryable:
            if pattern in name:
                return False

        return False

    @staticmethod
    def _parse_response(response, model_name: str) -> LLMResponse:
        """解析 Gemini API 响应为框架统一的 LLMResponse。"""
        candidate = response.candidates[0] if response.candidates else None

        if candidate is None or not candidate.content:
            return LLMResponse(
                content="",
                usage=_extract_usage(response),
                finish_reason=_map_finish_reason(candidate and candidate.finish_reason),
                model=model_name,
            )

        texts: list[str] = []
        tc_list: list[ToolCall] = []

        for part in candidate.content.parts:
            if part.text:
                texts.append(part.text)
            elif part.function_call:
                fc = part.function_call
                tc_list.append(ToolCall(
                    id=fc.id or "",
                    name=fc.name or "",
                    arguments=fc.args or {},
                ))

        if tc_list:
            return LLMResponse(
                tool_calls=tc_list,
                usage=_extract_usage(response),
                finish_reason=_map_finish_reason(candidate.finish_reason),
                model=model_name,
            )

        return LLMResponse(
            content="\n".join(texts),
            usage=_extract_usage(response),
            finish_reason=_map_finish_reason(candidate.finish_reason),
            model=model_name,
        )


def _extract_usage(response) -> TokenUsage:
    """从 Gemini usage_metadata 提取 TokenUsage。"""
    um = response.usage_metadata
    if um is None:
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=um.prompt_token_count or 0,
        completion_tokens=um.candidates_token_count or 0,
        total_tokens=um.total_token_count or 0,
    )


def _map_finish_reason(reason) -> str:
    """将 Gemini FinishReason 枚举映射为字符串。"""
    if reason is None:
        return "stop"
    mapping = {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "safety",
        "RECITATION": "content_filter",
    }
    reason_name = reason.name if hasattr(reason, "name") else str(reason)
    return mapping.get(reason_name, "stop")
