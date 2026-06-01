"""BaseLLM — LLM Provider 的统一抽象接口。

所有 LLM Provider (OpenAI, Anthropic, Ollama 等) 必须实现此接口。

设计理念：
- 异步优先：LLM 调用和工具执行都是 I/O 密集，async/await 是刚需
- 统一消息模型：使用 nexus.core.types.Message，各 Provider 内部转换
- Tool Calling 支持：原生 Tool Calling (OpenAI/Anthropic 均支持)
- 内置重试：指数退避重试，应对网络抖动和速率限制
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from nexus.core.types import LLMResponse, Message, ToolCall

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """LLM Provider 的统一抽象接口。

    用法:
        llm = OpenAIProvider(api_key="sk-...", default_model="gpt-4o")
        response = await llm.generate(messages, tools=[...])

    关键设计点:
    - generate() 和 generate_stream() 是核心能力
    - count_tokens() 用于上下文管理，Agent 用它判断是否需要裁剪
    - tools 参数接收 JSON Schema 格式的 dict 列表，与 OpenAI 兼容
    - 内置指数退避重试：自动重试网络错误、速率限制和服务端错误
    """

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # ── 元数据 ─────────────────────────────────

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """返回 Provider 名称，如 'openai', 'anthropic'。"""
        ...

    @property
    @abstractmethod
    def default_model(self) -> str:
        """返回该 Provider 的默认模型名称。"""
        ...

    @property
    @abstractmethod
    def supported_models(self) -> list[str]:
        """返回该 Provider 支持的模型列表（非穷举，列举主要模型即可）。"""
        ...

    # ── 重试机制 ─────────────────────────────────

    def _is_retryable_error(self, error: Exception) -> bool:
        """判断异常是否可重试（网络错误、速率限制、服务端错误）。

        Provider 可 override 此方法以添加 SDK 特有的错误类型。
        """
        status = getattr(error, "status_code", None)
        if status is not None:
            if status == 429:
                return True
            if status >= 500:
                return True
            if status in (401, 403):
                return False

        name = type(error).__name__
        retryable_patterns = ("ConnectionError", "Timeout", "RateLimit", "InternalServer")
        for pattern in retryable_patterns:
            if pattern in name:
                return True

        if isinstance(error, (ConnectionError, TimeoutError, OSError)):
            return True

        return False

    async def _retry_call(self, call_fn, operation_name: str = "LLM 调用"):
        """带指数退避的重试包装器。

        Args:
            call_fn: 无参异步可调用对象，返回 API 响应
            operation_name: 操作名称，用于日志

        Returns:
            call_fn 的返回值

        Raises:
            最后一次重试的异常（如果不可重试或超过最大次数）
        """
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return await call_fn()
            except Exception as exc:
                last_error = exc
                if not self._is_retryable_error(exc) or attempt >= self.max_retries:
                    raise
                delay = self.retry_delay * (2 ** attempt)
                logger.warning(
                    "%s 失败 (第 %d/%d 次): %s，%.1fs 后重试...",
                    operation_name, attempt + 1, self.max_retries + 1, exc, delay,
                )
                await asyncio.sleep(delay)
        raise last_error  # type: ignore[return]

    # ── 核心生成能力 ─────────────────────────────

    @abstractmethod
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
        """生成回复（非流式），支持 Tool Calling。

        Args:
            messages: 对话历史，使用框架统一的 Message 格式
            tools: 工具定义列表（JSON Schema 格式，OpenAI 兼容）
            model: 指定模型，None 则使用 default_model
            temperature: 采样温度，0-2
            max_tokens: 最大输出 token 数
            stop_sequences: 停止序列

        Returns:
            LLMResponse: 包含 content 或 tool_calls，两者不共存
        """
        ...

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str | ToolCall]:
        """流式生成回复。

        逐步产出文本块(str)或工具调用(ToolCall)。
        Agent 的 stream() 方法基于此实现实时输出。

        Yields:
            str: 文本增量（流式输出的每个 chunk）
            ToolCall: 完整的工具调用对象
        """
        ...

    # ── 辅助能力 ─────────────────────────────────

    async def count_tokens(self, messages: list[Message]) -> int:
        """估算消息列表的 token 数量。

        默认使用字符数粗略估算（4 字符 ≈ 1 token）。
        Provider 可 override 此方法使用官方 tokenizer 获得精确计数。
        """
        total = 0
        for msg in messages:
            content = msg.get("content")
            if content:
                total += len(content) // 4
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    args_str = str(tc.get("arguments", {}))
                    total += len(tc["name"]) // 4 + len(args_str) // 4
        return max(total, 1)

    async def supports_tool_calling(self) -> bool:
        """当前模型是否支持原生 Tool Calling。

        默认 True，不支持 Tool Calling 的模型应 override 返回 False。
        Agent 会据此决定是否启用 Tool Calling 逻辑。
        """
        return True
