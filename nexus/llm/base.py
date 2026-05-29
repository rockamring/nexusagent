"""BaseLLM — LLM Provider 的统一抽象接口。

所有 LLM Provider (OpenAI, Anthropic, Ollama 等) 必须实现此接口。

设计理念：
- 异步优先：LLM 调用和工具执行都是 I/O 密集，async/await 是刚需
- 统一消息模型：使用 nexus.core.types.Message，各 Provider 内部转换
- Tool Calling 支持：原生 Tool Calling (OpenAI/Anthropic 均支持)
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from nexus.core.types import LLMResponse, Message, ToolCall


class BaseLLM(ABC):
    """LLM Provider 的统一抽象接口。

    用法:
        llm = OpenAIProvider(api_key="sk-...", default_model="gpt-4o")
        response = await llm.generate(messages, tools=[...])

    关键设计点:
    - generate() 和 generate_stream() 是核心能力
    - count_tokens() 用于上下文管理，Agent 用它判断是否需要裁剪
    - tools 参数接收 JSON Schema 格式的 dict 列表，与 OpenAI 兼容
    """

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
