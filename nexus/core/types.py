"""框架内所有模块共享的公共类型定义。"""

from dataclasses import dataclass, field
from typing import Literal, NotRequired, TypedDict


class ToolCall(TypedDict):
    """LLM 返回的工具调用。"""
    id: str
    name: str
    arguments: dict[str, object]


class Message(TypedDict, total=False):
    """统一消息模型，兼容 OpenAI / Anthropic 格式。

    total=False 表示所有字段都是可选的，实际使用时根据 role 决定哪些字段必须存在。
    """
    role: Literal["system", "user", "assistant", "tool"]
    content: NotRequired[str | None]
    tool_calls: NotRequired[list[ToolCall] | None]
    tool_call_id: NotRequired[str | None]
    name: NotRequired[str | None]


@dataclass
class TokenUsage:
    """LLM 调用的 Token 消耗统计。"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResponse:
    """LLM 返回的统一响应。

    一次调用要么有 content（文本回复），
    要么有 tool_calls（要求执行工具），
    两者不会同时存在（OpenAI/Anthropic 的行为一致）。
    """
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: str = "stop"
    model: str = ""


@dataclass
class ToolCallRecord:
    """记录一次工具调用的完整信息（工具名、参数、结果、耗时）。"""
    tool_name: str
    arguments: dict[str, object]
    result: str
    error: str | None = None
    elapsed_ms: float = 0


@dataclass
class AgentResult:
    """Agent.run() 的返回结果。"""
    content: str
    iterations: int
    tool_calls_history: list[ToolCallRecord] = field(default_factory=list)
    finish_reason: str = "stop"
