"""LLM Provider 抽象层。

提供统一的 LLM 调用接口，支持多 Provider (OpenAI, Anthropic) 切换。
"""

from nexus.llm.base import BaseLLM
from nexus.llm.registry import LLMRegistry

__all__ = ["BaseLLM", "LLMRegistry"]
