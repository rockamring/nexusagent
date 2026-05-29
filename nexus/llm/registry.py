"""LLM Registry — LLM Provider 的工厂注册表。

使用模块级字典注册 Provider 类（非实例），
通过 get() / create() 获取或创建 Provider 实例。

用法:
    # 注册
    LLMRegistry.register("openai", OpenAIProvider)

    # 创建实例
    llm = LLMRegistry.create("openai", api_key="sk-...", default_model="gpt-4o")

    # 或直接获取已注册的类
    provider_cls = LLMRegistry.get("openai")
"""

from __future__ import annotations

from typing import Any

from nexus.llm.base import BaseLLM


class LLMRegistry:
    """LLM Provider 注册表。

    使用类方法操作模块级字典，不需要实例化。
    设计为类方法而非单例，避免全局状态污染测试。
    """

    _providers: dict[str, type[BaseLLM]] = {}

    @classmethod
    def register(cls, name: str, provider_cls: type[BaseLLM]) -> None:
        """注册一个 Provider 类。

        Args:
            name: Provider 名称，如 'openai', 'anthropic'
            provider_cls: 继承 BaseLLM 的 Provider 类
        """
        if not issubclass(provider_cls, BaseLLM):
            raise TypeError(f"{provider_cls.__name__} 必须继承 BaseLLM")
        cls._providers[name] = provider_cls

    @classmethod
    def get(cls, name: str) -> type[BaseLLM]:
        """获取已注册的 Provider 类。

        Raises:
            KeyError: 未找到指定的 Provider
        """
        if name not in cls._providers:
            available = ", ".join(cls._providers.keys()) or "(无)"
            raise KeyError(f"未找到 Provider '{name}'，可用: {available}")
        return cls._providers[name]

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseLLM:
        """创建 Provider 实例。

        Args:
            name: Provider 名称
            **kwargs: 传递给 Provider 构造函数的参数
        """
        provider_cls = cls.get(name)
        return provider_cls(**kwargs)

    @classmethod
    def list_providers(cls) -> list[str]:
        """列出所有已注册的 Provider 名称。"""
        return list(cls._providers.keys())

    @classmethod
    def clear(cls) -> None:
        """清空注册表（测试用）。"""
        cls._providers.clear()
