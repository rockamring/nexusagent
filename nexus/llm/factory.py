"""LLM 工厂函数 — 从配置创建 LLM 实例。

用法:
    # 从配置文件自动创建
    llm = create_llm()

    # 指定 Provider
    llm = create_llm("anthropic")

    # 传入配置对象
    config = NexusConfig.from_yaml("config.yaml")
    llm = create_llm(config=config)

    # 覆盖配置中的值
    llm = create_llm(api_key="sk-...", default_model="gpt-4o-mini")
"""

from __future__ import annotations

from nexus.core.config import NexusConfig
from nexus.llm.base import BaseLLM
from nexus.llm.registry import LLMRegistry

# 内置 Provider 注册（延迟导入，避免循环依赖）
_builtin_registered = False


def _register_builtin_providers() -> None:
    """注册内置的 Provider 类。"""
    global _builtin_registered
    if _builtin_registered:
        return

    from nexus.llm.providers.anthropic import AnthropicProvider
    from nexus.llm.providers.openai import OpenAIProvider

    LLMRegistry.register("openai", OpenAIProvider)
    LLMRegistry.register("anthropic", AnthropicProvider)
    _builtin_registered = True


def create_llm(
    provider: str | None = None,
    *,
    config: NexusConfig | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    default_model: str | None = None,
    **extra_kwargs,
) -> BaseLLM:
    """从配置创建 LLM 实例。

    自动注册内置 Provider（openai, anthropic），
    从配置文件获取 api_key / base_url / default_model，
    并允许通过参数覆盖。

    Args:
        provider: Provider 名称（如 'openai', 'anthropic'），默认从 config 读取
        config: NexusConfig 对象，未提供时自动从配置文件或环境变量加载
        api_key: 覆盖配置中的 api_key
        base_url: 覆盖配置中的 base_url
        default_model: 覆盖配置中的 default_model
        **extra_kwargs: 传递给 Provider 构造函数的额外参数

    Returns:
        BaseLLM 实例

    Raises:
        KeyError: Provider 未注册或未找到配置
    """
    _register_builtin_providers()

    if config is None:
        config = NexusConfig.from_file()

    provider_name = provider or config.default_provider

    # 尝试从配置获取 Provider 配置
    try:
        provider_cfg = config.get_provider_config(provider_name)
    except KeyError:
        provider_cfg = None

    # 构建参数：代码传入 > 配置文件 > 默认值
    final_api_key = api_key
    final_base_url = base_url
    final_model = default_model

    if provider_cfg:
        if final_api_key is None:
            final_api_key = provider_cfg.api_key
        if final_base_url is None:
            final_base_url = provider_cfg.base_url or None
        if final_model is None:
            final_model = provider_cfg.default_model or None

    # 回退到顶层 default_model
    if final_model is None:
        final_model = config.default_model

    kwargs: dict = {}
    if final_api_key:
        kwargs["api_key"] = final_api_key
    if final_base_url:
        kwargs["base_url"] = final_base_url
    if final_model:
        kwargs["default_model"] = final_model
    kwargs.update(extra_kwargs)

    return LLMRegistry.create(provider_name, **kwargs)
