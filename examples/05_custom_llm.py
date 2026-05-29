"""Example 5: 自定义 LLM Provider — 演示如何接入新的 LLM 服务。

演示接入一个自定义的 LLM Provider（兼容 OpenAI API 格式的第三方服务）。

运行:
    python examples/05_custom_llm.py
"""

import asyncio

from nexus.agent import ReActAgent
from nexus.llm.base import BaseLLM
from nexus.llm.registry import LLMRegistry


async def main():
    # 使用 OpenAIProvider 的 base_url 参数接入任意兼容的服务
    # 例如：本地 Ollama、vLLM、LiteLLM 代理等

    from nexus.llm.providers.openai import OpenAIProvider

    # 示例：接入本地 Ollama 服务
    # llm = OpenAIProvider(
    #     api_key="ollama",  # Ollama 不需要真实 key
    #     base_url="http://localhost:11434/v1",
    #     default_model="llama3",
    # )

    print("自定义 LLM Provider 示例")
    print("=" * 50)
    print("使用 OpenAIProvider + base_url 参数可以接入任意兼容 OpenAI API 的服务：")
    print()
    print("  # 本地 Ollama")
    print('  llm = OpenAIProvider(api_key="ollama", base_url="http://localhost:11434/v1", default_model="llama3")')
    print()
    print("  # vLLM")
    print('  llm = OpenAIProvider(api_key="none", base_url="http://localhost:8000/v1", default_model="qwen2.5")')
    print()
    print("  # LiteLLM 代理")
    print('  llm = OpenAIProvider(api_key="sk-...", base_url="http://localhost:4000/v1", default_model="gpt-4o")')
    print()
    print("也可以通过 LLMRegistry 注册自定义 Provider：")
    print()
    print("  LLMRegistry.register('my_provider', MyCustomProvider)")
    print('  llm = LLMRegistry.create("my_provider", ...)')

    # 显示当前已注册的 Provider
    print()
    print("当前已注册的 Provider（需要在代码中 import 并 register）：")

    # 手动注册以演示
    from nexus.llm.providers.openai import OpenAIProvider
    from nexus.llm.providers.anthropic import AnthropicProvider

    LLMRegistry.register("openai", OpenAIProvider)
    LLMRegistry.register("anthropic", AnthropicProvider)

    print(f"  {LLMRegistry.list_providers()}")


if __name__ == "__main__":
    asyncio.run(main())
