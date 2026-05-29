"""Example 5: 自定义 LLM — 演示接入第三方服务和注册自定义 Provider。

演示:
1. 通过 config.yaml 配置 base_url 接入 Ollama / vLLM / LiteLLM
2. 通过 LLMRegistry 注册自定义 Provider
3. 使用 create_llm() 自动加载配置

运行:
    python examples/05_custom_llm.py
"""

import asyncio

from nexus.core.config import NexusConfig
from nexus.llm import create_llm, LLMRegistry
from nexus.agent import ReActAgent


async def main():
    print("自定义 LLM 配置示例")
    print("=" * 50)
    print()
    print("方式 1: 通过 config.yaml 配置（推荐）")
    print("-" * 30)
    print("""
  # config.yaml
  default_provider: ollama

  providers:
    ollama:
      api_key: "ollama"   # Ollama 不需要真实 key
      base_url: "http://localhost:11434/v1"
      default_model: "llama3"

    vllm:
      api_key: "none"
      base_url: "http://localhost:8000/v1"
      default_model: "qwen2.5"

    litellm:
      api_key: "${LITELLM_API_KEY}"
      base_url: "http://localhost:4000/v1"
      default_model: "gpt-4o"
""")
    print("然后一行代码即可创建:")
    print('  llm = create_llm("ollama")')
    print()

    print("方式 2: 从代码构造配置")
    print("-" * 30)
    config = NexusConfig.from_file()  # 自动查找 config.yaml
    if config.providers:
        print(f"  已加载 Provider: {list(config.providers.keys())}")
        for name, cfg in config.providers.items():
            print(f"    {name}: model={cfg.default_model}, base_url={cfg.base_url or '(默认)'}")
    else:
        print("  未找到配置文件（使用环境变量或默认值）")
        print("  提示: 复制 config.yaml 到项目根目录并编辑")

    print()
    print("方式 3: 注册自定义 Provider")
    print("-" * 30)
    from nexus.llm.providers.openai import OpenAIProvider
    from nexus.llm.providers.anthropic import AnthropicProvider

    LLMRegistry.register("openai", OpenAIProvider)
    LLMRegistry.register("anthropic", AnthropicProvider)
    print(f"  已注册: {LLMRegistry.list_providers()}")


if __name__ == "__main__":
    asyncio.run(main())
