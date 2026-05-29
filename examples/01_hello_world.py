"""Example 1: 最简 Agent — 演示框架最基本的用法。

运行前请设置环境变量:
    export OPENAI_API_KEY="sk-..."

运行:
    python examples/01_hello_world.py
"""

import asyncio
import os

from nexus.llm.providers.openai import OpenAIProvider
from nexus.agent import ReActAgent


async def main():
    # 1. 创建 LLM Provider
    api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    llm = OpenAIProvider(api_key=api_key, default_model="gpt-4o-mini")

    # 2. 创建 Agent（不需要工具，纯对话）
    agent = ReActAgent(
        name="助手",
        llm=llm,
        system_prompt="你是一个友好的 AI 助手，请用中文回答问题。",
    )

    # 3. 执行
    result = await agent.run("你好！请用一句话介绍一下你自己。")
    print(f"[回复] {result.content}")
    print(f"[循环次数] {result.iterations}")


if __name__ == "__main__":
    asyncio.run(main())
