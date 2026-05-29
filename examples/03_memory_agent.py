"""Example 3: 带记忆的 Agent — 演示跨轮对话记忆。

运行前请设置:
    export OPENAI_API_KEY="sk-..."

运行:
    python examples/03_memory_agent.py
"""

import asyncio
import os

from nexus.llm.providers.openai import OpenAIProvider
from nexus.agent import ReActAgent
from nexus.memory import BufferMemory


async def main():
    api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    llm = OpenAIProvider(api_key=api_key, default_model="gpt-4o-mini")

    # 创建带短期记忆的 Agent
    memory = BufferMemory(max_messages=20)

    agent = ReActAgent(
        name="记忆助手",
        llm=llm,
        memory=memory,
        system_prompt="你是一个记住用户说过的话的 AI 助手。每次回复时请确认你是否记住了之前的信息。",
    )

    # 模拟多轮对话
    print("=" * 50)
    print("第 1 轮")
    print("=" * 50)
    result = await agent.run("我叫小明，我喜欢 Python 编程。")
    print(f"[用户] 我叫小明，我喜欢 Python 编程。")
    print(f"[助手] {result.content}")
    print()

    print("=" * 50)
    print("第 2 轮")
    print("=" * 50)
    result = await agent.run("能推荐一些适合我的学习资源吗？")
    print(f"[用户] 能推荐一些适合我的学习资源吗？")
    print(f"[助手] {result.content}")
    print()

    print("=" * 50)
    print("第 3 轮")
    print("=" * 50)
    result = await agent.run("我叫什么名字？我喜欢什么？")
    print(f"[用户] 我叫什么名字？我喜欢什么？")
    print(f"[助手] {result.content}")
    print()
    print(f"[记忆中的消息数] {len(memory._messages)}")


if __name__ == "__main__":
    asyncio.run(main())
