"""Example 2: 带工具的 Agent — 演示 Tool Calling。

运行前设置环境变量或创建 config.yaml:
    export OPENAI_API_KEY="sk-..."

运行:
    python examples/02_tool_agent.py
"""

import asyncio

from nexus.agent import ReActAgent
from nexus.llm import create_llm
from nexus.tools.builtin.calculator import calculator
from nexus.tools.builtin.file_ops import list_files, read_file, write_file
from nexus.tools.registry import ToolRegistry


async def main():
    # 1. 创建 LLM — 自动从 config.yaml 或环境变量加载配置
    llm = create_llm()

    # 2. 注册工具
    tools = ToolRegistry()
    tools.register(calculator)
    tools.register(read_file)
    tools.register(write_file)
    tools.register(list_files)

    # 3. 创建 Agent
    agent = ReActAgent(
        name="工具助手",
        llm=llm,
        tools=tools,
        system_prompt=(
            "你是一个可以帮助用户完成任务的 AI 助手。"
            "你可以使用计算器、读写文件、列出目录等工具。"
        ),
    )

    # 4. 执行带工具的任务
    print("=" * 50)
    print("任务 1: 数学计算")
    print("=" * 50)
    result = await agent.run("请帮我计算 (123 + 456) * 789 的结果")
    print(f"[回复] {result.content}")
    print(f"[循环次数] {result.iterations}")
    print(f"[工具调用] {[r.tool_name for r in result.tool_calls_history]}")
    print()

    print("=" * 50)
    print("任务 2: 文件操作")
    print("=" * 50)
    result = await agent.run("在当前目录创建一个名为 hello.txt 的文件，内容写 'Hello NexusAgent!'")
    print(f"[回复] {result.content}")
    print(f"[工具调用] {[r.tool_name for r in result.tool_calls_history]}")


if __name__ == "__main__":
    asyncio.run(main())
