"""Example 4: 多 Agent 协作 — 演示三种编排模式。

运行前设置环境变量或创建 config.yaml:
    export OPENAI_API_KEY="sk-..."

运行:
    python examples/04_multi_agent.py
"""

import asyncio

from nexus.agent import ReActAgent
from nexus.llm import create_llm
from nexus.orchestrator import (
    END,
    GraphOrchestrator,
    SequentialOrchestrator,
)
from nexus.orchestrator.state import OrchestratorState
from nexus.tools.builtin.calculator import calculator
from nexus.tools.registry import ToolRegistry


async def demo_sequential():
    """演示顺序编排：分析 -> 写作 -> 润色"""
    print("=" * 50)
    print("1. 顺序编排 (Sequential)")
    print("=" * 50)

    # 不同 Agent 可共享同一个 LLM 实例
    llm = create_llm()

    analyzer = ReActAgent(
        name="分析员",
        llm=llm,
        system_prompt="你是需求分析员。请分析以下需求并列出关键点。",
    )

    writer = ReActAgent(
        name="写作者",
        llm=llm,
        system_prompt="你是内容写作者。根据分析结果写出完整的方案。",
    )

    pipeline = SequentialOrchestrator([
        (analyzer, "请分析以下需求：{input}"),
        (writer, "根据以下分析写出完整方案：{input}"),
    ])

    result = await pipeline.run("设计一个简单的待办事项应用")
    print(f"[最终输出] {result.final_output[:500]}...")
    print(f"[参与 Agent] {result.agent_names}")
    print()


async def demo_graph():
    """演示图式编排：条件分支"""
    print("=" * 50)
    print("2. 图式编排 (Graph)")
    print("=" * 50)

    llm = create_llm()

    math_agent = ReActAgent(
        name="数学助手",
        llm=llm,
        tools=ToolRegistry(),
        system_prompt="你是数学专家，请解答数学问题。",
    )
    math_agent._tools.register(calculator)

    general_agent = ReActAgent(
        name="通用助手",
        llm=llm,
        system_prompt="你是通用助手，请回答各类问题。",
    )

    graph = GraphOrchestrator()

    def route(state: OrchestratorState) -> str:
        task = state.get("task", "")
        if "计算" in task or "算" in task:
            return "math"
        return "general"

    graph.add_node("math", math_agent)
    graph.add_node("general", general_agent)
    graph.set_entry_point("general")
    graph.add_conditional_edges("general", route)
    graph.add_edge("math", END)
    graph.add_edge("general", END)

    result = await graph.run("请帮我计算 12345 * 67890 的结果")
    print("[路由结果] math")
    print(f"[最终输出] {result.final_output[:500]}...")
    print()


async def main():
    await demo_sequential()
    await demo_graph()


if __name__ == "__main__":
    asyncio.run(main())
