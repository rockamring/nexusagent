"""Orchestrator 编排系统测试。"""

import pytest

from nexus.agent.base import BaseAgent
from nexus.core.types import AgentResult
from nexus.orchestrator.graph import END, GraphOrchestrator
from nexus.orchestrator.sequential import SequentialOrchestrator
from nexus.orchestrator.state import OrchestratorState

# ── Mock Agent ──────────────────────────────

class EchoAgent(BaseAgent):
    """回显 Agent，在输入前加上名字标签。"""

    def __init__(self, name: str):
        self.name = name

    async def run(self, user_input: str, **kwargs) -> AgentResult:
        return AgentResult(
            content=f"[{self.name}] {user_input}",
            iterations=1,
        )


class UppercaseAgent(BaseAgent):
    """转大写 Agent。"""

    name = "uppercase"

    async def run(self, user_input: str, **kwargs) -> AgentResult:
        return AgentResult(
            content=user_input.upper(),
            iterations=1,
        )


# ── Sequential Orchestrator ─────────────────

@pytest.mark.asyncio
async def test_sequential_orchestrator():
    pipeline = SequentialOrchestrator([
        (EchoAgent("A"), "{input}"),
        (EchoAgent("B"), "处理: {input}"),
        (EchoAgent("C"), "最终: {input}"),
    ])

    result = await pipeline.run("hello")

    assert result.final_output == "[C] 最终: [B] 处理: [A] hello"
    assert len(result.agent_names) == 3
    assert result.total_iterations == 3


# ── OrchestratorState ─────────────────────

def test_state_get_set():
    state = OrchestratorState()
    state.set("key", "value")
    assert state.get("key") == "value"
    assert "key" in state

    assert state.get("missing", "default") == "default"

    state.delete("key")
    assert "key" not in state


def test_state_update():
    state = OrchestratorState({"a": 1})
    state.update({"b": 2, "c": 3})
    assert state.to_dict() == {"a": 1, "b": 2, "c": 3}


# ── Graph Orchestrator ─────────────────────

async def _classifier(state: OrchestratorState) -> str:
    """分类函数：根据 task 内容决定路由。"""
    task = state.get("task", "")
    if "研究" in task or "research" in task.lower():
        return "research"
    return "direct"


@pytest.mark.asyncio
async def test_graph_orchestrator_simple():
    graph = GraphOrchestrator()

    graph.add_node("start", EchoAgent("start"))
    graph.add_node("end", EchoAgent("end"))
    graph.add_edge("start", "end")
    graph.add_edge("end", END)
    graph.set_entry_point("start")

    result = await graph.run("test task")
    assert result.final_output == "[end] [start] test task"
    assert len(result.agent_names) == 2


@pytest.mark.asyncio
async def test_graph_orchestrator_conditional():
    graph = GraphOrchestrator()

    graph.add_node("classifier", _classifier)
    graph.add_node("research", EchoAgent("研究员"))
    graph.add_node("direct", EchoAgent("直接回复"))

    graph.set_entry_point("classifier")

    def route(state: OrchestratorState) -> str:
        task = state.get("task", "")
        if "研究" in task:
            return "research"
        return "direct"

    graph.add_conditional_edges("classifier", route)
    graph.add_edge("research", END)
    graph.add_edge("direct", END)

    result = await graph.run("请研究一下 AI Agent 框架")
    assert "研究员" in result.final_output
