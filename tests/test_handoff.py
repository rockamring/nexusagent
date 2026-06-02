"""Agent Handoff 和 Orchestrator 测试。"""

import pytest

from nexus.agent.handoff import HandoffTool
from nexus.core.types import AgentResult
from nexus.orchestrator.sequential import SequentialOrchestrator

# ── 简单 Agent（用于编排测试）────────────────

class SimpleAgent:
    """最小 Agent 实现，用于测试编排。"""

    def __init__(self, name: str, response: str = "default"):
        self.name = name
        self.response = response

    async def run(self, user_input: str, **kwargs) -> AgentResult:
        return AgentResult(
            content=f"[{self.name}] 收到: {user_input} → {self.response}",
            iterations=1,
        )


# ── HandoffTool 测试 ────────────────────────

def test_handoff_tool_schema():
    """HandoffTool 自动生成正确的 schema。"""
    source = SimpleAgent(name="主Agent")
    target = SimpleAgent(name="子Agent")

    handoff = HandoffTool(source_agent=source, target_agent=target)

    assert handoff.schema["name"] == "handoff_to_子Agent"
    assert "交接" in handoff.schema["description"]
    assert "context" in handoff.schema["parameters"]["properties"]


def test_handoff_tool_custom_name():
    """支持自定义 tool name。"""
    source = SimpleAgent(name="主")
    target = SimpleAgent(name="子")

    handoff = HandoffTool(
        source_agent=source,
        target_agent=target,
        name="delegate_task",
        description="委派任务给子Agent",
    )

    assert handoff.schema["name"] == "delegate_task"
    assert handoff.schema["description"] == "委派任务给子Agent"


@pytest.mark.asyncio
async def test_handoff_tool_execute():
    """Handoff 执行后目标 Agent 被调用。"""
    source = SimpleAgent(name="主Agent")
    target = SimpleAgent(name="研究员", response="研究完成")

    handoff = HandoffTool(
        source_agent=source,
        target_agent=target,
        handoff_prompt="请研究以下内容：\n{context}",
    )

    result = await handoff.execute(context="Python 异步编程")
    assert "研究员" in result
    assert "研究完成" in result
    assert "Python 异步编程" in result


@pytest.mark.asyncio
async def test_handoff_tool_default_prompt():
    """默认 handoff_prompt 直接使用 context。"""
    source = SimpleAgent(name="主")
    target = SimpleAgent(name="子", response="完成")

    handoff = HandoffTool(source_agent=source, target_agent=target)
    result = await handoff.execute(context="直接传递")
    assert "完成" in result


# ── SequentialOrchestrator 测试 ──────────────

@pytest.mark.asyncio
async def test_sequential_orchestrator_basic():
    """基本流水线：两个 Agent 顺序执行。"""
    agent_a = SimpleAgent(name="A", response="A的结果")
    agent_b = SimpleAgent(name="B", response="B的结果")

    orchestrator = SequentialOrchestrator([
        (agent_a, "处理: {input}"),
        (agent_b, "基于上一步: {input}"),
    ])

    result = await orchestrator.run("初始任务")
    assert "B的结果" in result.final_output
    assert "A的结果" in result.final_output
    assert result.agent_names == ["A", "B"]
    assert result.total_iterations == 2


@pytest.mark.asyncio
async def test_sequential_orchestrator_add_stage():
    """动态添加阶段。"""
    agent_a = SimpleAgent(name="A", response="A的输出")

    orchestrator = SequentialOrchestrator()
    orchestrator.add_stage(agent_a, "第一步: {input}")
    orchestrator.add_stage(SimpleAgent(name="B", response="B的输出"), "第二步: {input}")

    result = await orchestrator.run("任务")
    assert "B的输出" in result.final_output
    assert len(result.agent_names) == 2


@pytest.mark.asyncio
async def test_sequential_orchestrator_single_stage():
    """只有一个阶段也可以运行。"""
    agent = SimpleAgent(name="唯一", response="处理完成")

    orchestrator = SequentialOrchestrator([(agent, "{input}")])
    result = await orchestrator.run("测试")

    assert "处理完成" in result.final_output
    assert result.agent_names == ["唯一"]


@pytest.mark.asyncio
async def test_sequential_orchestrator_empty():
    """空阶段列表时直接返回任务输入。"""
    orchestrator = SequentialOrchestrator()
    result = await orchestrator.run("原始任务")

    assert result.final_output == "原始任务"
    assert result.agent_names == []


@pytest.mark.asyncio
async def test_sequential_data_flow():
    """验证数据从前一个 Agent 流到后一个 Agent。"""
    agent_a = SimpleAgent(name="A", response="数据A")
    agent_b = SimpleAgent(name="B", response="数据B")

    orchestrator = SequentialOrchestrator([
        (agent_a, "{input}"),
        (agent_b, "{input}"),
    ])

    result = await orchestrator.run("初始输入")
    # agent_b 收到了 agent_a 的完整输出
    assert "数据A" in result.final_output
    assert "数据B" in result.final_output


# ── OrchestratorResult 测试 ─────────────────

def test_orchestrator_result_successful():
    """有 final_output 时为成功。"""
    from nexus.orchestrator.base import OrchestratorResult

    result = OrchestratorResult(final_output="完成了")
    assert result.successful is True


def test_orchestrator_result_not_successful():
    """空 final_output 时为失败。"""
    from nexus.orchestrator.base import OrchestratorResult

    result = OrchestratorResult(final_output="")
    assert result.successful is False


# ── OrchestratorEvent 测试 ──────────────────

def test_orchestrator_event():
    """事件包含正确的字段。"""
    from nexus.orchestrator.base import OrchestratorEvent

    event = OrchestratorEvent(
        event_type="agent_start",
        agent_name="MyAgent",
        content="开始执行",
    )
    assert event.event_type == "agent_start"
    assert event.agent_name == "MyAgent"
    assert event.error == ""
