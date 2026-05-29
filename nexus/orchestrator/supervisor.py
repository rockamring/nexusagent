"""SupervisorOrchestrator — 监督者编排模式。

参考 CrewAI 的 Manager-Worker 模式 + OpenAI Agents SDK 的 Handoff 机制。

工作原理:
- Supervisor Agent 负责分析任务、分配任务给 Worker Agent
- 每个 Worker 通过 Handoff 机制被调用
- Supervisor 可通过 Tool Calling 决定委托给哪个 Worker

架构图:
                     ┌──────────────────┐
                     │  Supervisor Agent │  (任务分解 + 分配)
                     └────────┬─────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
     ┌────────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
     │ Researcher    │ │  Coder      │ │  Reviewer   │
     │ Agent         │ │  Agent      │ │  Agent      │
     └───────────────┘ └─────────────┘ └─────────────┘
"""

from __future__ import annotations

from nexus.agent.base import BaseAgent
from nexus.agent.handoff import HandoffTool
from nexus.orchestrator.base import BaseOrchestrator, OrchestratorResult
from nexus.tools.registry import ToolRegistry
from nexus.utils import get_logger

logger = get_logger(__name__)


class SupervisorOrchestrator(BaseOrchestrator):
    """监督者编排器。

    Supervisor Agent 通过 Handoff Tool 将任务分派给 Worker Agent。

    用法:
        supervisor = ReActAgent(name="管理者", llm=llm, system_prompt="你是项目经理...")

        orchestrator = SupervisorOrchestrator(
            supervisor=supervisor,
            workers={
                "研究员": researcher_agent,
                "程序员": coder_agent,
                "审核者": reviewer_agent,
            },
        )

        result = await orchestrator.run("开发一个 Web 应用的需求分析")
    """

    def __init__(
        self,
        supervisor: BaseAgent,
        workers: dict[str, BaseAgent],
    ):
        self._supervisor = supervisor
        self._workers = workers
        self._tools = ToolRegistry()

        # 为每个 Worker 创建 Handoff Tool 并注册到 Supervisor 的工具集
        for worker_name, worker_agent in workers.items():
            handoff = HandoffTool(
                source_agent=supervisor,
                target_agent=worker_agent,
                name=f"assign_to_{worker_name}",
                description=(
                    f"将任务分配给 {worker_name} 处理。"
                    f"当任务需要 {worker_name} 的专业能力时调用。"
                    f"提供的 context 应包含完整的任务描述和约束。"
                ),
            )
            self._tools.register(handoff)

        # 将 Handoff Tools 注入 Supervisor
        if hasattr(supervisor, "_tools"):
            for name in self._tools.tool_names:
                supervisor._tools.register(self._tools.get(name))

    async def run(self, task: str, **kwargs) -> OrchestratorResult:
        logger.info("supervisor_start", workers=list(self._workers.keys()))

        # Supervisor 通过 Handoff Tools 自动分配任务给 Worker
        result = await self._supervisor.run(task, **kwargs)

        logger.info(
            "supervisor_done",
            iterations=result.iterations,
            tools_called=[r.tool_name for r in result.tool_calls_history],
        )

        return OrchestratorResult(
            final_output=result.content,
            agent_names=["supervisor"] + [r.tool_name for r in result.tool_calls_history],
            total_iterations=result.iterations,
        )
