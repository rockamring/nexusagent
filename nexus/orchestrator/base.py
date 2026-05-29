"""BaseOrchestrator — 编排器的统一抽象接口。

编排器负责协调多个 Agent 协作完成复杂任务。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field


@dataclass
class OrchestratorResult:
    """编排执行的结果。"""
    final_output: str
    intermediate_outputs: list = field(default_factory=list)
    agent_names: list[str] = field(default_factory=list)
    total_iterations: int = 0

    @property
    def successful(self) -> bool:
        return bool(self.final_output)


@dataclass
class OrchestratorEvent:
    """编排过程中的事件。"""
    event_type: str  # "agent_start", "agent_done", "agent_error", "orchestrator_done"
    agent_name: str = ""
    content: str = ""
    error: str = ""


class BaseOrchestrator(ABC):
    """编排器抽象基类。

    子类必须实现：run()。
    可选实现：stream() 用于流式事件输出。
    """

    @abstractmethod
    async def run(self, task: str, **kwargs) -> OrchestratorResult:
        """执行编排，返回最终结果。"""
        ...

    async def stream(self, task: str, **kwargs) -> AsyncIterator[OrchestratorEvent]:
        """流式执行编排，实时产出事件。"""
        result = await self.run(task, **kwargs)
        yield OrchestratorEvent(
            event_type="orchestrator_done",
            content=result.final_output,
        )
