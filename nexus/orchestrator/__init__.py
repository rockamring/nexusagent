"""Orchestrator 编排系统。

支持三种编排模式：顺序流水线、监督者模式、图式编排。
"""

from nexus.orchestrator.base import BaseOrchestrator, OrchestratorEvent, OrchestratorResult
from nexus.orchestrator.graph import END, GraphOrchestrator
from nexus.orchestrator.sequential import SequentialOrchestrator
from nexus.orchestrator.state import OrchestratorState
from nexus.orchestrator.supervisor import SupervisorOrchestrator

__all__ = [
    "BaseOrchestrator",
    "OrchestratorResult",
    "OrchestratorEvent",
    "OrchestratorState",
    "SequentialOrchestrator",
    "SupervisorOrchestrator",
    "GraphOrchestrator",
    "END",
]
