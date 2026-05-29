"""GraphOrchestrator — 图式编排。

参考 LangGraph 的图式架构，是三种编排模式中最灵活的一种。

核心概念:
- Node: 编排中的一个处理单元（Agent 或普通函数）
- Edge: 节点之间的数据流
- Conditional Edge: 条件分支，根据状态决定路由
- Entry Point: 编排的起始节点

支持:
- 条件分支（if-else 逻辑）
- 循环（回边到之前的节点）
- 并行（多个下游节点）

用法:
    graph = GraphOrchestrator()
    graph.add_node("classifier", classifier_agent)
    graph.add_node("research", research_agent)
    graph.add_node("write", writer_agent)
    graph.add_node("review", reviewer_agent)

    graph.set_entry_point("classifier")

    # 条件分支：根据分类结果路由
    def route_after_classify(state):
        if state.get("task_type") == "research":
            return "research"
        return "write"

    graph.add_conditional_edges("classifier", route_after_classify)
    graph.add_edge("research", "write")
    graph.add_edge("write", "review")
    graph.add_edge("review", END)  # END 表示终止

    result = await graph.run("分析 AI Agent 框架的设计模式")
"""

from __future__ import annotations

from collections.abc import Callable

from nexus.agent.base import BaseAgent
from nexus.orchestrator.base import BaseOrchestrator, OrchestratorEvent, OrchestratorResult
from nexus.orchestrator.state import OrchestratorState
from nexus.utils import get_logger

logger = get_logger(__name__)

# 特殊节点名，表示编排结束
END = "__end__"
START = "__start__"


class GraphOrchestrator(BaseOrchestrator):
    """图式编排器。

    节点可以是:
    - BaseAgent 实例（有 run() 方法）
    - 普通 async 函数（接收 state，返回 None 或 str）
    """

    def __init__(self):
        self._nodes: dict[str, BaseAgent | Callable] = {}
        self._edges: dict[str, list[str]] = {}
        self._conditional_edges: dict[str, tuple[Callable, dict[str, str]]] = {}
        self._entry_point: str | None = None

    # ── 构建方法 ────────────────────────────

    def add_node(self, name: str, agent_or_func: BaseAgent | Callable) -> None:
        """添加一个节点。

        Args:
            name: 节点名称（唯一标识）
            agent_or_func: BaseAgent 实例 或 async callable
        """
        self._nodes[name] = agent_or_func

    def add_edge(self, from_node: str, to_node: str) -> None:
        """添加一条固定边：from_node → to_node。

        to_node 可以是 END（编排终止）。
        """
        if from_node not in self._edges:
            self._edges[from_node] = []
        self._edges[from_node].append(to_node)

    def add_conditional_edges(
        self,
        from_node: str,
        router: Callable[[OrchestratorState], str],
        route_map: dict[str, str] | None = None,
    ) -> None:
        """添加条件边：根据 router 函数的返回值路由到不同节点。

        Args:
            from_node: 源节点
            router: 接收 OrchestratorState，返回路由键
            route_map: 路由键 → 目标节点名的映射。
                      为 None 时，router 返回值直接作为目标节点名。
        """
        self._conditional_edges[from_node] = (router, route_map or {})

    def set_entry_point(self, node_name: str) -> None:
        """设置编排的起始节点。"""
        self._entry_point = node_name

    # ── 执行方法 ────────────────────────────

    async def run(self, task: str, **kwargs) -> OrchestratorResult:
        """执行图式编排。

        从 entry_point 开始，沿边遍历所有节点，直到遇到 END。
        使用简单的 BFS 策略（遇到条件边时评估路由函数）。
        """
        if not self._entry_point:
            raise ValueError("未设置 entry_point，调用 set_entry_point() 设置起始节点")

        state = OrchestratorState(initial_data={"task": task, "current_input": task})
        agent_names = []
        total_iterations = 0

        current = self._entry_point
        visited: set[str] = set()
        max_steps = 50  # 防止无限循环

        for _ in range(max_steps):
            if current == END:
                break

            if current in visited:
                logger.warning("graph_cycle", node=current)
                break
            visited.add(current)

            if current not in self._nodes:
                logger.error("graph_unknown_node", node=current)
                break

            logger.info("graph_node_start", node=current)

            node = self._nodes[current]
            current_input = state.get("current_input", task)

            # 执行节点
            if isinstance(node, BaseAgent):
                result = await node.run(current_input)
                output = result.content
                total_iterations += result.iterations
                agent_names.append(getattr(node, "name", current))
            else:
                output = await node(state)
                if output is None:
                    output = current_input
                agent_names.append(current)

            state.set("current_input", output)
            state.set(f"__{current}_output__", output)

            logger.info("graph_node_done", node=current, output_preview=str(output)[:200])

            # 确定下一个节点
            if current in self._conditional_edges:
                router, route_map = self._conditional_edges[current]
                route_key = router(state)
                next_node = route_map.get(route_key, route_key)
            elif current in self._edges:
                next_node = self._edges[current][0]  # 取第一条边（最简情况）
            else:
                break

            current = next_node

        return OrchestratorResult(
            final_output=state.get("current_input", ""),
            agent_names=agent_names,
            total_iterations=total_iterations,
        )

    async def stream(self, task: str, **kwargs) -> list[OrchestratorEvent]:
        """流式执行，每个节点完成时产出一个事件。"""
        import asyncio

        events: list[OrchestratorEvent] = []
        state = OrchestratorState(initial_data={"task": task, "current_input": task})
        current = self._entry_point
        visited: set[str] = set()
        max_steps = 50

        for _ in range(max_steps):
            if current == END:
                events.append(OrchestratorEvent(event_type="orchestrator_done", content=state.get("current_input", "")))
                break

            if current in visited:
                break
            visited.add(current)

            if current not in self._nodes:
                break

            events.append(OrchestratorEvent(event_type="agent_start", agent_name=current))

            node = self._nodes[current]
            current_input = state.get("current_input", task)

            try:
                if isinstance(node, BaseAgent):
                    output = ""
                    async for chunk in node.stream(current_input):
                        output += chunk
                else:
                    output = await node(state) or current_input
            except Exception as exc:
                events.append(OrchestratorEvent(event_type="agent_error", agent_name=current, error=str(exc)))
                break

            events.append(OrchestratorEvent(event_type="agent_done", agent_name=current, content=str(output)[:200]))
            state.set("current_input", output)

            if current in self._conditional_edges:
                router, route_map = self._conditional_edges[current]
                route_key = router(state)
                current = route_map.get(route_key, route_key)
            elif current in self._edges:
                current = self._edges[current][0]
            else:
                break

        return events
