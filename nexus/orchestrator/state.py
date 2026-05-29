"""编排状态管理。

在整个编排生命周期中维护共享状态，Agent 可以读写状态中的键值对。
用于在 Graph Orchestrator 中实现条件路由和数据传递。
"""

from __future__ import annotations

from typing import Any


class OrchestratorState:
    """编排共享状态。

    所有参与编排的 Agent 可以读/写此状态对象。
    Graph Orchestrator 使用状态字段值来决定条件路由。

    用法:
        state = OrchestratorState()
        state.set("task_type", "research")
        state.set("findings", {...})

        if state.get("task_type") == "research":
            route_to = "researcher"
    """

    def __init__(self, initial_data: dict[str, Any] | None = None):
        self._data: dict[str, Any] = dict(initial_data or {})

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def update(self, data: dict[str, Any]) -> None:
        self._data.update(data)

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def has(self, key: str) -> bool:
        return key in self._data

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"OrchestratorState({self._data})"
