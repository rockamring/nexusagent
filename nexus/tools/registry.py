"""ToolRegistry — 工具注册中心。

集中管理所有可用的 Tool 实例，Agent 通过 Registry 查找和执行工具。

用法:
    registry = ToolRegistry()
    registry.register(weather_tool)
    registry.register(calculator_tool)

    # 获取所有工具的 JSON Schema（传给 LLM）
    schemas = registry.get_schemas()

    # 按名称执行工具
    result = await registry.execute("calculator", expression="1+1")
"""

from __future__ import annotations

from nexus.tools.base import BaseTool, ToolResult


class ToolRegistry:
    """工具注册中心。

    存储 Tool 实例，提供按名称查找、批量获取 Schema、执行等功能。
    """

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """注册一个工具。

        Args:
            tool: Tool 实例

        Raises:
            ValueError: 同名工具已注册
        """
        name = tool.schema["name"]
        if name in self._tools:
            raise ValueError(f"工具 '{name}' 已注册，不允许重复")
        self._tools[name] = tool

    def register_many(self, tools: list[BaseTool]) -> None:
        """批量注册工具。"""
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> BaseTool:
        """按名称获取工具。

        Raises:
            KeyError: 工具未注册
        """
        if name not in self._tools:
            available = ", ".join(self._tools.keys()) or "(无)"
            raise KeyError(f"未找到工具 '{name}'，可用: {available}")
        return self._tools[name]

    def get_schemas(self) -> list[dict]:
        """获取所有已注册工具的 JSON Schema 列表（用于传给 LLM）。"""
        return [tool.schema for tool in self._tools.values()]

    async def execute(self, name: str, **arguments) -> ToolResult:
        """按名称执行工具。

        Args:
            name: 工具名称
            **arguments: 传递给工具的命名参数

        Returns:
            ToolResult: 执行结果（包含成功/失败状态）
        """
        tool = self.get(name)
        return await ToolResult.from_execution(tool, arguments)

    async def execute_many(
        self,
        calls: list[dict],
    ) -> list[ToolResult]:
        """批量执行工具调用。

        Args:
            calls: 工具调用列表，格式 [{"name": "...", "arguments": {...}}, ...]

        Returns:
            与输入顺序对应的 ToolResult 列表
        """
        results = []
        for call in calls:
            result = await self.execute(call["name"], **call.get("arguments", {}))
            results.append(result)
        return results

    @property
    def tool_names(self) -> list[str]:
        """所有已注册工具的名称。"""
        return list(self._tools.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
