"""BaseTool / Tool — 工具系统的核心抽象。

设计目标：一个普通 Python 函数 + 类型注解 = 一个可被 LLM 调用的 Tool。

用法:
    # 方式 1: 装饰器（推荐，简单场景）
    @Tool.from_function(name="get_weather", description="查询指定城市的天气")
    async def get_weather(city: str, date: str = "today") -> str:
        return f"{city} {date}: 晴, 28°C"

    # 方式 2: 继承 BaseTool（复杂场景）
    class DatabaseTool(BaseTool):
        @property
        def schema(self) -> dict:
            return {...}

        async def execute(self, **kwargs) -> str:
            ...
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Callable

from nexus.tools.schema import build_schema


class BaseTool(ABC):
    """Tool 的抽象基类。

    所有工具必须实现：
    - execute(): 执行工具逻辑，返回文本结果
    - schema():   返回工具的 JSON Schema 定义（OpenAI 兼容格式）
    """

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """执行工具逻辑。

        Args:
            **kwargs: LLM 传递的参数，key 对应 schema 中定义的参数名

        Returns:
            文本格式的执行结果（会作为 ToolMessage 返回给 LLM）
        """
        ...

    @property
    @abstractmethod
    def schema(self) -> dict:
        """返回工具的 JSON Schema 定义。

        格式与 OpenAI Function Calling 兼容:
        {
            "name": "tool_name",
            "description": "工具描述",
            "parameters": {
                "type": "object",
                "properties": {...},
                "required": [...]
            }
        }
        """
        ...


class Tool(BaseTool):
    """从 Python 函数创建的 Tool。

    使用 from_function 装饰器将普通 async 函数包装为 Tool。
    自动从函数签名和类型注解推导 JSON Schema。
    """

    def __init__(self, name: str, description: str, func: Callable, schema: dict):
        self._name = name
        self._description = description
        self._func = func
        self._schema = schema

    @staticmethod
    def from_function(name: str, description: str) -> Callable:
        """装饰器：把 async 函数包装为 Tool。

        自动从函数签名生成 JSON Schema，参数名和类型从类型注解推导。

        Args:
            name: 工具名称（暴露给 LLM 的函数名）
            description: 工具描述（帮助 LLM 理解何时使用此工具）

        Example:
            @Tool.from_function(name="calculator", description="执行数学运算")
            async def calculator(expression: str) -> str:
                return str(eval(expression))
        """
        def decorator(func: Callable) -> Tool:
            schema = build_schema(func, name, description)
            return Tool(name=name, description=description, func=func, schema=schema)
        return decorator

    @property
    def schema(self) -> dict:
        return self._schema

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    async def execute(self, **kwargs: Any) -> str:
        result = self._func(**kwargs)
        # 支持同步和异步函数
        if hasattr(result, "__await__"):
            result = await result
        return str(result)


class ToolResult:
    """一次工具执行的结果记录，包含成功/失败信息和耗时。"""

    def __init__(self, tool_name: str, arguments: dict, result: str, elapsed_ms: float, error: str | None = None):
        self.tool_name = tool_name
        self.arguments = arguments
        self.result = result
        self.elapsed_ms = elapsed_ms
        self.error = error

    @property
    def success(self) -> bool:
        return self.error is None

    @classmethod
    async def from_execution(cls, tool: BaseTool, arguments: dict) -> "ToolResult":
        """执行工具并记录结果。"""
        start = time.perf_counter()
        try:
            result = await tool.execute(**arguments)
            elapsed_ms = (time.perf_counter() - start) * 1000
            return cls(
                tool_name=tool.schema["name"],
                arguments=arguments,
                result=result,
                elapsed_ms=elapsed_ms,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            error_msg = f"{type(exc).__name__}: {exc}"
            return cls(
                tool_name=tool.schema["name"],
                arguments=arguments,
                result=error_msg,
                elapsed_ms=elapsed_ms,
                error=error_msg,
            )
