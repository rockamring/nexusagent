"""Tool 系统。

一个普通 Python 函数 + 类型注解 = 一个可被 LLM 调用的 Tool。
"""

from nexus.tools.base import BaseTool, Tool
from nexus.tools.registry import ToolRegistry

__all__ = ["BaseTool", "Tool", "ToolRegistry"]
