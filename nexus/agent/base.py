"""BaseAgent — Agent 的统一抽象接口。

所有 Agent 实现必须遵循此接口。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from nexus.core.types import AgentResult


class BaseAgent(ABC):
    """Agent 抽象基类。

    子类最小实现：__init__ + run()。
    可选实现：stream() 用于流式输出。
    """

    @abstractmethod
    async def run(self, user_input: str, **kwargs) -> AgentResult:
        """执行 Agent，处理用户输入并返回结果。"""
        ...

    async def stream(self, user_input: str, **kwargs) -> AsyncIterator[str]:
        """流式执行 Agent，逐步产出文本。

        默认实现直接调用 run() 并一次性返回结果。
        支持流式的 Agent（如 ReActAgent）应 override 此方法。
        """
        result = await self.run(user_input, **kwargs)
        yield result.content
