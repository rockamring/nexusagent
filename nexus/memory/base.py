"""BaseMemory — 记忆系统的统一抽象接口。

设计理念（参考人类记忆模型）：
- 工作记忆（短期）= 滑动窗口 Buffer，保留最近 N 轮精确保留
- 长期记忆（语义）= 向量存储 + 语义检索，跨会话持久化
- 摘要压缩 = 上下文过长时用 LLM 压缩早期对话
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from nexus.core.types import Message


class BaseMemory(ABC):
    """记忆系统的统一抽象。

    子类必须实现：add(), get_context(), clear()

    设计原则：
    - add() 追记消息到记忆
    - get_context() 返回当前上下文的记忆消息，支持语义检索
    - clear() 清空所有记忆
    """

    @abstractmethod
    async def add(self, message: Message) -> None:
        """添加一条消息到记忆存储。

        Args:
            message: 框架统一 Message 格式
        """
        ...

    @abstractmethod
    async def get_context(
        self,
        *,
        query: str | None = None,
        max_tokens: int | None = None,
        k: int = 5,
    ) -> list[Message]:
        """获取当前上下文的记忆消息。

        Args:
            query: 用于语义搜索的查询文本（长期记忆使用）。为 None 则返回最近的记忆。
            max_tokens: 返回消息的总 token 上限（估算），为 None 则不限制。
            k: 语义搜索返回的最相关条目数。

        Returns:
            记忆消息列表，按相关性/时间排序
        """
        ...

    @abstractmethod
    async def clear(self) -> None:
        """清空所有记忆。"""
        ...
