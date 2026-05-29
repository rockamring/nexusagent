"""CompositeMemory — 多策略组合记忆。

将多个 Memory 实现组合在一起，统一调度：
- add: 写入所有子 Memory
- get_context: 合并所有子 Memory 的上下文
"""

from __future__ import annotations

from nexus.core.types import Message
from nexus.memory.base import BaseMemory


class CompositeMemory(BaseMemory):
    """组合多种记忆策略。

    用法:
        memory = CompositeMemory([
            BufferMemory(max_messages=20),
            VectorStoreMemory(persist_dir="./memories"),
        ])
        await memory.add(msg)  # 同时写入短期和长期记忆
        ctx = await memory.get_context(query="Python")  # 合并两者的结果
    """

    def __init__(self, memories: list[BaseMemory] | None = None):
        self._memories = memories or []

    def add_memory(self, memory: BaseMemory) -> None:
        """添加一个子记忆模块。"""
        self._memories.append(memory)

    async def add(self, message: Message) -> None:
        for memory in self._memories:
            await memory.add(message)

    async def get_context(
        self,
        *,
        query: str | None = None,
        max_tokens: int | None = None,
        k: int = 5,
    ) -> list[Message]:
        all_messages: list[Message] = []

        # 为每个子 Memory 分配均等的 token 预算
        sub_budget = (max_tokens // len(self._memories)) if max_tokens and self._memories else None

        for memory in self._memories:
            msgs = await memory.get_context(query=query, max_tokens=sub_budget, k=k)
            all_messages.extend(msgs)

        return all_messages

    async def clear(self) -> None:
        for memory in self._memories:
            await memory.clear()
