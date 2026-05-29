"""BufferMemory — 基于滑动窗口的短期记忆。

保留最近 N 条消息，自动丢弃旧消息。
始终保留 System 消息（如果 reserve_system=True）。
"""

from __future__ import annotations

from nexus.core.types import Message
from nexus.memory.base import BaseMemory


class BufferMemory(BaseMemory):
    """滑动窗口短期记忆。

    保留最近 max_messages 条非 system 消息。
    System 消息始终保留（因为定义了 Agent 的行为边界）。

    用法:
        memory = BufferMemory(max_messages=20)
        await memory.add({"role": "user", "content": "你好"})
        ctx = await memory.get_context()
    """

    def __init__(self, max_messages: int = 20, reserve_system: bool = True):
        self.max_messages = max_messages
        self.reserve_system = reserve_system
        self._messages: list[Message] = []

    async def add(self, message: Message) -> None:
        self._messages.append(message)
        self._trim()

    async def get_context(
        self,
        *,
        query: str | None = None,
        max_tokens: int | None = None,
        k: int = 5,
    ) -> list[Message]:
        messages = list(self._messages)

        if max_tokens:
            messages = self._trim_by_tokens(messages, max_tokens)

        return messages

    async def clear(self) -> None:
        self._messages.clear()

    def _trim(self) -> None:
        """保留最近 max_messages 条非 system 消息。"""
        system_msgs = [m for m in self._messages if m["role"] == "system"] if self.reserve_system else []
        others = [m for m in self._messages if m["role"] != "system"]
        self._messages = system_msgs + others[-self.max_messages:]

    @staticmethod
    def _trim_by_tokens(messages: list[Message], max_tokens: int) -> list[Message]:
        """从最新向旧保留，直到超过 token 限制。"""
        system_msgs = [m for m in messages if m["role"] == "system"]
        others = [m for m in messages if m["role"] != "system"]

        total = sum(len(m.get("content", "") or "") for m in system_msgs) // 4
        kept = []
        for m in reversed(others):
            t = len(m.get("content", "") or "") // 4
            if total + t > max_tokens:
                break
            kept.insert(0, m)
            total += t

        return system_msgs + kept
