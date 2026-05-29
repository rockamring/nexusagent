"""SummaryMemory — 基于 LLM 摘要压缩的记忆。

参考 Anthropic 的 Compaction 策略：
当对话上下文超过 token 预算时，使用一个便宜的 LLM 对早期对话进行摘要压缩。
摘要作为 System 级上下文注入，保证 Agent 不会丢失早期对话的关键信息。
"""

from __future__ import annotations

from nexus.core.types import Message
from nexus.llm.base import BaseLLM
from nexus.memory.base import BaseMemory


class SummaryMemory(BaseMemory):
    """LLM 摘要压缩记忆。

    当上下文超过 trigger_tokens 时，调用 LLM 对早期对话生成摘要。

    策略：
    1. 保留最近 keep_recent 条完整消息
    2. 对更早的消息，调用 summarizer LLM 生成摘要
    3. 摘要以 system 消息形式注入到上下文最前面

    用法:
        summary_llm = OpenAIProvider(api_key="...", default_model="gpt-4o-mini")
        memory = SummaryMemory(summarizer_llm=summary_llm, trigger_tokens=4000)
    """

    def __init__(
        self,
        summarizer_llm: BaseLLM,
        trigger_tokens: int = 4000,
        keep_recent: int = 10,
    ):
        self._summarizer = summarizer_llm
        self._trigger_tokens = trigger_tokens
        self._keep_recent = keep_recent
        self._messages: list[Message] = []
        self._summary: str = ""

    async def add(self, message: Message) -> None:
        self._messages.append(message)

    async def get_context(
        self,
        *,
        query: str | None = None,
        max_tokens: int | None = None,
        k: int = 5,
    ) -> list[Message]:
        threshold = max_tokens or self._trigger_tokens
        current_tokens = self._count_tokens(self._messages)

        if current_tokens > threshold and len(self._messages) > self._keep_recent:
            await self._compress()

        result: list[Message] = []
        if self._summary:
            result.append({"role": "system", "content": f"[对话历史摘要]\n{self._summary}"})

        result.extend(self._messages[-self._keep_recent:])
        return result

    async def clear(self) -> None:
        self._messages.clear()
        self._summary = ""

    async def _compress(self) -> None:
        """调用 LLM 对早期对话进行摘要。"""
        split = max(0, len(self._messages) - self._keep_recent)
        early = self._messages[:split]
        recent = self._messages[split:]

        prompt = "请用 2-3 句话总结以下对话的关键信息，保留重要事实、决定和上下文：\n\n"
        for m in early:
            content = m.get("content", "")
            if content:
                prompt += f"[{m.get('role', 'unknown')}]: {content}\n"

        try:
            response = await self._summarizer.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3,
            )
            self._summary = response.content or ""
        except Exception:
            self._summary = f"(摘要生成失败，共 {len(early)} 条早期消息)"

        self._messages = recent

    @staticmethod
    def _count_tokens(messages: list[Message]) -> int:
        return sum(len(m.get("content", "") or "") for m in messages) // 4
