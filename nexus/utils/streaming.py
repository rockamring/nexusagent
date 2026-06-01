"""流式处理工具。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator


async def merge_streams(*streams: AsyncIterator) -> AsyncIterator:
    """合并多个异步流，按完成顺序产出。

    用于并行 Agent 执行时的流式输出合并。
    """
    queue: asyncio.Queue = asyncio.Queue()

    async def _feed(stream: AsyncIterator):
        async for item in stream:
            await queue.put(item)
        await queue.put(None)  # sentinel

    tasks = [asyncio.create_task(_feed(s)) for s in streams]
    done_count = 0

    while done_count < len(streams):
        item = await queue.get()
        if item is None:
            done_count += 1
        else:
            yield item

    for t in tasks:
        t.cancel()
