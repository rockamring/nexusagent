"""Memory 系统测试。"""

import pytest

from nexus.memory.buffer import BufferMemory
from nexus.memory.composite import CompositeMemory


@pytest.mark.asyncio
async def test_buffer_memory_add_and_get():
    memory = BufferMemory(max_messages=5)

    for i in range(7):
        await memory.add({"role": "user", "content": f"消息{i}"})

    ctx = await memory.get_context()
    assert len(ctx) == 5
    assert ctx[0]["content"] == "消息2"
    assert ctx[-1]["content"] == "消息6"


@pytest.mark.asyncio
async def test_buffer_memory_reserve_system():
    memory = BufferMemory(max_messages=3, reserve_system=True)

    await memory.add({"role": "system", "content": "系统提示"})
    await memory.add({"role": "user", "content": "用户消息1"})
    await memory.add({"role": "user", "content": "用户消息2"})
    await memory.add({"role": "user", "content": "用户消息3"})

    ctx = await memory.get_context()
    assert len(ctx) == 4  # 1 system + 3 user
    assert ctx[0]["role"] == "system"


@pytest.mark.asyncio
async def test_buffer_clear():
    memory = BufferMemory()
    await memory.add({"role": "user", "content": "测试"})
    await memory.clear()
    ctx = await memory.get_context()
    assert len(ctx) == 0


@pytest.mark.asyncio
async def test_composite_memory():
    buf1 = BufferMemory(max_messages=3)
    buf2 = BufferMemory(max_messages=3)

    composite = CompositeMemory([buf1, buf2])

    for i in range(5):
        await composite.add({"role": "user", "content": f"msg{i}"})

    ctx = await composite.get_context()
    # 两个 buffer 各保留 3 条
    assert len(ctx) == 6

    await composite.clear()
    ctx = await composite.get_context()
    assert len(ctx) == 0
