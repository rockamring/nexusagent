"""Example 9: 向量长期记忆 — 演示基于 ChromaDB 的语义记忆检索。

无需 API Key 即可运行（使用 SHA-256 哈希作为回退嵌入）。

演示:
    1. VectorStoreMemory vs BufferMemory 的区别
    2. 存储会话记忆 → 语义检索历史
    3. 跨"会话"持久化（模拟）
    4. 与 Agent 集成

运行:
    python examples/09_vector_memory.py
"""

import asyncio
import tempfile
from pathlib import Path

from nexus.memory.buffer import BufferMemory
from nexus.memory.vector_store import VectorStoreMemory

# ── 演示 1: BufferMemory vs VectorStoreMemory ─────

async def demo_comparison():
    """对比两种记忆模式的特性和适用场景。"""
    print("=" * 60)
    print("演示 1: BufferMemory vs VectorStoreMemory")
    print("=" * 60)
    print()

    print("┌────────────────────┬──────────────────────────────┐")
    print("│ BufferMemory       │ VectorStoreMemory            │")
    print("├────────────────────┼──────────────────────────────┤")
    print("│ 短期记忆           │ 长期记忆                     │")
    print("│ 最近 N 条消息      │ 全部历史消息                 │")
    print("│ 按时间顺序         │ 按语义相关度                 │")
    print("│ 内存中（进程内）   │ ChromaDB 持久化              │")
    print("│ 适合单轮对话       │ 适合跨会话 + 大规模记忆      │")
    print("│ max_messages 限制  │ k 参数控制检索数量           │")
    print("└────────────────────┴──────────────────────────────┘")
    print()

    # BufferMemory 示例
    buffer = BufferMemory(max_messages=4)
    await buffer.add({"role": "user", "content": "我喜欢 Python"})
    await buffer.add({"role": "assistant", "content": "Python 是很好的选择！"})
    await buffer.add({"role": "user", "content": "我也喜欢 JavaScript"})
    await buffer.add({"role": "assistant", "content": "JS 在前端很强大。"})
    await buffer.add({"role": "user", "content": "推荐一个框架"})

    ctx = await buffer.get_context()
    msg_count = len(buffer._messages)
    print("BufferMemory.get_context() 返回最近消息:")
    for msg in ctx:
        print(f"    [{msg['role']}] {msg['content'][:50]}")
    print(f"    (共添加 5 条，max_messages=4，自动丢弃了最早 1 条，保留 {msg_count} 条)")
    print()

    # VectorStoreMemory 示例
    tmpdir = Path(tempfile.mkdtemp(prefix="nexus_vecmem_example_"))
    vecmem = VectorStoreMemory(
        persist_dir=str(tmpdir / "vec_data"),
        collection_name="demo_comparison",
    )

    messages = [
        {"role": "user", "content": "我喜欢用 Python 做数据分析和机器学习。"},
        {"role": "assistant", "content": "Python 在数据科学领域确实很强，pandas 和 scikit-learn 都很好用。"},
        {"role": "user", "content": "最近在学 Rust，内存安全的概念很有意思。"},
        {"role": "assistant", "content": "Rust 的所有权系统是它最大的特色，学习曲线陡但收获大。"},
        {"role": "user", "content": "周末经常去爬山，减压效果很好。"},
        {"role": "assistant", "content": "户外运动确实能放松身心，注意安全。"},
        {"role": "user", "content": "Pandas 的 DataFrame 操作有没有好的学习资源？"},
        {"role": "assistant", "content": "推荐看 pandas 官方文档的 Getting Started 部分，很全面。"},
    ]

    for msg in messages:
        await vecmem.add(msg)

    print(f"VectorStoreMemory 共存储 {vecmem.count} 条记忆")
    print()

    # 语义检索：搜索"编程"相关 → 应该返回 Python/Rust 的记忆
    results = await vecmem.get_context(query="编程语言学习", k=3)
    print("搜索 '编程语言学习' → 返回:")
    for msg in results:
        print(f"    {msg['content'][:80]}")
    print()

    # 语义检索：搜索"周末" → 应该返回户外运动的记忆
    results = await vecmem.get_context(query="周末活动", k=3)
    print("搜索 '周末活动' → 返回:")
    for msg in results:
        print(f"    {msg['content'][:80]}")
    print()

    # 语义检索：搜索"数据分析" → 应该返回 pandas 相关的记忆
    results = await vecmem.get_context(query="数据分析工具", k=3)
    print("搜索 '数据分析工具' → 返回:")
    for msg in results:
        print(f"    {msg['content'][:80]}")
    print()

    await vecmem.clear()

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


# ── 演示 2: 跨会话持久化 ─────────────────────────

async def demo_persistence():
    """模拟两次"会话"，展示向量记忆的持久化能力。"""
    print("=" * 60)
    print("演示 2: 跨会话持久化")
    print("=" * 60)
    print()

    tmpdir = Path(tempfile.mkdtemp(prefix="nexus_vecmem_persist_"))
    persist_dir = str(tmpdir / "persist_data")

    # ── 会话 1: 存入知识 ──
    print("── 会话 1: 存入知识 ──")
    memory1 = VectorStoreMemory(
        persist_dir=persist_dir,
        collection_name="agent_knowledge",
    )

    knowledge = [
        {"role": "user", "content": "项目使用 PostgreSQL 作为主数据库，端口 5432。"},
        {"role": "assistant", "content": "已记录。PostgreSQL 端口 5432，请确保防火墙已开放。"},
        {"role": "user", "content": "API 服务部署在 8080 端口，使用 FastAPI 框架。"},
        {"role": "assistant", "content": "了解。FastAPI 在 8080 端口，需要配置反向代理。"},
        {"role": "user", "content": "日志文件位于 /var/log/myapp/，按天滚动。"},
        {"role": "assistant", "content": "日志路径已记录。建议定期清理避免磁盘占满。"},
        {"role": "user", "content": "Redis 缓存使用 6379 端口，密码在 .env 文件的 REDIS_PASSWORD。"},
        {"role": "assistant", "content": "Redis 配置已记录。注意不要将 .env 提交到版本控制。"},
    ]

    for msg in knowledge:
        await memory1.add(msg)

    print(f"  已存储 {memory1.count} 条知识")
    print()

    # ── 会话 2: 检索知识（新实例，相同 persist_dir）──
    print("── 会话 2: 检索知识（新实例，相同 persist_dir）──")
    memory2 = VectorStoreMemory(
        persist_dir=persist_dir,
        collection_name="agent_knowledge",
    )

    print(f"  加载到 {memory2.count} 条历史知识")
    print()

    queries = [
        ("数据库配置", "数据库相关"),
        ("API 端口号", "API 相关"),
        ("日志在哪里", "日志相关"),
        ("Redis 密码", "Redis 相关"),
    ]

    for query, category in queries:
        results = await memory2.get_context(query=query, k=1)
        if results:
            print(f"  查询 '{query}' ({category}): {results[0]['content'][:80]}")
    print()

    print("VectorStoreMemory 的关键能力:")
    print("  1. 记忆持久化到磁盘（ChromaDB PersistentClient）")
    print("  2. 新会话自动加载历史记忆")
    print("  3. 语义检索 vs 关键词检索（'数据库配置' 匹配 'PostgreSQL'）")
    print("  4. 不像 BufferMemory 受消息数量限制")
    print()

    await memory1.clear()

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


# ── 演示 3: 与 Agent 集成 ─────────────────────────

async def demo_agent_integration():
    """展示 VectorStoreMemory 如何注入 Agent 作为上下文。"""
    print("=" * 60)
    print("演示 3: 与 Agent 集成")
    print("=" * 60)
    print()

    print("集成方式与 BufferMemory 完全相同：")
    print()
    print("  from nexus.memory.vector_store import VectorStoreMemory")
    print()
    print("  memory = VectorStoreMemory(")
    print("      persist_dir='./agent_memory',")
    print("      collection_name='my_agent',")
    print("  )")
    print()
    print("  agent = ReActAgent(")
    print("      name='助手',")
    print("      llm=llm,")
    print("      memory=memory,  # ← 直接注入，与其他 Memory 完全兼容")
    print("  )")
    print()

    print("Agent 每次 run() 时：")
    print("  1. 用 user input 作为 query 检索最相关的 k 条历史记忆")
    print("  2. 将检索结果以 [历史记忆 - role] 格式注入 system prompt")
    print("  3. LLM 看到这些历史记忆后可参考它们回答问题")
    print()

    # 实际演示（使用 Mock 数据）
    tmpdir = Path(tempfile.mkdtemp(prefix="nexus_vecmem_agent_"))
    memory = VectorStoreMemory(
        persist_dir=str(tmpdir / "agent_data"),
        collection_name="demo_agent",
    )

    # 模拟之前会话中存储的信息
    await memory.add({"role": "user", "content": "我叫张三，是一名后端工程师，主要用 Go 和 Python。"})
    await memory.add({"role": "user", "content": "我正在开发一个微服务项目，包含 5 个服务。"})
    await memory.add({"role": "user", "content": "我们的数据库是 PostgreSQL 15，部署在 AWS RDS 上。"})

    # 新会话中，Agent 自动获取相关上下文
    ctx = await memory.get_context(query="我的技术栈", k=3)
    print("Agent 自动注入的上下文（基于用户输入 '我的技术栈是什么？'）:")
    for msg in ctx:
        print(f"    {msg['content']}")
    print()

    print("LLM 看到这些记忆后，可以回答：")
    print("  '根据之前的对话，您是后端工程师，主要使用 Go 和 Python...'")
    print()

    await memory.clear()

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


# ── 主流程 ──────────────────────────────────────

async def main():
    await demo_comparison()
    await demo_persistence()
    await demo_agent_integration()
    print("=" * 60)
    print("全部向量记忆演示完成。")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
