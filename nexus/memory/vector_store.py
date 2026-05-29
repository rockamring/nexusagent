"""VectorStoreMemory — 基于向量存储的长期记忆。

使用 ChromaDB 作为向量数据库，支持跨会话的语义检索。

原理：
- 每条消息通过 Embedding 模型向量化
- 存储到 ChromaDB 持久化集合中
- 检索时：对查询向量化 → 在 ChromaDB 中搜索最相似的 k 条记忆
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from nexus.core.types import Message
from nexus.memory.base import BaseMemory


class VectorStoreMemory(BaseMemory):
    """基于 ChromaDB 的长期向量记忆。

    支持跨会话持久化，通过语义相似度检索最相关的历史记忆。

    用法:
        memory = VectorStoreMemory(
            persist_dir="./agent_memory",
            collection_name="my_agent",
        )
        await memory.add({"role": "user", "content": "我喜欢 Python"})
        # ... 下次会话 ...
        ctx = await memory.get_context(query="编程语言")
    """

    def __init__(
        self,
        persist_dir: str = "./memory_data",
        collection_name: str = "agent_memory",
        embedding_fn: callable | None = None,
    ):
        import chromadb

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedding_fn = embedding_fn or self._default_embedding

    @staticmethod
    async def _default_embedding(text: str) -> list[float]:
        """默认的简单 Embedding（基于字符哈希）。

        生产环境应替换为真实的 Embedding 模型，如：
        - OpenAI text-embedding-3-small
        - sentence-transformers
        - ollama embedding
        """
        # 使用简单的词袋哈希向量作为占位实现
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        # 生成 384 维向量（兼容常见的 Embedding 维度）
        return [float(b) / 255.0 for b in h * 12][:384]

    async def add(self, message: Message) -> None:
        content = message.get("content")
        if not content:
            return

        embedding = await self._embedding_fn(content)

        self._collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding],
            documents=[content],
            metadatas=[{
                "role": message.get("role", "unknown"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }],
        )

    async def get_context(
        self,
        *,
        query: str | None = None,
        max_tokens: int | None = None,
        k: int = 5,
    ) -> list[Message]:
        if not query or self._collection.count() == 0:
            return []

        try:
            query_embedding = await self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, self._collection.count()),
            )
        except Exception:
            return []

        messages: list[Message] = []
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for doc, meta in zip(docs, metadatas):
            role = (meta or {}).get("role", "memory")
            messages.append({
                "role": "system",
                "content": f"[历史记忆 - {role}] {doc}",
            })

        return messages

    async def clear(self) -> None:
        # ChromaDB 不支持直接清空集合，需要删除后重建
        name = self._collection.name
        try:
            self._client.delete_collection(name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        """当前存储的记忆条数。"""
        return self._collection.count()
