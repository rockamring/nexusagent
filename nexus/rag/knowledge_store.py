"""KnowledgeStore — 基于 ChromaDB 的文档向量存储。

专为文档块（非对话消息）设计。复用 BaseEmbeddingProvider 依赖注入模式。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from nexus.embeddings.base import BaseEmbeddingProvider
from nexus.rag.types import Document, SearchResult


class KnowledgeStore:
    """基于 ChromaDB 的文档向量存储。

    与 VectorStoreMemory 的关键区别：
    - 存储 Document 对象（content + metadata）而非 Message 对象（role + content）
    - 支持按 source 批量删除（delete_by_source）
    - 返回 SearchResult（含相似度分数）而非 Message 列表
    - 专为读多写少的检索场景设计

    用法:
        store = KnowledgeStore(
            persist_dir="./knowledge_base",
            collection_name="my_docs",
            embedding_provider=provider,
        )

        docs = [Document(content="...", metadata={"source": "readme.md"})]
        store.add_documents(docs)

        results = await store.search("如何安装?", k=5)
        for r in results:
            print(f"[{r.score:.3f}] {r.content[:100]}...")
    """

    def __init__(
        self,
        persist_dir: str = "./knowledge_data",
        collection_name: str = "knowledge_base",
        embedding_provider: BaseEmbeddingProvider | None = None,
    ):
        import chromadb

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedding_provider = embedding_provider

    async def _embed(self, text: str) -> list[float]:
        """获取文本向量，优先用 provider，否则用 SHA-256 回退。"""
        if self._embedding_provider:
            return await self._embedding_provider.embed(text)
        return await self._default_embedding(text)

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量获取文本向量。"""
        if self._embedding_provider:
            return await self._embedding_provider.embed_batch(texts)
        results: list[list[float]] = []
        for t in texts:
            results.append(await self._default_embedding(t))
        return results

    @staticmethod
    async def _default_embedding(text: str) -> list[float]:
        """字符 bigram 频率向量（384 维），作为默认回退嵌入。

        原理：提取文本中的字符 bigram，哈希映射到 384 个桶中，
        归一化后得到频率向量。相似文本共享更多 bigram，余弦距离更近。

        仅用于测试或无 API Key 的场景，生产环境应提供真实 EmbeddingProvider。
        """
        import hashlib
        dim = 384
        vector = [0.0] * dim
        text_lower = text.lower()
        for i in range(len(text_lower) - 1):
            bigram = text_lower[i:i + 2]
            bucket = int(hashlib.md5(bigram.encode()).hexdigest(), 16) % dim
            vector[bucket] += 1.0
        total = sum(vector)
        if total > 0:
            vector = [v / total for v in vector]
        return vector

    async def add_documents(self, docs: list[Document]) -> int:
        """嵌入并存储文档块。

        Args:
            docs: 待存储的 Document 列表

        Returns:
            成功添加的文档数
        """
        if not docs:
            return 0

        contents = [doc.content for doc in docs]
        embeddings = await self._embed_batch(contents)

        ids = [str(uuid.uuid4()) for _ in docs]
        metadatas = []
        for doc in docs:
            meta = dict(doc.metadata)
            meta["_stored_at"] = datetime.now(timezone.utc).isoformat()
            metadatas.append(self._sanitize_metadata(meta))

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas,
        )
        return len(docs)

    async def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """语义检索与查询最相关的文档块。

        异步嵌入查询文本后调用 ChromaDB 同步查询（本地操作）。

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            SearchResult 列表，按相关度排序（score 越低越相似）
        """
        if not query or self._collection.count() == 0:
            return []

        query_embedding = await self._embed(query)

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, self._collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        search_results: list[SearchResult] = []
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metadatas, distances):
            clean_meta = {k: v for k, v in (meta or {}).items() if not k.startswith("_")}
            search_results.append(SearchResult(
                content=doc or "",
                metadata=clean_meta,
                score=dist,
            ))

        return search_results

    def delete_by_source(self, source: str) -> int:
        """删除指定来源的所有文档块。

        优先按 source 精确匹配（绝对路径），匹配不到则回退按 file_name 匹配。
        这种两层回退使得调用方既可以用完整路径也可以用文件名删除。

        Args:
            source: 来源标识（完整路径或文件名）

        Returns:
            删除的块数量
        """
        try:
            ids_to_delete = self._find_ids_by_metadata("source", source)
            if not ids_to_delete:
                ids_to_delete = self._find_ids_by_metadata("file_name", source)
            if ids_to_delete:
                self._collection.delete(ids=ids_to_delete)
            return len(ids_to_delete)
        except Exception:
            return 0

    def _find_ids_by_metadata(self, key: str, value: str) -> list[str]:
        """按元数据字段精确匹配，返回对应的文档 ID 列表。"""
        try:
            results = self._collection.get(
                where={key: value},
                include=[],
            )
            return results.get("ids", [])
        except Exception:
            return []

    def clear(self) -> None:
        """清空所有文档。"""
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
        """当前存储的文档块数。"""
        return self._collection.count()

    @staticmethod
    def _sanitize_metadata(meta: dict) -> dict:
        """确保元数据值为 ChromaDB 兼容类型（str/int/float/bool）。

        ChromaDB 不支持列表或字典作为元数据值，需转换为 JSON 字符串。
        """
        safe = {}
        for key, value in meta.items():
            if isinstance(value, (str, int, float, bool)):
                safe[key] = value
            elif value is None:
                safe[key] = ""
            elif isinstance(value, (list, dict)):
                safe[key] = json.dumps(value, ensure_ascii=False)
            else:
                safe[key] = str(value)
        return safe
