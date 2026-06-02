"""Retriever — 检索器，封装 KnowledgeStore 提供格式化输出。

当前为薄封装层，设计上可扩展：
- 重排序：使用交叉编码器对 top-k 结果重新评分
- 混合检索：结合稠密向量 + 稀疏关键词（BM25）
- 元数据过滤：检索前/后按条件筛选
- 查询改写：基于 LLM 的查询扩展/拆解
"""

from __future__ import annotations

from nexus.rag.knowledge_store import KnowledgeStore
from nexus.rag.types import SearchResult


class Retriever:
    """检索器，封装 KnowledgeStore 并提供格式化输出。

    用法:
        retriever = Retriever(knowledge_store)
        results = await retriever.retrieve("如何安装?", k=5)
        context = await retriever.retrieve_formatted("如何安装?", k=5)
    """

    def __init__(self, knowledge_store: KnowledgeStore):
        """初始化检索器。

        Args:
            knowledge_store: 底层 KnowledgeStore 实例
        """
        self._store = knowledge_store

    async def retrieve(self, query: str, k: int = 5) -> list[SearchResult]:
        """检索与查询相关的文档块。

        Args:
            query: 查询文本
            k: 返回结果数

        Returns:
            按相关度排序的 SearchResult 列表
        """
        return await self._store.search(query, k=k)

    async def retrieve_formatted(
        self,
        query: str,
        k: int = 5,
        include_scores: bool = False,
    ) -> str:
        """检索并格式化为可直接注入 LLM 提示词的上下文字符串。

        Args:
            query: 查询文本
            k: 返回结果数
            include_scores: 是否在输出中包含相似度分数

        Returns:
            格式化后的上下文字符串

        输出示例:
            [Source: /docs/readme.md]
            This is the first chunk of relevant content...

            ---

            [Source: /docs/guide.md, chunk 3/10]
            This is the second chunk...
        """
        results = await self.retrieve(query, k=k)
        if not results:
            return "未找到相关文档。"

        parts: list[str] = []
        for i, r in enumerate(results, start=1):
            source = r.metadata.get("source", r.metadata.get("file_name", "unknown"))
            chunk_info = ""
            if "chunk_index" in r.metadata:
                total = r.metadata.get("total_chunks", "?")
                chunk_info = f", chunk {r.metadata['chunk_index'] + 1}/{total}"

            header = f"[Source: {source}{chunk_info}]"
            if include_scores:
                header += f" (score: {r.score:.3f})"

            parts.append(f"{header}\n{r.content}")

        return "\n\n---\n\n".join(parts)
