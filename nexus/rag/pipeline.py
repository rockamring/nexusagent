"""IngestionPipeline — 编排完整的文档采集流程：Load → Split → Embed → Store。

这是 RAG 模块的高层入口，组合加载器、切分器和知识库三个管道阶段。

用法:
    pipeline = IngestionPipeline(
        loader=DirectoryLoader(),
        splitter=RecursiveCharacterTextSplitter(chunk_size=500),
        knowledge_store=KnowledgeStore(embedding_provider=provider),
    )
    num_chunks = await pipeline.ingest_directory("./docs/")
    print(f"已采集 {num_chunks} 个块")
"""

from __future__ import annotations

from nexus.rag.knowledge_store import KnowledgeStore
from nexus.rag.loaders.base import BaseLoader
from nexus.rag.splitters.base import BaseSplitter


class IngestionPipeline:
    """文档采集流水线：Load → Split → Embed → Store。

    组合加载器、切分器和知识库三个独立组件，
    提供一键式文件/目录采集入口。
    """

    def __init__(
        self,
        loader: BaseLoader,
        splitter: BaseSplitter,
        knowledge_store: KnowledgeStore,
    ):
        """初始化流水线。

        Args:
            loader: 文档加载器（TextLoader / DirectoryLoader 等）
            splitter: 文本切分器（CharacterTextSplitter / RecursiveCharacterTextSplitter 等）
            knowledge_store: 用于嵌入和存储的知识库
        """
        self._loader = loader
        self._splitter = splitter
        self._store = knowledge_store

    @property
    def store(self) -> KnowledgeStore:
        """底层的 KnowledgeStore 实例。"""
        return self._store

    async def ingest_file(self, path: str) -> int:
        """采集单个文件：加载 → 切分 → 存储。

        Args:
            path: 文件路径

        Returns:
            存储的块数量
        """
        documents = await self._loader.load(path)
        chunks = self._splitter.split_documents(documents)
        return await self._store.add_documents(chunks)

    async def ingest_directory(self, dir_path: str) -> int:
        """采集目录中所有文件：加载 → 切分 → 存储。

        Args:
            dir_path: 目录路径

        Returns:
            所有文件合计存储的块数量
        """
        documents = await self._loader.load(dir_path)
        chunks = self._splitter.split_documents(documents)
        return await self._store.add_documents(chunks)
