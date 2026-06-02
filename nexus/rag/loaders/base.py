"""BaseLoader — 文档加载器抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from nexus.rag.types import Document


class BaseLoader(ABC):
    """文档加载器抽象基类。

    每个加载器负责从特定来源（文件、目录、URL、数据库等）读取文档，
    返回 Document 对象列表。

    子类必须实现 load()。
    """

    @abstractmethod
    async def load(self, path: str) -> list[Document]:
        """从给定路径加载文档。

        Args:
            path: 文档来源路径（文件路径、目录路径、URL 等）

        Returns:
            Document 对象列表，每个包含原始文本和来源元数据
        """
        ...
