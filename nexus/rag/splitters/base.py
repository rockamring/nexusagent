"""BaseSplitter — 文本切分器抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from nexus.rag.types import Document


class BaseSplitter(ABC):
    """文本切分器抽象基类。

    将文档切分为更小的块，以便嵌入和检索。
    子类必须实现 split_text()。
    """

    @abstractmethod
    def split_text(self, text: str, metadata: dict | None = None) -> list[Document]:
        """将原始文本切分为 Document 块。

        Args:
            text: 待切分的文本
            metadata: 附加到每个输出 Document 的元数据（会补充 chunk_index）

        Returns:
            Document 对象列表，每个为输入文本的一个块
        """
        ...

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """将多个 Document 切分为更小的 Document 块。

        便利方法：遍历每个 Document，对其内容切分，保留并补充原始元数据。

        Args:
            documents: 待切分的 Document 列表

        Returns:
            切分后的 Document 块列表
        """
        result: list[Document] = []
        for doc in documents:
            chunks = self.split_text(doc.content, metadata=doc.metadata)
            result.extend(chunks)
        return result
