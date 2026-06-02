"""CharacterTextSplitter — 固定大小字符级文本切分器。

按 chunk_size 切分文本，保留 chunk_overlap 重叠。尽量在分隔符边界断开。
"""

from __future__ import annotations

from nexus.rag.splitters.base import BaseSplitter
from nexus.rag.types import Document


class CharacterTextSplitter(BaseSplitter):
    """按字符数固定大小切分文本，块间有重叠。

    用法:
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator="\n")
        chunks = splitter.split_text(long_text, metadata={"source": "doc.txt"})
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n",
    ):
        """初始化字符切分器。

        Args:
            chunk_size: 每块最大字符数
            chunk_overlap: 相邻块间重叠的字符数
            separator: 优先在此分隔符处断开

        Raises:
            ValueError: chunk_overlap >= chunk_size
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) 必须小于 chunk_size ({chunk_size})"
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separator = separator

    def split_text(self, text: str, metadata: dict | None = None) -> list[Document]:
        """按固定大小切分文本。"""
        if not text:
            return []

        meta = dict(metadata or {})
        chunks: list[Document] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self._chunk_size, text_len)

            # 在 chunk 边界附近寻找更自然的分隔点
            if end < text_len:
                sep_pos = text.rfind(self._separator, start, end)
                if sep_pos != -1 and sep_pos > start + self._chunk_size // 2:
                    end = sep_pos + len(self._separator)

            chunk_text = text[start:end]
            chunks.append(Document(
                content=chunk_text,
                metadata={**meta, "chunk_index": len(chunks)},
            ))

            # 下一块的起始位置（考虑重叠）
            start = end - self._chunk_overlap
            if start >= text_len or end >= text_len:
                break

        return chunks
