"""RecursiveCharacterTextSplitter — 递归层级文本切分器。

RAG 模块最核心的组件。算法：
1. 用最高优先级分隔符（如 "\\n\\n" 段落）尝试切分文本
2. 若某块仍超过 chunk_size，递归用下一级分隔符（如 "\\n" 行）切分
3. 继续降级：中文句号 → 英文句号+空格 → 空格 → 字符
4. 无分隔符可用时，按字符数硬切分

这样生成的块尽量保持自然语义边界（段落、句子），只在必要时做硬断。
"""

from __future__ import annotations

import re

from nexus.rag.splitters.base import BaseSplitter
from nexus.rag.types import Document


class RecursiveCharacterTextSplitter(BaseSplitter):
    """递归按分隔符优先级切分文本，尽量保持语义边界。

    默认分隔符优先级（与 LangChain 的经典顺序一致）：
        "\\n\\n"（段落）→ "\\n"（行）→ "。"（中文句号）
        → ". "（英文句子）→ " "（词）→ ""（字符）

    用法:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(long_text)
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", "。", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ):
        """初始化递归切分器。

        Args:
            chunk_size: 每块最大字符数
            chunk_overlap: 相邻块间重叠的字符数
            separators: 有序分隔符列表，默认 DEFAULT_SEPARATORS
                       空字符串 "" 必须为最后一个分隔符（字符级切分）

        Raises:
            ValueError: chunk_overlap >= chunk_size
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) 必须小于 chunk_size ({chunk_size})"
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators or self.DEFAULT_SEPARATORS

    def split_text(self, text: str, metadata: dict | None = None) -> list[Document]:
        """递归切分文本。"""
        meta = dict(metadata or {})
        raw_chunks = self._split_recursive(text, self._separators)
        return self._merge_with_overlap(raw_chunks, meta)

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """核心递归切分算法。

        Args:
            text: 待切分文本
            separators: 从当前索引到末尾的剩余分隔符列表

        Returns:
            文本块列表，每块长度 <= chunk_size（或尽力接近）
        """
        # 递归基：文本已足够短
        if len(text) <= self._chunk_size:
            return [text] if text else []

        separator = separators[0]
        remaining = separators[1:]

        if separator == "":
            # 最后手段：无剩余递归层级，按字符硬切
            return self._character_split(text)

        splits = self._split_with_separator(text, separator)

        result: list[str] = []
        for split_text in splits:
            if len(split_text) <= self._chunk_size:
                result.append(split_text)
            elif remaining:
                # 递归用下一级分隔符切分
                sub_chunks = self._split_recursive(split_text, remaining)
                result.extend(sub_chunks)
            else:
                # 无更多分隔符，按字符硬切
                result.extend(self._character_split(split_text))

        return result

    def _split_with_separator(self, text: str, separator: str) -> list[str]:
        """按分隔符切分文本，分隔符保留在块末尾。

        使用正则 lookbehind 在不消耗分隔符的情况下切分，
        这样每个块末尾保留换行/句号等语义标记。
        """
        escaped = re.escape(separator)
        parts = re.split(f"(?<={escaped})", text)
        return [p for p in parts if p]

    def _character_split(self, text: str) -> list[str]:
        """按 chunk_size 字符硬切分，不保留语义边界（最后手段）。"""
        return [text[i:i + self._chunk_size] for i in range(0, len(text), self._chunk_size)]

    def _merge_with_overlap(self, chunks: list[str], metadata: dict) -> list[Document]:
        """为相邻块添加重叠前缀，提升检索连续性。

        每块（除第一块外）以上一块末尾文本为前缀，
        使块与块之间保持 chunk_overlap 字符的上下文重叠。
        """
        if not chunks:
            return []

        documents: list[Document] = []

        for i, chunk in enumerate(chunks):
            chunk_content = chunk
            if i > 0 and self._chunk_overlap > 0:
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk[-self._chunk_overlap:]
                # 仅当重叠文本与当前块开头不同时才添加（避免冗余重复）
                if overlap_text and overlap_text != chunk[:len(overlap_text)]:
                    chunk_content = overlap_text + chunk

            documents.append(Document(
                content=chunk_content,
                metadata={**metadata, "chunk_index": i, "total_chunks": len(chunks)},
            ))

        return documents
