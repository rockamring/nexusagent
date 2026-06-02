"""Embedding Provider 抽象基类。

所有 Embedding 服务必须实现 embed() 和 embed_batch() 两个方法。

用法:
    class MyEmbedding(BaseEmbeddingProvider):
        async def embed(self, text: str) -> list[float]:
            ...

        async def embed_batch(self, texts: list[str]) -> list[list[float]]:
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbeddingProvider(ABC):
    """Embedding Provider 抽象基类。

    将文本转换为向量表示，用于语义搜索和向量存储。
    子类必须实现 embed() 和 embed_batch()。
    """

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """将单条文本转换为向量。

        Args:
            text: 输入文本

        Returns:
            浮点数列表（向量），维度取决于模型
        """
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量将文本转换为向量。

        Args:
            texts: 输入文本列表

        Returns:
            向量列表，每个向量对应一条输入文本
        """
        ...
