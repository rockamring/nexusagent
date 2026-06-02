"""Embedding 系统 — 将文本转换为向量表示。"""

from nexus.embeddings.base import BaseEmbeddingProvider
from nexus.embeddings.providers.openai import OpenAIEmbeddingProvider

__all__ = [
    "BaseEmbeddingProvider",
    "OpenAIEmbeddingProvider",
]
