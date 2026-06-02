"""RAG（检索增强生成）模块。

组件:
- loaders:    文档加载器（TextLoader, DirectoryLoader）
- splitters:  文本切分器（CharacterTextSplitter, RecursiveCharacterTextSplitter）
- knowledge_store: 基于 ChromaDB 的文档向量存储
- retriever:  检索器（封装 KnowledgeStore，提供格式化输出）
- pipeline:   采集流水线（Load → Split → Embed → Store）
"""

from nexus.rag.knowledge_store import KnowledgeStore
from nexus.rag.loaders.base import BaseLoader
from nexus.rag.loaders.directory_loader import DirectoryLoader
from nexus.rag.loaders.text_loader import TextLoader
from nexus.rag.pipeline import IngestionPipeline
from nexus.rag.retriever import Retriever
from nexus.rag.splitters.base import BaseSplitter
from nexus.rag.splitters.character_splitter import CharacterTextSplitter
from nexus.rag.splitters.recursive_splitter import RecursiveCharacterTextSplitter
from nexus.rag.types import Document, SearchResult

__all__ = [
    "Document",
    "SearchResult",
    "BaseLoader",
    "TextLoader",
    "DirectoryLoader",
    "BaseSplitter",
    "CharacterTextSplitter",
    "RecursiveCharacterTextSplitter",
    "KnowledgeStore",
    "Retriever",
    "IngestionPipeline",
]
