"""文本切分器 — 将文档切分为适合嵌入和检索的小块。"""

from nexus.rag.splitters.base import BaseSplitter
from nexus.rag.splitters.character_splitter import CharacterTextSplitter
from nexus.rag.splitters.recursive_splitter import RecursiveCharacterTextSplitter

__all__ = ["BaseSplitter", "CharacterTextSplitter", "RecursiveCharacterTextSplitter"]
