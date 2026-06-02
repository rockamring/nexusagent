"""RAG 模块核心数据类型 — Document 和 SearchResult。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    """表示一个文本文档块。

    这是 RAG 管道中流通的基本单元：
    Loader 产出 Document → Splitter 变换 Document → KnowledgeStore 消费 Document。

    Attributes:
        content: 文档文本内容
        metadata: 附加元数据（来源路径、文件类型、块序号等）
    """

    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """KnowledgeStore / Retriever 返回的单条检索结果。

    Attributes:
        content: 检索到的文档块文本
        metadata: 原始文档的元数据（来源、文件类型等）
        score: 相似度分数（余弦距离，越低越相似：0=完全相同，2=完全相反）
    """

    content: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0
