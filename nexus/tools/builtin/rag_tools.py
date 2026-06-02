"""RAG 工具 — search_knowledge_base 和 index_document。

通过 set_rag_retriever() 注入 Retriever 实例后，
Agent 即可通过工具调用检索和索引文档。

用法:
    from nexus.rag import Retriever, KnowledgeStore
    from nexus.tools.builtin.rag_tools import set_rag_retriever

    store = KnowledgeStore(embedding_provider=provider)
    retriever = Retriever(store)
    set_rag_retriever(retriever)

    # Agent 现在可以调用 search_knowledge_base 和 index_document
"""

from __future__ import annotations

from nexus.tools.base import Tool

# 模块级 Retriever 引用，由应用在初始化时注入
_retriever = None


def set_rag_retriever(retriever) -> None:
    """设置全局 Retriever 实例，激活 RAG 工具。

    应在创建 Agent 之前调用。

    Args:
        retriever: nexus.rag.retriever.Retriever 实例
    """
    global _retriever
    _retriever = retriever


def _get_retriever():
    """获取当前 Retriever，未配置时给出明确错误提示。"""
    if _retriever is None:
        raise RuntimeError(
            "RAG 工具未配置。请在初始化时调用 set_rag_retriever(retriever)。"
        )
    return _retriever


@Tool.from_function(
    name="search_knowledge_base",
    description=(
        "在知识库中搜索与查询相关的文档。"
        "当你需要从已索引的文档中查找信息时使用此工具。"
        "返回最相关的文本块及其来源信息。"
    ),
)
async def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """在知识库中搜索相关文档块。

    Args:
        query: 查询文本，请包含具体的关键词
        top_k: 返回结果数（默认 5）

    Returns:
        带来源信息的格式化检索结果
    """
    retriever = _get_retriever()
    return await retriever.retrieve_formatted(query, k=top_k, include_scores=True)


@Tool.from_function(
    name="index_document",
    description=(
        "将文件或目录索引到知识库中供后续检索。"
        "支持的文本格式包括 .txt/.md/.py/.json/.csv 等。"
        "返回索引的文档块数量。"
    ),
)
async def index_document(path: str) -> str:
    """将文档索引到知识库。

    Args:
        path: 待索引的文件或目录路径

    Returns:
        包含索引块数量的状态消息
    """
    from pathlib import Path

    from nexus.rag.loaders.directory_loader import DirectoryLoader
    from nexus.rag.loaders.text_loader import TextLoader
    from nexus.rag.pipeline import IngestionPipeline

    retriever = _get_retriever()

    target = Path(path).resolve()
    if not target.exists():
        return f"错误: 路径不存在: {path}"

    if target.is_file():
        loader = TextLoader()
    else:
        loader = DirectoryLoader()

    pipeline = IngestionPipeline(
        loader=loader,
        splitter=_get_default_splitter(),
        knowledge_store=retriever._store,
    )

    if target.is_file():
        count = await pipeline.ingest_file(str(target))
        return f"成功索引文件 '{target.name}': {count} 个块已存储。"
    else:
        count = await pipeline.ingest_directory(str(target))
        return f"成功索引目录 '{target.name}': {count} 个块已存储。"


def _get_default_splitter():
    """获取索引操作使用的默认切分器。"""
    from nexus.rag.splitters.recursive_splitter import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
