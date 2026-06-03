"""RAG 模块测试。"""

import os
import tempfile
from pathlib import Path

import pytest

from nexus.embeddings.base import BaseEmbeddingProvider
from nexus.rag.knowledge_store import KnowledgeStore
from nexus.rag.loaders.directory_loader import DirectoryLoader
from nexus.rag.loaders.text_loader import TextLoader
from nexus.rag.pipeline import IngestionPipeline
from nexus.rag.retriever import Retriever
from nexus.rag.splitters.character_splitter import CharacterTextSplitter
from nexus.rag.splitters.recursive_splitter import RecursiveCharacterTextSplitter
from nexus.rag.types import Document, SearchResult

# ── Mock Embedding Provider ──────────────────────

class MockEmbeddingProvider(BaseEmbeddingProvider):
    """模拟 Embedding Provider，返回固定维度向量。"""

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.embed_calls: list[str] = []
        self.batch_calls: list[list[str]] = []

    async def embed(self, text: str) -> list[float]:
        self.embed_calls.append(text)
        return [float(hash(text) % 100) / 100.0 for _ in range(self.dim)]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls.append(texts)
        return [[float(hash(t) % 100) / 100.0 for _ in range(self.dim)] for t in texts]


# ── Fixtures ─────────────────────────────────────

@pytest.fixture
def temp_text_file():
    """创建临时文本文件用于测试。"""
    content = "Hello\n\nThis is a test document.\nIt has multiple lines.\n\nGoodbye."
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def temp_directory():
    """创建含多个文本文件的临时目录。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        (base / "a.txt").write_text("File A content.", encoding="utf-8")
        (base / "b.md").write_text("# File B\n\nMarkdown content here.", encoding="utf-8")
        (base / "sub").mkdir()
        (base / "sub" / "c.py").write_text("# File C\nprint('hello')", encoding="utf-8")
        yield tmpdir


# ── Document 测试 ────────────────────────────────

def test_document_default_metadata():
    """Document 默认 metadata 为空字典。"""
    doc = Document(content="test")
    assert doc.content == "test"
    assert doc.metadata == {}


def test_document_custom_metadata():
    """Document 可携带自定义 metadata。"""
    doc = Document(content="hello", metadata={"source": "test.txt", "page": 1})
    assert doc.metadata["source"] == "test.txt"
    assert doc.metadata["page"] == 1


def test_search_result():
    """SearchResult 包含 content, metadata, score。"""
    r = SearchResult(content="chunk", metadata={"source": "a.txt"}, score=0.15)
    assert r.content == "chunk"
    assert r.score == 0.15
    assert r.metadata["source"] == "a.txt"


# ── TextLoader 测试 ──────────────────────────────

@pytest.mark.asyncio
async def test_text_loader_loads_file(temp_text_file):
    """TextLoader 正确加载文本文件。"""
    loader = TextLoader()
    docs = await loader.load(temp_text_file)
    assert len(docs) == 1
    assert "test document" in docs[0].content
    assert docs[0].metadata["file_type"] == ".txt"
    assert docs[0].metadata["source"] == str(Path(temp_text_file).resolve())


@pytest.mark.asyncio
async def test_text_loader_file_not_found():
    """加载不存在的文件抛出 FileNotFoundError。"""
    loader = TextLoader()
    with pytest.raises(FileNotFoundError):
        await loader.load("/nonexistent/file.txt")


@pytest.mark.asyncio
async def test_text_loader_encoding_fallback(tmp_path):
    """非 UTF-8 文件通过回退链正确读取。"""
    file_path = tmp_path / "gbk_file.txt"
    content = "中文测试内容"
    file_path.write_bytes(content.encode("gbk"))
    loader = TextLoader(encoding="utf-8", fallback_encodings=["gbk"])
    docs = await loader.load(str(file_path))
    assert content in docs[0].content


# ── DirectoryLoader 测试 ─────────────────────────

@pytest.mark.asyncio
async def test_directory_loader_loads_all_files(temp_directory):
    """DirectoryLoader 递归加载目录中所有支持的文件。"""
    loader = DirectoryLoader()
    docs = await loader.load(temp_directory)
    assert len(docs) == 3
    sources = {d.metadata["file_name"] for d in docs}
    assert sources == {"a.txt", "b.md", "c.py"}


@pytest.mark.asyncio
async def test_directory_loader_non_recursive(temp_directory):
    """recursive=False 仅加载顶层文件。"""
    loader = DirectoryLoader(recursive=False)
    docs = await loader.load(temp_directory)
    assert len(docs) == 2


@pytest.mark.asyncio
async def test_directory_loader_glob_pattern(temp_directory):
    """glob_pattern 仅加载匹配的文件。"""
    loader = DirectoryLoader(glob_pattern="**/*.md")
    docs = await loader.load(temp_directory)
    assert len(docs) == 1
    assert docs[0].metadata["file_name"] == "b.md"


@pytest.mark.asyncio
async def test_directory_loader_not_a_directory():
    """非目录路径抛出 NotADirectoryError。"""
    loader = DirectoryLoader()
    with pytest.raises(NotADirectoryError):
        await loader.load("/nonexistent/directory")


# ── CharacterTextSplitter 测试 ───────────────────

def test_character_splitter_basic():
    """CharacterTextSplitter 按 chunk_size 切分长文本。"""
    text = "a" * 3000
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    assert len(chunks) >= 3


def test_character_splitter_metadata():
    """切分器在 metadata 中添加 chunk_index。"""
    splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    text = "x" * 300
    chunks = splitter.split_text(text, metadata={"source": "test.txt"})
    assert len(chunks) >= 3
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_index"] == i
        assert chunk.metadata["source"] == "test.txt"


def test_character_splitter_empty():
    """空文本返回空列表。"""
    splitter = CharacterTextSplitter()
    assert splitter.split_text("") == []


def test_character_splitter_overlap_validation():
    """chunk_overlap >= chunk_size 抛出 ValueError。"""
    with pytest.raises(ValueError, match="overlap"):
        CharacterTextSplitter(chunk_size=100, chunk_overlap=100)


def test_character_splitter_split_documents():
    """split_documents 批量处理多个 Document。"""
    splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    docs = [
        Document(content="a" * 200, metadata={"source": "a.txt"}),
        Document(content="b" * 200, metadata={"source": "b.txt"}),
    ]
    chunks = splitter.split_documents(docs)
    assert len(chunks) == 4
    for chunk in chunks:
        assert "chunk_index" in chunk.metadata


# ── RecursiveCharacterTextSplitter 测试 ─────────

def test_recursive_splitter_short_text():
    """短于 chunk_size 的文本保持完整。"""
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = splitter.split_text(text)
    assert len(chunks) == 1


def test_recursive_splitter_splits_large_text():
    """长文本被递归切分到 chunk_size 以下。"""
    text = "short\n\n" + "x" * 3000
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    for chunk in chunks:
        assert len(chunk.content) <= 1000 + 200


def test_recursive_splitter_chinese_sentences():
    """中文句号作为分隔符正确切分。"""
    text = "第一句话。第二句话。第三句话。" * 50
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = splitter.split_text(text)
    for chunk in chunks:
        assert len(chunk.content) <= 200 + 50


def test_recursive_splitter_metadata():
    """切分器保留并补充 metadata。"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500)
    chunks = splitter.split_text("x" * 1500, metadata={"source": "test.txt"})
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_index"] == i
        assert chunk.metadata["source"] == "test.txt"
        assert "total_chunks" in chunk.metadata


def test_recursive_splitter_split_documents():
    """RecursiveCharacterTextSplitter.split_documents 正确工作。"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = [Document(content="x" * 1000, metadata={"source": "test.txt"})]
    chunks = splitter.split_documents(docs)
    assert all(len(c.content) <= 500 for c in chunks)


def test_recursive_splitter_empty():
    """空文本返回空列表。"""
    splitter = RecursiveCharacterTextSplitter()
    assert splitter.split_text("") == []


def test_recursive_splitter_single_long_word():
    """单个超长无分隔符文本也能正确切分。"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    text = "x" * 500
    chunks = splitter.split_text(text)
    assert len(chunks) == 5


# ── KnowledgeStore 测试 ──────────────────────────

@pytest.mark.asyncio
async def test_knowledge_store_add_and_count():
    """添加文档后 count 正确。"""
    provider = MockEmbeddingProvider(dim=128)
    store = KnowledgeStore(
        persist_dir="./temp/test_rag_data_add",
        collection_name="test_add_count",
        embedding_provider=provider,
    )
    docs = [
        Document(content="Python is a programming language.", metadata={"source": "a.txt"}),
        Document(content="Python is also a snake.", metadata={"source": "b.txt"}),
    ]
    count = await store.add_documents(docs)
    assert count == 2
    assert store.count == 2
    store.clear()


@pytest.mark.asyncio
async def test_knowledge_store_search():
    """语义搜索返回 SearchResult 列表。"""
    provider = MockEmbeddingProvider(dim=128)
    store = KnowledgeStore(
        persist_dir="./temp/test_rag_data_search",
        collection_name="test_search",
        embedding_provider=provider,
    )
    docs = [
        Document(content="Machine learning is fascinating.", metadata={"source": "ml.txt"}),
        Document(content="Cooking pasta is easy.", metadata={"source": "cooking.txt"}),
    ]
    await store.add_documents(docs)

    results = await store.search("artificial intelligence", k=2)
    assert len(results) == 2
    assert all(isinstance(r, SearchResult) for r in results)
    assert all(hasattr(r, "score") for r in results)
    store.clear()


@pytest.mark.asyncio
async def test_knowledge_store_empty_search():
    """空库搜索返回空列表。"""
    provider = MockEmbeddingProvider(dim=128)
    store = KnowledgeStore(
        persist_dir="./temp/test_rag_data_empty",
        collection_name="test_empty_search",
        embedding_provider=provider,
    )
    results = await store.search("anything")
    assert results == []


@pytest.mark.asyncio
async def test_knowledge_store_delete_by_source():
    """按 source 删除文档正确。"""
    provider = MockEmbeddingProvider(dim=128)
    store = KnowledgeStore(
        persist_dir="./temp/test_rag_data_delete",
        collection_name="test_delete",
        embedding_provider=provider,
    )
    docs = [
        Document(content="AAA", metadata={"source": "a.txt"}),
        Document(content="BBB", metadata={"source": "b.txt"}),
    ]
    await store.add_documents(docs)
    assert store.count == 2

    deleted = store.delete_by_source("a.txt")
    assert deleted == 1
    assert store.count == 1
    store.clear()


@pytest.mark.asyncio
async def test_knowledge_store_default_embedding():
    """未提供 EmbeddingProvider 时使用 SHA-256 回退。"""
    store = KnowledgeStore(
        persist_dir="./temp/test_rag_data_default",
        collection_name="test_default_emb",
    )
    docs = [Document(content="Test content for default embedding.")]
    count = await store.add_documents(docs)
    assert count == 1
    assert store.count == 1
    store.clear()


@pytest.mark.asyncio
async def test_knowledge_store_add_empty_docs():
    """添加空文档列表返回 0。"""
    provider = MockEmbeddingProvider(dim=128)
    store = KnowledgeStore(
        persist_dir="./temp/test_rag_data_empty_docs",
        collection_name="test_empty_docs",
        embedding_provider=provider,
    )
    count = await store.add_documents([])
    assert count == 0


# ── Retriever 测试 ───────────────────────────────

@pytest.mark.asyncio
async def test_retriever_retrieve():
    """Retriever.retrieve 返回 SearchResult 列表。"""
    provider = MockEmbeddingProvider(dim=128)
    store = KnowledgeStore(
        persist_dir="./temp/test_rag_data_retrieve",
        collection_name="test_retrieve",
        embedding_provider=provider,
    )
    retriever = Retriever(store)

    await store.add_documents([
        Document(content="RAG combines retrieval with generation.", metadata={"source": "rag.txt"}),
    ])

    results = await retriever.retrieve("retrieval augmented generation", k=1)
    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    store.clear()


@pytest.mark.asyncio
async def test_retriever_retrieve_formatted():
    """retrieve_formatted 返回带来源标注的格式化字符串。"""
    provider = MockEmbeddingProvider(dim=128)
    store = KnowledgeStore(
        persist_dir="./temp/test_rag_data_formatted",
        collection_name="test_formatted",
        embedding_provider=provider,
    )
    retriever = Retriever(store)

    await store.add_documents([
        Document(content="Test chunk content here.", metadata={"source": "/docs/test.txt"}),
    ])

    formatted = await retriever.retrieve_formatted("test", k=1)
    assert "[Source: /docs/test.txt]" in formatted
    assert "Test chunk content here" in formatted
    store.clear()


@pytest.mark.asyncio
async def test_retriever_retrieve_formatted_empty():
    """空库检索返回提示信息。"""
    provider = MockEmbeddingProvider(dim=128)
    store = KnowledgeStore(
        persist_dir="./temp/test_rag_data_empty_ret",
        collection_name="test_empty_ret",
        embedding_provider=provider,
    )
    retriever = Retriever(store)
    formatted = await retriever.retrieve_formatted("anything", k=5)
    assert "未找到" in formatted


# ── IngestionPipeline 测试 ──────────────────────

@pytest.mark.asyncio
async def test_pipeline_ingest_file(temp_text_file):
    """Pipeline 完整流程：加载 → 切分 → 存储。"""
    provider = MockEmbeddingProvider(dim=128)

    pipeline = IngestionPipeline(
        loader=TextLoader(),
        splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20),
        knowledge_store=KnowledgeStore(
            persist_dir="./temp/test_rag_data_pipeline",
            collection_name="test_pipeline_file",
            embedding_provider=provider,
        ),
    )

    count = await pipeline.ingest_file(temp_text_file)
    assert count > 0
    results = await pipeline.store.search("test document")
    assert len(results) > 0
    pipeline.store.clear()


@pytest.mark.asyncio
async def test_pipeline_ingest_directory(temp_directory):
    """Pipeline 加载整个目录。"""
    provider = MockEmbeddingProvider(dim=128)

    pipeline = IngestionPipeline(
        loader=DirectoryLoader(),
        splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50),
        knowledge_store=KnowledgeStore(
            persist_dir="./temp/test_rag_data_pipeline_dir",
            collection_name="test_pipeline_dir",
            embedding_provider=provider,
        ),
    )

    count = await pipeline.ingest_directory(temp_directory)
    assert count > 0
    assert pipeline.store.count == count
    pipeline.store.clear()
