"""Embedding 系统测试。

测试 BaseEmbeddingProvider、OpenAIEmbeddingProvider 和 VectorStoreMemory 集成。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus.embeddings.base import BaseEmbeddingProvider
from nexus.embeddings.providers.openai import OpenAIEmbeddingProvider
from nexus.memory.vector_store import VectorStoreMemory

# ── 自定义 Embedding Provider（用于测试） ──────────

class MockEmbeddingProvider(BaseEmbeddingProvider):
    """模拟 Embedding Provider，返回固定维度向量。"""

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.calls: list[str] = []
        self.batch_calls: list[list[str]] = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return [float(hash(text) % 100) / 100.0 for _ in range(self.dim)]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls.append(texts)
        return [[float(hash(t) % 100) / 100.0 for _ in range(self.dim)] for t in texts]


# ── BaseEmbeddingProvider 测试 ──────────────────

def test_base_embedding_provider_is_abstract():
    """BaseEmbeddingProvider 不可直接实例化。"""
    with pytest.raises(TypeError):
        BaseEmbeddingProvider()  # type: ignore[abstract]


def test_custom_provider_can_be_instantiated():
    """自定义子类可以正常实例化。"""
    provider = MockEmbeddingProvider(dim=256)
    assert provider.dim == 256


@pytest.mark.asyncio
async def test_custom_provider_embed():
    """自定义 Provider 的 embed 方法返回固定维度向量。"""
    provider = MockEmbeddingProvider(dim=64)
    result = await provider.embed("hello")
    assert len(result) == 64
    assert all(isinstance(v, float) for v in result)


@pytest.mark.asyncio
async def test_custom_provider_embed_batch():
    """自定义 Provider 的 embed_batch 返回多条向量。"""
    provider = MockEmbeddingProvider(dim=32)
    results = await provider.embed_batch(["a", "b", "c"])
    assert len(results) == 3
    assert all(len(v) == 32 for v in results)


# ── OpenAIEmbeddingProvider 测试 ────────────────

def test_openai_provider_default_model():
    """默认模型为 text-embedding-3-small。"""
    provider = OpenAIEmbeddingProvider(api_key="test-key")
    assert provider.model == "text-embedding-3-small"


def test_openai_provider_custom_model():
    """支持自定义模型名称和 base_url。"""
    provider = OpenAIEmbeddingProvider(
        api_key="test-key",
        model="text-embedding-3-large",
        base_url="http://localhost:8080/v1",
    )
    assert provider.model == "text-embedding-3-large"


@pytest.mark.asyncio
async def test_openai_provider_embed():
    """Mock OpenAI API，验证 embed 调用正确。"""
    mock_embedding = [0.1, 0.2, 0.3]

    with patch.object(OpenAIEmbeddingProvider, "__init__", lambda self, **kw: None):
        provider = OpenAIEmbeddingProvider.__new__(OpenAIEmbeddingProvider)
        provider._client = AsyncMock()
        provider._model = "text-embedding-3-small"

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=mock_embedding)]
        provider._client.embeddings.create = AsyncMock(return_value=mock_response)

        result = await provider.embed("你好")
        assert result == mock_embedding
        provider._client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input="你好",
        )


@pytest.mark.asyncio
async def test_openai_provider_embed_batch():
    """Mock OpenAI API，验证 embed_batch 调用正确。"""
    mock_embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

    with patch.object(OpenAIEmbeddingProvider, "__init__", lambda self, **kw: None):
        provider = OpenAIEmbeddingProvider.__new__(OpenAIEmbeddingProvider)
        provider._client = AsyncMock()
        provider._model = "text-embedding-3-small"

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=e) for e in mock_embeddings]
        provider._client.embeddings.create = AsyncMock(return_value=mock_response)

        results = await provider.embed_batch(["a", "b", "c"])
        assert results == mock_embeddings
        assert len(results) == 3
        provider._client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=["a", "b", "c"],
        )


# ── VectorStoreMemory + EmbeddingProvider 集成测试 ──

@pytest.mark.asyncio
async def test_vector_store_with_embedding_provider():
    """VectorStoreMemory 使用 EmbeddingProvider 进行语义检索。"""
    provider = MockEmbeddingProvider(dim=128)
    memory = VectorStoreMemory(
        persist_dir="./temp/test_memory_data_1",
        collection_name="test_provider",
        embedding_provider=provider,
    )

    await memory.add({"role": "user", "content": "我喜欢 Python 编程"})
    await memory.add({"role": "user", "content": "我喜欢打篮球"})
    await memory.add({"role": "assistant", "content": "Python 是一种优秀的编程语言"})

    # 语义查询应返回最相关的结果
    ctx = await memory.get_context(query="编程")
    assert len(ctx) > 0
    # Mock 是基于哈希的，所以检索会返回结果
    assert provider.calls  # Provider 被调用过

    await memory.clear()


@pytest.mark.asyncio
async def test_vector_store_default_embedding():
    """未提供 EmbeddingProvider 时使用 SHA-256 默认嵌入。"""
    memory = VectorStoreMemory(
        persist_dir="./temp/test_memory_data_2",
        collection_name="test_default",
    )

    await memory.add({"role": "user", "content": "测试消息"})
    ctx = await memory.get_context(query="测试")

    assert len(ctx) > 0
    assert "[历史记忆" in ctx[0]["content"]

    await memory.clear()
    assert memory.count == 0


@pytest.mark.asyncio
async def test_vector_store_custom_embedding_fn():
    """支持通过 embedding_fn 传入自定义嵌入函数。"""
    async def my_embed(text: str) -> list[float]:
        return [0.5] * 384

    memory = VectorStoreMemory(
        persist_dir="./temp/test_memory_data_3",
        collection_name="test_custom_fn",
        embedding_fn=my_embed,
    )

    await memory.add({"role": "user", "content": "hello"})
    ctx = await memory.get_context(query="world")

    assert memory.count == 1
    assert len(ctx) == 1

    await memory.clear()


@pytest.mark.asyncio
async def test_vector_store_empty_query():
    """空查询或无记忆时返回空列表。"""
    memory = VectorStoreMemory(
        persist_dir="./temp/test_memory_data_4",
        collection_name="test_empty",
    )
    # 无数据时应返回空
    ctx = await memory.get_context(query="something")
    assert ctx == []


@pytest.mark.asyncio
async def test_vector_store_add_empty_content():
    """content 为空的消息不存储。"""
    memory = VectorStoreMemory(
        persist_dir="./temp/test_memory_data_5",
        collection_name="test_empty_content",
    )
    await memory.add({"role": "user", "content": ""})
    assert memory.count == 0
    await memory.clear()


@pytest.mark.asyncio
async def test_vector_store_count():
    """count 属性返回正确的记忆条数。"""
    memory = VectorStoreMemory(
        persist_dir="./temp/test_memory_data_6",
        collection_name="test_count",
    )
    assert memory.count == 0
    await memory.add({"role": "user", "content": "消息1"})
    await memory.add({"role": "user", "content": "消息2"})
    assert memory.count == 2
    await memory.clear()
