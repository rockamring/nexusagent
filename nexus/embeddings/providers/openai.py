"""OpenAI Embedding Provider — 封装 OpenAI Embeddings API。

支持所有兼容 OpenAI Embeddings API 的服务（包括本地 ollama、vLLM 等）。
"""

from __future__ import annotations

from openai import AsyncOpenAI

from nexus.embeddings.base import BaseEmbeddingProvider


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI Embedding Provider。

    用法:
        provider = OpenAIEmbeddingProvider(api_key="sk-...")
        vector = await provider.embed("你好世界")

        # 使用兼容服务
        provider = OpenAIEmbeddingProvider(
            api_key="ollama",
            base_url="http://localhost:11434/v1",
            model="nomic-embed-text",
        )
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str | None = None,
    ):
        client_kwargs: dict = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**client_kwargs)
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def embed(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(
            model=self._model,
            input=text,
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [d.embedding for d in response.data]
