"""LLM Registry 和 Factory 测试。"""

import pytest

from nexus.llm.base import BaseLLM
from nexus.llm.registry import LLMRegistry

# ── 清理注册表 ──────────────────────────────

@pytest.fixture(autouse=True)
def clean_registry():
    """每个测试前后清空注册表。"""
    LLMRegistry.clear()
    yield
    LLMRegistry.clear()


# ── Mock Provider ───────────────────────────

class _MockProvider(BaseLLM):
    """模拟 Provider 用于测试 Registry。"""

    def __init__(self, api_key="", default_model="mock", **kwargs):
        super().__init__()
        self._default_model = default_model
        self._provider_name = "mock"
        self._supported_models = ["mock"]
        self.api_key = api_key

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def supported_models(self) -> list[str]:
        return self._supported_models

    async def generate(self, messages, tools=None, **kwargs):
        raise NotImplementedError

    async def generate_stream(self, messages, tools=None, **kwargs):
        raise NotImplementedError

    async def count_tokens(self, messages):
        return 0


# ── LLMRegistry 测试 ─────────────────────────

def test_register_and_get():
    """注册后可以通过 get 获取。"""
    LLMRegistry.register("mock", _MockProvider)
    cls = LLMRegistry.get("mock")
    assert cls is _MockProvider


def test_register_invalid_type():
    """注册非 BaseLLM 子类抛出 TypeError。"""
    class NotAProvider:
        pass

    with pytest.raises(TypeError, match="必须继承 BaseLLM"):
        LLMRegistry.register("bad", NotAProvider)


def test_get_missing():
    """获取未注册的 Provider 抛出 KeyError。"""
    with pytest.raises(KeyError, match="未找到 Provider"):
        LLMRegistry.get("nonexistent")


def test_create():
    """create 返回正确的 Provider 实例。"""
    LLMRegistry.register("mock", _MockProvider)
    instance = LLMRegistry.create("mock", api_key="test-key")
    assert isinstance(instance, _MockProvider)
    assert instance.api_key == "test-key"


def test_create_with_extra_kwargs():
    """额外的 kwargs 传递到 Provider 构造器。"""
    LLMRegistry.register("mock", _MockProvider)
    instance = LLMRegistry.create("mock", api_key="override-key", default_model="custom-model")
    assert instance.api_key == "override-key"
    assert instance.default_model == "custom-model"


def test_list_providers():
    """list_providers 返回所有已注册名称。"""
    assert LLMRegistry.list_providers() == []
    LLMRegistry.register("a", _MockProvider)
    LLMRegistry.register("b", _MockProvider)
    assert sorted(LLMRegistry.list_providers()) == ["a", "b"]


def test_clear():
    """clear 清空所有注册项。"""
    LLMRegistry.register("mock", _MockProvider)
    assert "mock" in LLMRegistry.list_providers()
    LLMRegistry.clear()
    assert LLMRegistry.list_providers() == []


def test_duplicate_register_overwrites():
    """重复注册同名 Provider 覆盖旧值。"""
    class MockV2(BaseLLM):
        def __init__(self, **kwargs):
            super().__init__()
            self._provider_name = "mock_v2"
            self._default_model = "v2"

        @property
        def provider_name(self) -> str:
            return self._provider_name

        @property
        def default_model(self) -> str:
            return self._default_model

        @property
        def supported_models(self) -> list[str]:
            return ["v2"]

        async def generate(self, messages, tools=None, **kwargs):
            raise NotImplementedError

        async def generate_stream(self, messages, tools=None, **kwargs):
            raise NotImplementedError

        async def count_tokens(self, messages):
            return 0

    LLMRegistry.register("mock", _MockProvider)
    LLMRegistry.register("mock", MockV2)
    cls = LLMRegistry.get("mock")
    assert cls is MockV2


# ── create_llm 测试 ─────────────────────────

def test_create_llm_basic():
    """create_llm 使用默认参数创建 Provider。"""
    from nexus.llm.factory import _register_builtin_providers, create_llm

    _register_builtin_providers()
    # 使用 mock 注册覆盖
    LLMRegistry.register("openai", _MockProvider)

    llm = create_llm("openai", api_key="test-key")
    assert isinstance(llm, _MockProvider)
    assert llm.api_key == "test-key"


def test_create_llm_with_model():
    """create_llm 支持指定模型。"""
    from nexus.llm.factory import _register_builtin_providers, create_llm

    _register_builtin_providers()
    LLMRegistry.register("openai", _MockProvider)

    llm = create_llm("openai", api_key="key", default_model="gpt-4o-mini")
    assert llm.default_model == "gpt-4o-mini"
