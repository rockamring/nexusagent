"""Config 配置系统测试。"""

import os
import tempfile

import pytest

from nexus.core.config import AgentConfig, LoggingConfig, NexusConfig, ProviderConfig, _resolve_string_env

# ── ProviderConfig 测试 ──────────────────────

def test_provider_config_defaults():
    """默认值：api_key/base_url/default_model 均为空字符串。"""
    cfg = ProviderConfig()
    assert cfg.api_key == ""
    assert cfg.base_url == ""
    assert cfg.default_model == ""


def test_provider_config_with_values():
    """所有字段正确赋值。"""
    cfg = ProviderConfig(api_key="sk-xxx", base_url="http://localhost:8080", default_model="gpt-4o")
    assert cfg.api_key == "sk-xxx"
    assert cfg.base_url == "http://localhost:8080"
    assert cfg.default_model == "gpt-4o"


# ── AgentConfig 测试 ─────────────────────────

def test_agent_config_defaults():
    cfg = AgentConfig()
    assert cfg.max_iterations == 10
    assert cfg.loop_detection_threshold == 3
    assert cfg.max_retries == 3
    assert cfg.retry_delay == 1.0


# ── NexusConfig 测试 ─────────────────────────

def test_nexus_config_defaults():
    """空构造使用默认值。"""
    cfg = NexusConfig()
    assert cfg.default_provider == "openai"
    assert cfg.default_model == "gpt-4o"
    assert cfg.providers == {}
    assert isinstance(cfg.agent, AgentConfig)
    assert isinstance(cfg.logging, LoggingConfig)


def test_nexus_config_with_providers():
    """Provider 字典正确解析。"""
    cfg = NexusConfig(
        providers={
            "openai": ProviderConfig(api_key="sk-xxx"),
            "anthropic": ProviderConfig(api_key="ant-xxx"),
        },
    )
    assert len(cfg.providers) == 2
    assert cfg.providers["openai"].api_key == "sk-xxx"


def test_get_provider_config_existing():
    """get_provider_config 返回已存在的 Provider 配置。"""
    cfg = NexusConfig(providers={"openai": ProviderConfig(api_key="test-key")})
    result = cfg.get_provider_config("openai")
    assert result.api_key == "test-key"


def test_get_provider_config_default():
    """未指定 provider 时使用 default_provider。"""
    cfg = NexusConfig(
        default_provider="anthropic",
        providers={"anthropic": ProviderConfig(api_key="ant-key")},
    )
    result = cfg.get_provider_config()
    assert result.api_key == "ant-key"


def test_get_provider_config_missing():
    """provider 不存在时抛出 KeyError。"""
    cfg = NexusConfig()
    with pytest.raises(KeyError, match="未找到 Provider"):
        cfg.get_provider_config("nonexistent")


# ── YAML 加载测试 ────────────────────────────

def test_from_yaml_basic():
    """从 YAML 文件加载基本配置。"""
    yaml_content = """
default_provider: anthropic
default_model: claude-sonnet-4-6

providers:
  openai:
    api_key: sk-yaml-key
    default_model: gpt-4o-mini
  anthropic:
    api_key: ant-yaml-key
    default_model: claude-sonnet-4-6

agent:
  max_iterations: 20
  loop_detection_threshold: 5

logging:
  level: DEBUG
  format: console
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        cfg = NexusConfig.from_yaml(temp_path)
        assert cfg.default_provider == "anthropic"
        assert cfg.default_model == "claude-sonnet-4-6"
        assert len(cfg.providers) == 2
        assert cfg.providers["openai"].api_key == "sk-yaml-key"
        assert cfg.providers["openai"].default_model == "gpt-4o-mini"
        assert cfg.agent.max_iterations == 20
        assert cfg.agent.loop_detection_threshold == 5
        assert cfg.logging.level == "DEBUG"
        assert cfg.logging.format == "console"
    finally:
        os.unlink(temp_path)


def test_from_yaml_empty():
    """空 YAML 文件使用默认值。"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write("")
        temp_path = f.name

    try:
        cfg = NexusConfig.from_yaml(temp_path)
        assert cfg.default_provider == "openai"
        assert cfg.providers == {}
    finally:
        os.unlink(temp_path)


# ── 环境变量加载测试 ─────────────────────────

def test_from_env_openai(monkeypatch):
    """从环境变量加载 OpenAI 配置。"""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://proxy:8080")

    cfg = NexusConfig.from_env()
    assert "openai" in cfg.providers
    assert cfg.providers["openai"].api_key == "sk-env-key"
    assert cfg.providers["openai"].base_url == "http://proxy:8080"


def test_from_env_anthropic(monkeypatch):
    """从环境变量加载 Anthropic 配置。"""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-env-key")

    cfg = NexusConfig.from_env()
    assert "anthropic" in cfg.providers
    assert cfg.providers["anthropic"].api_key == "ant-env-key"


def test_from_env_google(monkeypatch):
    """从环境变量加载 Google 配置。"""
    monkeypatch.setenv("GOOGLE_API_KEY", "google-env-key")

    cfg = NexusConfig.from_env()
    assert "google" in cfg.providers
    assert cfg.providers["google"].api_key == "google-env-key"


def test_from_env_no_keys(monkeypatch):
    """没有设置任何 Key 时不创建 Provider。"""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    cfg = NexusConfig.from_env()
    assert cfg.providers == {}


def test_from_env_custom_settings(monkeypatch):
    """自定义环境变量覆盖默认值。"""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-custom")
    monkeypatch.setenv("NEXUS_DEFAULT_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("NEXUS_DEFAULT_PROVIDER", "openai")
    monkeypatch.setenv("NEXUS_MAX_ITERATIONS", "15")
    monkeypatch.setenv("NEXUS_LOG_LEVEL", "DEBUG")

    cfg = NexusConfig.from_env()
    assert cfg.default_model == "gpt-4o-mini"
    assert cfg.default_provider == "openai"
    assert cfg.agent.max_iterations == 15
    assert cfg.logging.level == "DEBUG"


# ── from_file 自动发现测试 ───────────────────

def test_from_file_with_explicit_path():
    """显式指定路径加载。"""
    yaml_content = """
default_model: gpt-4.1
providers:
  openai:
    api_key: sk-file-key
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        cfg = NexusConfig.from_file(temp_path)
        assert cfg.default_model == "gpt-4.1"
        assert cfg.providers["openai"].api_key == "sk-file-key"
    finally:
        os.unlink(temp_path)


def test_from_file_not_found():
    """配置文件不存在时返回默认配置。"""
    old_path = os.getenv("NEXUS_CONFIG_PATH", "")
    try:
        os.environ["NEXUS_CONFIG_PATH"] = "/nonexistent/config.yaml"
        cfg = NexusConfig.from_file()
        assert cfg.default_provider == "openai"
    finally:
        if old_path:
            os.environ["NEXUS_CONFIG_PATH"] = old_path
        else:
            os.environ.pop("NEXUS_CONFIG_PATH", None)


# ── 环境变量解析测试 ─────────────────────────

def test_resolve_string_env_var(monkeypatch):
    """${VAR} 替换为环境变量值。"""
    monkeypatch.setenv("MY_API_KEY", "secret123")
    result = _resolve_string_env("${MY_API_KEY}")
    assert result == "secret123"


def test_resolve_string_no_match():
    """无匹配的 ${VAR} 保留原样。"""
    result = _resolve_string_env("${NONEXISTENT_VAR}")
    assert result == "${NONEXISTENT_VAR}"


def test_resolve_string_plain_text():
    """纯文本不变。"""
    result = _resolve_string_env("plain-text-key")
    assert result == "plain-text-key"


# ── LoggingConfig 测试 ───────────────────────

def test_logging_config_defaults():
    cfg = LoggingConfig()
    assert cfg.level == "INFO"
    assert cfg.format == "json"
