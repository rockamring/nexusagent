"""测试配置和共享 Fixture。"""

import pytest

from nexus.core.config import AgentConfig, LoggingConfig, NexusConfig, ProviderConfig


@pytest.fixture
def config():
    """返回测试用配置。"""
    return NexusConfig(
        default_provider="openai",
        default_model="gpt-4o",
        providers={
            "openai": ProviderConfig(api_key="test-key", default_model="gpt-4o"),
            "anthropic": ProviderConfig(api_key="test-key", default_model="claude-sonnet-4-6"),
        },
        agent=AgentConfig(max_iterations=10),
        logging=LoggingConfig(level="INFO", format="json"),
    )
