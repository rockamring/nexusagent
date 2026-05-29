"""测试配置和共享 Fixture。"""

import pytest

from nexus.core.config import NexusConfig


@pytest.fixture
def config():
    """返回测试用配置。"""
    return NexusConfig(
        openai_api_key="test-key",
        anthropic_api_key="test-key",
        default_model="gpt-4o",
    )
