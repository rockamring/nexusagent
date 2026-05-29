"""全局配置模型。"""

from pydantic import BaseModel, Field


class NexusConfig(BaseModel):
    """框架全局配置。

    可通过环境变量或代码直接构造。

    用法:
        config = NexusConfig(openai_api_key="sk-...", default_model="gpt-4o")
        config = NexusConfig.from_env()  # 从环境变量读取
    """

    openai_api_key: str = Field(default="", description="OpenAI API Key")
    openai_base_url: str = Field(default="", description="OpenAI 兼容 API 地址（可选）")
    anthropic_api_key: str = Field(default="", description="Anthropic API Key")
    google_api_key: str = Field(default="", description="Google AI API Key")

    default_model: str = Field(default="gpt-4o", description="默认使用的模型名称")
    default_provider: str = Field(default="openai", description="默认 LLM Provider")

    max_iterations: int = Field(default=10, description="Agent 最大循环轮数")
    loop_detection_threshold: int = Field(default=3, description="相同 tool_call 连续出现 N 次则判定为循环")

    log_level: str = Field(default="INFO", description="日志级别")
    log_format: str = Field(default="json", description="日志格式: json 或 console")

    @classmethod
    def from_env(cls) -> "NexusConfig":
        """从环境变量加载配置。"""
        import os

        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_base_url=os.getenv("OPENAI_BASE_URL", ""),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            default_model=os.getenv("NEXUS_DEFAULT_MODEL", "gpt-4o"),
            default_provider=os.getenv("NEXUS_DEFAULT_PROVIDER", "openai"),
            max_iterations=int(os.getenv("NEXUS_MAX_ITERATIONS", "10")),
            log_level=os.getenv("NEXUS_LOG_LEVEL", "INFO"),
        )
