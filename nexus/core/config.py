"""全局配置模型 — 支持 YAML 文件、环境变量和代码直接构造。

优先级（从高到低）：代码传入 > 环境变量 > YAML 文件 > 默认值
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """单个 LLM Provider 的配置。"""

    api_key: str = Field(default="", description="API Key（支持 ${ENV_VAR} 引用）")
    base_url: str = Field(default="", description="API Base URL（可选，用于 Ollama/vLLM/LiteLLM 等）")
    default_model: str = Field(default="", description="该 Provider 的默认模型")


class AgentConfig(BaseModel):
    """Agent 引擎配置。"""

    max_iterations: int = Field(default=10, description="Agent 最大循环轮数")
    loop_detection_threshold: int = Field(default=3, description="相同 tool_call 连续出现 N 次则判定为循环")
    max_retries: int = Field(default=3, description="LLM 调用最大重试次数")
    retry_delay: float = Field(default=1.0, description="重试初始延迟（秒），每次翻倍")


class LoggingConfig(BaseModel):
    """日志配置。"""

    level: str = Field(default="INFO", description="日志级别")
    format: str = Field(default="json", description="日志格式: json 或 console")


class NexusConfig(BaseModel):
    """框架全局配置。

    支持三种加载方式（按优先级排序）:

    1. 代码直接构造:
       config = NexusConfig(
           providers={"openai": ProviderConfig(api_key="sk-...")},
           default_model="gpt-4o",
       )

    2. YAML 配置文件:
       config = NexusConfig.from_yaml("config.yaml")

    3. 环境变量（兼容旧版）:
       config = NexusConfig.from_env()
    """

    default_provider: str = Field(default="openai", description="默认使用的 LLM Provider 名称")
    default_model: str = Field(default="gpt-4o", description="默认使用的模型名称")

    providers: dict[str, ProviderConfig] = Field(
        default_factory=dict,
        description="Provider 配置字典，key 为 provider 名称",
    )

    agent: AgentConfig = Field(default_factory=AgentConfig, description="Agent 引擎配置")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="日志配置")

    @classmethod
    def from_env(cls) -> "NexusConfig":
        """从环境变量加载配置（兼容旧版 API）。

        读取的环境变量:
        - OPENAI_API_KEY, OPENAI_BASE_URL
        - ANTHROPIC_API_KEY
        - GOOGLE_API_KEY
        - NEXUS_DEFAULT_MODEL, NEXUS_DEFAULT_PROVIDER
        - NEXUS_MAX_ITERATIONS
        - NEXUS_LOG_LEVEL
        """
        providers: dict[str, ProviderConfig] = {}

        openai_key = os.getenv("OPENAI_API_KEY", "")
        openai_url = os.getenv("OPENAI_BASE_URL", "")
        if openai_key or openai_url:
            providers["openai"] = ProviderConfig(
                api_key=openai_key,
                base_url=openai_url,
                default_model=os.getenv("NEXUS_DEFAULT_MODEL", "gpt-4o"),
            )

        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        if anthropic_key:
            providers["anthropic"] = ProviderConfig(
                api_key=anthropic_key,
                default_model="claude-sonnet-4-6",
            )

        google_key = os.getenv("GOOGLE_API_KEY", "")
        if google_key:
            providers["google"] = ProviderConfig(
                api_key=google_key,
                default_model="gemini-2.5-flash",
            )

        return cls(
            default_provider=os.getenv("NEXUS_DEFAULT_PROVIDER", "openai"),
            default_model=os.getenv("NEXUS_DEFAULT_MODEL", "gpt-4o"),
            providers=providers,
            agent=AgentConfig(
                max_iterations=int(os.getenv("NEXUS_MAX_ITERATIONS", "10")),
            ),
            logging=LoggingConfig(
                level=os.getenv("NEXUS_LOG_LEVEL", "INFO"),
            ),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NexusConfig":
        """从 YAML 文件加载配置。

        YAML 格式示例:
            default_provider: openai
            default_model: gpt-4o-mini

            providers:
              openai:
                api_key: "${OPENAI_API_KEY}"
                base_url: ""
                default_model: gpt-4o

              anthropic:
                api_key: "${ANTHROPIC_API_KEY}"
                default_model: claude-sonnet-4-6

            agent:
              max_iterations: 10
              loop_detection_threshold: 3

            logging:
              level: INFO
              format: json

        字符串中的 ${VAR} 会自动替换为对应的环境变量值。
        """
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            raw: dict = yaml.safe_load(f) or {}

        # 递归解析环境变量引用
        resolved = _resolve_config_env(raw)

        # 解析 providers
        providers: dict[str, ProviderConfig] = {}
        for name, pdata in resolved.pop("providers", {}).items():
            if isinstance(pdata, dict):
                providers[name] = ProviderConfig(**pdata)

        # 解析子配置
        agent = AgentConfig(**(resolved.pop("agent", {})))
        logging_cfg = LoggingConfig(**(resolved.pop("logging", {})))

        return cls(
            providers=providers,
            agent=agent,
            logging=logging_cfg,
            **{k: v for k, v in resolved.items() if k in cls.model_fields},
        )

    @classmethod
    def from_file(cls, path: str | Path | None = None) -> "NexusConfig":
        """自动发现并加载配置文件。

        查找顺序:
        1. NEXUS_CONFIG_PATH 环境变量
        2. 当前目录下的 config.yaml
        3. 当前目录下的 config.yml
        4. 当前目录下的 config.yaml.example (模板，供首次使用)

        未找到配置文件时返回空配置（全部使用默认值）。
        """
        if path:
            return cls.from_yaml(path)

        env_path = os.getenv("NEXUS_CONFIG_PATH", "")
        if env_path and Path(env_path).exists():
            return cls.from_yaml(env_path)

        for candidate in ("config.yaml", "config.yml", "config.yaml.example"):
            if Path(candidate).exists():
                return cls.from_yaml(candidate)

        return cls()

    def get_provider_config(self, provider: str | None = None) -> ProviderConfig:
        """获取指定 Provider 的配置。

        未指定时使用 default_provider。
        """
        name = provider or self.default_provider
        if name in self.providers:
            return self.providers[name]
        raise KeyError(
            f"未找到 Provider '{name}' 的配置，"
            f"可用: {list(self.providers.keys()) or '(无)'}"
        )


_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")


def _resolve_config_env(data: dict) -> dict:
    """递归解析字典中所有字符串值的 ${VAR} 环境变量引用。"""
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = _resolve_config_env(value)
        elif isinstance(value, list):
            result[key] = [
                _resolve_string_env(v) if isinstance(v, str) else v
                for v in value
            ]
        elif isinstance(value, str):
            result[key] = _resolve_string_env(value)
        else:
            result[key] = value
    return result


def _resolve_string_env(value: str) -> str:
    """替换字符串中的 ${VAR} 为环境变量值。

    ${OPENAI_API_KEY} → os.getenv("OPENAI_API_KEY", "")
    未匹配到的变量保留原样。
    """
    def _replace(m: re.Match) -> str:
        return os.getenv(m.group(1), m.group(0))

    return _ENV_VAR_RE.sub(_replace, value)
