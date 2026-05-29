"""框架统一异常体系。"""


class NexusError(Exception):
    """所有框架异常的基类。"""


class LLMError(NexusError):
    """LLM 调用相关异常（网络错误、API 错误、超时等）。"""


class ToolError(NexusError):
    """工具执行相关异常。"""


class ToolNotFoundError(ToolError):
    """Agent 尝试调用未注册的工具。"""


class MaxIterationError(NexusError):
    """Agent 超过最大循环次数仍无法完成任务。"""


class ConfigError(NexusError):
    """配置错误（缺少 API Key、无效参数等）。"""


class MemoryError(NexusError):
    """记忆系统异常。"""


class OrchestrationError(NexusError):
    """编排系统异常。"""
