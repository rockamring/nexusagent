"""NexusAgent - 一个模块化、可扩展的 AI Agent 学习框架。

核心架构分 5 层:
- llm:     LLM Provider 抽象层 (OpenAI, Anthropic)
- tools:   Tool 系统 (函数 → JSON Schema 自动推导)
- agent:   ReAct Agent 核心引擎
- memory:  Memory 系统 (短期/长期/摘要)
- orchestrator: 多 Agent 编排
"""

__version__ = "0.1.0"
