"""Agent Handoff — Agent 间任务交接机制。

参考 OpenAI Agents SDK 的 Handoff 设计：
一个 Agent 可以将任务"交接"给另一个 Agent，带上必要的上下文信息。

Handoff 本质上是一个特殊的 Tool：
当主 Agent 认为某个子 Agent 更适合处理当前任务时，
调用 handoff_to_<name> 工具，将任务和上下文传递给子 Agent。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nexus.core.types import AgentResult
from nexus.tools.base import BaseTool


@dataclass
class HandoffConfig:
    """定义从一个 Agent 到另一个 Agent 的交接配置。

    Args:
        target_agent: 目标 Agent 的引用（需要实现 BaseAgent.run()）
        name: 交接工具的名称，会暴露给 LLM 作为 tool name
        description: 交接工具的描述，帮助 LLM 判断何时交接
        input_filter: 可选，对交接内容的转换函数
    """
    target_agent: object  # BaseAgent，使用 object 避免循环导入
    name: str
    description: str
    input_filter: callable | None = None


@dataclass
class HandoffResult:
    """一次 Handoff 的执行结果。"""
    from_agent: str
    to_agent: str
    result: AgentResult
    handoff_context: str = ""


class HandoffTool(BaseTool):
    """Handoff 工具 — 将任务交接给另一个 Agent。

    LLM 调用此工具时，实际上是将控制权转移给目标 Agent。

    用法:
        researcher = ReActAgent(name="研究员", ...)
        writer = ReActAgent(name="写作者", ...)

        handoff_to_writer = HandoffTool(
            source_agent=researcher,
            target_agent=writer,
            handoff_prompt="以下是对你写作有用的研究资料：\n{context}",
        )

        researcher.tools.register(handoff_to_writer)
    """

    def __init__(
        self,
        source_agent: object,  # BaseAgent
        target_agent: object,  # BaseAgent
        *,
        name: str | None = None,
        description: str | None = None,
        handoff_prompt: str = "{context}",
    ):
        self._source = source_agent
        self._target = target_agent
        self._handoff_prompt = handoff_prompt

        target_name = getattr(target_agent, "name", "unknown")
        self._tool_name = name or f"handoff_to_{target_name}"
        self._tool_description = description or (
            f"将当前任务交接给 {target_name} 处理。"
            f"当你认为自己不适合处理此任务，而 {target_name} 更适合时调用。"
            f"需要提供 context 参数，简要说明当前进展和需要对方做什么。"
        )

    @property
    def schema(self) -> dict:
        return {
            "name": self._tool_name,
            "description": self._tool_description,
            "parameters": {
                "type": "object",
                "properties": {
                    "context": {
                        "type": "string",
                        "description": "交接给目标 Agent 的上下文信息，包括当前进展和需求。",
                    },
                },
                "required": ["context"],
            },
        }

    async def execute(self, context: str = "", **kwargs) -> str:
        """执行 Handoff：将任务交接给目标 Agent。"""
        source_name = getattr(self._source, "name", "unknown")

        # 构造交接提示
        prompt = self._handoff_prompt.format(context=context)

        target = self._target
        result = await target.run(prompt)

        return (
            f"[Handoff: {source_name} → {getattr(target, 'name', 'unknown')}]\n"
            f"结果: {result.content}"
        )
