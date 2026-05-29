"""SequentialOrchestrator — 顺序编排。

最简单的编排模式：多个 Agent 按顺序执行，前一个的输出作为后一个的输入。

适用场景:
- 论文写作: 检索 → 起草 → 润色
- 代码审查: 分析 → 建议 → 生成修复
- 数据处理: 提取 → 转换 → 加载

数据流:
    Input → [Agent A] → [Agent B] → [Agent C] → Output
"""

from __future__ import annotations

from nexus.agent.base import BaseAgent
from nexus.orchestrator.base import BaseOrchestrator, OrchestratorResult
from nexus.utils import get_logger

logger = get_logger(__name__)


class SequentialOrchestrator(BaseOrchestrator):
    """顺序流水线编排器。

    用法的：
        researcher = ReActAgent(name="研究员", ...)
        writer = ReActAgent(name="写作者", ...)
        reviewer = ReActAgent(name="审核者", ...)

        pipeline = SequentialOrchestrator([
            (researcher, "研究这个主题：{input}"),
            (writer, "根据以下研究写一篇报告：{input}"),
            (reviewer, "审核以下报告并给出改进建议：{input}"),
        ])

        result = await pipeline.run("AI Agent 框架对比")
    """

    def __init__(self, stages: list[tuple[BaseAgent, str]] = None):
        """初始化顺序编排器。

        Args:
            stages: 编排阶段列表，每个元素是 (Agent, prompt_template)。
                    prompt_template 中 {input} 会被替换为上一阶段的输出。
                    第一个阶段的 {input} 是 run(task) 的 task 参数。
        """
        self._stages = stages or []

    def add_stage(self, agent: BaseAgent, prompt_template: str = "{input}") -> None:
        """添加一个编排阶段。"""
        self._stages.append((agent, prompt_template))

    async def run(self, task: str, **kwargs) -> OrchestratorResult:
        intermediate = []
        agent_names = []
        total_iterations = 0
        current_input = task

        for agent, template in self._stages:
            agent_name = getattr(agent, "name", "unknown")
            logger.info("orchestrator_stage_start", agent=agent_name)

            prompt = template.format(input=current_input)
            result = await agent.run(prompt)

            current_input = result.content
            intermediate.append(result)
            agent_names.append(agent_name)
            total_iterations += result.iterations

            logger.info(
                "orchestrator_stage_done",
                agent=agent_name,
                iterations=result.iterations,
                output_preview=result.content[:200],
            )

        return OrchestratorResult(
            final_output=current_input,
            intermediate_outputs=intermediate,
            agent_names=agent_names,
            total_iterations=total_iterations,
        )
