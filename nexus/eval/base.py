"""Agent Eval 评估框架。

轻量级评估框架，用于定义评估用例、运行 Agent 并判断结果是否符合预期。

用法:
    from nexus.eval import EvalCase, EvalSuite

    suite = EvalSuite("数学能力测试")
    suite.add_case(EvalCase(
        name="简单加法",
        input="1+1等于多少？",
        expected="2",
    ))
    suite.add_case(EvalCase(
        name="调用计算器",
        input="计算 123*456",
        expected=["56088"],
        tools=["calculator"],
    ))

    results = await suite.run(agent)
    print(suite.summary(results))
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nexus.agent.base import BaseAgent


@dataclass
class EvalCase:
    """单个评估用例。

    Attributes:
        name: 用例名称（用于报告展示）
        input: Agent 的用户输入
        expected: 期望输出内容，支持以下匹配模式：
            - str: 子串匹配（expected in actual_output）
            - list[str]: 列表匹配（任一元素出现在 actual_output 中即通过）
            - None: 不检查输出内容
        tools: 期望 Agent 调用的工具名称列表，None 表示不检查
        tool_call_count: 期望工具调用总次数，None 表示不检查
    """

    name: str
    input: str
    expected: str | list[str] | None = None
    tools: list[str] | None = None
    tool_call_count: int | None = None


@dataclass
class EvalResult:
    """单个评估用例的执行结果。

    Attributes:
        case: 对应的评估用例
        passed: 是否通过
        actual_output: Agent 的实际输出
        tool_calls: Agent 实际调用的工具名称列表
        iterations: Agent 执行的循环次数
        error: 执行过程中的异常信息（如有）
        details: 未通过时的详细说明
    """

    case: EvalCase
    passed: bool
    actual_output: str
    tool_calls: list[str] = field(default_factory=list)
    iterations: int = 0
    error: str | None = None
    details: str = ""


class EvalSuite:
    """评估套件 — 批量运行评估用例并生成报告。

    用法:
        suite = EvalSuite("基础能力测试")
        suite.add_case(EvalCase(...))
        results = await suite.run(agent)
        print(suite.summary(results))
    """

    def __init__(self, name: str):
        self.name = name
        self.cases: list[EvalCase] = []

    def add_case(self, case: EvalCase) -> None:
        """添加评估用例。"""
        self.cases.append(case)

    def add_cases(self, cases: list[EvalCase]) -> None:
        """批量添加评估用例。"""
        self.cases.extend(cases)

    async def run(self, agent: BaseAgent) -> list[EvalResult]:
        """顺序执行所有评估用例。

        Args:
            agent: 实现了 BaseAgent 接口的 Agent 实例

        Returns:
            EvalResult 列表，每个元素对应一个评估用例
        """
        results: list[EvalResult] = []
        for case in self.cases:
            try:
                agent_result = await agent.run(case.input)
                result = self._evaluate(case, agent_result)
            except Exception as exc:
                result = EvalResult(
                    case=case,
                    passed=False,
                    actual_output="",
                    error=f"{type(exc).__name__}: {exc}",
                    details=f"Agent 执行异常: {exc}",
                )
            results.append(result)
        return results

    def summary(self, results: list[EvalResult]) -> str:
        """生成评估结果摘要。

        Args:
            results: run() 返回的结果列表

        Returns:
            格式化的评估报告字符串
        """
        passed = sum(1 for r in results if r.passed)
        rate = (passed / len(results) * 100) if results else 0

        lines = [
            f"{'=' * 60}",
            f"  评估套件: {self.name}",
            f"  结果: {passed}/{len(results)} 通过 ({rate:.1f}%)",
            f"{'=' * 60}",
        ]

        for r in results:
            status = "[通过]" if r.passed else "[失败]"
            lines.append(f"  {status} {r.case.name}")
            if r.details:
                lines.append(f"          {r.details}")
            if not r.passed and r.error:
                lines.append(f"          错误: {r.error}")

        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    @property
    def count(self) -> int:
        """当前套件中的用例总数。"""
        return len(self.cases)

    # ── 内部方法 ──────────────────────────────

    def _evaluate(self, case: EvalCase, agent_result) -> EvalResult:
        """对 Agent 结果进行评判。"""
        details_parts: list[str] = []

        actual_output = agent_result.content
        tool_names = [r.tool_name for r in agent_result.tool_calls_history]
        iterations = agent_result.iterations

        # 检查输出内容
        content_ok = True
        if case.expected is not None:
            if isinstance(case.expected, str):
                content_ok = case.expected in actual_output
                if not content_ok:
                    details_parts.append(
                        f"期望包含 '{case.expected[:80]}'，实际输出 '{actual_output[:80]}'"
                    )
            elif isinstance(case.expected, list):
                content_ok = any(e in actual_output for e in case.expected)
                if not content_ok:
                    details_parts.append(
                        f"期望包含 {case.expected} 之一，实际输出 '{actual_output[:80]}'"
                    )

        # 检查工具调用
        tools_ok = True
        if case.tools is not None:
            tools_ok = all(t in tool_names for t in case.tools)
            if not tools_ok:
                missing = [t for t in case.tools if t not in tool_names]
                details_parts.append(f"缺少工具调用: {missing}，实际调用: {tool_names}")

        # 检查工具调用次数
        count_ok = True
        if case.tool_call_count is not None:
            count_ok = len(tool_names) == case.tool_call_count
            if not count_ok:
                details_parts.append(
                    f"期望 {case.tool_call_count} 次工具调用，实际 {len(tool_names)} 次"
                )

        passed = content_ok and tools_ok and count_ok
        details = "; ".join(details_parts) if details_parts else ""

        return EvalResult(
            case=case,
            passed=passed,
            actual_output=actual_output,
            tool_calls=tool_names,
            iterations=iterations,
            details=details,
        )
