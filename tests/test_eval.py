"""Eval 评估框架测试。"""

import pytest

from nexus.agent.base import BaseAgent
from nexus.core.types import AgentResult
from nexus.eval.base import EvalCase, EvalResult, EvalSuite

# ── 最简 Agent（用于测试 Eval 框架）──────────────

class SimpleEvalAgent(BaseAgent):
    """固定输出 Agent，用于测试评估框架。"""

    def __init__(self, output: str = "固定回复", iterations: int = 1,
                 tool_names: list[str] | None = None):
        self.output = output
        self.iterations_count = iterations
        self.tool_names = tool_names or []

    async def run(self, user_input: str, **kwargs) -> AgentResult:
        from nexus.core.types import ToolCallRecord
        records = [
            ToolCallRecord(tool_name=name, arguments={}, result="ok")
            for name in self.tool_names
        ]
        return AgentResult(
            content=self.output,
            iterations=self.iterations_count,
            tool_calls_history=records,
        )


# ── EvalCase 测试 ─────────────────────────────

def test_eval_case_creation():
    """基本创建，默认值检查。"""
    case = EvalCase(name="测试", input="hello")
    assert case.name == "测试"
    assert case.input == "hello"
    assert case.expected is None
    assert case.tools is None
    assert case.tool_call_count is None


def test_eval_case_with_all_fields():
    """所有字段赋值正确。"""
    case = EvalCase(
        name="复杂测试",
        input="计算 1+1",
        expected="2",
        tools=["calculator"],
        tool_call_count=1,
    )
    assert case.expected == "2"
    assert case.tools == ["calculator"]
    assert case.tool_call_count == 1


def test_eval_case_expected_list():
    """expected 支持列表形式。"""
    case = EvalCase(
        name="多答案测试",
        input="你好",
        expected=["你好", "您好", "Hello"],
    )
    assert isinstance(case.expected, list)
    assert len(case.expected) == 3


# ── EvalSuite 测试 ────────────────────────────

def test_eval_suite_creation():
    """创建套件后 count 为 0。"""
    suite = EvalSuite("测试套件")
    assert suite.name == "测试套件"
    assert suite.count == 0
    assert suite.cases == []


def test_eval_suite_add_case():
    """add_case 增加用例数量。"""
    suite = EvalSuite("测试")
    suite.add_case(EvalCase(name="case1", input="hello"))
    suite.add_case(EvalCase(name="case2", input="world"))
    assert suite.count == 2


def test_eval_suite_add_cases():
    """add_cases 批量添加用例。"""
    suite = EvalSuite("批量测试")
    cases = [
        EvalCase(name="a", input="1"),
        EvalCase(name="b", input="2"),
        EvalCase(name="c", input="3"),
    ]
    suite.add_cases(cases)
    assert suite.count == 3


# ── EvalResult 测试 ───────────────────────────

def test_eval_result_passed():
    """通过的用例。"""
    case = EvalCase(name="测试", input="hello")
    result = EvalResult(
        case=case, passed=True, actual_output="世界",
    )
    assert result.passed is True
    assert result.error is None
    assert result.details == ""


def test_eval_result_failed_with_details():
    """失败的用例包含详细说明。"""
    case = EvalCase(name="失败测试", input="hello", expected="世界")
    result = EvalResult(
        case=case, passed=False, actual_output="你好",
        details="期望包含 '世界'，实际输出 '你好'",
    )
    assert result.passed is False
    assert "世界" in result.details


# ── EvalSuite.run 测试 ────────────────────────

@pytest.mark.asyncio
async def test_eval_suite_run_all_pass():
    """全部用例通过。"""
    suite = EvalSuite("通过套件")
    suite.add_case(EvalCase(name="测试1", input="hello", expected="你好"))
    suite.add_case(EvalCase(name="测试2", input="hello", expected="你好"))

    agent = SimpleEvalAgent(output="你好世界！")
    results = await suite.run(agent)

    assert len(results) == 2
    assert all(r.passed for r in results)


@pytest.mark.asyncio
async def test_eval_suite_run_mixed():
    """部分通过，部分失败。"""
    suite = EvalSuite("混合套件")
    suite.add_case(EvalCase(name="通过", input="a", expected="是"))
    suite.add_case(EvalCase(name="失败", input="b", expected="不存在的内容"))

    agent = SimpleEvalAgent(output="是的，好的。")
    results = await suite.run(agent)

    assert results[0].passed is True
    assert results[1].passed is False
    assert results[1].details != ""


@pytest.mark.asyncio
async def test_eval_suite_run_tool_check():
    """检查工具调用是否匹配。"""
    suite = EvalSuite("工具测试")
    suite.add_case(EvalCase(
        name="需要计算器",
        input="1+1",
        tools=["calculator"],
    ))

    agent = SimpleEvalAgent(output="答案是2", tool_names=["calculator"])
    results = await suite.run(agent)

    assert results[0].passed is True
    assert "calculator" in results[0].tool_calls


@pytest.mark.asyncio
async def test_eval_suite_run_tool_not_called():
    """期望的工具未调用时应失败。"""
    suite = EvalSuite("工具缺失测试")
    suite.add_case(EvalCase(
        name="需要搜索工具",
        input="搜索 Python",
        tools=["search"],
    ))

    agent = SimpleEvalAgent(output="搜索不到", tool_names=["calculator"])
    results = await suite.run(agent)

    assert results[0].passed is False
    assert "缺少工具调用" in results[0].details


@pytest.mark.asyncio
async def test_eval_suite_run_tool_count():
    """检查工具调用次数。"""
    suite = EvalSuite("调用次数测试")
    suite.add_case(EvalCase(
        name="应有2次调用",
        input="计算并搜索",
        tool_call_count=2,
    ))

    agent = SimpleEvalAgent(output="完成", tool_names=["calculator", "search"])
    results = await suite.run(agent)

    assert results[0].passed is True


@pytest.mark.asyncio
async def test_eval_suite_run_agent_exception():
    """Agent 抛出异常时结果仍记录但不崩溃。"""
    class FailingAgent(BaseAgent):
        async def run(self, user_input: str, **kwargs) -> AgentResult:
            raise RuntimeError("模拟异常")

    suite = EvalSuite("异常测试")
    suite.add_case(EvalCase(name="崩溃用例", input="test"))

    results = await suite.run(FailingAgent())

    assert len(results) == 1
    assert results[0].passed is False
    assert "RuntimeError" in results[0].error
    assert "模拟异常" in results[0].details


@pytest.mark.asyncio
async def test_eval_suite_run_expected_list():
    """expected 为列表时，任意匹配即通过。"""
    suite = EvalSuite("多答案测试")
    suite.add_case(EvalCase(
        name="问候",
        input="你好",
        expected=["Hello", "你好", "Hi"],
    ))

    agent = SimpleEvalAgent(output="你好，很高兴见到你！")
    results = await suite.run(agent)

    assert results[0].passed is True


# ── EvalSuite.summary 测试 ────────────────────

def test_summary_format():
    """验证 summary 输出格式。"""
    suite = EvalSuite("格式测试")
    case1 = EvalCase(name="通过", input="a", expected="ok")
    case2 = EvalCase(name="失败", input="b", expected="不存在")

    results = [
        EvalResult(case=case1, passed=True, actual_output="ok!"),
        EvalResult(case=case2, passed=False, actual_output="no",
                    details="期望包含 '不存在'"),
    ]

    text = suite.summary(results)
    assert "格式测试" in text
    assert "1/2" in text
    assert "50.0%" in text
    assert "[通过]" in text
    assert "[失败]" in text


def test_summary_all_pass():
    """全部通过时的摘要。"""
    suite = EvalSuite("全过套件")
    case = EvalCase(name="测试", input="x", expected="ok")
    results = [EvalResult(case=case, passed=True, actual_output="ok!")]
    text = suite.summary(results)
    assert "1/1" in text
    assert "100.0%" in text


def test_summary_empty():
    """空结果不会崩溃。"""
    suite = EvalSuite("空套件")
    text = suite.summary([])
    assert "0/0" in text
