"""Example 8: Human-in-the-Loop — 演示工具调用的安全审批机制。

无需 API Key 即可运行（使用 MockLLM）。

演示:
    1. CLIApproval 命令行审批器的工作原理
    2. 自定义审批回调：白名单、日志记录
    3. Agent 集成：审批拒绝后 Agent 如何响应

运行:
    python examples/08_human_in_the_loop.py
"""

import asyncio

from nexus.agent import ReActAgent
from nexus.agent.guard import CLIApproval
from nexus.core.types import LLMResponse
from nexus.tools.base import Tool
from nexus.tools.registry import ToolRegistry

# ── 危险工具（标记 requires_approval）────────────

@Tool.from_function(
    name="delete_files",
    description="删除指定路径的文件",
    requires_approval=True,  # 标记为需要人工审批
)
async def delete_files(path: str) -> str:
    return f"文件已删除: {path}"


@Tool.from_function(
    name="write_config",
    description="写入系统配置文件",
    requires_approval=True,  # 标记为需要人工审批
)
async def write_config(path: str, content: str) -> str:
    return f"配置已写入: {path}"


# ── 安全工具 ────────────────────────────────────

@Tool.from_function(
    name="read_file",
    description="读取文件内容",
)
async def read_file(path: str) -> str:
    return f"[文件内容] 示例数据来自 {path}"


@Tool.from_function(
    name="list_files",
    description="列出目录中的文件",
)
async def list_files(path: str = ".") -> str:
    return "file1.txt, file2.txt, config.yaml"


# ── Mock LLM ────────────────────────────────────

class MockLLM:
    """模拟 LLM，预设工具调用序列。"""

    def __init__(self, responses: list[LLMResponse]):
        self.responses = responses
        self.call_count = 0
        self._model = "mock"

    async def generate(self, messages, tools=None, **kwargs):
        if self.call_count >= len(self.responses):
            return LLMResponse(content="[无更多预设响应]")
        resp = self.responses[self.call_count]
        self.call_count += 1
        return resp

    async def generate_stream(self, messages, tools=None, **kwargs):
        resp = await self.generate(messages, tools, **kwargs)
        if resp.content:
            for char in resp.content:
                yield LLMResponse(content=char)

    def count_tokens(self, text: str) -> int:
        return len(text) // 4


# ── 演示 1: CLIApproval 工作原理 ──────────────────

async def demo_cli_approval_concept():
    """演示 CLIApproval 的基本机制。

    在实际使用中，CLIApproval 会在终端请求用户输入。
    这里我们展示其 API 和控制逻辑。
    """
    print("=" * 60)
    print("演示 1: CLIApproval 工作机制")
    print("=" * 60)
    print()

    approval = CLIApproval(enabled=True)

    print("CLIApproval 是同步回调，在每次工具调用前触发:")
    print()
    print("  async def __call__(self, tool_name, arguments) -> bool:")
    print("      print('工具:', tool_name)")
    print("      print('参数:', arguments)")
    print("      resp = input('批准? [y/N]: ')")
    print("      return resp.lower() in ('y', 'yes')")
    print()

    print("当 approval_callback 返回 True → 执行工具")
    print("当 approval_callback 返回 False → 跳过工具，Agent 收到拒绝通知")
    print()

    # 演示 enabled 开关
    approval.enabled = False
    result = await approval("delete_files", {"path": "/tmp/test.txt"})
    print(f"enabled=False 时自动批准: {result}")
    print()


# ── 演示 2: 自定义审批回调 ────────────────────────

async def demo_custom_approval():
    """演示三种自定义审批策略。"""
    print("=" * 60)
    print("演示 2: 自定义审批回调")
    print("=" * 60)
    print()

    # ── 策略 A: 白名单自动批准 ──
    class WhitelistApproval:
        """白名单内的工具自动批准，其他工具拒绝。"""

        SAFE_TOOLS = {"read_file", "list_files", "calculator", "web_search"}

        def __init__(self):
            self.approved: list[str] = []
            self.denied: list[str] = []

        async def __call__(self, tool_name: str, arguments: dict) -> bool:
            if tool_name in self.SAFE_TOOLS:
                self.approved.append(tool_name)
                return True
            self.denied.append(tool_name)
            print(f"    [安全策略] 拒绝危险工具: {tool_name}")
            return False

    whitelist = WhitelistApproval()
    print("策略 A: 白名单自动审批")
    print(f"  安全工具: {WhitelistApproval.SAFE_TOOLS}")
    print()

    for tool in ["read_file", "write_file", "list_files", "delete_files"]:
        ok = await whitelist(tool, {})
        status = "批准" if ok else "拒绝"
        print(f"    {tool}: {status}")
    print()

    # ── 策略 B: 日志记录审批 ──
    class AuditApproval:
        """记录所有审批请求，用于审计追踪。"""

        def __init__(self):
            self.audit_log: list[dict] = []

        async def __call__(self, tool_name: str, arguments: dict) -> bool:
            from datetime import datetime, timezone
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tool": tool_name,
                "arguments": arguments,
            }
            self.audit_log.append(entry)
            # 实际场景中会通知人类审批者
            return True  # 演示中直接批准

    audit = AuditApproval()
    print("策略 B: 审计日志记录")
    await audit("write_file", {"path": "/tmp/test.txt", "content": "hello"})
    await audit("read_file", {"path": "/tmp/test.txt"})
    for entry in audit.audit_log:
        print(f"    [{entry['timestamp'][:19]}] {entry['tool']}: {entry['arguments']}")
    print()

    # ── 策略 C: 参数级安全校验 ──
    class ParamGuardApproval:
        """基于参数的精细审批。"""

        DANGEROUS_PATHS = {"/etc", "/System", "C:\\Windows", "C:\\Windows\\System32"}

        async def __call__(self, tool_name: str, arguments: dict) -> bool:
            path = arguments.get("path", "")
            if tool_name in ("write_file",) and any(
                str(path).startswith(d) for d in self.DANGEROUS_PATHS
            ):
                print(f"    [安全策略] 禁止写入系统路径: {path}")
                return False
            return True

    guard = ParamGuardApproval()
    print("策略 C: 参数级安全校验")
    print(f"  受保护路径: {ParamGuardApproval.DANGEROUS_PATHS}")
    print()

    for path in ["/tmp/notes.txt", "/etc/passwd", "C:\\Windows\\hosts"]:
        ok = await guard("write_file", {"path": path, "content": "data"})
        status = "批准" if ok else "拒绝"
        print(f"    write_file({path}): {status}")
    print()


# ── 演示 3: Agent 集成 ────────────────────────────

async def demo_agent_integration():
    """演示审批回调如何影响 Agent 的工具调用流程。

    Agent 尝试调用 write_file，审批拒绝后 Agent 收到错误通知。
    """
    print("=" * 60)
    print("演示 3: Agent 集成（审批拒绝后 Agent 的响应）")
    print("=" * 60)
    print()

    # 创建一个始终拒绝 delete_files 的审批回调
    class DenyDeleteApproval:
        async def __call__(self, tool_name: str, arguments: dict) -> bool:
            if tool_name == "delete_files":
                print(f"    [审批] 拒绝 delete_files({arguments.get('path', '?')})")
                return False
            return True

    tools = ToolRegistry()
    tools.register(delete_files)  # requires_approval=True
    tools.register(read_file)
    tools.register(list_files)

    agent = ReActAgent(
        name="受限助手",
        llm=MockLLM([
            # 第 1 轮: Agent 尝试删除文件
            LLMResponse(
                content="",
                tool_calls=[{
                    "id": "call_1",
                    "name": "delete_files",
                    "arguments": {"path": "/tmp/secret.txt"},
                }],
            ),
            # 第 2 轮: 收到拒绝通知后，Agent 调整策略
            LLMResponse(content=(
                "我尝试删除文件但被安全策略拒绝了。"
                "这表明删除 /tmp/secret.txt 是不被允许的。"
                "建议您手动检查文件权限或联系管理员。"
            )),
        ]),
        tools=tools,
        approval_callback=DenyDeleteApproval(),
        system_prompt="你是一个受安全策略约束的 AI 助手。",
    )

    result = await agent.run("请删除 /tmp/secret.txt 文件")
    print(f"    [Agent 输出] {result.content}")
    records = [(r.tool_name, "error=" + r.error if r.error else "ok") for r in result.tool_calls_history]
    print(f"    [工具调用记录] {records}")
    print()

    print("审批流程:")
    print("  1. LLM 产生 delete_files 的 tool_call")
    print("  2. ReActAgent._act() 检查 requires_approval 属性")
    print("  3. 调用 approval_callback → 拒绝")
    print("  4. 不执行工具，追加 tool 拒绝消息到对话历史")
    print("  5. LLM 看到拒绝消息，调整回答策略")
    print()


# ── 主流程 ──────────────────────────────────────

async def main():
    await demo_cli_approval_concept()
    await demo_custom_approval()
    await demo_agent_integration()
    print("=" * 60)
    print("全部 Human-in-the-Loop 演示完成。")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
