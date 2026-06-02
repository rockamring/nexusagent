"""Human-in-the-Loop 安全审批机制。

在危险操作执行前请求人类确认，这是生产级 Agent 必需的安全机制。

提供:
- CLIApproval: 命令行交互式审批器

自定义审批回调只需实现:
    async def my_approval(tool_name: str, arguments: dict) -> bool:
        return True  # 批准，或 False 拒绝

用法:
    from nexus.agent.guard import CLIApproval

    agent = ReActAgent(
        name="助手", llm=llm, tools=tools,
        approval_callback=CLIApproval(),
    )
"""

from __future__ import annotations


class CLIApproval:
    """CLI 交互式审批器。

    在终端询问用户是否批准工具执行。
    可通过 enabled 属性全局开关审批功能。

    用法:
        approval = CLIApproval()
        approved = await approval("delete_files", {"path": "/tmp/test.txt"})
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    async def __call__(self, tool_name: str, arguments: dict) -> bool:
        """在 CLI 中询问用户是否批准执行。

        Args:
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            True 批准执行，False 拒绝
        """
        if not self.enabled:
            return True

        print(f"\n{'=' * 50}")
        print("[审批] 工具调用请求")
        print(f"  工具: {tool_name}")
        print(f"  参数: {arguments}")
        print(f"{'=' * 50}")

        while True:
            resp = input("  批准执行? [y/N]: ").strip().lower()
            if resp in ("y", "yes"):
                return True
            elif resp in ("", "n", "no"):
                return False
            print("  请输入 y (批准) 或 n (拒绝)")
