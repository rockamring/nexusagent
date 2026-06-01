"""NexusAgent Gradio Web UI — 通过浏览器与 Agent 交互。

启动:
    python -m nexus.ui
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import gradio as gr

from nexus.agent import ReActAgent
from nexus.core.config import NexusConfig
from nexus.llm.factory import create_llm
from nexus.memory import BufferMemory
from nexus.tools.builtin.calculator import calculator
from nexus.tools.builtin.file_ops import list_files, read_file, write_file
from nexus.tools.registry import ToolRegistry

# ============================================================
# 应用配置（启动时自动加载）
# ============================================================
_config = NexusConfig.from_file()


def _default_provider() -> str:
    return _config.default_provider


def _default_model() -> str:
    return _config.default_model


def _default_system_prompt() -> str:
    return (
        "你是一个友好的 AI 助手，名字叫 Nexus。"
        "请用中文回答问题。"
        "当用户要求计算时，使用 calculator 工具。"
        "当用户要求读写文件时，使用对应的文件工具。"
    )


# ============================================================
# 聊天核心逻辑
# ============================================================

async def _chat_fn(
    message: str,
    history: list[dict],
    provider: str,
    model: str,
    system_prompt: str,
    max_iterations: int,
    enable_memory: bool,
):
    """处理每条用户消息，流式返回助手回复。

    Gradio 5.x ChatInterface 的 fn 签名: (message, history) -> generator of messages
    支持通过 additional_inputs 传入额外参数。
    """
    # 1. 创建 LLM
    try:
        llm = create_llm(
            provider=provider,
            default_model=model,
        )
    except Exception as exc:
        error_msg = f"**错误**: 无法创建 LLM 实例。请检查 config.yaml 或环境变量中的 API Key 配置。\n\n```\n{exc}\n```"
        yield error_msg
        return

    # 2. 创建工具
    tools = ToolRegistry()
    tools.register(calculator)
    tools.register(read_file)
    tools.register(write_file)
    tools.register(list_files)

    # 3. 创建记忆（恢复历史）
    memory = None
    if enable_memory:
        memory = BufferMemory(max_messages=30)
        for entry in history:
            if entry.get("role") == "user":
                await memory.add({"role": "user", "content": entry["content"]})
            elif entry.get("role") == "assistant":
                await memory.add({"role": "assistant", "content": entry["content"]})

    # 4. 创建 Agent
    agent = ReActAgent(
        name="Nexus",
        llm=llm,
        tools=tools,
        memory=memory,
        system_prompt=system_prompt,
        max_iterations=max_iterations,
    )

    # 5. 执行——因为 stream() 在 tool_call 轮次会跳过流式，
    #    这里改用 run() 获取完整结果，再提取工具调用信息。
    result = await agent.run(message)

    # 6. 插入工具调用步骤到回复
    if result.tool_calls_history:
        parts = []
        for tc in result.tool_calls_history:
            args_str = ", ".join(f"{k}={v}" for k, v in tc.arguments.items())
            if tc.error:
                parts.append(f"🔧 `{tc.tool_name}({args_str})`\n> ❌ {tc.error}")
            else:
                result_preview = tc.result[:200] + ("..." if len(tc.result) > 200 else "")
                parts.append(f"🔧 `{tc.tool_name}({args_str})`\n> {result_preview}")
        tool_section = "\n\n".join(parts)
        final = f"{tool_section}\n\n---\n\n{result.content}"
    else:
        final = result.content

    yield final


# ============================================================
# UI 构建
# ============================================================

def create_ui() -> gr.Blocks:
    """创建 Gradio UI。"""
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    )

    with gr.Blocks(theme=theme, title="NexusAgent Chat") as demo:
        gr.Markdown("# NexusAgent Chat")

        with gr.Row():
            # ---- 侧边栏 ----
            with gr.Column(scale=1, min_width=240):
                gr.Markdown("### 配置")

                provider = gr.Dropdown(
                    choices=["openai", "anthropic"],
                    value=_default_provider(),
                    label="Provider",
                    interactive=True,
                )
                model = gr.Textbox(
                    value=_default_model(),
                    label="Model",
                    placeholder="gpt-4o-mini",
                )
                system_prompt = gr.Textbox(
                    value=_default_system_prompt(),
                    label="System Prompt",
                    lines=4,
                    placeholder="输入系统提示词...",
                )
                max_iterations = gr.Slider(
                    minimum=1,
                    maximum=30,
                    value=_config.agent.max_iterations,
                    step=1,
                    label="Max Iterations",
                )
                enable_memory = gr.Checkbox(
                    value=True,
                    label="启用记忆（跨轮对话）",
                )

            # ---- 聊天区域 ----
            with gr.Column(scale=3):
                chatbot = gr.ChatInterface(
                    fn=_chat_fn,
                    type="messages",
                    additional_inputs=[
                        provider,
                        model,
                        system_prompt,
                        max_iterations,
                        enable_memory,
                    ],
                    examples=[
                        "你好！请介绍一下你自己。",
                        "计算 (123 + 456) * 789",
                        "列出当前目录下的文件",
                    ],
                    title="",
                    description="",
                )

        gr.Markdown(
            "<small>API Key 从 config.yaml 或环境变量自动加载，不暴露在 UI 中。</small>",
            elem_id="footer",
        )

    return demo


def launch(**kwargs):
    """启动 Gradio Web UI。

    Args: 传递给 demo.launch() 的参数，如 server_name, server_port, share 等。
    """
    defaults = {
        "server_name": "127.0.0.1",
        "server_port": 7860,
        "show_error": True,
    }
    defaults.update(kwargs)

    demo = create_ui()
    print(f"启动 NexusAgent Chat: http://{defaults['server_name']}:{defaults['server_port']}")
    demo.launch(**defaults)


if __name__ == "__main__":
    launch()
