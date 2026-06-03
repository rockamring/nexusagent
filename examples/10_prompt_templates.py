"""Example 10: 提示词模板 — 演示 PromptTemplate 和预设模板。

无需 API Key 即可运行（纯模板渲染演示）。

演示:
    1. PromptTemplate 基本用法：变量替换
    2. 从文件加载模板
    3. 预设模板：RAG_QA、RAG_CITE、CODE_REVIEWER 等
    4. 自定义模板 + 组合使用

运行:
    python examples/10_prompt_templates.py
"""

import asyncio
import tempfile
from pathlib import Path

from nexus.prompts import presets
from nexus.prompts.template import PromptTemplate

# ── 演示 1: 基本变量替换 ─────────────────────────

async def demo_basic():
    """演示 {{ variable }} 语法和渲染规则。"""
    print("=" * 60)
    print("演示 1: 基本变量替换")
    print("=" * 60)
    print()

    # 创建模板
    t = PromptTemplate(
        "你是一个{{ role }}，请用{{ language }}回答问题。\n"
        "你的专长领域是{{ expertise }}。"
    )

    print(f"模板: {t}")
    print()

    # 完整渲染
    result = t.render(role="Python 导师", language="中文", expertise="Web 开发")
    print("完整渲染:")
    print(result)
    print()

    # 部分渲染（缺少 expertise）
    result = t.render(role="助手", language="英文")
    print("部分渲染（缺少 expertise）:")
    print(result)
    print("  ↑ 未传入的变量保留 {{ }} 原样，不会崩溃")
    print()

    # 缺少全部变量
    result = t.render()
    print("无参数渲染:")
    print(result)
    print("  ↑ 全部变量保留原样，可以看到模板结构")
    print()


# ── 演示 2: 从文件加载模板 ────────────────────────

async def demo_file_loading():
    """演示 PromptTemplate.from_file() 加载外部模板文件。"""
    print("=" * 60)
    print("演示 2: 从文件加载模板")
    print("=" * 60)
    print()

    # 创建临时模板文件
    tmpdir = Path(tempfile.mkdtemp(prefix="nexus_prompt_"))
    template_file = tmpdir / "code_review.txt"
    template_file.write_text(
        "你是一个资深代码审查专家。\n\n"
        "审查重点:\n"
        "1. {{ focus_1 }}\n"
        "2. {{ focus_2 }}\n"
        "3. {{ focus_3 }}\n\n"
        "请对以下 {{ language }} 代码进行审查:\n"
        "```\n"
        "{{ code }}\n"
        "```\n\n"
        "输出格式: {{ output_format }}",
        encoding="utf-8",
    )

    # 从文件加载
    t = PromptTemplate.from_file(str(template_file))
    print(f"从文件加载: {template_file}")
    print(f"模板预览: {t}")
    print()

    # 渲染
    result = t.render(
        focus_1="代码安全性（SQL 注入、XSS 等）",
        focus_2="性能优化（算法复杂度、缓存策略）",
        focus_3="代码可维护性（命名、结构、注释）",
        language="Python",
        code='query = f"SELECT * FROM users WHERE id = {user_id}"',
        output_format="Markdown 表格，列出问题、严重程度、修复建议",
    )
    print("渲染结果:")
    print(result)
    print()

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


# ── 演示 3: 预设模板一览 ──────────────────────────

async def demo_presets():
    """展示所有预设模板及其渲染效果。"""
    print("=" * 60)
    print("演示 3: 预设模板一览")
    print("=" * 60)
    print()

    # ── 通用助手 ──
    print("── 通用助手 ──")
    print(f"  ASSISTANT: {presets.ASSISTANT.render()}")
    print()
    print(f"  CHINESE_ASSISTANT: {presets.CHINESE_ASSISTANT.render(language='中文', role='客服')}")
    print()

    # ── 代码相关 ──
    print("── 代码相关 ──")
    print("  CODE_REVIEWER:")
    print(presets.CODE_REVIEWER.render(
        language="Python",
        focus_areas="安全性、性能、可维护性",
    ))
    print()
    print("  CODE_WRITER:")
    print(presets.CODE_WRITER.render(
        language="Python",
        requirements="实现一个 LRU 缓存，支持 O(1) 的 get/put 操作",
    ))
    print()

    # ── 角色扮演 ──
    print("── 角色扮演 ──")
    print("  ROLE_PLAY:")
    print(presets.ROLE_PLAY.render(
        role="技术面试官",
        background="你在一家顶尖科技公司工作，正在面试一位高级后端工程师",
        goal="评估候选人的系统设计能力",
    ))
    print()

    # ── 数据分析 ──
    print("── 数据分析 ──")
    print("  DATA_ANALYST:")
    print(presets.DATA_ANALYST.render(
        tool="Pandas",
        format="Markdown 图表",
        dimensions="时间趋势、地域分布、用户分群",
    ))
    print()

    # ── 多 Agent 协作 ──
    print("── 多 Agent 协作 ──")
    print("  SUPERVISOR:")
    print(presets.SUPERVISOR.render(
        team_size="3",
        members="产品分析师、系统架构师、前端开发",
        goal="设计一个在线教育平台的 MVP 方案",
    ))
    print()
    print("  WORKER:")
    print(presets.WORKER.render(
        role="前端开发",
        expertise="React 18 + TypeScript + Tailwind CSS",
    ))
    print()


# ── 演示 4: RAG 提示词模板 ────────────────────────

async def demo_rag_templates():
    """演示 RAG_QA 和 RAG_CITE 模板的实际用法。"""
    print("=" * 60)
    print("演示 4: RAG 提示词模板（RAG_QA / RAG_CITE）")
    print("=" * 60)
    print()

    # 模拟从知识库检索到的上下文
    context = """[Source: docs/product.md, chunk 1/3]
NexusAgent 使用 ChromaDB 作为向量数据库，支持余弦距离（cosine distance）。

[Source: docs/guide.md, chunk 2/5]
Embedding 模型支持 OpenAI text-embedding-3-small 和 SHA-256 回退。
生产环境建议使用 OpenAI Embedding，测试环境可使用回退方案。"""

    question = "NexusAgent 使用什么向量数据库？"

    # RAG_QA: 简洁问答
    print("── RAG_QA 模板 ──")
    qa_prompt = presets.RAG_QA.render(context=context, question=question)
    print(qa_prompt)
    print()

    # RAG_CITE: 带来源引用
    print("── RAG_CITE 模板 ──")
    cite_prompt = presets.RAG_CITE.render(context=context, question=question)
    print(cite_prompt)
    print()

    print("两种模板的区别:")
    print("  RAG_QA  → 简洁回答，不需要标注来源")
    print("  RAG_CITE → 每个关键信息标注 [Source: 文件名]")
    print()


# ── 演示 5: 自定义模板 + 组合使用 ─────────────────

async def demo_custom_composition():
    """演示模板组合和实际使用场景。"""
    print("=" * 60)
    print("演示 5: 自定义模板组合")
    print("=" * 60)
    print()

    # 场景: 创建一个带 RAG 的代码助手模板
    code_rag_template = PromptTemplate(
        "你是一个{{ language }}编程助手。\n\n"
        "规则:\n"
        "1. 优先根据以下参考文档回答问题\n"
        "2. 文档中找不到的信息，基于你的知识回答但要注明\n"
        "3. 代码示例要完整可运行\n\n"
        "参考文档:\n"
        "{{ retrieved_context }}\n\n"
        "用户问题: {{ question }}\n\n"
        "回答:"
    )

    # 模拟检索到的文档
    retrieved = """[Source: nexus/docs/api.md]
ReActAgent.run(user_input: str) -> AgentResult
  执行 Think→Act→Observe 主循环，返回最终回复和工具调用历史。

[Source: nexus/docs/guide.md]
创建 Agent 示例:
  agent = ReActAgent(name="助手", llm=llm, tools=tools, memory=memory)
  result = await agent.run("用户输入")"""

    prompt = code_rag_template.render(
        language="Python",
        retrieved_context=retrieved,
        question="如何创建一个带记忆和工具的 Agent？",
    )

    print(prompt)
    print()

    print("模板组合的优势:")
    print("  1. 分离关注点：模板结构 vs 变量值")
    print("  2. 版本控制友好：模板文件可纳入 Git")
    print("  3. 非技术人员可编辑：只需了解 {{ var }} 语法")
    print("  4. A/B 测试：切换模板即可对比效果")
    print()


# ── 主流程 ──────────────────────────────────────

async def main():
    await demo_basic()
    await demo_file_loading()
    await demo_presets()
    await demo_rag_templates()
    await demo_custom_composition()
    print("=" * 60)
    print("全部提示词模板演示完成。")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
