"""Example 6: RAG 文档问答 — 演示完整的检索增强生成流程。

无需 API Key 即可运行（使用 SHA-256 哈希作为回退嵌入）。

流程:
    1. 创建示例文档（产品手册、FAQ）
    2. 加载 → 切分 → 嵌入 → 存储（IngestionPipeline）
    3. 语义检索（Retriever）
    4. 结合 RAG 提示词模板生成 LLM 就绪的 prompt

运行:
    python examples/06_rag_qa.py
"""

import asyncio
import tempfile
from pathlib import Path

from nexus.prompts.presets import RAG_CITE, RAG_QA
from nexus.rag.knowledge_store import KnowledgeStore
from nexus.rag.loaders.directory_loader import DirectoryLoader
from nexus.rag.pipeline import IngestionPipeline
from nexus.rag.retriever import Retriever
from nexus.rag.splitters.recursive_splitter import RecursiveCharacterTextSplitter

# ── 示例文档 ────────────────────────────────────

SAMPLE_DOCS = {
    "产品手册.txt": """
NexusAgent 产品手册 v2.0
======================

概述
----
NexusAgent 是一个模块化 AI Agent 框架，专为学习和原型设计而开发。
它采用 5 层架构设计，从底到顶分别为：
  - Core 层：公共类型定义 (Message, AgentResult, ToolCallRecord)
  - LLM 层：统一大模型调用接口，支持 OpenAI / Anthropic / Google
  - Tools 层：函数 → JSON Schema 自动推导，支持 Tool.from_function 装饰器
  - Agent 层：ReAct 主循环（Think → Act → Observe）
  - Memory 层：短期/长期/向量记忆

安装
----
需要 Python 3.11+，通过 pip 安装：

    pip install -e ".[dev]"

快速开始
--------
    from nexus.llm import create_llm
    from nexus.agent import ReActAgent

    llm = create_llm()
    agent = ReActAgent(name="助手", llm=llm)
    result = await agent.run("你好！")

配置
----
支持通过 config.yaml 或环境变量配置：

    default_provider: openai
    providers:
      openai:
        api_key: "${OPENAI_API_KEY}"
        default_model: "gpt-4o-mini"

RAG 模块
--------
Tier 4 新增的 RAG（检索增强生成）模块支持：
  - 文档加载：TextLoader（18 种文本格式）、DirectoryLoader（递归目录）
  - 文本切分：CharacterTextSplitter、RecursiveCharacterTextSplitter
  - 向量存储：基于 ChromaDB 的 KnowledgeStore
  - 检索器：Retriever 封装检索 + 格式化输出
  - 采集流水线：IngestionPipeline（Load → Split → Embed → Store）
""",

    "FAQ.txt": """
常见问题 (FAQ)
==============

Q: NexusAgent 支持哪些 LLM Provider？
A: 目前支持 OpenAI、Anthropic Claude、Google Gemini 三个主流 Provider。
   通过统一的 BaseLLM 接口，切换 Provider 只需一行代码。

Q: 如何添加自定义工具？
A: 使用 @Tool.from_function 装饰器：

    from nexus.tools.base import Tool

    @Tool.from_function(name="my_tool", description="我的工具")
    async def my_tool(param: str) -> str:
        return f"处理结果: {param}"

   装饰器会自动从类型注解推导 JSON Schema。

Q: Memory 和 KnowledgeStore 有什么区别？
A: Memory 存储对话消息（Message），用于 Agent 的对话上下文。
   KnowledgeStore 存储文档块（Document），用于 RAG 检索。
   两者数据和生命周期不同，不共用 ChromaDB collection。

Q: 支持流式输出吗？
A: 支持。LLM Provider 实现 generate_stream() 方法后，
   Agent 通过 run_stream() 返回异步迭代器，可逐 token 输出。

Q: 如何处理工具调用的安全问题？
A: 通过 Human-in-the-Loop 机制。使用 CLIApproval 回调，
   在危险操作（如文件删除）执行前请求人类确认。
""",

    "最佳实践.md": """
# RAG 最佳实践

## 文档切分策略

- 代码文件：使用 CharacterTextSplitter，chunk_size=500, overlap=50
- 长文文档：使用 RecursiveCharacterTextSplitter，chunk_size=1000, overlap=200
- 中文文档：RecursiveCharacterTextSplitter 会自动在中文句号处切分

## Embedding 选择

- 生产环境建议使用 OpenAI text-embedding-3-small（1536 维）
- 测试环境可使用 SHA-256 哈希回退（无需 API Key）
- 通过 BaseEmbeddingProvider 依赖注入，切换 Embedding 不影响上层代码

## 检索优化

- top_k 建议设置为 3-5，过多无关内容会干扰 LLM 回答
- 使用 RAG_CITE 模板可要求 LLM 标注信息来源
- 元数据过滤可提升检索精度（如按文件类型、日期筛选）

## 提示词设计

- RAG_QA 模板：简洁问答，适合直接回答的场景
- RAG_CITE 模板：带来源引用，适合需要溯源可查的场景
- 始终在提示词中明确"不要编造信息"，降低幻觉风险
""",
}


def create_sample_docs(base_dir: Path) -> Path:
    """创建示例文档目录。"""
    docs_dir = base_dir / "docs"
    docs_dir.mkdir(exist_ok=True)
    for filename, content in SAMPLE_DOCS.items():
        (docs_dir / filename).write_text(content, encoding="utf-8")
    return docs_dir


# ── 主流程 ──────────────────────────────────────

async def main():
    print("=" * 60)
    print("RAG 文档问答示例")
    print("=" * 60)
    print()

    # ── 步骤 1: 创建示例文档 ─────────────────
    print("[步骤 1] 创建示例文档...")
    tmpdir = Path(tempfile.mkdtemp(prefix="nexus_rag_example_"))
    docs_dir = create_sample_docs(tmpdir)
    for f in sorted(docs_dir.iterdir()):
        print(f"    {f.name} ({len(f.read_text(encoding='utf-8'))} 字符)")
    print()

    # ── 步骤 2: 构建知识库并索引 ─────────────
    print("[步骤 2] 索引文档到向量知识库...")

    # 使用 SHA-256 回退嵌入（无需 API Key）
    # 生产环境可替换为:
    #   from nexus.embeddings import OpenAIEmbeddingProvider
    #   provider = OpenAIEmbeddingProvider(api_key="sk-...")
    store = KnowledgeStore(
        persist_dir=str(tmpdir / "rag_data"),
        collection_name="example_kb",
    )

    pipeline = IngestionPipeline(
        loader=DirectoryLoader(),
        splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100),
        knowledge_store=store,
    )

    chunk_count = await pipeline.ingest_directory(str(docs_dir))
    print(f"    共索引 {chunk_count} 个文本块")
    print("    向量维度: 384 (SHA-256 回退)")
    print()

    # ── 步骤 3: 语义检索 ─────────────────────
    print("[步骤 3] 语义检索演示")
    print("-" * 40)

    retriever = Retriever(store)

    queries = [
        "如何安装这个框架？",
        "支持哪些大模型？",
        "怎么添加自定义工具？",
        "RAG 的 chunk_size 应该设多少？",
    ]

    for query in queries:
        results = await retriever.retrieve(query, k=2)
        print(f"  查询: {query}")
        for i, r in enumerate(results):
            source = r.metadata.get("file_name", "unknown")
            print(f"    [{i+1}] {source} (score={r.score:.3f}): {r.content[:80]}...")
        print()

    # ── 步骤 4: 格式化检索结果 ───────────────
    print("[步骤 4] 格式化上下文（retrieve_formatted）")
    print("-" * 40)

    formatted = await retriever.retrieve_formatted(
        "RAG 模块有哪些组件？", k=3
    )
    print(formatted)
    print()

    # ── 步骤 5: 结合 RAG 提示词模板 ──────────
    print("[步骤 5] 结合 RAG 提示词模板")
    print("-" * 40)

    # 检索相关上下文
    context = await retriever.retrieve_formatted(
        "RAG 模块的文档切分策略", k=3
    )

    # 使用 RAG_QA 模板
    rag_prompt = RAG_QA.render(
        context=context,
        question="RAG 模块应该使用什么切分策略？",
    )
    print("── RAG_QA 模板渲染结果 ──")
    print(rag_prompt[:600])
    print("...(截断)")
    print()

    # 使用 RAG_CITE 模板
    cite_prompt = RAG_CITE.render(
        context=context,
        question="RAG 模块的 Embedding 选择有什么建议？",
    )
    print("── RAG_CITE 模板渲染结果 ──")
    print(cite_prompt[:600])
    print("...(截断)")
    print()

    # ── 步骤 6: 按源过滤和删除 ───────────────
    print("[步骤 6] 知识库管理")
    print("-" * 40)
    print(f"    索引前文档数: {store.count}")

    deleted = store.delete_by_source("FAQ.txt")
    print(f"    删除 'FAQ.txt' 的 {deleted} 个块后: {store.count}")

    # 重新搜索（FAQ 结果消失）
    results = await retriever.retrieve("如何添加自定义工具？", k=2)
    sources = [r.metadata.get("file_name", "") for r in results]
    print(f"    搜索 '如何添加自定义工具' 来源: {sources}")
    print("    (FAQ.txt 已删除，应由其他文档匹配)")
    print()

    # ── 清理 ──────────────────────────────────
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("[完成] 示例执行完毕。")


if __name__ == "__main__":
    asyncio.run(main())
