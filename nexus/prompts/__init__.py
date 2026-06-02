"""Prompt 模板系统 — 动态构建提示词。

提供:
- PromptTemplate: 支持 {{ variable }} 语法的模板引擎
- presets: 开箱即用的预设模板集合

用法:
    from nexus.prompts import PromptTemplate, presets

    # 使用预设
    prompt = presets.CHINESE_ASSISTANT.render(language="中文", role="客服")

    # 自定义模板
    template = PromptTemplate("你是{{ role }}，请用{{ language }}回答。")
    prompt = template.render(role="数据分析师", language="中文")

    # 在 Agent 中使用
    agent = ReActAgent(name="助手", llm=llm, system_prompt=prompt)
"""

from nexus.prompts.template import PromptTemplate

from . import presets

__all__ = ["PromptTemplate", "presets"]
