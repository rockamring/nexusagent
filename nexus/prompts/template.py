"""Prompt 模板引擎 — 支持 {{ variable }} 语法的轻量模板系统。

设计理念:
- 零外部依赖：用正则实现变量替换，不依赖 Jinja2
- Jinja2 兼容语法：{{ var }} 格式，未来可无缝切换
- 简单安全：未传入的变量保留原样，不会崩溃
- 文件加载：支持从外部 .txt 文件加载大型提示词

用法:
    template = PromptTemplate("你是{{ role }}，请用{{ language }}回答。")
    prompt = template.render(role="助手", language="中文")
"""

from __future__ import annotations

import re
from pathlib import Path

_VAR_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")


class PromptTemplate:
    r"""支持 {{ variable }} 语法的提示词模板。

    变量名规则: 字母、数字、下划线（\w+），花括号内空格可选。

    用法:
        template = PromptTemplate("你是{{ role }}，请用{{ language }}回答。")
        prompt = template.render(role="助手", language="中文")

        # 从文件加载
        template = PromptTemplate.from_file("prompts/analyst.txt")
    """

    def __init__(self, template: str):
        self.template = template

    def render(self, **variables: str) -> str:
        """将模板中的 {{ var }} 替换为对应变量值。

        未通过关键字参数传入的变量保留原样（不替换），
        这样可以看到哪些变量没有被填充。

        Args:
            **variables: 变量名和值的映射

        Returns:
            渲染后的字符串
        """
        def _replace(m: re.Match) -> str:
            key = m.group(1)
            if key in variables:
                return str(variables[key])
            return m.group(0)

        return _VAR_RE.sub(_replace, self.template)

    @classmethod
    def from_file(cls, path: str | Path) -> "PromptTemplate":
        """从文件加载模板内容。

        Args:
            path: 模板文件路径

        Returns:
            PromptTemplate 实例
        """
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return cls(content)

    def __repr__(self) -> str:
        preview = self.template[:60].replace("\n", "\\n")
        if len(self.template) > 60:
            preview += "..."
        return f"PromptTemplate({preview!r})"
