"""Prompt 模板系统测试。"""

import tempfile
from pathlib import Path

from nexus.prompts import PromptTemplate, presets


class TestPromptTemplate:
    """PromptTemplate 核心功能测试。"""

    def test_simple_render(self):
        t = PromptTemplate("你是{{ role }}，请用{{ language }}回答。")
        result = t.render(role="助手", language="中文")
        assert result == "你是助手，请用中文回答。"

    def test_multiple_variables(self):
        t = PromptTemplate("{{ greeting }}，我是{{ name }}，{{ action }}")
        result = t.render(greeting="你好", name="小明", action="请多指教")
        assert result == "你好，我是小明，请多指教"

    def test_unreplaced_variable(self):
        t = PromptTemplate("{{ a }} 和 {{ b }}")
        result = t.render(a="1")
        assert result == "1 和 {{ b }}"

    def test_no_variables(self):
        t = PromptTemplate("纯文本提示词，无变量。")
        result = t.render()
        assert result == "纯文本提示词，无变量。"

    def test_whitespace_in_braces(self):
        t = PromptTemplate("{{  name  }} 你好")
        result = t.render(name="World")
        assert result == "World 你好"

    def test_render_with_extra_kwargs(self):
        t = PromptTemplate("{{ x }}")
        result = t.render(x="1", y="2", z="3")
        assert result == "1"

    def test_numeric_variable(self):
        t = PromptTemplate("最高迭代{{ count }}次")
        result = t.render(count="10")
        assert result == "最高迭代10次"

    def test_from_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("你是{{ role }}，负责{{ task }}。")
            tmp_path = f.name

        try:
            t = PromptTemplate.from_file(tmp_path)
            result = t.render(role="测试员", task="质量保证")
            assert result == "你是测试员，负责质量保证。"
        finally:
            Path(tmp_path).unlink()

    def test_repr(self):
        t = PromptTemplate("你是助手")
        r = repr(t)
        assert r.startswith("PromptTemplate")
        assert "你是助手" in r

    def test_long_repr_truncated(self):
        long_template = "A" * 100
        t = PromptTemplate(long_template)
        r = repr(t)
        assert "..." in r

    def test_preserve_special_chars(self):
        t = PromptTemplate("JSON: {{ data }}")
        result = t.render(data='{"key": "value"}')
        assert result == 'JSON: {"key": "value"}'


class TestPresets:
    """预设模板测试。"""

    def test_assistant(self):
        result = presets.ASSISTANT.render()
        assert "有帮助" in result or "AI" in result

    def test_chinese_assistant(self):
        result = presets.CHINESE_ASSISTANT.render(language="中文", role="客服")
        assert "中文" in result
        assert "客服" in result

    def test_code_reviewer(self):
        result = presets.CODE_REVIEWER.render(
            language="Python",
            focus_areas="安全性、性能",
        )
        assert "Python" in result
        assert "安全性" in result

    def test_code_writer(self):
        result = presets.CODE_WRITER.render(
            language="Go",
            requirements="高性能并发处理",
        )
        assert "Go" in result
        assert "并发" in result

    def test_role_play(self):
        result = presets.ROLE_PLAY.render(
            role="医生",
            background="三甲医院主任医师",
            goal="回答患者的医学咨询",
        )
        assert "医生" in result
        assert "三甲医院" in result
        assert "医学咨询" in result

    def test_data_analyst(self):
        result = presets.DATA_ANALYST.render(
            tool="Pandas",
            format="Markdown 表格",
            dimensions="营收、用户增长",
        )
        assert "Pandas" in result
        assert "Markdown" in result
        assert "营收" in result

    def test_supervisor(self):
        result = presets.SUPERVISOR.render(
            team_size="3",
            members="分析师、开发者、测试员",
            goal="构建用户画像系统",
        )
        assert "3" in result
        assert "分析师" in result
        assert "用户画像" in result

    def test_worker(self):
        result = presets.WORKER.render(
            role="前端",
            expertise="React、TypeScript、CSS-in-JS",
        )
        assert "前端" in result
        assert "React" in result
