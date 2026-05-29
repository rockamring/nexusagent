"""内置工具：文件操作。"""

import os

from nexus.tools.base import Tool


@Tool.from_function(
    name="read_file",
    description="读取指定文件的内容。返回文件全文（文本文件）。",
)
async def read_file(filepath: str) -> str:
    """读取文本文件内容。"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        if len(content) > 10000:
            content = content[:10000] + "\n...(内容过长，已截断)"
        return content
    except FileNotFoundError:
        return f"文件不存在: {filepath}"
    except PermissionError:
        return f"没有读取权限: {filepath}"
    except Exception as exc:
        return f"读取文件失败: {exc}"


@Tool.from_function(
    name="write_file",
    description="将内容写入指定文件。会覆盖已有文件，请谨慎使用。",
)
async def write_file(filepath: str, content: str) -> str:
    """写入文本文件。"""
    try:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"文件写入成功: {filepath} ({len(content)} 字符)"
    except Exception as exc:
        return f"写入文件失败: {exc}"


@Tool.from_function(
    name="list_files",
    description="列出指定目录下的所有文件和子目录。",
)
async def list_files(directory: str = ".") -> str:
    """列出目录内容。"""
    try:
        entries = os.listdir(directory)
        if not entries:
            return f"目录 '{directory}' 为空"
        lines = []
        for name in sorted(entries):
            full = os.path.join(directory, name)
            tag = "[DIR]" if os.path.isdir(full) else "[FILE]"
            lines.append(f"  {tag} {name}")
        return f"目录 '{directory}' 的内容:\n" + "\n".join(lines)
    except FileNotFoundError:
        return f"目录不存在: {directory}"
    except Exception as exc:
        return f"列出目录失败: {exc}"
