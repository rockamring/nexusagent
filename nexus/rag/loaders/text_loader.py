"""TextLoader — 文本文件加载器。

支持 .txt/.md/.py/.json/.csv/.yaml 等常见文本格式。
"""

from __future__ import annotations

from pathlib import Path

from nexus.rag.loaders.base import BaseLoader
from nexus.rag.types import Document


class TextLoader(BaseLoader):
    """加载纯文本文件。

    支持的文件扩展名见 SUPPORTED_EXTENSIONS，编码回退链处理中文等编码。

    用法:
        loader = TextLoader()
        docs = await loader.load("path/to/file.md")
        # 返回 [Document(content="...全文...", metadata={"source": ..., "file_name": ..., "file_type": ".md"})]
    """

    SUPPORTED_EXTENSIONS = frozenset({
        ".txt", ".md", ".py", ".json", ".csv", ".yaml", ".yml",
        ".xml", ".html", ".css", ".js", ".ts", ".sh", ".bat",
        ".ini", ".cfg", ".toml", ".rst", ".tex", ".log",
    })

    def __init__(self, encoding: str = "utf-8", fallback_encodings: list[str] | None = None):
        """初始化文本加载器。

        Args:
            encoding: 首选编码
            fallback_encodings: 编码回退列表，默认 ["gbk", "latin-1"]
        """
        self._encoding = encoding
        self._fallback_encodings = fallback_encodings or ["gbk", "latin-1"]

    async def load(self, path: str) -> list[Document]:
        """加载单个文本文件为 Document。"""
        file_path = Path(path).resolve()
        if not file_path.is_file():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        content = self._read_with_fallback(file_path)

        return [Document(
            content=content,
            metadata={
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": file_path.suffix.lower(),
            },
        )]

    def _read_with_fallback(self, path: Path) -> str:
        """尝试用首选编码读取，失败则按回退链依次尝试。"""
        encodings = [self._encoding] + self._fallback_encodings
        for enc in encodings:
            try:
                return path.read_text(encoding=enc)
            except (UnicodeDecodeError, UnicodeError):
                continue
        # 最后兜底：按字节读取，用 errors='replace' 替换无法解码的字符
        return path.read_bytes().decode(self._encoding, errors="replace")
