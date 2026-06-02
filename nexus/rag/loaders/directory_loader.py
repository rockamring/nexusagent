"""DirectoryLoader — 递归加载目录中所有支持的文本文件。"""

from __future__ import annotations

from pathlib import Path

from nexus.rag.loaders.base import BaseLoader
from nexus.rag.loaders.text_loader import TextLoader
from nexus.rag.types import Document


class DirectoryLoader(BaseLoader):
    """递归加载目录中所有支持格式的文本文件。

    将单个文件的加载委托给 TextLoader。

    用法:
        loader = DirectoryLoader()
        docs = await loader.load("path/to/docs/")
        # 返回 [Document(...), Document(...), ...] — 每个文件一个 Document

        # 仅加载 Markdown 文件
        loader = DirectoryLoader(glob_pattern="**/*.md")
        docs = await loader.load("path/to/docs/")
    """

    def __init__(
        self,
        text_loader: TextLoader | None = None,
        recursive: bool = True,
        glob_pattern: str | None = None,
    ):
        """初始化目录加载器。

        Args:
            text_loader: 用于加载单个文件的 TextLoader 实例，内部创建默认为 None
            recursive: 是否递归进入子目录
            glob_pattern: 可选的 glob 模式过滤文件（如 "**/*.md"）
                          指定后覆盖 SUPPORTED_EXTENSIONS 过滤
        """
        self._text_loader = text_loader or TextLoader()
        self._recursive = recursive
        self._glob_pattern = glob_pattern

    async def load(self, path: str) -> list[Document]:
        """加载目录中所有支持的文件。"""
        dir_path = Path(path).resolve()
        if not dir_path.is_dir():
            raise NotADirectoryError(f"不是目录: {dir_path}")

        if self._glob_pattern:
            file_paths = list(dir_path.glob(self._glob_pattern))
        else:
            if self._recursive:
                file_paths = list(dir_path.rglob("*"))
            else:
                file_paths = list(dir_path.glob("*"))
            file_paths = [
                fp for fp in file_paths
                if fp.is_file() and fp.suffix.lower() in TextLoader.SUPPORTED_EXTENSIONS
            ]

        documents: list[Document] = []
        for file_path in sorted(file_paths):
            try:
                docs = await self._text_loader.load(str(file_path))
                documents.extend(docs)
            except Exception:
                # 单个文件加载失败不中断整个目录加载
                continue

        return documents
