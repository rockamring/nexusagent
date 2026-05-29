"""内置工具：Web 搜索。"""

from nexus.tools.base import Tool


@Tool.from_function(
    name="web_search",
    description="搜索互联网获取信息。输入搜索查询字符串，返回相关结果的摘要。",
)
async def web_search(query: str) -> str:
    """执行 Web 搜索。

    注意：这是一个示例实现，实际使用时需要接入真实的搜索 API
    （如 Google Custom Search、Bing Search API、SerpAPI 等）。
    """
    import urllib.request
    import urllib.parse
    import json

    # 使用 DuckDuckGo 的 Instant Answer API（免费，无需 API Key）
    url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1&skip_disambig=1"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "NexusAgent/0.1"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        results = []

        # Abstract（摘要答案）
        abstract = data.get("AbstractText", "")
        if abstract:
            source = data.get("AbstractSource", "")
            results.append(f"摘要: {abstract}")
            if source:
                results.append(f"来源: {source}")

        # Related Topics
        related = data.get("RelatedTopics", [])
        for topic in related[:5]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(f"- {topic['Text']}")

        if not results:
            return f"未找到与 '{query}' 相关的结果。"

        return "\n".join(results)

    except urllib.error.URLError as exc:
        return f"搜索请求失败（网络错误）: {exc}"
    except Exception as exc:
        return f"搜索失败: {exc}"
