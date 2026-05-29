"""Tool 系统测试。"""

import pytest

from nexus.tools.base import Tool
from nexus.tools.registry import ToolRegistry
from nexus.tools.schema import build_schema


# ── Schema 生成测试 ──────────────────────────

def test_build_schema_simple():
    async def test_fn(name: str, count: int = 1) -> str:
        return "ok"

    schema = build_schema(test_fn, "test_tool", "测试工具")

    assert schema["name"] == "test_tool"
    assert schema["description"] == "测试工具"
    assert schema["parameters"]["type"] == "object"
    assert "name" in schema["parameters"]["properties"]
    assert "count" in schema["parameters"]["properties"]
    assert schema["parameters"]["properties"]["name"]["type"] == "string"
    assert schema["parameters"]["properties"]["count"]["type"] == "integer"
    assert "name" in schema["parameters"]["required"]
    assert "count" not in schema["parameters"]["required"]


def test_build_schema_types():
    async def test_fn(
        text: str,
        number: int,
        price: float,
        flag: bool,
        items: list,
    ) -> str:
        return "ok"

    schema = build_schema(test_fn, "types", "类型测试")
    props = schema["parameters"]["properties"]

    assert props["text"]["type"] == "string"
    assert props["number"]["type"] == "integer"
    assert props["price"]["type"] == "number"
    assert props["flag"]["type"] == "boolean"
    assert props["items"]["type"] == "array"


# ── Tool 装饰器测试 ──────────────────────────

@pytest.mark.asyncio
async def test_tool_decorator():
    @Tool.from_function(name="greet", description="打招呼")
    async def greet(name: str) -> str:
        return f"你好, {name}!"

    assert greet.name == "greet"
    assert greet.description == "打招呼"
    assert greet.schema["name"] == "greet"

    result = await greet.execute(name="世界")
    assert result == "你好, 世界!"


@pytest.mark.asyncio
async def test_tool_sync_function():
    @Tool.from_function(name="add", description="加法")
    def add(a: int, b: int) -> int:
        return a + b

    result = await add.execute(a=1, b=2)
    assert result == "3"


# ── Registry 测试 ──────────────────────────

@pytest.mark.asyncio
async def test_registry_register_and_execute():
    @Tool.from_function(name="echo", description="回显")
    async def echo(message: str) -> str:
        return message

    registry = ToolRegistry()
    registry.register(echo)

    assert "echo" in registry
    assert len(registry) == 1

    result = await registry.execute("echo", message="hello")
    assert result.success
    assert result.result == "hello"


def test_registry_duplicate_error():
    @Tool.from_function(name="dup", description="重复")
    async def dup() -> str:
        return "dup"

    registry = ToolRegistry()
    registry.register(dup)

    with pytest.raises(ValueError, match="已注册"):
        registry.register(dup)


def test_registry_get_schemas():
    @Tool.from_function(name="t1", description="工具1")
    async def t1() -> str:
        return "1"

    @Tool.from_function(name="t2", description="工具2")
    async def t2() -> str:
        return "2"

    registry = ToolRegistry()
    registry.register_many([t1, t2])

    schemas = registry.get_schemas()
    assert len(schemas) == 2
    names = {s["name"] for s in schemas}
    assert names == {"t1", "t2"}
