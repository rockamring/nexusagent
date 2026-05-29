"""内置工具：计算器。"""

from nexus.tools.base import Tool


@Tool.from_function(
    name="calculator",
    description="执行数学表达式计算。支持 +, -, *, /, **, sqrt, sin, cos 等基本运算。输入为一个数学表达式字符串。",
)
async def calculator(expression: str) -> str:
    """安全地计算数学表达式。

    使用受限的 eval 环境，仅暴露 math 模块中的安全函数。
    """
    import math

    allowed = {
        "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
        "sqrt": math.sqrt, "pow": pow,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "asin": math.asin, "acos": math.acos, "atan": math.atan,
        "log": math.log, "log10": math.log10, "log2": math.log2,
        "pi": math.pi, "e": math.e, "tau": math.tau,
        "ceil": math.ceil, "floor": math.floor,
        "radians": math.radians, "degrees": math.degrees,
    }

    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as exc:
        return f"计算错误: {exc}"
