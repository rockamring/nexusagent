"""从 Python 函数签名自动生成 JSON Schema。

核心功能:
- 解析函数参数名、类型注解、默认值
- 映射 Python 类型到 JSON Schema 类型
- 自动识别必填/可选参数

类型映射:
    str   → {"type": "string"}
    int   → {"type": "integer"}
    float → {"type": "number"}
    bool  → {"type": "boolean"}
    list  → {"type": "array", "items": {...}}
    dict  → {"type": "object"}
"""

from __future__ import annotations

import inspect
from typing import Callable, get_type_hints


def build_schema(
    func: Callable,
    name: str,
    description: str,
) -> dict:
    """从函数签名和类型注解生成 OpenAI 兼容的 JSON Schema。

    Args:
        func: 源函数
        name: 工具名称（暴露给 LLM）
        description: 工具描述（LLM 用它判断何时调用）

    Returns:
        OpenAI 兼容的工具定义 dict:
        {
            "name": "get_weather",
            "description": "查询指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "..."}
                },
                "required": ["city"]
            }
        }
    """
    hints = get_type_hints(func)
    sig = inspect.signature(func)

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        param_type = hints.get(param_name, str)
        param_schema = _type_to_json_schema(param_type)

        # 从 docstring 中提取参数描述（仅 :param name: description 格式）
        param_desc = _extract_param_desc(func, param_name)
        if param_desc:
            param_schema["description"] = param_desc

        properties[param_name] = param_schema

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def _type_to_json_schema(py_type: type) -> dict:
    """Python 类型 → JSON Schema 类型映射。"""
    origin = getattr(py_type, "__origin__", None)

    if py_type is str:
        return {"type": "string"}
    if py_type is int:
        return {"type": "integer"}
    if py_type is float:
        return {"type": "number"}
    if py_type is bool:
        return {"type": "boolean"}
    if py_type is list or origin is list:
        args = getattr(py_type, "__args__", ())
        item_type = args[0] if args else str
        return {"type": "array", "items": _type_to_json_schema(item_type)}
    if origin is dict:
        return {"type": "object"}
    if py_type is type(None):
        return {"type": "null"}

    # 处理 Optional[X] → Union[X, None]
    if origin is not None:
        import types
        if origin is types.UnionType:
            args = getattr(py_type, "__args__", ())
            non_none = [a for a in args if a is not type(None)]
            if non_none:
                return _type_to_json_schema(non_none[0])

    return {"type": "string"}


def _extract_param_desc(func: Callable, param_name: str) -> str | None:
    """从函数的 Google/NumPy 风格 docstring 中提取参数描述。"""
    doc = inspect.getdoc(func)
    if not doc:
        return None

    for line in doc.split("\n"):
        line = line.strip()
        # 匹配 :param name: description 格式
        if line.startswith(f":param {param_name}:"):
            return line.split(":", 2)[-1].strip()
        # 匹配 Args: 块中的 name: description 格式
        if line.startswith(f"{param_name}:"):
            return line.split(":", 1)[-1].strip()

    return None
