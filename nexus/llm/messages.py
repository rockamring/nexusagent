"""消息格式转换工具。

将框架内部的统一 Message 格式转换为各 Provider 的原生格式。
"""

from nexus.core.types import Message, ToolCall


def to_openai_messages(messages: list[Message]) -> list[dict]:
    """将框架 Message 列表转为 OpenAI Messages API 格式。

    处理要点：
    - System/User 消息：直接映射
    - Assistant 消息（有 tool_calls）：映射为带 tool_calls 的 assistant 消息
    - Tool 消息：映射为 tool 角色消息，关联 tool_call_id
    """
    result = []
    for i, msg in enumerate(messages):
        role = msg["role"]

        if role == "system":
            result.append({"role": "system", "content": msg.get("content", "")})
        elif role == "user":
            result.append({"role": "user", "content": msg.get("content", "")})
        elif role == "assistant":
            entry: dict = {"role": "assistant", "content": msg.get("content")}
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                entry["tool_calls"] = _tool_calls_to_openai(tool_calls)
                entry["content"] = None
            result.append(entry)
        elif role == "tool":
            result.append({
                "role": "tool",
                "tool_call_id": msg.get("tool_call_id", ""),
                "content": msg.get("content", ""),
            })

    # 校验消息顺序：每条 tool 消息前必须能找到对应的 assistant(tool_calls)
    for j, r in enumerate(result):
        if r["role"] == "tool":
            # 向前查找最近的非 tool 消息，必须是 assistant(tool_calls)
            ok = False
            for k in range(j - 1, -1, -1):
                if result[k]["role"] != "tool":
                    ok = (
                        result[k]["role"] == "assistant"
                        and "tool_calls" in result[k]
                    )
                    break
            if not ok:
                raise ValueError(
                    f"消息顺序错误: 第 {j} 条是 tool 消息，但前面没有对应的 assistant(tool_calls)。"
                    f" 实际角色序列: {[m['role'] for m in result]}\n"
                    f" 请检查 react_agent.py 中 assistant 消息是否在 _act 之前追加。"
                )

    return result


def to_anthropic_messages(messages: list[Message]) -> tuple[str | None, list[dict]]:
    """将框架 Message 列表转为 Anthropic Messages API 格式。

    Anthropic 的 system 是独立的 top-level 参数，不作为 message。
    返回 (system_prompt, api_messages)。

    Anthropic 的 tool_use 和 tool_result 使用 content blocks 而非顶层字段，
    这和 OpenAI 不同。
    """
    system_prompt = None
    api_messages = []

    for msg in messages:
        role = msg["role"]

        if role == "system":
            system_prompt = msg.get("content", "")
        elif role == "user":
            api_messages.append({"role": "user", "content": msg.get("content", "")})
        elif role == "assistant":
            content = _build_anthropic_assistant_content(msg)
            api_messages.append({"role": "assistant", "content": content})
        elif role == "tool":
            api_messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", ""),
                }],
            })

    return system_prompt, api_messages


def _tool_calls_to_openai(tool_calls: list[ToolCall]) -> list[dict]:
    return [
        {
            "id": tc["id"],
            "type": "function",
            "function": {
                "name": tc["name"],
                "arguments": _dump_json(tc.get("arguments", {})),
            },
        }
        for tc in tool_calls
    ]


def _build_anthropic_assistant_content(msg: Message) -> str | list[dict]:
    tool_calls = msg.get("tool_calls")
    text = msg.get("content")

    if not tool_calls:
        return text or ""

    blocks = []
    if text:
        blocks.append({"type": "text", "text": text})
    for tc in tool_calls:
        blocks.append({
            "type": "tool_use",
            "id": tc["id"],
            "name": tc["name"],
            "input": tc.get("arguments", {}),
        })
    return blocks


def tools_to_openai_format(tools: list[dict]) -> list[dict]:
    """将框架通用 JSON Schema 格式转为 OpenAI Tool 格式。"""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {}),
            },
        }
        for t in tools
    ]


def tools_to_anthropic_format(tools: list[dict]) -> list[dict]:
    """将框架通用 JSON Schema 格式转为 Anthropic Tool 格式。"""
    return [
        {
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t.get("parameters", {"type": "object", "properties": {}, "required": []}),
        }
        for t in tools
    ]


def _dump_json(obj: object) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False)
