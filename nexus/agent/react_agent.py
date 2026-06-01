"""ReActAgent — 框架的核心引擎，实现 Think → Act → Observe 主循环。

这是整个框架最关键的文件，实现了标准的 ReAct (Reasoning + Acting) 模式:

┌──────────────────────────────────────────┐
│  1. [Observe]  接收用户输入               │
│       ↓                                  │
│  2. [Think]    LLM 推理, 决定是否调用工具   │
│       ↓                                  │
│  3. [Check]    有 tool_calls?            │
│       ├── YES → 4. [Act] 执行工具         │
│       └── NO  → 5. [Respond] 返回结果     │
│       ↓                                  │
│  4. [Act]      执行工具, 追加结果到历史     │
│       ↓ 返回步骤 2                        │
│  5. [Respond]  返回最终回复               │
└──────────────────────────────────────────┘

关键设计点:
- 依赖注入: Agent 不创建任何组件，全部通过构造器注入
- 消息序列: System → User → Assistant(tool_calls) → Tool(result) → ... → Assistant(final)
- 循环检测: 检测连续相同的 tool_call，防止陷入死循环
- Token 预算: 支持上下文 token 上限，超出自动裁剪
- Hooks: pre_model_hook / post_model_hook 支持自定义中间件
- 错误隔离: Tool 执行异常封装为错误信息返回 LLM，不中断循环
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable

from nexus.agent.base import BaseAgent
from nexus.core.errors import MaxIterationError, ToolNotFoundError
from nexus.core.types import AgentResult, LLMResponse, Message, ToolCallRecord
from nexus.llm.base import BaseLLM
from nexus.memory.base import BaseMemory
from nexus.tools.registry import ToolRegistry
from nexus.utils import get_logger

logger = get_logger(__name__)


class ReActAgent(BaseAgent):
    """ReAct (Reasoning + Acting) Agent。

    用法:
        llm = OpenAIProvider(api_key="...")
        tools = ToolRegistry()
        tools.register(calculator_tool)

        agent = ReActAgent(
            name="助手",
            llm=llm,
            tools=tools,
            system_prompt="你是一个有帮助的助手。",
            max_iterations=10,
        )

        result = await agent.run("计算 123 * 456 的结果")
        print(result.content)
    """

    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        tools: ToolRegistry | None = None,
        memory: BaseMemory | None = None,
        *,
        system_prompt: str = "你是一个有帮助的 AI 助手。",
        max_iterations: int = 10,
        token_budget: int | None = None,
        loop_detection_threshold: int = 3,
        pre_model_hook: Callable[[list[Message]], list[Message]] | None = None,
        post_model_hook: Callable[[LLMResponse], LLMResponse] | None = None,
    ):
        self.name = name
        self._llm = llm
        self._tools = tools or ToolRegistry()
        self._memory = memory
        self._system_prompt = system_prompt
        self._max_iterations = max_iterations
        self._token_budget = token_budget
        self._loop_detection_threshold = loop_detection_threshold
        self._pre_model_hook = pre_model_hook
        self._post_model_hook = post_model_hook

        # 追踪状态
        self._iteration = 0
        self._tool_call_log: list[ToolCallRecord] = []

    async def run(self, user_input: str, **kwargs) -> AgentResult:
        """执行 ReAct 主循环。

        流程:
        1. 构建初始消息列表 (system + memory + user_input)
        2. 进入循环: Think → Check → (Act → 追加 Tool Result) → Think → ... → Respond
        3. 返回 AgentResult

        Args:
            user_input: 用户输入文本

        Returns:
            AgentResult: 包含最终回复、循环次数、工具调用历史

        Raises:
            MaxIterationError: 超过最大循环次数仍未完成
        """
        self._iteration = 0
        self._tool_call_log.clear()

        # Step 1: 构建初始消息
        messages = await self._build_initial_messages(user_input)

        # Step 1b: 将用户输入写入 Memory（必须在 get_context 之后）
        if self._memory:
            await self._memory.add({"role": "user", "content": user_input})

        logger.info("agent_start", agent=self.name, input=user_input[:200])

        # Step 2: ReAct 主循环
        while self._iteration < self._max_iterations:
            self._iteration += 1

            # 2a: Token 预算裁剪
            if self._token_budget:
                messages = self._trim_messages(messages)

            # 2b: pre_model_hook (如摘要压缩)
            if self._pre_model_hook:
                messages = self._pre_model_hook(messages)

            # 2c: [Think] LLM 推理
            tool_schemas = self._tools.get_schemas() if self._tools else None
            response = await self._llm.generate(
                messages=messages,
                tools=tool_schemas,
                model=kwargs.get("model"),
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 4096),
            )

            # 2d: post_model_hook
            if self._post_model_hook:
                response = self._post_model_hook(response)

            logger.debug(
                "agent_think",
                agent=self.name,
                iteration=self._iteration,
                has_tool_calls=bool(response.tool_calls),
                content_preview=(response.content or "")[:100],
            )

            # 2e: [Check] 是否有工具调用
            if response.tool_calls:
                # 循环检测（检测到循环时注入干预消息）
                await self._check_loop(response, messages)

                # 将 Assistant 消息（含 tool_calls）追加到历史
                # 必须在工具结果之前，否则 OpenAI 会报错：
                # "Messages with role 'tool' must be a response to a preceding message with 'tool_calls'"
                assistant_msg: Message = {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": response.tool_calls,
                }
                messages.append(assistant_msg)

                # 同步写入 Memory
                if self._memory:
                    await self._memory.add(assistant_msg)

                # [Act] 执行工具
                await self._act(response.tool_calls, messages)
                continue
            else:
                # [Respond] 最终回复
                logger.info(
                    "agent_done",
                    agent=self.name,
                    iterations=self._iteration,
                    tools_called=len(self._tool_call_log),
                )

                # 同步写入 Memory: 最终回复
                if self._memory:
                    await self._memory.add({
                        "role": "assistant",
                        "content": response.content or "",
                    })

                return AgentResult(
                    content=response.content or "",
                    iterations=self._iteration,
                    tool_calls_history=list(self._tool_call_log),
                    finish_reason=response.finish_reason,
                )

        raise MaxIterationError(
            f"Agent '{self.name}' 超过最大循环次数 {self._max_iterations}，"
            f"共调用 {len(self._tool_call_log)} 次工具: "
            f"{[r.tool_name for r in self._tool_call_log]}"
        )

    async def stream(self, user_input: str, **kwargs) -> AsyncIterator[str]:
        """流式执行 Agent——每一轮都使用流式输出。

        核心改进：不再先用非流式检查 tool_calls，而是直接流式接收 LLM 输出。
        - text chunk → 立即 yield 给用户
        - tool_call → 累积，流结束后执行工具，进入下一轮循环
        - 无 tool_call → 最终回复已流式输出完毕，返回

        这样用户在工具调用轮次也能看到模型的"思考"过程（如 "Let me calculate..."）。
        """
        self._iteration = 0
        self._tool_call_log.clear()

        messages = await self._build_initial_messages(user_input)

        if self._memory:
            await self._memory.add({"role": "user", "content": user_input})

        while self._iteration < self._max_iterations:
            self._iteration += 1

            if self._token_budget:
                messages = self._trim_messages(messages)

            if self._pre_model_hook:
                messages = self._pre_model_hook(messages)

            tool_schemas = self._tools.get_schemas() if self._tools else None

            # 每轮都使用流式——text 直接 yield，tool_call 累积
            text_parts: list[str] = []
            tool_calls_received: list = []

            async for chunk in self._llm.generate_stream(
                messages=messages,
                tools=tool_schemas,
                model=kwargs.get("model"),
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 4096),
            ):
                if isinstance(chunk, str):
                    text_parts.append(chunk)
                    yield chunk
                else:
                    tool_calls_received.append(chunk)

            # 有工具调用 → 执行后继续循环
            if tool_calls_received:
                loop_response = LLMResponse(
                    content="".join(text_parts) if text_parts else None,
                    tool_calls=tool_calls_received,
                    finish_reason="tool_calls",
                )
                await self._check_loop(loop_response, messages)

                assistant_msg: Message = {
                    "role": "assistant",
                    "content": loop_response.content,
                    "tool_calls": loop_response.tool_calls,
                }
                messages.append(assistant_msg)

                if self._memory:
                    await self._memory.add(assistant_msg)

                await self._act(tool_calls_received, messages)
                continue

            # 无工具调用 → 最终回复已通过 yield 流式输出完毕
            full_text = "".join(text_parts)
            if self._memory and full_text:
                await self._memory.add({
                    "role": "assistant",
                    "content": full_text,
                })
            return

        raise MaxIterationError(f"Agent '{self.name}' 超过最大循环次数 {self._max_iterations}")

    # ── 内部方法 ─────────────────────────────────

    async def _build_initial_messages(self, user_input: str) -> list[Message]:
        """构建初始消息列表：System Prompt + Memory Context + User Input。"""
        messages: list[Message] = []

        # System Prompt
        messages.append({"role": "system", "content": self._system_prompt})

        # Memory Context（来自 Memory 系统）
        if self._memory:
            memory_msgs = await self._memory.get_context(query=user_input, max_tokens=4000)
            for m in memory_msgs:
                messages.append(m)

        # User Input
        messages.append({"role": "user", "content": user_input})

        return messages

    async def _act(self, tool_calls: list, messages: list[Message]) -> None:
        """执行工具调用并追加 Tool Result 到消息历史。

        每个 tool_call 对应一条 tool 角色消息。
        工具执行异常会被捕获并作为错误信息返回给 LLM。
        """
        for tc in tool_calls:
            tool_name = tc["name"]
            arguments = tc.get("arguments", {})
            call_id = tc["id"]

            start = time.perf_counter()
            try:
                result = await self._tools.execute(tool_name, **arguments)
                elapsed = (time.perf_counter() - start) * 1000
                result_text = result.result
            except (KeyError, ToolNotFoundError):
                elapsed = (time.perf_counter() - start) * 1000
                result_text = f"错误: 工具 '{tool_name}' 未注册"
            except Exception as exc:
                elapsed = (time.perf_counter() - start) * 1000
                result_text = f"工具执行异常: {type(exc).__name__}: {exc}"

            # 记录到日志
            self._tool_call_log.append(ToolCallRecord(
                tool_name=tool_name,
                arguments=arguments,
                result=result_text,
                error=None if "错误" not in result_text and "异常" not in result_text else result_text,
                elapsed_ms=elapsed,
            ))

            logger.info(
                "agent_act",
                agent=self.name,
                tool=tool_name,
                arguments=str(arguments)[:100],
                elapsed_ms=f"{elapsed:.0f}",
                error=result_text if "错误" in result_text or "异常" in result_text else None,
            )

            # 追加 Tool Result 到消息历史
            messages.append({
                "role": "tool",
                "content": result_text,
                "tool_call_id": call_id,
                "name": tool_name,
            })

            # 同步写入 Memory
            if self._memory:
                await self._memory.add({
                    "role": "tool",
                    "content": f"[{tool_name}] {result_text}",
                    "tool_call_id": call_id,
                })

    async def _check_loop(self, response: LLMResponse, messages: list[Message]) -> None:
        """检测 Agent 是否陷入循环并主动注入干预消息。

        如果连续 N 次（loop_detection_threshold）调用相同的工具和参数，
        则向消息历史注入一条 user 消息提示 Agent 尝试其他方法。
        这比被动日志更有效——LLM 看到这条消息后通常会调整策略。
        """
        if self._iteration < self._loop_detection_threshold:
            return

        if not response.tool_calls or not self._tool_call_log:
            return

        recent = self._tool_call_log[-(self._loop_detection_threshold):]
        if len(recent) < self._loop_detection_threshold:
            return

        first = recent[0]
        if all(
            r.tool_name == first.tool_name and r.arguments == first.arguments
            for r in recent
        ):
            logger.warning(
                "agent_loop_detected",
                agent=self.name,
                tool=first.tool_name,
                arguments=str(first.arguments)[:100],
                count=self._loop_detection_threshold,
            )

            # 注入干预消息，引导 Agent 跳出循环
            intervention = (
                f"你似乎重复了相同的操作（连续调用 `{first.tool_name}` {self._loop_detection_threshold} 次），"
                f"请尝试其他方法，或直接给出你能提供的最佳回答。"
            )
            messages.append({"role": "user", "content": intervention})
            if self._memory:
                await self._memory.add({"role": "user", "content": intervention})

    def _trim_messages(self, messages: list[Message]) -> list[Message]:
        """根据 token 预算裁剪消息历史。

        策略：保留 system + 最近的消息，使总 token 不超过预算。
        """
        system_msgs = [m for m in messages if m["role"] == "system"]
        others = [m for m in messages if m["role"] != "system"]

        # 从后往前保留
        total = sum(self._est_tokens(m) for m in system_msgs)
        kept = []
        for m in reversed(others):
            t = self._est_tokens(m)
            if total + t > (self._token_budget or 8000):
                break
            kept.insert(0, m)
            total += t

        return system_msgs + kept

    @staticmethod
    def _est_tokens(msg: Message) -> int:
        """粗略估算单条消息的 token 数（4 字符 ≈ 1 token）。"""
        content = msg.get("content")
        return len(content) // 4 if content else 0
