# cerebrum/community/camel_adapter.py

import time
import uuid
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from camel.messages import OpenAIMessage
from camel.models.base_model import BaseModelBackend
from camel.types import (
    ChatCompletion,
    ChatCompletionMessage,
)
from camel.types.openai_types import Choice, CompletionUsage
from camel.utils import BaseTokenCounter, OpenAITokenCounter
from openai import Stream, AsyncStream

from .adapter import get_request_func
from cerebrum.llm.apis import LLMQuery
# from cerebrum.utils.logger import get_logger
# logger = get_logger(__name__)

class AiosCamelModelBackend(BaseModelBackend):
    """
    用 AIOS 作为后端的 CAMEL ModelBackend。

    """

    def __init__(
        self,
        agent_name: str,
        model_type: Union[str, Any] = "gpt-4o-mini",
        model_config_dict: Optional[Dict[str, Any]] = None,
        token_counter: Optional[BaseTokenCounter] = None,
        timeout: Optional[float] = 60.0,
    ) -> None:
        # CAMEL 的 BaseModelBackend 的 __init__ 签名：
        # (model_type, model_config_dict={}, api_key=None, url=None, token_counter=None, timeout=None)
        if model_config_dict is None:
            model_config_dict = {}

        super().__init__(
            model_type=model_type,
            model_config_dict=model_config_dict,
            api_key=None,
            url=None,
            token_counter=token_counter,
            timeout=timeout,
        )
        self.agent_name = agent_name

    @property
    def token_counter(self) -> BaseTokenCounter:
        """
        CAMEL 要求每个 backend 有一个 token_counter。
        简单起见，直接用 OpenAITokenCounter，实际只用来算 token 数。
        """
        if not self._token_counter:
            self._token_counter = OpenAITokenCounter(self.model_type)
        return self._token_counter
# cerebrum/community/adapter/camel_adapter.py

    def prepare_camel():
        print("Preparing CAMEL Framework...")
        # 这里可以添加初始化代码，比如注册模型、配置等

    def _build_prompt(self, messages: List[OpenAIMessage]) -> str:
        """
        把 OpenAI 风格的 messages 拉平成一个简单文本 prompt。
        只是为了浅集成 demo。
        """
        lines: List[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        text_parts.append(str(part["text"]))
                    else:
                        text_parts.append(str(part))
                text = "".join(text_parts)
            else:
                text = str(content)

            lines.append(f"{role}: {text}")
        lines.append("assistant:")
        return "\n".join(lines)

    def _run(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatCompletion | Stream:
        """
        真正跑一次模型推理。

        浅集成版本：
        - 不支持 tools / response_format / streaming
        - 直接一次性发给 AIOS，拿到字符串回复
        - 包成一个 ChatCompletion 返回
        """
        # 先拿到 AIOS 的 request 函数
        request_fn = get_request_func()
        if request_fn is None:
            raise RuntimeError(
                "AIOS request function 未设置，请确认在 Agent.run() 里调用了 "
                "set_request_func(send_request, agent_name)。"
            )

        llm_query = LLMQuery(
            messages=messages,
            tools=tools,
        )

        try:
            resp = request_fn(query=llm_query)
            resp = resp["response"]
            reply_text = resp.get("response_message", "")
        except Exception as e:
            reply_text = f"[AIOS backend error] {e}"

        # 计算 token 使用情况（大致估计）
        try:
            prompt_tokens = self.token_counter.count_tokens_from_messages(messages)
            completion_tokens = len(self.token_counter.encode(reply_text))
        except Exception:
            prompt_tokens = 0
            completion_tokens = 0

        usage = CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        )

        # 构造 ChatCompletionMessage
        message = ChatCompletionMessage(
            role="assistant",
            content=reply_text,
        )

        choice = Choice(
            index=0,
            message=message,
            finish_reason="stop",
            logprobs=None,
        )

        completion = ChatCompletion(
            id=f"aios-{uuid.uuid4().hex}",
            model=str(self.model_type),
            created=int(time.time()),
            object="chat.completion",
            choices=[choice],
            usage=usage,
            service_tier=None,
            system_fingerprint=None,
        )

        return completion

    # ========== 异步 _arun：简单版直接复用同步逻辑 ==========
    async def _arun(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatCompletion | AsyncStream:
        return self._run(messages, response_format, tools)
