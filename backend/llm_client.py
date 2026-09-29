# -*- coding: utf-8 -*-
"""
llm_client.py - 多Provider LLM客户端（技术栈2.1：DashScope / 多Provider 自动降级）
=============================================
功能：
  1. 按 priority 顺序维护多个LLM Provider（OpenAI兼容接口）
  2. 调用失败时自动降级到下一个可用Provider
  3. 支持流式与非流式，兼容 DeepSeek / DashScope(通义千问) / OpenAI / Moonshot 等
  4. 自动跳过未配置 API Key 的 Provider
  5. 记录最终使用的 Provider 与 Token 用量（供 ECharts 统计）

使用方式：
  from llm_client import LLMClient
  client = LLMClient()
  resp = client.chat(messages=[...])          # 非流式
  for chunk in client.chat_stream(messages):  # 流式
      ...
"""

import time
from typing import List, Dict, Any, Optional, Tuple, Iterator

from loguru import logger

from config import settings


class LLMProvider:
    """单个LLM Provider（OpenAI兼容接口）"""

    def __init__(self, name: str, model: str, api_key: str, base_url: str, priority: int = 1):
        self.name = name
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.priority = priority

    @property
    def available(self) -> bool:
        """是否已配置API Key"""
        return bool(self.api_key)

    def get_client(self):
        """获取OpenAI客户端（惰性）"""
        from openai import OpenAI
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            # 关键：显式超时，避免LLM无响应时无限挂起（SDK默认600秒）
            timeout=30.0,     # 连接+读取超时30秒
            max_retries=0,    # 失败立即降级到下一个Provider，不做长重试
        )

    def __repr__(self):
        return f"<Provider {self.name} model={self.model} base={self.base_url} priority={self.priority}>"


class LLMClient:
    """
    多Provider自动降级客户端

    调用流程：按 priority 排序，跳过未配置Key的Provider；
    当前Provider抛异常时自动降级到下一个；全部失败则抛最终异常。
    """

    def __init__(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        初始化多Provider客户端

        Args:
            temperature: 生成温度（None则用配置）
            max_tokens: 最大生成token数（None则用配置）
        """
        self.temperature = temperature if temperature is not None else settings.model.llm_temperature
        self.max_tokens = max_tokens if max_tokens is not None else settings.model.llm_max_tokens

        # 构建Provider列表（按priority升序）
        providers = list(settings.model.providers)
        providers.sort(key=lambda p: p.priority)
        self.providers: List[LLMProvider] = [LLMProvider(**p.model_dump()) for p in providers]

        # 若所有Provider都未配置Key，用旧字段兜底
        if not any(p.available for p in self.providers):
            fallback = LLMProvider(
                name="default",
                model=settings.model.llm_model,
                api_key=settings.model.llm_api_key,
                base_url=settings.model.llm_base_url,
            )
            self.providers.append(fallback)

        logger.info(
            f"LLMClient初始化: {len(self.providers)} 个Provider, "
            f"可用 {sum(1 for p in self.providers if p.available)} 个: "
            + ", ".join(f"{p.name}({p.model})" for p in self.providers if p.available)
        )

    @property
    def active_provider(self) -> Optional[LLMProvider]:
        """当前生效的Provider（供Token统计）"""
        return getattr(self, "_active_provider", None)

    def _try_providers(self, func):
        """
        通用降级执行器

        Args:
            func: 接收 provider 返回结果的函数

        Returns:
            三元组: (result, provider, error_list)
        """
        errors: List[str] = []
        available = [p for p in self.providers if p.available]
        if not available:
            raise RuntimeError("所有LLM Provider均未配置API Key（请在config.yaml或环境变量中配置）")

        for provider in available:
            try:
                result = func(provider)
                self._active_provider = provider
                if errors:
                    logger.warning(f"Provider降级成功: {provider.name}（此前失败: {errors[-1]}）")
                return result, provider, errors
            except Exception as e:
                errors.append(f"{provider.name}: {str(e)}")
                logger.warning(f"Provider {provider.name} 调用失败，尝试降级: {str(e)}")

        raise RuntimeError(f"所有Provider均调用失败: {' | '.join(errors)}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Tuple[str, LLMProvider, Dict[str, Any]]:
        """
        非流式对话（自动降级）

        Args:
            messages: OpenAI格式消息列表
            **kwargs: 透传参数（temperature/max_tokens等）

        Returns:
            (answer, provider, usage): 回答文本、生效Provider、token用量
        """
        def _call(provider: LLMProvider):
            client = provider.get_client()
            resp = client.chat.completions.create(
                model=provider.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                stream=False,
            )
            return resp

        resp, provider, errors = self._try_providers(_call)
        answer = (resp.choices[0].message.content or "").strip()
        usage = resp.usage.model_dump() if resp.usage else {"prompt_tokens": 0, "completion_tokens": 0}
        return answer, provider, usage

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Iterator[str]:
        """
        流式对话（自动降级）

        以生成器方式逐token输出；Provider失败时在生成器内抛异常，
        由调用方捕获（流式降级由上层决定是否重试）。

        Args:
            messages: OpenAI格式消息列表
            **kwargs: 透传参数

        Yields:
            str: 内容增量token
        """
        errors: List[str] = []
        available = [p for p in self.providers if p.available]
        if not available:
            raise RuntimeError("所有LLM Provider均未配置API Key（请在config.yaml或环境变量中配置）")

        for provider in available:
            try:
                client = provider.get_client()
                stream = client.chat.completions.create(
                    model=provider.model,
                    messages=messages,
                    temperature=kwargs.get("temperature", self.temperature),
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                    stream=True,
                )
                self._active_provider = provider
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return  # 流式正常结束
            except Exception as e:
                errors.append(f"{provider.name}: {str(e)}")
                logger.warning(f"Provider {provider.name} 流式调用失败，尝试降级: {str(e)}")

        raise RuntimeError(f"所有Provider均调用失败: {' | '.join(errors)}")
