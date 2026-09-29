# -*- coding: utf-8 -*-
"""
rag_chain.py - RAG问答链模块
=============================================
功能：
  1. Prompt工程：系统Prompt明确角色、规则、输出格式
  2. LLM调用：支持OpenAI兼容API（GPT/DeepSeek/通义千问等）
  3. 引用标注：回答末尾标注 [来源:文件名-第X页]
  4. 非流式生成和流式生成（SSE）两种模式
  5. 无相关信息时明确回答"无法回答"
  6. 低temperature保证事实性，减少幻觉

Prompt规则：
  - 只根据参考资料回答，不使用外部知识
  - 参考资料中无相关信息时，回答"根据现有资料无法回答该问题"
  - 回答末尾标注引用来源
  - 引用格式：[来源:文件名-第X页]
"""

import time
import json
import asyncio
import queue as queue_mod
import threading
from typing import List, Optional, Dict, Any, AsyncIterator, Tuple
from dataclasses import dataclass, field

from langchain_core.documents import Document
from loguru import logger

from config import settings
from retriever import HybridRetriever, RetrieveResult
from fastapi.concurrency import run_in_threadpool  # 同步检索操作放入线程池，避免阻塞FastAPI事件循环

# LLM流式生成超时兜底（防止Provider无响应时前端无限等待）
LLM_PER_CHUNK_TIMEOUT = 30   # 单块超过30秒无输出视为挂起
LLM_TOTAL_TIMEOUT = 90       # 整个生成过程最多90秒


# ============================================================
# 问答结果数据结构
# ============================================================

@dataclass
class SourceInfo:
    """引用来源信息"""
    filename: str
    page: int
    score: float
    chunk_id: str = ""
    content_preview: str = ""


@dataclass
class AnswerResult:
    """
    问答结果数据结构

    Attributes:
        answer: 生成的回答文本
        sources: 引用来源列表
        retrieve_time: 检索耗时（毫秒）
        generate_time: 生成耗时（毫秒）
        total_time: 总耗时（毫秒）
        query: 用户原始问题
        provider: 实际生效的LLM Provider名称（多Provider降级）
        token_usage: Token用量统计 {prompt_tokens, completion_tokens}
    """
    answer: str = ""
    sources: List[SourceInfo] = field(default_factory=list)
    retrieve_time: float = 0.0
    generate_time: float = 0.0
    total_time: float = 0.0
    query: str = ""
    provider: str = ""
    token_usage: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转为字典（用于API返回）"""
        return {
            "answer": self.answer,
            "sources": [
                {
                    "filename": s.filename,
                    "page": s.page,
                    "score": round(s.score, 4),
                    "chunk_id": s.chunk_id,
                    "content_preview": s.content_preview,
                }
                for s in self.sources
            ],
            "retrieve_time": round(self.retrieve_time, 2),
            "generate_time": round(self.generate_time, 2),
            "total_time": round(self.total_time, 2),
            "provider": self.provider,
            "token_usage": self.token_usage,
        }


# ============================================================
# Prompt 模板定义
# ============================================================

# 系统Prompt：定义AI角色和回答规则
SYSTEM_PROMPT = """你是一个专业的文档问答助手。请严格遵守以下规则：

【核心规则】
1. 只根据下方提供的【参考资料】回答问题，绝对不要使用参考资料以外的知识。
2. 如果参考资料中没有与问题相关的内容，请明确回答："根据现有资料，无法回答该问题。"
3. 回答要准确、简洁、有条理，避免冗余和猜测。
4. 回答末尾必须标注引用来源，格式为：[来源:文件名-第X页]
5. 如果多个来源都有相关内容，可以引用多个来源。
6. 不要编造参考资料中不存在的信息。

【输出格式】
直接给出回答，不需要重复问题。回答末尾附上引用来源。
"""

# 用户Prompt模板：将问题和参考资料组合
USER_PROMPT_TEMPLATE = """【参考资料】
{context}

【用户问题】
{query}

请根据上述参考资料回答问题。"""


# ============================================================
# RAG 问答链核心类
# ============================================================

class RAGChain:
    """
    RAG问答链

    整合检索引擎和LLM，实现完整的问答流程：
      1. 用户提问 -> 检索相关文档
      2. 构建Prompt（系统Prompt + 参考资料 + 用户问题）
      3. 调用LLM生成回答
      4. 提取引用来源并返回结果
    """

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        初始化RAG问答链

        Args:
            retriever: 混合检索引擎实例，None则自动创建
            system_prompt: 自定义系统Prompt，None则使用默认
        """
        self.retriever = retriever or HybridRetriever()
        self.system_prompt = system_prompt or SYSTEM_PROMPT

        # 多Provider自动降级客户端（技术栈2.1：DashScope / 多Provider）
        from llm_client import LLMClient
        self.llm_client = LLMClient(
            temperature=settings.model.llm_temperature,
            max_tokens=settings.model.llm_max_tokens,
        )

        # LLM配置（默认展示）
        self.llm_model = settings.model.llm_model
        self.api_key = settings.model.llm_api_key
        self.base_url = settings.model.llm_base_url
        self.temperature = settings.model.llm_temperature
        self.max_tokens = settings.model.llm_max_tokens

        logger.info(
            f"RAG问答链初始化: 默认模型={self.llm_model}, "
            f"temperature={self.temperature}, max_tokens={self.max_tokens}, "
            f"Providers={len(self.llm_client.providers)}"
        )

    def _get_client(self):
        """兼容占位：实际LLM调用走多Provider客户端（llm_client）"""
        return self.llm_client

    def _build_context(self, retrieve_result: RetrieveResult) -> Tuple[str, List[SourceInfo]]:
        """
        从检索结果构建参考资料文本和来源信息

        Args:
            retrieve_result: 检索结果

        Returns:
            Tuple[str, List[SourceInfo]]: (格式化的参考资料文本, 来源信息列表)
        """
        context_parts = []
        sources = []

        for i, (doc, score) in enumerate(zip(retrieve_result.docs, retrieve_result.scores)):
            filename = doc.metadata.get("filename", "未知文件")
            page = doc.metadata.get("page", 0)
            chunk_id = doc.metadata.get("chunk_id", "")

            # 构建来源信息
            source_info = SourceInfo(
                filename=filename,
                page=page,
                score=score,
                chunk_id=chunk_id,
                content_preview=doc.page_content[:100],
            )
            sources.append(source_info)

            # 构建参考资料段落（带编号和来源标记）
            context_part = f"[资料{i+1}] [来源:{filename}-第{page}页]\n{doc.page_content}"
            context_parts.append(context_part)

        context_text = "\n\n".join(context_parts)
        return context_text, sources

    def _build_messages(self, query: str, context: str) -> List[Dict[str, str]]:
        """
        构建OpenAI格式的消息列表

        Args:
            query: 用户问题
            context: 参考资料文本

        Returns:
            List[Dict]: messages列表
        """
        user_content = USER_PROMPT_TEMPLATE.format(context=context, query=query)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        return messages


    def _validate_answer(self, answer: str, context: str) -> str:
        """
        输出校验（文档3.4节第4层幻觉控制）

        检查答案中的关键实体（数字、日期、百分比等）是否出现在检索上下文中。
        如果答案包含上下文中完全没有的数字类实体，追加提示以降低幻觉风险。

        Args:
            answer: LLM生成的原始回答
            context: 检索到的参考资料文本

        Returns:
            str: 校验后的回答（可能追加来源核实提示）
        """
        import re

        if not answer or not context:
            return answer

        # 提取答案中的数字类实体（含百分号/单位的数字、年份、日期）
        number_pattern = re.compile(r"\d+(?:\.\d+)?%?|\d{4}年?")
        answer_numbers = set(number_pattern.findall(answer))
        context_numbers = set(number_pattern.findall(context))

        # 找出答案中有但参考资料中没有的数字
        hallucinated = [n for n in answer_numbers if n not in context_numbers and len(n) >= 2]

        if hallucinated:
            logger.warning(f"输出校验：发现答案中可能无依据的数字: {hallucinated[:5]}")
            # 追加温和提示，提醒用户核实关键数据
            answer = answer.rstrip() + "\n\n（注：以上数字信息请以引用来源原文为准。）"

        return answer

    def query(self, question: str, use_reranker: bool = True) -> AnswerResult:
        """
        非流式问答（主入口）

        Args:
            question: 用户问题
            use_reranker: 是否使用Reranker精排

        Returns:
            AnswerResult: 问答结果
        """
        total_start = time.time()
        logger.info(f"收到问答请求: {question[:50]}...")

        # ---------- 第一步：检索 ----------
        retrieve_start = time.time()
        retrieve_result = self.retriever.retrieve(
            query=question,
            use_reranker=use_reranker,
        )
        retrieve_time = (time.time() - retrieve_start) * 1000

        # 如果没有检索到相关文档
        if not retrieve_result.docs:
            answer = "根据现有资料，无法回答该问题。知识库中可能没有相关文档。"
            total_time = (time.time() - total_start) * 1000
            return AnswerResult(
                answer=answer,
                sources=[],
                retrieve_time=retrieve_time,
                generate_time=0,
                total_time=total_time,
                query=question,
            )

        # ---------- 第二步：构建Prompt ----------
        context, sources = self._build_context(retrieve_result)
        messages = self._build_messages(question, context)

        # ---------- 第三步：调用LLM生成（多Provider自动降级） ----------
        generate_start = time.time()
        try:
            answer, provider, usage = self.llm_client.chat(messages=messages)
            # 第4层幻觉控制：输出校验——检查答案中的关键实体是否出现在检索上下文中
            answer = self._validate_answer(answer, context)
        except Exception as e:
            logger.error(f"LLM调用失败: {str(e)}")
            answer = f"抱歉，生成回答时出现错误：{str(e)}"
            sources = []
            provider = None
            usage = {}

        generate_time = (time.time() - generate_start) * 1000
        total_time = (time.time() - total_start) * 1000

        result = AnswerResult(
            answer=answer,
            sources=sources,
            retrieve_time=retrieve_time,
            generate_time=generate_time,
            total_time=total_time,
            query=question,
            provider=provider.name if provider else "",
            token_usage=usage,
        )

        logger.info(
            f"问答完成: 总耗时 {total_time:.1f}ms "
            f"(检索 {retrieve_time:.1f}ms + 生成 {generate_time:.1f}ms), "
            f"引用 {len(sources)} 个来源"
        )

        return result

    async def _llm_stream_guard(self, messages: List[Dict[str, str]],):
        """
        带超时兜底的异步LLM流式包装

        将同步的 llm_client.chat_stream 放入后台线程，通过队列逐块取回；
        单块超过 LLM_PER_CHUNK_TIMEOUT 无输出、或总时长超过 LLM_TOTAL_TIMEOUT
        时抛异常（由调用方转为SSE error事件），保证前端不会无限等待。
        """
        loop = asyncio.get_event_loop()
        q: "queue_mod.Queue" = queue_mod.Queue(maxsize=64)
        sentinel = object()

        def _producer():
            try:
                for chunk in self.llm_client.chat_stream(messages=messages):
                    q.put(chunk)
            except Exception as e:  # Provider失败/超时等
                q.put(e)
            finally:
                q.put(sentinel)

        thread = threading.Thread(target=_producer, daemon=True, name="llm-stream")
        thread.start()

        total_start = time.time()
        while True:
            remaining = LLM_TOTAL_TIMEOUT - (time.time() - total_start)
            if remaining <= 0:
                raise TimeoutError(f"LLM生成超过{LLM_TOTAL_TIMEOUT}秒，已自动终止")
            try:
                item = await loop.run_in_executor(
                    None,
                    lambda: q.get(timeout=min(LLM_PER_CHUNK_TIMEOUT, remaining)),
                )
            except queue_mod.Empty:
                raise TimeoutError(f"LLM超过{LLM_PER_CHUNK_TIMEOUT}秒无输出，已自动终止")

            if item is sentinel:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    async def query_stream(
        self,
        question: str,
        use_reranker: bool = True,
    ) -> AsyncIterator[str]:
        """
        流式问答（SSE格式）

        以异步生成器方式逐token输出，适合SSE流式响应

        Args:
            question: 用户问题
            use_reranker: 是否使用Reranker

        Yields:
            str: SSE格式的事件数据，每行以 "data: " 开头
        """
        total_start = time.time()

        # 第一步：检索（同步CPU/GPU密集操作，放入线程池，避免阻塞事件循环）
        retrieve_result = await run_in_threadpool(
            self.retriever.retrieve,
            query=question,
            use_reranker=use_reranker,
        )
        retrieve_time = (time.time() - total_start) * 1000

        # 发送检索元信息
        meta_data = {
            "type": "meta",
            "retrieve_time": round(retrieve_time, 2),
            "source_count": len(retrieve_result.docs),
            "sources": [
                {
                    "filename": doc.metadata.get("filename", ""),
                    "page": doc.metadata.get("page", 0),
                    "score": round(float(score), 4),
                }
                for doc, score in zip(retrieve_result.docs, retrieve_result.scores)
            ],
        }
        yield f"data: {json.dumps(meta_data, ensure_ascii=False)}\n\n"

        # 如果没有检索结果
        if not retrieve_result.docs:
            no_answer = {"type": "content", "content": "根据现有资料，无法回答该问题。"}
            yield f"data: {json.dumps(no_answer, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # 第二步：构建Prompt
        context, sources = self._build_context(retrieve_result)
        messages = self._build_messages(question, context)

        # 第三步：流式调用LLM（多Provider自动降级 + 超时兜底）
        try:
            async for content in self._llm_stream_guard(messages):
                data = {"type": "content", "content": content}
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

            # 流式结束后补充生效Provider信息（可选）
            provider = self.llm_client.active_provider
            if provider:
                info_data = {"type": "provider", "name": provider.name, "model": provider.model}
                yield f"data: {json.dumps(info_data, ensure_ascii=False)}\n\n"

        except Exception as e:
            error_data = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

        # 发送结束标记
        total_time = (time.time() - total_start) * 1000
        done_data = {
            "type": "done",
            "total_time": round(total_time, 2),
        }
        yield f"data: {json.dumps(done_data, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"


# ============================================================
# 便捷函数
# ============================================================

def create_rag_chain() -> RAGChain:
    """
    创建RAG问答链实例（便捷函数）

    Returns:
        RAGChain: 初始化好的问答链
    """
    return RAGChain()


# ============================================================
# 模块自测
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RAG问答链模块自测")
    print("=" * 60)

    # 检查API Key配置
    if not settings.model.llm_api_key:
        print("\n【警告】未配置 LLM API Key")
        print("请在 config.yaml 中设置 model.llm_api_key，或创建 .env 文件")
        print("示例 .env 内容: OPENAI_API_KEY=你的api_key")
    else:
        print(f"\nLLM API Key: 已配置")
        print(f"LLM模型: {settings.model.llm_model}")
        print(f"API Base: {settings.model.llm_base_url}")

    print(f"\n系统Prompt预览:")
    print("-" * 40)
    print(SYSTEM_PROMPT[:200] + "...")
    print("-" * 40)

    print("\n提示: 完整问答测试需要先构建知识库")
    print("运行 python main.py chat \"你的问题\" 进行测试")
    print("=" * 60)
