# -*- coding: utf-8 -*-
"""
retriever.py - 检索引擎模块
=============================================
功能：
  1. 向量检索：基于Embedding相似度的语义检索（Top20）
  2. BM25关键词检索：基于词频匹配的关键词检索（Top20）
  3. 混合检索：向量权重0.5 + BM25权重0.5，加权融合去重
  4. Reranker精排：使用bge-reranker-v2-m3对召回结果重排序（Top5）
  5. 上下文排序：最相关文档放开头和结尾，缓解"Lost in the Middle"问题
  6. 父子块策略支持：子块检索命中后返回对应父块

检索流程：
  用户Query
     ↓
  ┌─────────────┐    ┌─────────────┐
  │ 向量检索Top20│    │ BM25检索Top20│
  └──────┬──────┘    └──────┬──────┘
         └────────┬─────────┘
                  ↓
           混合融合 + 去重
                  ↓
           Reranker精排 Top5
                  ↓
           上下文重排序
                  ↓
           返回检索结果
"""

import time
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

import numpy as np
from langchain_core.documents import Document
from loguru import logger

from config import settings
from vector_store import BaseVectorStore, get_vector_store, BGEEmbedding


# ============================================================
# 检索结果数据结构
# ============================================================

@dataclass
class RetrieveResult:
    """
    检索结果数据结构

    Attributes:
        docs: 检索到的文档块列表（已按相关性排序）
        scores: 对应的相关度分数列表
        retrieve_time: 检索总耗时（毫秒）
        vector_count: 向量检索命中数
        bm25_count: BM25检索命中数
        reranked: 是否经过Reranker精排
    """
    docs: List[Document] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    retrieve_time: float = 0.0
    vector_count: int = 0
    bm25_count: int = 0
    reranked: bool = False

    def __len__(self) -> int:
        return len(self.docs)

    def get_sources(self) -> List[Dict[str, Any]]:
        """获取来源信息列表（用于API返回）"""
        sources = []
        for doc, score in zip(self.docs, self.scores):
            sources.append({
                "filename": doc.metadata.get("filename", ""),
                "page": doc.metadata.get("page", 0),
                "score": round(float(score), 4),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "content_preview": doc.page_content[:100],
            })
        return sources


# ============================================================
# BM25 关键词检索器
# ============================================================

class BM25Retriever:
    """
    BM25关键词检索器

    基于 rank_bm25 库实现，对知识库中所有文档块建立BM25索引
    支持中文分词（使用简单的字符级n-gram分词，避免额外依赖）
    """

    def __init__(self, documents: Optional[List[Document]] = None):
        """
        初始化BM25检索器

        Args:
            documents: 初始文档列表，用于构建索引
        """
        self._bm25 = None
        self._documents: List[Document] = []
        self._tokenized_corpus: List[List[str]] = []

        if documents:
            self.build_index(documents)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        简单中文分词：字符级二元分词（bigram）
        对中文效果较好，无需额外分词器依赖

        Args:
            text: 输入文本

        Returns:
            List[str]: token列表
        """
        # 去除空白和特殊字符
        text = text.strip()
        if len(text) < 2:
            return list(text)

        # 生成字符bigram
        tokens = []
        for i in range(len(text) - 1):
            bigram = text[i:i+2]
            tokens.append(bigram)
        # 也加入单字（提高召回）
        tokens.extend(list(text))
        return tokens

    def build_index(self, documents: List[Document]) -> None:
        """
        构建BM25索引

        Args:
            documents: 文档块列表
        """
        if not documents:
            logger.warning("BM25索引构建：文档列表为空")
            return

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("请安装 rank-bm25: pip install rank-bm25")

        self._documents = documents
        self._tokenized_corpus = [self._tokenize(doc.page_content) for doc in documents]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info(f"BM25索引构建完成: {len(documents)} 个文档块")

    def add_documents(self, documents: List[Document]) -> None:
        """
        增量添加文档并重建索引
        （BM25不支持真正的增量，需要重建）

        Args:
            documents: 新增文档列表
        """
        all_docs = self._documents + documents
        self.build_index(all_docs)

    def remove_by_filename(self, filename: str) -> None:
        """
        按文件名移除文档并重建索引

        Args:
            filename: 要移除的文件名
        """
        remaining = [doc for doc in self._documents if doc.metadata.get("filename") != filename]
        if len(remaining) < len(self._documents):
            self.build_index(remaining)
            logger.info(f"BM25索引移除文档: {filename}, 剩余 {len(remaining)} 个块")

    def search(self, query: str, top_k: int = 20) -> List[Tuple[Document, float]]:
        """
        BM25关键词检索

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            List[Tuple[Document, float]]: (文档, BM25分数) 列表
        """
        if self._bm25 is None or not self._documents:
            return []

        # 对查询分词
        query_tokens = self._tokenize(query)

        # 计算BM25分数
        scores = self._bm25.get_scores(query_tokens)

        # 获取TopK索引
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 只返回分数大于0的
                results.append((self._documents[idx], float(scores[idx])))

        return results

    def __len__(self) -> int:
        return len(self._documents)


# ============================================================
# Reranker 精排器
# ============================================================

class Reranker:
    """
    Reranker精排器

    使用 BAAI/bge-reranker-v2-m3 模型对召回结果进行重排序
    该模型计算query和document的相关性分数，比向量相似度更精准
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        初始化Reranker

        Args:
            model_name: 模型名称，默认从配置读取
            device: 推理设备
        """
        self.model_name = model_name or settings.model.reranker_model
        self.device = device or settings.model.device
        self._model = None
        logger.info(f"初始化Reranker模型: {self.model_name}")

    def _load_model(self):
        """延迟加载模型"""
        if self._model is None:
            try:
                from FlagEmbedding import FlagReranker
                self._model = FlagReranker(
                    self.model_name,
                    use_fp16=True,  # 使用半精度加速
                )
                logger.info(f"Reranker模型加载成功: {self.model_name}")
            except Exception as e:
                logger.error(f"Reranker模型加载失败: {str(e)}")
                raise

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        对文档列表进行精排

        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_k: 返回前K个

        Returns:
            List[Tuple[Document, float]]: 精排后的 (文档, 相关性分数) 列表
        """
        if not documents:
            return []

        self._load_model()

        # 构建 (query, document) 对
        pairs = [[query, doc.page_content] for doc in documents]

        # 计算相关性分数
        # use_granularity 参数控制分数粒度，False返回原始logits
        scores = self._model.compute_score(pairs, normalize=True)

        # 确保scores是列表
        if not isinstance(scores, list):
            scores = [scores]

        # 按分数降序排序
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 返回TopK
        results = scored_docs[:top_k]

        logger.info(
            f"Reranker精排: {len(documents)} -> {len(results)} 个, "
            f"最高分: {results[0][1]:.4f}" if results else "Reranker精排: 无结果"
        )

        return results


# ============================================================
# 混合检索引擎（核心类）
# ============================================================

class HybridRetriever:
    """
    混合检索引擎

    整合向量检索、BM25检索和Reranker精排
    实现三级检索架构
    """

    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        embedding_model: Optional[BGEEmbedding] = None,
        reranker: Optional[Reranker] = None,
        enable_reranker: bool = True,
    ):
        """
        初始化混合检索引擎

        Args:
            vector_store: 向量存储实例，None则自动创建
            embedding_model: Embedding模型实例
            reranker: Reranker实例
            enable_reranker: 是否启用Reranker精排
        """
        self.embedding_model = embedding_model or BGEEmbedding()
        self.vector_store = vector_store or get_vector_store(embedding_model=self.embedding_model)
        self.reranker = reranker or Reranker()
        self.enable_reranker = enable_reranker

        # BM25检索器（从向量库加载所有文档构建索引）
        self.bm25_retriever = BM25Retriever()
        self._rebuild_bm25_index()

        # 检索配置
        self.vector_top_k = settings.retrieval.vector_top_k
        self.bm25_top_k = settings.retrieval.bm25_top_k
        self.reranker_top_k = settings.retrieval.reranker_top_k
        self.vector_weight = settings.retrieval.vector_weight
        self.bm25_weight = settings.retrieval.bm25_weight

    def _rebuild_bm25_index(self):
        """从向量库重新构建BM25索引"""
        all_docs = self.vector_store.get_all_chunks()
        if all_docs:
            self.bm25_retriever.build_index(all_docs)
        else:
            logger.info("向量库为空，BM25索引暂不构建")

    def refresh_index(self):
        """刷新索引（添加/删除文档后调用）"""
        self._rebuild_bm25_index()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_reranker: Optional[bool] = None,
    ) -> RetrieveResult:
        """
        执行混合检索（主入口）

        Args:
            query: 用户查询
            top_k: 最终返回数量，默认从配置读取
            use_reranker: 是否使用Reranker，None则使用默认设置

        Returns:
            RetrieveResult: 检索结果
        """
        start_time = time.time()
        top_k = top_k or self.reranker_top_k
        use_reranker = use_reranker if use_reranker is not None else self.enable_reranker

        logger.info(f"开始检索: query='{query[:50]}...'")

        # ---------- 第一步：向量检索 ----------
        vector_results = self.vector_store.similarity_search(
            query=query,
            k=self.vector_top_k,
        )
        vector_count = len(vector_results)
        logger.debug(f"向量检索命中: {vector_count} 个")

        # ---------- 第二步：BM25检索 ----------
        bm25_results = self.bm25_retriever.search(
            query=query,
            top_k=self.bm25_top_k,
        )
        bm25_count = len(bm25_results)
        logger.debug(f"BM25检索命中: {bm25_count} 个")

        # ---------- 第三步：混合融合 + 去重 ----------
        merged_docs = self._merge_results(vector_results, bm25_results)
        logger.debug(f"混合融合后: {len(merged_docs)} 个（去重后）")

        # ---------- 第四步：Reranker精排 ----------
        if use_reranker and merged_docs:
            reranked = self.reranker.rerank(
                query=query,
                documents=[doc for doc, _ in merged_docs],
                top_k=top_k,
            )
            final_docs = [doc for doc, _ in reranked]
            final_scores = [score for _, score in reranked]
            reranked_flag = True
        else:
            # 不使用Reranker，直接取融合后的TopK
            merged_docs = merged_docs[:top_k]
            final_docs = [doc for doc, _ in merged_docs]
            final_scores = [score for _, score in merged_docs]
            reranked_flag = False

        # ---------- 第五步：上下文重排序（Lost in the Middle优化） ----------
        final_docs, final_scores = self._context_reorder(final_docs, final_scores)

        elapsed = (time.time() - start_time) * 1000  # 转毫秒

        result = RetrieveResult(
            docs=final_docs,
            scores=final_scores,
            retrieve_time=elapsed,
            vector_count=vector_count,
            bm25_count=bm25_count,
            reranked=reranked_flag,
        )

        logger.info(
            f"检索完成: 耗时 {elapsed:.1f}ms, "
            f"向量={vector_count}, BM25={bm25_count}, "
            f"最终={len(final_docs)}, Reranker={reranked_flag}"
        )

        return result

    def _merge_results(
        self,
        vector_results: List[Tuple[Document, float]],
        bm25_results: List[Tuple[Document, float]],
    ) -> List[Tuple[Document, float]]:
        """
        混合融合向量检索和BM25检索结果

        采用统一归一化 + 加权融合 + 按chunk_id去重。
        注意：对重复chunk保留排序靠前的分数，不做分数累加
        （避免chunk数量多的文档获得不公平的高分）。

        Args:
            vector_results: 向量检索结果 (doc, similarity)
            bm25_results: BM25检索结果 (doc, bm25_score)

        Returns:
            List[Tuple[Document, float]]: 融合后的结果，按综合分数降序
        """
        # 对两路检索分数做统一的归一化，再按照实验得到的权重进行加权融合
        merged: Dict[str, Tuple[Document, float]] = {}

        # 处理向量结果（先归一化）
        if vector_results:
            vec_scores = [score for _, score in vector_results]
            max_vec = max(vec_scores) if vec_scores else 1.0
            min_vec = min(vec_scores) if vec_scores else 0.0
            vec_range = max_vec - min_vec if max_vec > min_vec else 1.0

            for doc, score in vector_results:
                chunk_id = doc.metadata.get("chunk_id", "")
                # 向量分数归一化到[0,1]
                normalized_score = (score - min_vec) / vec_range
                weighted_score = normalized_score * self.vector_weight

                if chunk_id in merged:
                    # 已存在，保留较高分数（不累加，避免chunk数量多的文档获得不公平高分）
                    existing_doc, existing_score = merged[chunk_id]
                    if weighted_score > existing_score:
                        merged[chunk_id] = (existing_doc, weighted_score)
                else:
                    merged[chunk_id] = (doc, weighted_score)

        # 处理BM25结果（先归一化）
        if bm25_results:
            bm25_scores = [score for _, score in bm25_results]
            max_bm25 = max(bm25_scores) if bm25_scores else 1.0
            min_bm25 = min(bm25_scores) if bm25_scores else 0.0
            bm25_range = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1.0

            for doc, score in bm25_results:
                chunk_id = doc.metadata.get("chunk_id", "")
                # 归一化到 [0, 1]
                normalized_score = (score - min_bm25) / bm25_range
                weighted_score = normalized_score * self.bm25_weight

                if chunk_id in merged:
                    # 已存在，保留较高分数（不累加）
                    existing_doc, existing_score = merged[chunk_id]
                    if weighted_score > existing_score:
                        merged[chunk_id] = (existing_doc, weighted_score)
                else:
                    merged[chunk_id] = (doc, weighted_score)

        # 按综合分数降序排序
        result_list = list(merged.values())
        result_list.sort(key=lambda x: x[1], reverse=True)

        return result_list

    @staticmethod
    def _context_reorder(
        docs: List[Document],
        scores: List[float],
    ) -> Tuple[List[Document], List[float]]:
        """
        实验性上下文重排序（Context Reordering）

        针对LLM对上下文位置的敏感性做的一个实验性重排策略。
        研究表明，LLM对放在开头和结尾的上下文注意力更高，
        对中间的上下文容易忽略。因此将最相关的文档放在
        开头和结尾，次相关的放在中间。
        注意：这是实验性策略，不是Lost in the Middle的标准解决方案。

        排序策略：
          原顺序（按相关性降序）: [1, 2, 3, 4, 5, 6]
          重排后: [1, 3, 5, 6, 4, 2]
          即：奇数位正序放前半，偶数位逆序放后半

        Args:
            docs: 按相关性降序的文档列表
            scores: 对应的分数列表

        Returns:
            Tuple[List[Document], List[float]]: 重排后的文档和分数
        """
        if len(docs) <= 2:
            return docs, scores

        # 奇数索引（0, 2, 4...）放前面，偶数索引（1, 3, 5...）逆序放后面
        odd_docs = [docs[i] for i in range(0, len(docs), 2)]
        odd_scores = [scores[i] for i in range(0, len(scores), 2)]
        even_docs = [docs[i] for i in range(1, len(docs), 2)][::-1]  # 逆序
        even_scores = [scores[i] for i in range(1, len(scores), 2)][::-1]

        return odd_docs + even_docs, odd_scores + even_scores


# ============================================================
# 模块自测
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("检索引擎模块自测")
    print("=" * 60)

    # 构造测试文档
    test_docs = [
        Document(
            page_content="RAG（检索增强生成）是一种结合信息检索和大语言模型的技术，通过从外部知识库检索相关文档来生成更准确的回答。",
            metadata={"filename": "rag_intro.txt", "page": 1, "chunk_id": "rag_1", "file_type": ".txt"},
        ),
        Document(
            page_content="向量数据库是专门用于存储和检索向量嵌入的数据库，支持相似度搜索和最近邻查询。",
            metadata={"filename": "vector_db.txt", "page": 1, "chunk_id": "vec_1", "file_type": ".txt"},
        ),
        Document(
            page_content="BM25是一种基于词频的信息检索算法，是传统搜索引擎的核心算法之一。",
            metadata={"filename": "bm25.txt", "page": 1, "chunk_id": "bm25_1", "file_type": ".txt"},
        ),
        Document(
            page_content="BGE是北京智源人工智能研究院开发的中文Embedding模型，在中文检索任务上表现优异。",
            metadata={"filename": "bge.txt", "page": 1, "chunk_id": "bge_1", "file_type": ".txt"},
        ),
    ]

    # 测试BM25
    print("\n【BM25检索测试】")
    bm25 = BM25Retriever(test_docs)
    results = bm25.search("什么是向量数据库", top_k=2)
    for doc, score in results:
        print(f"  BM25分数: {score:.4f}, 内容: {doc.page_content[:40]}...")

    print("\n提示: 完整的混合检索测试需要先运行 vector_store 添加文档")
    print("可通过 main.py 的 build_index 命令构建完整知识库后测试")
    print("=" * 60)
