# -*- coding: utf-8 -*-
"""
evaluator.py - RAGAS 评估模块
=============================================
功能：
  1. 使用RAGAS框架自动化评估RAG系统
  2. 核心指标：
     - faithfulness（忠实度）：回答是否基于上下文，无幻觉
     - answer_relevancy（答案相关性）：回答与问题的相关程度
     - context_recall（上下文召回率）：相关上下文是否被检索到
     - context_precision（上下文精确率）：检索到的上下文是否相关
  3. 支持自定义测试集
  4. 输出评估报告，支持JSON和CSV格式

使用方式：
  1. 准备测试集（questions + ground_truths）
  2. 运行评估
  3. 查看评估报告

注意：RAGAS评估需要调用LLM，会消耗API token
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import pandas as pd
from loguru import logger

from config import settings
from rag_chain import RAGChain
from retriever import HybridRetriever
from vector_store import BGEEmbedding  # 复用本地BGE Embedding，避免依赖OpenAI Embedding接口


# ============================================================
# 测试集数据结构
# ============================================================

class TestDataset:
    """
    测试集管理

    测试集格式：
    [
        {
            "question": "问题1",
            "ground_truth": "标准答案1"
        },
        ...
    ]
    """

    def __init__(self, data: Optional[List[Dict[str, str]]] = None):
        self.data = data or []

    @classmethod
    def from_json(cls, file_path: str) -> "TestDataset":
        """从JSON文件加载测试集"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    @classmethod
    def from_csv(cls, file_path: str) -> "TestDataset":
        """从CSV文件加载测试集（需包含 question 和 ground_truth 列）"""
        df = pd.read_csv(file_path)
        data = df[["question", "ground_truth"]].to_dict("records")
        return cls(data)

    def to_json(self, file_path: str):
        """保存为JSON文件"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ============================================================
# RAGAS 评估器
# ============================================================

class RAGASEvaluator:
    """
    RAGAS评估器

    评估流程：
      1. 对测试集中每个问题，使用RAG系统生成回答和检索上下文
      2. 使用RAGAS指标评估回答质量和检索质量
      3. 汇总指标，生成评估报告
    """

    def __init__(
        self,
        rag_chain: Optional[RAGChain] = None,
        output_dir: str = "./evaluations",
    ):
        """
        初始化评估器

        Args:
            rag_chain: RAG问答链实例，None则自动创建
            output_dir: 评估结果输出目录
        """
        self.rag_chain = rag_chain or RAGChain()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # RAGAS需要的LLM和Embedding配置
        self._init_ragas_components()

    def _init_ragas_components(self):
        """初始化RAGAS所需的组件"""
        try:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
            )
            self._ragas_available = True
            self._metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

            # RAGAS使用的LLM（用于评估）
            self._ragas_llm = ChatOpenAI(
                model=settings.model.llm_model,
                api_key=settings.model.llm_api_key,
                base_url=settings.model.llm_base_url,
                temperature=0,
                n=1,  # DeepSeek 仅支持 n=1，RAGAS 部分指标默认 n=2 会报 400
            )

            # RAGAS使用的Embedding（用于评估 answer_relevancy）
            # DeepSeek 不提供 /embeddings 接口，直接复用项目已加载的本地 BGE 模型
            # 复用 rag_chain.retriever 中已初始化的实例，避免重复加载模型占用显存
            self._ragas_embeddings = self.rag_chain.retriever.embedding_model

            logger.info("RAGAS组件初始化成功")

        except ImportError as e:
            self._ragas_available = False
            logger.warning(f"RAGAS未安装或初始化失败: {str(e)}")
            logger.warning("将使用基础评估模式（仅统计检索和生成指标）")

    def generate_rag_data(self, test_dataset: TestDataset) -> pd.DataFrame:
        """
        使用RAG系统为测试集生成回答和上下文

        Args:
            test_dataset: 测试集

        Returns:
            pd.DataFrame: 包含 question, answer, contexts, ground_truth 的DataFrame
        """
        logger.info(f"开始为 {len(test_dataset)} 个测试问题生成RAG数据...")

        records = []
        for i, item in enumerate(test_dataset):
            question = item["question"]
            ground_truth = item.get("ground_truth", "")

            logger.info(f"处理问题 {i+1}/{len(test_dataset)}: {question[:50]}...")

            # 使用RAG系统生成回答
            result = self.rag_chain.query(question)

            # 从检索结果获取完整上下文（RAGAS需要原始文档正文用于计算context_recall/precision）
            # 注意：result.sources 是 SourceInfo 元数据列表（仅文件名/页码/100字预览），不含完整正文
            retrieve_result = self.rag_chain.retriever.retrieve(question)
            contexts = [doc.page_content for doc in retrieve_result.docs]

            records.append({
                "question": question,
                "answer": result.answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
            })

        df = pd.DataFrame(records)
        logger.info(f"RAG数据生成完成，共 {len(df)} 条")
        return df

    def evaluate(
        self,
        test_dataset: TestDataset,
        save_report: bool = True,
    ) -> Dict[str, Any]:
        """
        执行完整评估

        Args:
            test_dataset: 测试集
            save_report: 是否保存评估报告

        Returns:
            Dict: 评估结果
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("开始RAGAS评估")
        logger.info("=" * 60)

        # 第一步：生成RAG数据
        rag_df = self.generate_rag_data(test_dataset)

        # 保存原始数据
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_data_path = self.output_dir / f"rag_data_{timestamp}.csv"
        rag_df.to_csv(raw_data_path, index=False, encoding="utf-8-sig")
        logger.info(f"RAG原始数据已保存: {raw_data_path}")

        # 第二步：RAGAS评估
        if self._ragas_available:
            try:
                from datasets import Dataset
                from ragas import evaluate

                # 转为HuggingFace Dataset格式
                hf_dataset = Dataset.from_pandas(rag_df)

                # 执行评估
                result = evaluate(
                    hf_dataset,
                    metrics=self._metrics,
                    llm=self._ragas_llm,
                    embeddings=self._ragas_embeddings,
                )

                # ragas 0.1.x 的 Result 对象用 to_pandas() 取每条样本的得分
                df_scores = result.to_pandas()
                # 取数值列的平均值作为整体指标
                avg_metrics = {
                    k: round(float(v), 4)
                    for k, v in df_scores.mean(numeric_only=True).items()
                }
                # 同时保存每条样本的详细得分
                detail_path = self.output_dir / (
                    f"ragas_detail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )
                df_scores.to_csv(detail_path, index=False, encoding="utf-8-sig")
                logger.info(f"RAGAS逐条得分已保存: {detail_path}")

                logger.info("RAGAS评估结果:")
                for k, v in avg_metrics.items():
                    logger.info(f"  {k}: {v:.4f}")

            except Exception as e:
                logger.error(f"RAGAS评估失败: {str(e)}")
                avg_metrics = {"error": str(e)}
        else:
            logger.info("RAGAS不可用，使用基础评估模式")
            avg_metrics = self._basic_evaluation(rag_df)

        # 第三步：生成报告
        total_time = time.time() - start_time
        report = {
            "timestamp": timestamp,
            "test_count": len(test_dataset),
            "total_time_seconds": round(total_time, 2),
            "metrics": avg_metrics,
            "config": {
                "embedding_model": settings.model.embedding_model,
                "reranker_model": settings.model.reranker_model,
                "llm_model": settings.model.llm_model,
                "vector_top_k": settings.retrieval.vector_top_k,
                "bm25_top_k": settings.retrieval.bm25_top_k,
                "reranker_top_k": settings.retrieval.reranker_top_k,
                "chunk_size": settings.chunking.chunk_size,
            },
        }

        # 保存报告
        if save_report:
            report_path = self.output_dir / f"evaluation_report_{timestamp}.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info(f"评估报告已保存: {report_path}")

        logger.info(f"评估完成，总耗时: {total_time:.2f}s")
        return report

    def _basic_evaluation(self, rag_df: pd.DataFrame) -> Dict[str, float]:
        """
        基础评估模式（RAGAS不可用时的降级方案）

        简单统计：
        - 回答长度
        - 上下文数量
        - 无法回答的比例
        """
        total = len(rag_df)
        if total == 0:
            return {}

        # 统计无法回答的比例
        no_answer_count = rag_df["answer"].str.contains("无法回答", na=False).sum()
        no_answer_rate = no_answer_count / total

        # 平均回答长度
        avg_answer_length = rag_df["answer"].str.len().mean()

        # 平均上下文数量
        avg_context_count = rag_df["contexts"].apply(len).mean()

        return {
            "no_answer_rate": round(float(no_answer_rate), 4),
            "avg_answer_length": round(float(avg_answer_length), 2),
            "avg_context_count": round(float(avg_context_count), 2),
            "note": "基础评估模式（RAGAS不可用），建议安装ragas获取完整指标",
        }


# ============================================================
# 生成测试集模板
# ============================================================

def generate_test_set_template(output_path: str = "./evaluations/test_set_template.json"):
    """
    生成测试集模板，供用户填写

    Args:
        output_path: 输出路径
    """
    template = [
        {
            "question": "请在此处输入测试问题1",
            "ground_truth": "请在此处输入标准答案1"
        },
        {
            "question": "请在此处输入测试问题2",
            "ground_truth": "请在此处输入标准答案2"
        },
        {
            "question": "请在此处输入测试问题3",
            "ground_truth": "请在此处输入标准答案3"
        },
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)

    print(f"测试集模板已生成: {output_path}")
    print("请填写问题和标准答案后运行评估")


# ============================================================
# 模块自测
# ============================================================

if __name__ == "__main__":
    # import sys
    #
    # print("=" * 60)
    # print("RAGAS评估模块")
    # print("=" * 60)
    #
    # if len(sys.argv) > 1 and sys.argv[1] == "template":
    #     # 生成测试集模板
    #     generate_test_set_template()
    # else:
    #     print("\n使用方式:")
    #     print("  1. 生成测试集模板: python evaluator.py template")
    #     print("  2. 填写测试集后运行评估（在代码中调用）")
    #     print("\n评估示例代码:")
    #     print("""
    #     from evaluator import RAGASEvaluator, TestDataset
    #
    #     # 加载测试集
    #     test_set = TestDataset.from_json("./evaluations/test_set_template.json")
    #
    #     # 创建评估器并运行
    #     evaluator = RAGASEvaluator()
    #     report = evaluator.evaluate(test_set)
    #
    #     print("评估结果:", report)
    #     """)
    #
    # print("=" * 60)

    # from evaluator import RAGASEvaluator, TestDataset

    # 加载测试集
    test_set = TestDataset.from_json("./evaluations/test_set_template.json")

    # 创建评估器并运行
    evaluator = RAGASEvaluator()
    report = evaluator.evaluate(test_set)

    print("评估结果:", report)
