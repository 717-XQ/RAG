# -*- coding: utf-8 -*-
"""
fts5.py - SQLite FTS5 全文检索（技术栈2.1：中文知识关键词检索）
=============================================
功能：
  1. 基于 SQLite FTS5 虚拟表建立知识库关键词索引
  2. 中文检索：优先使用 trigram 分词器（SQLite≥3.34），不满足则回退 unicode61
  3. 与向量库/BM25 并列的关键词检索实现，供混合检索调用
  4. 支持增量添加、按文件名删除、关键词检索

使用方式：
  from fts5 import FTS5Index
  index = FTS5Index()
  index.add_chunks(documents)
  results = index.search("退货政策", top_k=20)
"""

import os
import sqlite3
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

from langchain_core.documents import Document
from loguru import logger

from config import settings


class FTS5Index:
    """
    SQLite FTS5 关键词索引

    存储结构：
      fts_index 虚拟表字段：id, filename, page, text, file_type
      text 字段使用 trigram 分词器（中文按3字符gram切分，支持子串检索）
    """

    def __init__(self, db_path: Optional[str] = None, tokenizer: Optional[str] = None):
        """
        初始化FTS5索引

        Args:
            db_path: SQLite数据库路径，默认从配置读取
            tokenizer: 分词器 trigram / unicode61，默认从配置读取
        """
        self.db_path = db_path or settings.fts5.db_path
        self.tokenizer = tokenizer or settings.fts5.tokenizer

        # 检查SQLite版本是否支持trigram
        sqlite_version = tuple(int(x) for x in sqlite3.sqlite_version.split(".")[:2])
        if self.tokenizer == "trigram" and sqlite_version < (3, 34):
            logger.warning(
                f"当前SQLite版本 {sqlite3.sqlite_version} 不支持trigram分词器，"
                f"回退到 unicode61（中文检索效果会下降）"
            )
            self.tokenizer = "unicode61"

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ============================================================
    # 内部：建表
    # ============================================================

    def _connect(self) -> sqlite3.Connection:
        """获取连接（每操作独立连接，线程安全）"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        """初始化FTS5虚拟表"""
        conn = self._connect()
        try:
            conn.execute(
                f"""CREATE VIRTUAL TABLE IF NOT EXISTS fts_index USING fts5(
                    id UNINDEXED,
                    filename UNINDEXED,
                    page UNINDEXED,
                    text,
                    file_type UNINDEXED,
                    tokenize='{self.tokenizer}'
                )"""
            )
            conn.commit()
            logger.info(f"FTS5索引初始化完成: {self.db_path} (tokenizer={self.tokenizer})")
        except sqlite3.Error as e:
            logger.error(f"FTS5索引初始化失败: {str(e)}")
            raise
        finally:
            conn.close()

    # ============================================================
    # 索引维护
    # ============================================================

    def add_chunks(self, documents: List[Document]) -> int:
        """
        添加文档块到FTS索引（按chunk_id幂等：先删后插）

        Args:
            documents: 文档块列表

        Returns:
            int: 添加数量
        """
        if not documents:
            return 0

        conn = self._connect()
        added = 0
        try:
            for doc in documents:
                chunk_id = doc.metadata.get("chunk_id", "")
                if not chunk_id:
                    continue
                # 幂等：先删除旧记录
                conn.execute("DELETE FROM fts_index WHERE id = ?", (chunk_id,))
                conn.execute(
                    "INSERT INTO fts_index (id, filename, page, text, file_type) VALUES (?, ?, ?, ?, ?)",
                    (
                        chunk_id,
                        doc.metadata.get("filename", ""),
                        doc.metadata.get("page", 0),
                        doc.page_content,
                        doc.metadata.get("file_type", ""),
                    ),
                )
                added += 1
            conn.commit()
            logger.info(f"FTS5索引添加完成: {added} 个块")
        except sqlite3.Error as e:
            logger.error(f"FTS5索引添加失败: {str(e)}")
            conn.rollback()
        finally:
            conn.close()
        return added

    def delete_by_filename(self, filename: str) -> int:
        """
        按文件名删除FTS索引记录

        Args:
            filename: 文件名

        Returns:
            int: 删除数量
        """
        conn = self._connect()
        try:
            cursor = conn.execute("DELETE FROM fts_index WHERE filename = ?", (filename,))
            conn.commit()
            deleted = cursor.rowcount
            logger.info(f"FTS5索引删除文档: {filename}, 共 {deleted} 条")
            return deleted
        except sqlite3.Error as e:
            logger.error(f"FTS5索引删除失败: {str(e)}")
            return 0
        finally:
            conn.close()

    def clear(self) -> None:
        """清空索引"""
        conn = self._connect()
        try:
            conn.execute("DELETE FROM fts_index")
            conn.commit()
        finally:
            conn.close()

    def count(self) -> int:
        """索引记录数"""
        conn = self._connect()
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM fts_index")
            return int(cursor.fetchone()[0])
        finally:
            conn.close()

    # ============================================================
    # 检索
    # ============================================================

    def search(self, query: str, top_k: int = 20) -> List[Tuple[Document, float]]:
        """
        FTS5关键词检索

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            List[Tuple[Document, float]]: (文档, BM25相关度分数) 列表
            分数越小越相关（FTS5 bm25() 返回负分，取反后越大越相关）
        """
        if not query.strip():
            return []

        # 构造FTS5查询：trigram下用短语查询（双引号包裹）效果更好
        match_expr = self._build_match_expr(query)

        conn = self._connect()
        try:
            cursor = conn.execute(
                """SELECT id, filename, page, text, file_type, bm25(fts_index) AS score
                   FROM fts_index
                   WHERE fts_index MATCH ?
                   ORDER BY score
                   LIMIT ?""",
                (match_expr, top_k),
            )
            rows = cursor.fetchall()
        except sqlite3.Error as e:
            logger.warning(f"FTS5检索失败: {str(e)}（query={query[:30]}）")
            return []
        finally:
            conn.close()

        results = []
        for row in rows:
            doc = Document(
                page_content=row["text"],
                metadata={
                    "filename": row["filename"],
                    "page": row["page"],
                    "chunk_id": row["id"],
                    "file_type": row["file_type"],
                },
            )
            # bm25()返回负相关度分数（越小越相关），取反变为越大越相关，便于与向量分数统一
            score = -float(row["score"])
            results.append((doc, score))

        return results

    @staticmethod
    def _build_match_expr(query: str) -> str:
        """
        构造FTS5 MATCH表达式

        策略（召回优先，排序交给 bm25() 与后续 Reranker 精排）：
          - 单个词：短语查询 "..."（trigram 支持子串匹配）
          - 多个词：OR 组合（任意词命中即召回，全命中的因 bm25 自然排前）
        """
        query = query.strip().replace('"', " ")
        # 简单分词：按空白/标点切分
        import re
        terms = [t for t in re.split(r"[\s，。、；：？！,.!?;:'\"()\[\]{}<>《》]+", query) if t]

        if not terms:
            return '""'

        if len(terms) == 1:
            # 短语查询，保留完整词
            return f'"{terms[0]}"'

        # 多词：OR 连接，提高召回（bm25() 会优先排序多词全命中的文档）
        return " OR ".join(f'"{t}"' for t in terms[:10])


# ============================================================
# 便捷函数：单例索引
# ============================================================

_fts_index: Optional[FTS5Index] = None


def get_fts_index() -> FTS5Index:
    """获取全局FTS5索引单例"""
    global _fts_index
    if _fts_index is None:
        _fts_index = FTS5Index()
    return _fts_index


# ============================================================
# 模块自测
# ============================================================

if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("FTS5检索模块自测")
    print("=" * 60)
    print(f"SQLite版本: {sqlite3.sqlite_version}")

    with tempfile.TemporaryDirectory() as tmp:
        index = FTS5Index(db_path=str(Path(tmp) / "test_fts.db"))
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
                page_content="BGE是北京智源人工智能研究院开发的中文Embedding模型，在中文检索任务上表现优异。",
                metadata={"filename": "bge.txt", "page": 1, "chunk_id": "bge_1", "file_type": ".txt"},
            ),
        ]
        index.add_chunks(test_docs)
        print(f"索引数量: {index.count()}")

        for q in ["什么是RAG", "向量数据库 检索", "中文Embedding模型"]:
            results = index.search(q, top_k=2)
            print(f"\n查询「{q}」命中 {len(results)} 条:")
            for doc, score in results:
                print(f"  score={score:.2f}, {doc.page_content[:36]}...")
    print("=" * 60)
