# -*- coding: utf-8 -*-
"""
vector_store.py - 向量存储模块
=============================================
功能：
  1. 封装 Embedding 模型加载（BAAI/bge-large-zh-v1.5，1024维）
  2. 封装 Chroma 向量数据库操作（开发环境）
  3. 支持向量归一化（normalize_embeddings=True），使用余弦相似度
  4. 支持批量向量化与增量添加
  5. 支持按文档删除向量数据
  6. 预留 Milvus 适配接口（生产环境）

核心类：
  - EmbeddingModel: Embedding模型封装
  - VectorStore: 向量存储统一接口（Chroma实现）
"""

import os
import time
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from loguru import logger

from config import settings


# ============================================================
# Embedding 模型封装
# ============================================================

class BGEEmbedding(Embeddings):
    """
    BGE中文Embedding模型封装
    兼容 LangChain Embeddings 接口

    使用模型: BAAI/bge-large-zh-v1.5
    输出维度: 1024
    支持归一化（余弦相似度）
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ):
        """
        初始化Embedding模型

        Args:
            model_name: 模型名称或本地路径，默认从配置读取
            device: 推理设备 cuda/cpu，默认从配置读取
            normalize_embeddings: 是否归一化向量（余弦相似度必需）
        """
        self.model_name = model_name or settings.model.embedding_model
        self.device = device or settings.model.device
        self.normalize_embeddings = normalize_embeddings
        self._model = None

        logger.info(f"初始化Embedding模型: {self.model_name}, 设备: {self.device}")

    def _load_model(self):
        """延迟加载模型（首次调用时加载，避免启动慢）"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                )
                logger.info(f"Embedding模型加载成功: {self.model_name}")
            except Exception as e:
                logger.error(f"Embedding模型加载失败: {str(e)}")
                raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量向量化文档文本

        Args:
            texts: 文本列表

        Returns:
            List[List[float]]: 向量列表，每个向量是1024维float列表
        """
        self._load_model()

        if not texts:
            return []

        # 使用sentence-transformers编码
        embeddings = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        向量化查询文本（单条）

        Args:
            text: 查询文本

        Returns:
            List[float]: 1024维向量
        """
        self._load_model()

        # BGE模型要求查询前加 "为这个句子生成表示以用于检索相关文章：" 前缀
        # 这是bge-large-zh-v1.5的官方推荐用法
        query_text = f"为这个句子生成表示以用于检索相关文章：{text}"

        embedding = self._model.encode(
            [query_text],
            batch_size=1,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )

        return embedding[0].tolist()

    def get_dimension(self) -> int:
        """获取向量维度"""
        return 1024  # bge-large-zh-v1.5 固定输出1024维


# ============================================================
# 向量存储基类（定义统一接口）
# ============================================================

class BaseVectorStore:
    """向量存储基类，定义统一操作接口"""

    def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档到向量库"""
        raise NotImplementedError

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        相似度搜索

        Returns:
            List[Tuple[Document, float]]: (文档, 相似度分数) 列表
        """
        raise NotImplementedError

    def delete_by_filename(self, filename: str) -> int:
        """按文件名删除向量，返回删除数量"""
        raise NotImplementedError

    def delete_by_ids(self, ids: List[str]) -> None:
        """按ID删除"""
        raise NotImplementedError

    def count(self) -> int:
        """获取向量总数"""
        raise NotImplementedError

    def list_documents(self) -> List[Dict[str, Any]]:
        """列出所有文档（按文件名聚合）"""
        raise NotImplementedError

    def persist(self) -> None:
        """持久化到磁盘"""
        raise NotImplementedError


# ============================================================
# Chroma 向量存储实现
# ============================================================

class ChromaVectorStore(BaseVectorStore):
    """
    Chroma向量数据库实现（开发环境推荐）

    特点：
      - 轻量级，无需额外服务
      - 数据持久化到本地磁盘
      - 支持元数据过滤
      - 内置余弦相似度计算
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[BGEEmbedding] = None,
    ):
        """
        初始化Chroma向量存储

        Args:
            persist_directory: 持久化目录，默认从配置读取
            collection_name: 集合名称，默认从配置读取
            embedding_model: Embedding模型实例，None则自动创建
        """
        self.persist_directory = persist_directory or settings.vector_store.persist_directory
        self.collection_name = collection_name or settings.vector_store.collection_name
        self.embedding_model = embedding_model or BGEEmbedding()

        # 确保持久化目录存在
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        self._client = None
        self._collection = None
        self._init_chroma()

    def _init_chroma(self):
        """初始化Chroma客户端和集合"""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            raise ImportError("请安装 chromadb: pip install chromadb")

        # 创建Chroma客户端（持久化模式）
        self._client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,  # 关闭遥测
                allow_reset=True,
            ),
        )

        # 获取或创建集合
        # 使用 cosine 距离（Chroma中cosine距离 = 1 - cosine_similarity）
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"Chroma向量库初始化成功: 集合={self.collection_name}, "
            f"当前向量数={self._collection.count()}"
        )

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        批量添加文档到向量库

        Args:
            documents: 文档块列表（需包含chunk_id元数据）

        Returns:
            List[str]: 添加的文档ID列表
        """
        if not documents:
            return []

        logger.info(f"开始向量化并添加 {len(documents)} 个文档块...")
        start_time = time.time()

        # 提取文本、元数据、ID
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [doc.metadata.get("chunk_id", f"chunk_{i}_{time.time()}") for i, doc in enumerate(documents)]

        # 批量向量化
        embeddings = self.embedding_model.embed_documents(texts)

        # 分批添加到Chroma（避免单次数据过大）
        batch_size = 100
        added_ids = []
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            self._collection.add(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end],
                documents=texts[i:batch_end],
                metadatas=metadatas[i:batch_end],
            )
            added_ids.extend(ids[i:batch_end])

        elapsed = time.time() - start_time
        logger.info(f"向量添加完成: {len(documents)} 个块, 耗时 {elapsed:.2f}s")
        return added_ids

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回数量
            filter: 元数据过滤条件

        Returns:
            List[Tuple[Document, float]]: (Document, 相似度分数) 列表
            相似度分数范围 [0, 1]，越接近1越相关
        """
        # 向量化查询
        query_embedding = self.embedding_model.embed_query(query)

        # Chroma查询
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self._collection.count()),
            where=filter,
            include=["documents", "metadatas", "distances"],
        )

        # 解析结果
        docs_with_scores = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                doc = Document(
                    page_content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i],
                )
                # Chroma返回的是distance，转换为similarity (cosine距离 -> 相似度)
                distance = results["distances"][0][i]
                similarity = 1.0 - distance  # cosine距离转相似度
                docs_with_scores.append((doc, similarity))

        return docs_with_scores

    def delete_by_filename(self, filename: str) -> int:
        """
        按文件名删除所有相关向量

        Args:
            filename: 文件名

        Returns:
            int: 删除的向量数量
        """
        # 查询该文件名的所有向量ID
        results = self._collection.get(
            where={"filename": filename},
            include=[],
        )

        ids_to_delete = results["ids"]
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
            logger.info(f"删除文档向量: {filename}, 共 {len(ids_to_delete)} 个块")

        return len(ids_to_delete)

    def delete_by_ids(self, ids: List[str]) -> None:
        """按ID列表删除向量"""
        if ids:
            self._collection.delete(ids=ids)
            logger.info(f"按ID删除 {len(ids)} 个向量")

    def count(self) -> int:
        """获取向量总数"""
        return self._collection.count()

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        列出知识库中的所有文档（按文件名聚合）

        Returns:
            List[Dict]: 每个文档的信息 {filename, chunk_count, upload_time, file_type}
        """
        # 获取所有元数据
        results = self._collection.get(include=["metadatas"])

        # 按文件名聚合
        doc_map = {}
        for meta in results["metadatas"]:
            if not meta:
                continue
            filename = meta.get("filename", "unknown")
            if filename not in doc_map:
                doc_map[filename] = {
                    "filename": filename,
                    "chunk_count": 0,
                    "upload_time": meta.get("upload_time", ""),
                    "file_type": meta.get("file_type", ""),
                    "source": meta.get("source", ""),
                }
            doc_map[filename]["chunk_count"] += 1

        return list(doc_map.values())

    def persist(self) -> None:
        """Chroma PersistentClient 自动持久化，此方法留作接口兼容"""
        logger.debug("Chroma数据已自动持久化到磁盘")

    def get_all_chunks(self) -> List[Document]:
        """获取所有文档块（用于BM25索引构建等）"""
        results = self._collection.get(include=["documents", "metadatas"])
        docs = []
        for i in range(len(results["ids"])):
            docs.append(Document(
                page_content=results["documents"][i],
                metadata=results["metadatas"][i],
            ))
        return docs


# ============================================================
# Milvus 向量存储实现（生产环境，预留接口）
# ============================================================

class MilvusVectorStore(BaseVectorStore):
    """
    Milvus向量数据库实现（生产环境）

    使用前需要：
      1. 安装 pymilvus: pip install pymilvus
      2. 启动 Milvus 服务（Docker Compose）

    注意：本类为框架预留，核心方法已实现，可根据需要完善
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[BGEEmbedding] = None,
    ):
        self.host = host or settings.vector_store.milvus_host
        self.port = port or settings.vector_store.milvus_port
        self.collection_name = collection_name or settings.vector_store.collection_name
        self.embedding_model = embedding_model or BGEEmbedding()
        self._collection = None
        self._connect()

    def _connect(self):
        """连接Milvus并初始化集合"""
        try:
            from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
        except ImportError:
            raise ImportError("请安装 pymilvus: pip install pymilvus")

        connections.connect(host=self.host, port=self.port)

        # 如果集合不存在则创建
        if not utility.has_collection(self.collection_name):
            fields = [
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="page", dtype=DataType.INT64),
            ]
            schema = CollectionSchema(fields, description="RAG文档向量库")
            self._collection = Collection(self.collection_name, schema)
            # 创建IVF_FLAT索引
            index_params = {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 1024}}
            self._collection.create_index(field_name="embedding", index_params=index_params)
        else:
            self._collection = Collection(self.collection_name)

        self._collection.load()
        logger.info(f"Milvus连接成功: {self.host}:{self.port}, 集合: {self.collection_name}")

    def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档到Milvus"""
        if not documents:
            return []

        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_model.embed_documents(texts)
        ids = [doc.metadata.get("chunk_id", "") for doc in documents]
        filenames = [doc.metadata.get("filename", "") for doc in documents]
        pages = [doc.metadata.get("page", 0) for doc in documents]

        entities = [ids, embeddings, texts, filenames, pages]
        self._collection.insert(entities)
        self._collection.flush()
        logger.info(f"Milvus添加 {len(documents)} 个文档块")
        return ids

    def similarity_search(self, query, k=5, filter=None):
        """相似度搜索"""
        query_emb = self.embedding_model.embed_query(query)
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = self._collection.search(
            data=[query_emb],
            anns_field="embedding",
            param=search_params,
            limit=k,
            output_fields=["text", "filename", "page"],
        )

        docs_with_scores = []
        for hit in results[0]:
            meta = {
                "filename": hit.entity.get("filename"),
                "page": hit.entity.get("page"),
                "chunk_id": hit.id,
            }
            doc = Document(page_content=hit.entity.get("text"), metadata=meta)
            docs_with_scores.append((doc, hit.score))

        return docs_with_scores

    def delete_by_filename(self, filename):
        """按文件名删除"""
        expr = f'filename == "{filename}"'
        self._collection.delete(expr)
        return -1  # Milvus不直接返回删除数量

    def delete_by_ids(self, ids):
        """按ID删除"""
        expr = f'chunk_id in {ids}'
        self._collection.delete(expr)

    def count(self):
        return self._collection.num_entities

    def list_documents(self):
        """列出文档（Milvus需要额外查询，简化实现）"""
        return []

    def persist(self):
        """Milvus自动持久化"""
        pass


# ============================================================
# 向量存储工厂
# ============================================================

def get_vector_store(
    store_type: Optional[str] = None,
    embedding_model: Optional[BGEEmbedding] = None,
) -> BaseVectorStore:
    """
    根据配置创建向量存储实例

    Args:
        store_type: "chroma" 或 "milvus"，None则从配置读取
        embedding_model: 共享的Embedding模型实例

    Returns:
        BaseVectorStore: 向量存储实例
    """
    store_type = store_type or settings.vector_store.type

    if store_type.lower() == "chroma":
        return ChromaVectorStore(embedding_model=embedding_model)
    elif store_type.lower() == "milvus":
        return MilvusVectorStore(embedding_model=embedding_model)
    else:
        raise ValueError(f"不支持的向量库类型: {store_type}")


# ============================================================
# 模块自测
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("向量存储模块自测")
    print("=" * 60)

    # 注意：首次运行会下载BGE模型，需要网络连接
    print("\n提示: 首次运行将下载 BAAI/bge-large-zh-v1.5 模型（约1.3GB）")
    print("如果无法下载，请确保网络连接或使用本地模型路径")

    # 创建Embedding模型（使用CPU避免CUDA问题）
    embedding = BGEEmbedding(device="cpu")

    # 测试向量化
    test_texts = ["这是一个测试文档", "RAG系统可以提高问答准确性"]
    embeddings = embedding.embed_documents(test_texts)
    print(f"\n向量化测试: {len(test_text)} 条文本, 向量维度: {len(embeddings[0])}")

    # 创建Chroma向量库（使用临时目录）
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ChromaVectorStore(
            persist_directory=tmpdir,
            embedding_model=embedding,
        )

        # 添加测试文档
        test_docs = [
            Document(
                page_content="RAG是检索增强生成技术，结合检索和大模型生成准确回答。",
                metadata={"filename": "test.txt", "page": 1, "chunk_id": "test_1", "file_type": ".txt"},
            ),
            Document(
                page_content="向量数据库用于存储文档的向量表示，支持相似度搜索。",
                metadata={"filename": "test.txt", "page": 1, "chunk_id": "test_2", "file_type": ".txt"},
            ),
        ]
        store.add_documents(test_docs)
        print(f"向量库当前数量: {store.count()}")

        # 测试搜索
        results = store.similarity_search("什么是RAG？", k=2)
        print(f"\n搜索结果: {len(results)} 条")
        for doc, score in results:
            print(f"  相似度: {score:.4f}, 内容: {doc.page_content[:50]}...")

    print("\n" + "=" * 60)
