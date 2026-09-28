# -*- coding: utf-8 -*-
"""
api.py - FastAPI 服务模块
=============================================
功能：
  1. POST /upload          上传文档并加入知识库
  2. POST /chat            非流式问答
  3. POST /chat/stream     流式问答（SSE）
  4. GET  /docs            列出知识库中的文档列表
  5. DELETE /docs/{filename} 删除文档及其向量数据
  6. GET  /health          健康检查
  7. 支持CORS跨域
  8. 自动初始化RAG系统组件

启动方式：
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
  或: python api.py
"""

import os
import time
import shutil
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool  # 同步检索放入线程池，避免阻塞事件循环
from pydantic import BaseModel, Field
from loguru import logger

from config import settings
from document_loader import load_document, DocumentLoaderFactory
from text_splitter import split_documents
from vector_store import get_vector_store, BGEEmbedding
from retriever import HybridRetriever, BM25Retriever, Reranker
from rag_chain import RAGChain, AnswerResult


# ============================================================
# 请求/响应数据模型
# ============================================================

class ChatRequest(BaseModel):
    """问答请求体"""
    query: str = Field(..., description="用户问题", min_length=1, max_length=1000)
    use_reranker: bool = Field(True, description="是否使用Reranker精排")


class ChatResponse(BaseModel):
    """问答响应体"""
    answer: str
    sources: List[Dict[str, Any]]
    retrieve_time: float
    generate_time: float
    total_time: float


class UploadResponse(BaseModel):
    """上传响应体"""
    status: str
    filename: str
    chunks_added: int
    doc_id: str
    message: str


class DocInfo(BaseModel):
    """文档信息"""
    filename: str
    chunk_count: int
    upload_time: str
    file_type: str
    source: str


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    vector_count: int
    doc_count: int
    model_settings: Dict[str, Any]


# ============================================================
# 全局单例：系统组件（启动时初始化一次）
# ============================================================

class RAGSystem:
    """
    RAG系统单例，管理所有核心组件
    避免每次请求重复加载模型
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        logger.info("正在初始化RAG系统组件...")

        # 1. Embedding模型
        self.embedding_model = BGEEmbedding()

        # 2. 向量存储
        self.vector_store = get_vector_store(embedding_model=self.embedding_model)

        # 3. Reranker
        self.reranker = Reranker()

        # 4. 混合检索引擎
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedding_model=self.embedding_model,
            reranker=self.reranker,
        )

        # 5. RAG问答链
        self.rag_chain = RAGChain(retriever=self.retriever)

        # 上传目录
        self.upload_dir = Path(settings.storage.upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        logger.info("RAG系统组件初始化完成！")

    def add_document(self, file_path: str) -> Tuple[int, str]:
        """
        添加文档到知识库

        Args:
            file_path: 文档路径

        Returns:
            tuple: (添加的chunk数量, doc_id)
        """
        doc_id = str(uuid.uuid4())

        # 1. 加载文档
        documents = load_document(file_path)
        if not documents:
            return 0, doc_id

        # 2. 文本切分
        chunks = split_documents(documents)
        # 处理父子块情况（返回元组时取子块用于向量化）
        if isinstance(chunks, tuple):
            parent_chunks, child_chunks = chunks
            # 父子块策略：用子块检索，父块存储
            # 简化实现：将父块加入向量库
            chunks_to_add = parent_chunks
        else:
            chunks_to_add = chunks

        # 3. 添加到向量库
        self.vector_store.add_documents(chunks_to_add)

        # 4. 刷新BM25索引
        self.retriever.refresh_index()

        return len(chunks_to_add), doc_id

    def delete_document(self, filename: str) -> int:
        """
        删除文档及其向量数据

        Args:
            filename: 文件名

        Returns:
            int: 删除的chunk数量
        """
        # 从向量库删除
        deleted = self.vector_store.delete_by_filename(filename)

        # 从BM25索引删除
        self.retriever.bm25_retriever.remove_by_filename(filename)

        # 删除上传的文件
        file_path = self.upload_dir / filename
        if file_path.exists():
            file_path.unlink()

        return deleted


# 全局系统实例
rag_system = RAGSystem()


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(
    title="RAG智能文档问答系统",
    description="基于检索增强生成的智能文档问答API",
    version="1.0.0",
)

# CORS中间件（允许跨域，方便前端调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 生产环境建议限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# API 路由
# ============================================================

@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """
    健康检查接口
    返回系统状态、向量数量、文档数量等信息
    """
    docs = rag_system.vector_store.list_documents()
    return HealthResponse(
        status="healthy",
        vector_count=rag_system.vector_store.count(),
        doc_count=len(docs),
        model_settings={
            "embedding_model": settings.model.embedding_model,
            "reranker_model": settings.model.reranker_model,
            "llm_model": settings.model.llm_model,
            "vector_store": settings.vector_store.type,
        },
    )


@app.post("/upload", response_model=UploadResponse, tags=["文档管理"])
async def upload_document(file: UploadFile = File(...)):
    """
    上传文档并加入知识库

    支持格式: PDF, DOCX, TXT, HTML
    """
    # 检查文件格式
    if not DocumentLoaderFactory.is_supported(file.filename):
        supported = ", ".join(settings.storage.allowed_extensions)
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式。支持: {supported}"
        )

    # 保存上传文件
    file_path = rag_system.upload_dir / file.filename
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}")

    # 添加到知识库
    try:
        chunks_added, doc_id = rag_system.add_document(str(file_path))
    except Exception as e:
        # 失败时删除已保存的文件
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")

    return UploadResponse(
        status="success",
        filename=file.filename,
        chunks_added=chunks_added,
        doc_id=doc_id,
        message=f"文档上传成功，已添加 {chunks_added} 个文本块到知识库",
    )


@app.post("/chat", response_model=ChatResponse, tags=["问答"])
async def chat(request: ChatRequest):
    """
    非流式问答接口

    输入问题，返回回答和引用来源
    """
    try:
        # 检索+生成是CPU/GPU密集的同步操作，放入线程池，避免阻塞事件循环
        result: AnswerResult = await run_in_threadpool(
            rag_system.rag_chain.query,
            question=request.query,
            use_reranker=request.use_reranker,
        )
        return ChatResponse(
            answer=result.answer,
            sources=result.to_dict()["sources"],
            retrieve_time=result.retrieve_time,
            generate_time=result.generate_time,
            total_time=result.total_time,
        )
    except Exception as e:
        logger.error(f"问答失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"问答失败: {str(e)}")


@app.post("/chat/stream", tags=["问答"])
async def chat_stream(request: ChatRequest):
    """
    流式问答接口（SSE - Server-Sent Events）

    逐token输出回答，适合实时显示
    返回 Content-Type: text/event-stream
    """
    async def event_generator():
        try:
            async for chunk in rag_system.rag_chain.query_stream(
                question=request.query,
                use_reranker=request.use_reranker,
            ):
                yield chunk
        except Exception as e:
            error_data = {"type": "error", "message": str(e)}
            import json
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用Nginx缓冲
        },
    )


@app.get("/docs", response_model=List[DocInfo], tags=["文档管理"])
async def list_documents():
    """
    列出知识库中的所有文档
    """
    docs = rag_system.vector_store.list_documents()
    return [DocInfo(**doc) for doc in docs]


@app.delete("/docs/{filename}", tags=["文档管理"])
async def delete_document(filename: str):
    """
    删除指定文档及其向量数据

    Args:
        filename: 要删除的文件名
    """
    try:
        deleted_count = rag_system.delete_document(filename)
        if deleted_count == 0:
            raise HTTPException(status_code=404, detail=f"未找到文档: {filename}")
        return {
            "status": "success",
            "filename": filename,
            "deleted_chunks": deleted_count,
            "message": f"文档 {filename} 已删除，共移除 {deleted_count} 个向量",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


@app.get("/", tags=["系统"])
async def root():
    """根路径，返回API信息"""
    return {
        "name": "RAG智能文档问答系统",
        "version": "1.0.0",
        "docs": "/docs",       # Swagger UI
        "redoc": "/redoc",     # ReDoc
        "endpoints": {
            "upload": "POST /upload",
            "chat": "POST /chat",
            "chat_stream": "POST /chat/stream",
            "list_docs": "GET /docs",
            "delete_doc": "DELETE /docs/{filename}",
            "health": "GET /health",
        },
    }


# ============================================================
# 启动入口
# ============================================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("RAG智能文档问答系统 - API服务启动")
    print("=" * 60)
    print(f"服务地址: http://{settings.server.host}:{settings.server.port}")
    print(f"API文档: http://{settings.server.host}:{settings.server.port}/docs")
    print(f"向量库: {settings.vector_store.type} ({settings.vector_store.collection_name})")
    print(f"Embedding模型: {settings.model.embedding_model}")
    print(f"LLM模型: {settings.model.llm_model}")
    print("=" * 60)

    uvicorn.run(
        "api:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=False,        # 生产环境关闭reload
        log_level=settings.logging.level.lower(),
    )
