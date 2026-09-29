# -*- coding: utf-8 -*-
"""
api.py - FastAPI 服务模块
=============================================
功能（已按《技术栈规范》2.1 对齐扩展）：
  1. POST /auth/register       注册（bcrypt密码哈希）
  2. POST /auth/login          登录（JWT签发，python-jose）
  3. POST /auth/refresh        刷新Token（前端Token自动刷新）
  4. POST /upload              上传文档并加入知识库
  5. POST /chat                非流式问答（记录会话/Token用量）
  6. POST /chat/stream         流式问答（SSE）
  7. GET  /docs                列出知识库文档
  8. DELETE /docs/{filename}   删除文档
  9. POST /export/pdf          WeasyPrint 导出PDF报告
  10. POST /export/docx        python-docx 导出Word报告
  11. GET  /stats/token-usage  Token用量统计（ECharts图表）
  12. GET  /health             健康检查
  13. 支持CORS跨域

启动方式：
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
  或: python api.py
"""

import os
import time
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

# ---------- 技术栈2.1 新增：认证 / 数据库 / 导出 ----------
from auth import (
    RegisterRequest, LoginRequest, RefreshRequest, TokenResponse, UserOut,
    register_user, login_user, refresh_access_token, create_access_token,
    create_refresh_token, get_current_user,
)
from database import SessionLocal, init_db, get_db
from models import User, DocumentRecord, ChatSession, ChatMessage, TokenUsage
from export import export_pdf, export_docx


# ============================================================
# 请求/响应数据模型
# ============================================================

class ChatRequest(BaseModel):
    """问答请求体"""
    query: str = Field(..., description="用户问题", min_length=1, max_length=1000)
    use_reranker: bool = Field(True, description="是否使用Reranker精排")
    session_id: Optional[str] = Field(None, description="会话ID（不传则自动创建）")


class ChatResponse(BaseModel):
    """问答响应体"""
    answer: str
    sources: List[Dict[str, Any]]
    retrieve_time: float
    generate_time: float
    total_time: float
    session_id: str = ""
    provider: str = ""
    token_usage: Dict[str, Any] = Field(default_factory=dict)


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


class ExportRequest(BaseModel):
    """报告导出请求体"""
    question: str = Field(..., description="问题")
    answer: str = Field(..., description="回答")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="引用来源")
    title: Optional[str] = Field(None, description="报告标题")


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

        # 0. 初始化关系数据库（建表，技术栈2.1）
        try:
            init_db()
        except Exception as e:
            logger.error(f"关系数据库初始化失败（系统将继续以向量库模式运行）: {str(e)}")

        # 1. Embedding模型
        self.embedding_model = BGEEmbedding()

        # 2. 向量存储
        self.vector_store = get_vector_store(embedding_model=self.embedding_model)

        # 3. Reranker
        self.reranker = Reranker()

        # 4. 混合检索引擎（关键词路：BM25 / SQLite FTS5）
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedding_model=self.embedding_model,
            reranker=self.reranker,
        )

        # 5. RAG问答链（多Provider自动降级）
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

        # 4. 刷新关键词索引（BM25重建 / FTS5增量，技术栈2.1）
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

        # 从关键词索引删除（BM25 / FTS5）
        if hasattr(self.retriever.keyword_retriever, "remove_by_filename"):
            self.retriever.keyword_retriever.remove_by_filename(filename)
        elif hasattr(self.retriever.keyword_retriever, "delete_by_filename"):
            self.retriever.keyword_retriever.delete_by_filename(filename)

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
    description="基于检索增强生成的智能文档问答API（技术栈2.1对齐）",
    version="2.0.0",
    docs_url="/swagger",   # 避免与业务路由 /docs（知识库文档列表）冲突
    redoc_url="/redoc",
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
# 认证路由（技术栈2.1：python-jose + bcrypt JWT）
# ============================================================

@app.post("/auth/register", response_model=TokenResponse, tags=["认证"])
async def auth_register(req: RegisterRequest, db=Depends(get_db)):
    """用户注册，注册成功即返回Token"""
    user = register_user(db, req)
    return TokenResponse(
        access_token=create_access_token(user.username),
        refresh_token=create_refresh_token(user.username),
        expires_in=settings.auth.access_token_expire_minutes * 60,
    )


@app.post("/auth/login", response_model=TokenResponse, tags=["认证"])
async def auth_login(req: LoginRequest, db=Depends(get_db)):
    """用户登录"""
    user = login_user(db, req)
    return TokenResponse(
        access_token=create_access_token(user.username),
        refresh_token=create_refresh_token(user.username),
        expires_in=settings.auth.access_token_expire_minutes * 60,
    )


@app.post("/auth/refresh", response_model=TokenResponse, tags=["认证"])
async def auth_refresh(req: RefreshRequest):
    """用刷新Token换取新的访问Token"""
    new_access = refresh_access_token(req.refresh_token)
    return TokenResponse(
        access_token=new_access,
        refresh_token=req.refresh_token,
        expires_in=settings.auth.access_token_expire_minutes * 60,
    )


@app.get("/auth/me", response_model=UserOut, tags=["认证"])
async def auth_me(user: User = Depends(get_current_user)):
    """当前用户信息"""
    if user is None:
        raise HTTPException(status_code=401, detail="认证未启用")
    return UserOut(
        id=user.id,
        username=user.username,
        email=user.email,
        is_active=user.is_active,
        created_at=user.created_at.strftime("%Y-%m-%d %H:%M:%S") if user.created_at else None,
    )


# ============================================================
# 工具函数：会话与Token记录
# ============================================================

def _get_or_create_session(db, session_id: str, user: Optional[User]) -> str:
    """获取或创建会话，返回session_id"""
    if not session_id:
        session_id = str(uuid.uuid4())
    existing = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    if not existing:
        db.add(ChatSession(
            session_id=session_id,
            user_id=user.id if user else None,
            title=f"会话 {session_id[:8]}",
        ))
        db.commit()
    return session_id


def _save_chat_record(
    db,
    session_id: str,
    question: str,
    answer: str,
    sources: List[Dict[str, Any]],
    provider: str,
    token_usage: Dict[str, Any],
    user: Optional[User],
) -> None:
    """保存问答记录（会话消息 + Token用量）"""
    try:
        session_id = _get_or_create_session(db, session_id, user)

        db.add(ChatMessage(
            session_id=session_id,
            role="user",
            content=question,
            token_count=int(token_usage.get("prompt_tokens", 0) or 0),
            model=provider,
        ))
        db.add(ChatMessage(
            session_id=session_id,
            role="assistant",
            content=answer,
            sources=sources,
            token_count=int(token_usage.get("completion_tokens", 0) or 0),
            model=provider,
        ))
        db.add(TokenUsage(
            user_id=user.id if user else None,
            session_id=session_id,
            input_tokens=int(token_usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(token_usage.get("completion_tokens", 0) or 0),
            model=provider,
        ))
        db.commit()
    except Exception as e:
        logger.warning(f"会话记录保存失败（不影响问答主流程）: {str(e)}")


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
            "keyword_retriever": settings.retrieval.keyword_retriever,
            "auth": settings.auth.enabled,
            "providers": [p.name for p in settings.model.providers],
        },
    )


@app.post("/upload", response_model=UploadResponse, tags=["文档管理"])
async def upload_document(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
):
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

    # 同步文档记录到关系数据库（技术栈2.1）
    try:
        db = SessionLocal()
        try:
            ext = Path(file.filename).suffix.lower()
            existing = db.query(DocumentRecord).filter(DocumentRecord.filename == file.filename).first()
            if existing:
                existing.chunk_count = chunks_added
                existing.uploader_id = user.id if user else None
            else:
                db.add(DocumentRecord(
                    doc_id=doc_id,
                    filename=file.filename,
                    file_type=ext,
                    source=str(file_path),
                    chunk_count=chunks_added,
                    uploader_id=user.id if user else None,
                ))
            db.commit()
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"文档记录入库失败: {str(e)}")

    return UploadResponse(
        status="success",
        filename=file.filename,
        chunks_added=chunks_added,
        doc_id=doc_id,
        message=f"文档上传成功，已添加 {chunks_added} 个文本块到知识库",
    )


@app.post("/chat", response_model=ChatResponse, tags=["问答"])
async def chat(
    request: ChatRequest,
    user: User = Depends(get_current_user),
):
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

        # 记录会话与Token用量（异步线程池，不阻塞主流程）
        session_id = request.session_id or str(uuid.uuid4())
        db = SessionLocal()
        try:
            _save_chat_record(
                db, session_id, request.query, result.answer,
                result.to_dict().get("sources", []),
                result.provider, result.token_usage, user,
            )
        finally:
            db.close()

        return ChatResponse(
            answer=result.answer,
            sources=result.to_dict()["sources"],
            retrieve_time=result.retrieve_time,
            generate_time=result.generate_time,
            total_time=result.total_time,
            session_id=session_id,
            provider=result.provider,
            token_usage=result.token_usage,
        )
    except Exception as e:
        logger.error(f"问答失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"问答失败: {str(e)}")


@app.post("/chat/stream", tags=["问答"])
async def chat_stream(
    request: ChatRequest,
    user: User = Depends(get_current_user),
):
    """
    流式问答接口（SSE - Server-Sent Events）

    逐token输出回答，适合实时显示
    返回 Content-Type: text/event-stream
    """
    async def event_generator():
        import json
        collected: List[str] = []
        provider_used = ""
        session_id = request.session_id or str(uuid.uuid4())

        try:
            async for chunk in rag_system.rag_chain.query_stream(
                question=request.query,
                use_reranker=request.use_reranker,
            ):
                yield chunk
                # 解析并收集内容（用于流式结束后入库）
                try:
                    if chunk.startswith("data: "):
                        data = json.loads(chunk[6:].strip())
                        if data.get("type") == "content":
                            collected.append(data.get("content", ""))
                        elif data.get("type") == "provider":
                            provider_used = data.get("name", "")
                except Exception:
                    pass
        except Exception as e:
            error_data = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
        finally:
            # 流式结束后异步保存记录
            if collected:
                full_answer = "".join(collected)
                db = SessionLocal()
                try:
                    _save_chat_record(
                        db, session_id, request.query, full_answer,
                        [], provider_used, {}, user,
                    )
                finally:
                    db.close()

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
async def list_documents(user: User = Depends(get_current_user)):
    """
    列出知识库中的所有文档
    """
    docs = rag_system.vector_store.list_documents()
    return [DocInfo(**doc) for doc in docs]


@app.delete("/docs/{filename}", tags=["文档管理"])
async def delete_document(
    filename: str,
    user: User = Depends(get_current_user),
):
    """
    删除指定文档及其向量数据

    Args:
        filename: 要删除的文件名
    """
    try:
        deleted_count = rag_system.delete_document(filename)
        if deleted_count == 0:
            raise HTTPException(status_code=404, detail=f"未找到文档: {filename}")

        # 同步删除数据库文档记录
        try:
            db = SessionLocal()
            try:
                record = db.query(DocumentRecord).filter(DocumentRecord.filename == filename).first()
                if record:
                    db.delete(record)
                    db.commit()
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"文档记录删除失败: {str(e)}")

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


# ============================================================
# 导出路由（技术栈2.1：WeasyPrint PDF / python-docx Word）
# ============================================================

@app.post("/export/pdf", tags=["导出"])
async def export_report_pdf(
    req: ExportRequest,
    user: User = Depends(get_current_user),
):
    """WeasyPrint 导出PDF问答报告（HTML转PDF）"""
    try:
        data = await run_in_threadpool(export_pdf, req.question, req.answer, req.sources, req.title or "")
        return StreamingResponse(
            iter([data]),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="rag_report_{int(time.time())}.pdf"'},
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF导出失败: {str(e)}")


@app.post("/export/docx", tags=["导出"])
async def export_report_docx(
    req: ExportRequest,
    user: User = Depends(get_current_user),
):
    """python-docx 导出Word问答报告"""
    try:
        data = await run_in_threadpool(export_docx, req.question, req.answer, req.sources, req.title or "")
        return StreamingResponse(
            iter([data]),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="rag_report_{int(time.time())}.docx"'},
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Word导出失败: {str(e)}")


# ============================================================
# 统计路由（技术栈2.2：ECharts Token用量统计图表）
# ============================================================

@app.get("/stats/token-usage", tags=["统计"])
async def token_usage_stats(
    days: int = 7,
    user: User = Depends(get_current_user),
):
    """
    Token用量统计（按天聚合，供前端ECharts图表）

    Args:
        days: 最近N天，默认7
    """
    from datetime import datetime, timedelta, date
    from sqlalchemy import func

    days = max(1, min(days, 90))
    # 统计最近 days 天（含今天）：从 0 点往前推 days-1 天
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    since = today - timedelta(days=days - 1)

    db = SessionLocal()
    try:
        rows = (
            db.query(
                func.date(TokenUsage.created_at).label("day"),
                func.sum(TokenUsage.input_tokens).label("input_tokens"),
                func.sum(TokenUsage.output_tokens).label("output_tokens"),
                func.count(TokenUsage.id).label("calls"),
            )
            .filter(TokenUsage.created_at >= since)
            .group_by(func.date(TokenUsage.created_at))
            .order_by(func.date(TokenUsage.created_at))
            .all()
        )

        # 补全缺失日期（0值）
        result = []
        for i in range(days):
            d = since.date() + timedelta(days=i)
            day_str = d.strftime("%Y-%m-%d")
            matched = next((r for r in rows if str(r.day) == day_str), None)
            result.append({
                "date": day_str,
                "input_tokens": int(matched.input_tokens) if matched else 0,
                "output_tokens": int(matched.output_tokens) if matched else 0,
                "total_tokens": (int(matched.input_tokens) + int(matched.output_tokens)) if matched else 0,
                "calls": int(matched.calls) if matched else 0,
            })
        return {"days": days, "data": result}
    finally:
        db.close()


@app.get("/stats/sessions", tags=["统计"])
async def session_list(
    user: User = Depends(get_current_user),
):
    """会话列表（最近20个）"""
    db = SessionLocal()
    try:
        sessions = (
            db.query(ChatSession)
            .order_by(ChatSession.updated_at.desc())
            .limit(20)
            .all()
        )
        return [
            {
                "session_id": s.session_id,
                "title": s.title,
                "created_at": s.created_at.strftime("%Y-%m-%d %H:%M:%S") if s.created_at else "",
                "updated_at": s.updated_at.strftime("%Y-%m-%d %H:%M:%S") if s.updated_at else "",
            }
            for s in sessions
        ]
    finally:
        db.close()


@app.get("/", tags=["系统"])
async def root():
    """根路径，返回API信息"""
    return {
        "name": "RAG智能文档问答系统",
        "version": "2.0.0",
        "swagger": "/swagger",   # Swagger UI
        "redoc": "/redoc",       # ReDoc
        "endpoints": {
            "auth_register": "POST /auth/register",
            "auth_login": "POST /auth/login",
            "auth_refresh": "POST /auth/refresh",
            "upload": "POST /upload",
            "chat": "POST /chat",
            "chat_stream": "POST /chat/stream",
            "list_docs": "GET /docs",
            "delete_doc": "DELETE /docs/{filename}",
            "export_pdf": "POST /export/pdf",
            "export_docx": "POST /export/docx",
            "token_stats": "GET /stats/token-usage",
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
    print(f"关键词检索: {settings.retrieval.keyword_retriever}")
    print(f"Embedding模型: {settings.model.embedding_model}")
    print(f"LLM Provider: {', '.join(p.name for p in settings.model.providers)}")
    print(f"认证: {'启用' if settings.auth.enabled else '禁用'}")
    print(f"数据库: {settings.database.url}")
    print("=" * 60)

    uvicorn.run(
        "api:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=False,        # 生产环境关闭reload
        log_level=settings.logging.level.lower(),
    )
