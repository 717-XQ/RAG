# -*- coding: utf-8 -*-
"""
models.py - ORM 数据模型（技术栈2.1：SQLAlchemy）
=============================================
业务数据持久化模型：
  - User:          用户（JWT认证）
  - DocumentRecord:知识库文档元数据
  - ChatSession:   问答会话
  - ChatMessage:   会话消息
  - TokenUsage:    Token用量统计（前端ECharts图表数据源）
"""

from __future__ import annotations  # Python 3.9 支持前向引用

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    String, Integer, Float, Text, DateTime, Boolean, ForeignKey, UniqueConstraint, JSON
)
from sqlalchemy.orm import Mapped, mapped_column, relationship, foreign

from database import Base


class User(Base):
    """用户表（认证）"""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    documents: Mapped[list["DocumentRecord"]] = relationship(back_populates="uploader")
    sessions: Mapped[list["ChatSession"]] = relationship(back_populates="user")


class DocumentRecord(Base):
    """知识库文档记录表（与向量库/FTS5索引同步）"""
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    file_type: Mapped[str] = mapped_column(String(16), default="")
    source: Mapped[str] = mapped_column(String(512), default="")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    upload_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    uploader_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)

    uploader: Mapped[Optional["User"]] = relationship(back_populates="documents")


class ChatSession(Base):
    """问答会话表"""
    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)
    title: Mapped[str] = mapped_column(String(255), default="新会话")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)

    user: Mapped[Optional["User"]] = relationship(back_populates="sessions")
    # session_id 为业务唯一键（String），需显式指定 join 条件与 foreign 标注
    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        primaryjoin="ChatSession.session_id == foreign(ChatMessage.session_id)",
    )


class ChatMessage(Base):
    """会话消息表"""
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)          # user / assistant
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)      # 引用来源
    token_count: Mapped[int] = mapped_column(Integer, default=0)           # 该条消息token数
    model: Mapped[str] = mapped_column(String(64), default="")             # 使用的模型
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    session: Mapped["ChatSession"] = relationship(
        back_populates="messages",
        primaryjoin="foreign(ChatMessage.session_id) == ChatSession.session_id",
    )


class TokenUsage(Base):
    """Token用量统计表（ECharts图表数据源）"""
    __tablename__ = "token_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    model: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
