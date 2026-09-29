# -*- coding: utf-8 -*-
"""
database.py - 关系数据库接入（技术栈2.1：SQLAlchemy 2.0+ / MySQL开发 / PostgreSQL生产）
=============================================
功能：
  1. 创建 SQLAlchemy Engine 与 Session
  2. 默认连接 MySQL（配置 database.url），失败时按配置回退 SQLite 便于本地开发
  3. 提供 init_db() 建表、get_db() FastAPI 依赖

使用方式：
  from database import SessionLocal, init_db, get_db
  init_db()
  db = SessionLocal()
"""

from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.exc import SQLAlchemyError
from loguru import logger

from config import settings


class Base(DeclarativeBase):
    """ORM 声明基类"""
    pass


_engine = None
_SessionLocal = None


def _build_engine(url: str, echo: bool):
    """构建SQLAlchemy引擎（通用参数，兼容SQLite/MySQL/PostgreSQL）"""
    if url.startswith("sqlite"):
        # SQLite 需要 check_same_thread=False（FastAPI线程池场景）
        return create_engine(url, echo=echo, connect_args={"check_same_thread": False})
    return create_engine(url, echo=echo, pool_pre_ping=True, pool_recycle=3600)


def get_engine():
    """
    获取全局 Engine（惰性初始化）

    优先使用配置的关系数据库（MySQL/PostgreSQL），
    连接失败且开启 fallback_sqlite 时回退到本地 SQLite。
    """
    global _engine, _SessionLocal
    if _engine is not None:
        return _engine

    url = settings.database.url
    echo = settings.database.echo
    try:
        _engine = _build_engine(url, echo)
        # 实际连接探测（不建表，只验证连通性）
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info(f"关系数据库连接成功: {_engine.url.drivername}")
    except SQLAlchemyError as e:
        logger.error(f"关系数据库连接失败: {str(e)}")
        if settings.database.fallback_sqlite:
            sqlite_url = f"sqlite:///{settings.database.sqlite_path}"
            logger.warning(f"回退到本地SQLite: {sqlite_url}（生产请配置MySQL/PostgreSQL）")
            _engine = _build_engine(sqlite_url, echo)
        else:
            raise

    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, expire_on_commit=False)
    return _engine


def SessionLocal():
    """获取数据库会话"""
    get_engine()
    return _SessionLocal()


def get_db() -> Generator:
    """FastAPI 依赖：请求级数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """初始化数据库：创建所有表（幂等）"""
    from models import User, DocumentRecord, ChatSession, ChatMessage, TokenUsage  # noqa: F401
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    logger.info(f"数据库表初始化完成: {engine.url.drivername}")
