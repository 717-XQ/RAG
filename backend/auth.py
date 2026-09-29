# -*- coding: utf-8 -*-
"""
auth.py - JWT 认证模块（技术栈2.1：python-jose + bcrypt）
=============================================
功能：
  1. 用户注册 / 登录（bcrypt 密码哈希）
  2. JWT 访问Token签发与校验（python-jose，HS256）
  3. 刷新Token（长期有效，用于前端 Token 自动刷新）
  4. FastAPI 依赖 get_current_user，保护业务API

使用方式：
  from fastapi import Depends
  from auth import get_current_user, UserOut
  @app.get("/protected")
  async def protected(user: UserOut = Depends(get_current_user)):
      return user
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import jwt, JWTError
import bcrypt
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from sqlalchemy.orm import Session
from loguru import logger

from config import settings
from database import get_db
from models import User

# ============================================================
# OAuth2 方案：前端从 /auth/login 拿 token，之后带 Bearer Token
# ============================================================
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


# ============================================================
# Pydantic 模型
# ============================================================

class RegisterRequest(BaseModel):
    """注册请求"""
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    """登录请求"""
    username: str
    password: str


class TokenResponse(BaseModel):
    """Token响应"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshRequest(BaseModel):
    """刷新Token请求"""
    refresh_token: str


class UserOut(BaseModel):
    """用户信息响应"""
    id: int
    username: str
    email: str
    is_active: bool
    created_at: Optional[str] = None

    class Config:
        from_attributes = True


# ============================================================
# 密码哈希（bcrypt）
# ============================================================

def hash_password(password: str) -> str:
    """bcrypt 密码哈希"""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """校验密码"""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


# ============================================================
# JWT 签发与校验（python-jose）
# ============================================================

def _create_token(subject: str, token_type: str, expires_delta: timedelta) -> str:
    """签发JWT"""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "type": token_type,
        "iat": now,
        "exp": now + expires_delta,
    }
    return jwt.encode(payload, settings.auth.jwt_secret, algorithm=settings.auth.jwt_algorithm)


def create_access_token(username: str) -> str:
    """签发访问Token"""
    return _create_token(
        username,
        "access",
        timedelta(minutes=settings.auth.access_token_expire_minutes),
    )


def create_refresh_token(username: str) -> str:
    """签发刷新Token"""
    return _create_token(
        username,
        "refresh",
        timedelta(days=settings.auth.refresh_token_expire_days),
    )


def decode_token(token: str, expected_type: Optional[str] = None) -> Optional[str]:
    """
    解析JWT并返回subject（username）

    Args:
        token: JWT字符串
        expected_type: 期望的token类型（access/refresh），None则不校验类型

    Returns:
        str | None: username；无效/过期返回None
    """
    try:
        payload = jwt.decode(
            token,
            settings.auth.jwt_secret,
            algorithms=[settings.auth.jwt_algorithm],
        )
        if expected_type and payload.get("type") != expected_type:
            return None
        return payload.get("sub")
    except JWTError:
        return None


# ============================================================
# FastAPI 依赖：当前用户
# ============================================================

def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    FastAPI 依赖：从 Bearer Token 解析当前用户

    Raises:
        HTTPException 401: Token缺失/无效/用户不存在
    """
    if not settings.auth.enabled:
        # 认证关闭时放行（开发模式）
        return None  # type: ignore

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未提供认证Token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    username = decode_token(token, expected_type="access")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token无效或已过期",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(User.username == username).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在或已禁用",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


# ============================================================
# 业务函数：注册 / 登录 / 刷新
# ============================================================

def register_user(db: Session, req: RegisterRequest) -> User:
    """注册用户（用户名/邮箱唯一）"""
    username = req.username.strip()
    email = req.email.strip().lower()
    if not username or not email or not req.password:
        raise HTTPException(status_code=400, detail="用户名、邮箱、密码不能为空")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="密码至少6位")

    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="用户名已存在")
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=400, detail="邮箱已被注册")

    user = User(
        username=username,
        email=email,
        password_hash=hash_password(req.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info(f"新用户注册: {username}")
    return user


def login_user(db: Session, req: LoginRequest) -> User:
    """用户登录（校验密码）"""
    user = db.query(User).filter(User.username == req.username.strip()).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )
    if not user.is_active:
        raise HTTPException(status_code=403, detail="账号已被禁用")
    return user


def refresh_access_token(refresh_token: str) -> str:
    """用刷新Token换取新的访问Token"""
    username = decode_token(refresh_token, expected_type="refresh")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="刷新Token无效或已过期",
        )
    return create_access_token(username)
