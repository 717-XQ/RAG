# -*- coding: utf-8 -*-
"""
config.py - 配置管理模块
=============================================
功能：
  1. 从 config.yaml 加载系统配置
  2. 支持环境变量覆盖（如 OPENAI_API_KEY）
  3. 提供全局单例配置对象，供其他模块引用
  4. 自动创建必要的目录（上传目录、日志目录、向量库目录）

使用方式：
  from config import settings
  print(settings.model.embedding_model)
"""

import os

# ============================================================
# 【国内网络适配】设置 HuggingFace 国内镜像
# 解决国内无法访问 huggingface.co 导致模型下载超时的问题
# 如需使用官方源，可设置环境变量 HF_ENDPOINT=https://huggingface.co
# ============================================================
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 关闭 Chroma 遥测（避免 telemetry 相关的错误日志）
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import yaml
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量（如果存在）
load_dotenv()


# ============================================================
# Pydantic 配置模型定义（类型安全的配置结构）
# ============================================================

class ModelConfig(BaseModel):
    """模型相关配置"""
    embedding_model: str = "BAAI/bge-large-zh-v1.5"    # 嵌入模型，用于文本向量化
    reranker_model: str = "BAAI/bge-reranker-v2-m3"    # 重排序模型，用于优化检索结果
    llm_model: str = "gpt-3.5-turbo"                   # 大语言模型，用于生成回答
    llm_api_key: str = ""                              # LLM API密钥
    llm_base_url: str = "https://api.openai.com/v1"    # LLM API基础URL
    llm_temperature: float = 0.1                       # LLM温度参数，控制生成随机性
    llm_max_tokens: int = 1024                         # LLM最大生成token数
    device: str = "cuda"                               # 推理设备：cuda / cpu


class RetrievalConfig(BaseModel):
    """检索相关配置"""
    vector_top_k: int = 20                            # 向量检索TopK
    bm25_top_k: int = 20                              # BM25检索TopK
    reranker_top_k: int = 5                           # 重排序TopK
    vector_weight: float = 0.5                        # 向量权重
    bm25_weight: float = 0.5                          # BM25权重


class ChunkingConfig(BaseModel):
    """文本切分配置"""
    chunk_size: int = 500                             # 文本切分大小
    chunk_overlap: int = 100                          # 文本切分重叠
    enable_parent_child: bool = False                 # 是否启用父子关系
    child_chunk_size: int = 200                       # 子块大小
    parent_chunk_size: int = 1000                     # 父块大小


class VectorStoreConfig(BaseModel):
    """向量数据库配置"""
    type: str = "chroma"                              # 向量数据库类型
    persist_directory: str = "./chroma_db"            # 向量数据库持久化目录
    collection_name: str = "rag_docs"                 # 向量数据库集合名
    milvus_host: str = "localhost"                    # Milvus主机
    milvus_port: str = "19530"                        # Milvus端口


class StorageConfig(BaseModel):
    """文件存储配置"""
    upload_dir: str = "./uploaded_docs"               # 上传文档目录
    allowed_extensions: List[str] = [".pdf", ".docx", ".txt", ".html", ".htm"]  # 允许的文件扩展名


class ServerConfig(BaseModel):
    """服务配置"""
    host: str = "0.0.0.0"                             # 服务主机
    port: int = 8000                                  # 服务端口


class LoggingConfig(BaseModel):
    """日志配置"""
    level: str = "INFO"                               # 日志级别
    file: str = "./logs/rag.log"                      # 日志文件


class Settings(BaseModel):
    """全局配置根对象"""
    model: ModelConfig = Field(default_factory=ModelConfig)                     # 模型配置
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)         # 检索配置
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)            # 文本切分配置
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)  # 向量数据库配置
    storage: StorageConfig = Field(default_factory=StorageConfig)               # 文件存储配置
    server: ServerConfig = Field(default_factory=ServerConfig)                  # 服务配置
    logging: LoggingConfig = Field(default_factory=LoggingConfig)               # 日志配置


# ============================================================
# 配置加载函数
# ============================================================

def load_config(config_path: str = "config.yaml") -> Settings:
    """
    从YAML文件加载配置，并与环境变量合并

    Args:
        config_path: 配置文件路径，默认项目根目录下的 config.yaml

    Returns:
        Settings: 全局配置对象
    """
    # 确定配置文件的绝对路径（基于本文件所在目录）
    base_dir = Path(__file__).parent  # 获取当前文件所在目录的绝对路径
    full_path = base_dir / config_path  # 拼接配置文件的完整路径

    config_dict = {}  # 初始化一个空字典用于存储配置信息

    # 如果配置文件存在则读取
    if full_path.exists():  # 检查配置文件是否存在
        with open(full_path, "r", encoding="utf-8") as f:  # 以UTF-8编码打开文件
            config_dict = yaml.safe_load(f) or {}  # 加载YAML内容，如果为空则使用空字典
    else:
        print(f"[警告] 配置文件 {full_path} 不存在，使用默认配置")  # 文件不存在时打印警告

    # 创建Settings对象（Pydantic会自动做类型校验和默认值填充）
    settings = Settings(**config_dict)  # 使用配置字典创建Settings实例

    # ---------- 环境变量覆盖 ----------
    # 优先使用环境变量中的 OPENAI_API_KEY（如果配置文件中为空）
    env_api_key = os.getenv("OPENAI_API_KEY", "")  # 从环境变量获取API密钥
    if env_api_key and not settings.model.llm_api_key:  # 如果环境变量中有值且配置文件中为空
        settings.model.llm_api_key = env_api_key  # 使用环境变量的值覆盖配置

    # 优先使用环境变量中的 LLM_BASE_URL
    env_base_url = os.getenv("LLM_BASE_URL", "")  # 从环境变量获取基础URL
    if env_base_url:  # 如果环境变量中有值
        settings.model.llm_base_url = env_base_url  # 使用环境变量的值覆盖配置

    # ---------- 自动创建必要目录 ----------
    _ensure_directories(settings, base_dir)  # 确保必要的目录存在

    return settings  # 返回配置对象


def _ensure_directories(settings: Settings, base_dir: Path) -> None:
    """
    自动创建项目运行所需的目录

    该函数根据配置对象和项目根路径，创建必要的目录结构，确保程序运行时所有需要的目录都已存在。
    使用了exist_ok=True参数，确保即使目录已存在也不会抛出异常。
    Args:
        settings: 配置对象，包含项目运行所需的各种配置信息
        base_dir: 项目根目录的Path对象，作为其他目录的基准路径
    """
    # 需要创建的目录列表（将相对路径转为基于项目根目录的绝对路径）
    dirs_to_create = [
        base_dir / settings.vector_store.persist_directory,  # 向量库持久化目录，用于存储向量数据库数据
        base_dir / settings.storage.upload_dir,              # 上传文档目录，用于存储用户上传的文档
        base_dir / Path(settings.logging.file).parent,       # 日志目录，存储程序运行日志
        base_dir / "data",                                   # 数据目录，用于存储程序运行时产生的数据
        base_dir / "evaluations",                            # 评估结果目录，存储模型评估相关结果
    ]

    # 遍历所有需要创建的目录路径
    for dir_path in dirs_to_create:
        # 创建目录，包括所有必要的父目录
        # parents=True表示创建所有父目录（如果不存在）
        # exist_ok=True表示如果目录已存在则不抛出异常
        dir_path.mkdir(parents=True, exist_ok=True)


# ============================================================
# 全局单例：项目启动时加载一次，其他模块直接 import 使用
# ============================================================
settings = load_config()


# 模块导出测试
if __name__ == "__main__":
    print("=" * 60)
    print("RAG系统配置加载成功！")
    print("=" * 60)
    print(f"Embedding模型: {settings.model.embedding_model}")
    print(f"Reranker模型: {settings.model.reranker_model}")
    print(f"LLM模型: {settings.model.llm_model}")
    print(f"LLM API Key: {'已配置' if settings.model.llm_api_key else '【未配置 - 请在config.yaml或.env中填写】'}")
    print(f"推理设备: {settings.model.device}")
    print(f"向量库类型: {settings.vector_store.type}")
    print(f"Chunk大小: {settings.chunking.chunk_size}")
    print(f"服务端口: {settings.server.port}")
    print("=" * 60)
