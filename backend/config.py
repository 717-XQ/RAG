# -*- coding: utf-8 -*-
"""
config.py - 配置管理模块
=============================================
功能：
  1. 从 config.yaml 加载系统配置
  2. 支持环境变量覆盖（如 OPENAI_API_KEY、DASHSCOPE_API_KEY、JWT_SECRET、DATABASE_URL）
  3. 提供全局单例配置对象，供其他模块引用
  4. 自动创建必要的目录（上传目录、日志目录、向量库目录）

已按《技术栈规范》2.1 对齐扩展：数据库/认证/导出/FTS5/多Provider

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

class LLMProviderConfig(BaseModel):
    """单个LLM Provider配置（技术栈2.1：多Provider自动降级）"""
    name: str = "deepseek"                          # Provider名称
    model: str = "deepseek-chat"                    # 模型名称
    api_key: str = ""                               # API密钥（建议环境变量）
    base_url: str = "https://api.deepseek.com/v1"   # API Base URL
    priority: int = 1                               # 优先级（数字小优先）


class ModelConfig(BaseModel):
    """模型相关配置"""
    embedding_model: str = "BAAI/bge-large-zh-v1.5"    # 嵌入模型，用于文本向量化
    reranker_model: str = "BAAI/bge-reranker-v2-m3"    # 重排序模型，用于优化检索结果
    llm_model: str = "gpt-3.5-turbo"                   # 默认大语言模型（兼容旧配置）
    llm_api_key: str = ""                              # 默认LLM API密钥
    llm_base_url: str = "https://api.openai.com/v1"    # 默认LLM API基础URL
    llm_temperature: float = 0.1                       # LLM温度参数，控制生成随机性
    llm_max_tokens: int = 1024                         # LLM最大生成token数
    device: str = "cuda"                               # 推理设备：cuda / cpu
    providers: List[LLMProviderConfig] = Field(default_factory=list)  # 多Provider列表


class RetrievalConfig(BaseModel):
    """检索相关配置"""
    vector_top_k: int = 20                            # 向量检索TopK
    bm25_top_k: int = 20                              # BM25/FTS5检索TopK
    keyword_retriever: str = "fts5"                   # 关键词检索实现: bm25 / fts5（技术栈2.1）
    reranker_top_k: int = 5                           # 重排序TopK
    vector_weight: float = 0.5                        # 向量权重
    bm25_weight: float = 0.5                          # 关键词权重


class FTS5Config(BaseModel):
    """SQLite FTS5 全文检索配置（技术栈2.1）"""
    db_path: str = "./data/fts5.db"                   # FTS5索引数据库路径
    tokenizer: str = "trigram"                        # 分词器: trigram（中文推荐）/ unicode61


class ChunkingConfig(BaseModel):
    """文本切分配置"""
    chunk_size: int = 500                             # 文本切分大小
    chunk_overlap: int = 100                          # 文本切分重叠
    enable_parent_child: bool = False                 # 是否启用父子关系
    child_chunk_size: int = 200                       # 子块大小
    parent_chunk_size: int = 1000                     # 父块大小


class VectorStoreConfig(BaseModel):
    """向量数据库配置"""
    type: str = "chroma"                              # 向量数据库类型: chroma / milvus
    persist_directory: str = "./chroma_db"            # 向量数据库持久化目录
    collection_name: str = "rag_docs"                 # 向量数据库集合名
    milvus_host: str = "localhost"                    # Milvus主机
    milvus_port: str = "19530"                        # Milvus端口


class DatabaseConfig(BaseModel):
    """关系数据库配置（技术栈2.1：MySQL/PostgreSQL + SQLAlchemy + Alembic）"""
    url: str = "mysql+pymysql://root:CHANGE_ME@127.0.0.1:3306/rag_kb?charset=utf8mb4"
    echo: bool = False                                # SQL日志
    fallback_sqlite: bool = True                      # 连接失败回退SQLite
    sqlite_path: str = "./data/rag.db"                # 回退SQLite路径


class AuthConfig(BaseModel):
    """JWT认证配置（技术栈2.1：python-jose + bcrypt）"""
    enabled: bool = True                              # 是否启用认证
    jwt_secret: str = "rag-system-change-me"          # JWT密钥（生产务必更换）
    jwt_algorithm: str = "HS256"                      # JWT算法
    access_token_expire_minutes: int = 30             # 访问Token有效期（分钟）
    refresh_token_expire_days: int = 7                # 刷新Token有效期（天）


class ExportConfig(BaseModel):
    """文档导出配置（技术栈2.1：WeasyPrint PDF / python-docx Word）"""
    report_title: str = "RAG 智能问答报告"            # 报告标题
    pdf_css_font: str = "Microsoft YaHei"             # PDF中文字体


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
    fts5: FTS5Config = Field(default_factory=FTS5Config)                        # FTS5全文检索配置
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)            # 文本切分配置
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)  # 向量数据库配置
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)            # 关系数据库配置
    auth: AuthConfig = Field(default_factory=AuthConfig)                        # 认证配置
    export: ExportConfig = Field(default_factory=ExportConfig)                  # 导出配置
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
    # 优先使用环境变量中的 OPENAI_API_KEY（兼容旧逻辑）
    env_api_key = os.getenv("OPENAI_API_KEY", "")  # 从环境变量获取API密钥
    if env_api_key and not settings.model.llm_api_key:  # 如果环境变量中有值且配置文件中为空
        settings.model.llm_api_key = env_api_key  # 使用环境变量的值覆盖配置

    # 优先使用环境变量中的 LLM_BASE_URL
    env_base_url = os.getenv("LLM_BASE_URL", "")  # 从环境变量获取基础URL
    if env_base_url:  # 如果环境变量中有值
        settings.model.llm_base_url = env_base_url  # 使用环境变量的值覆盖配置

    # 多Provider：用环境变量补齐各Provider的API Key（不写死密钥在配置文件中）
    # 规则：DEEPSEEK_API_KEY -> deepseek；DASHSCOPE_API_KEY -> dashscope；通用 OPENAI_API_KEY
    provider_env_map = {
        "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com/v1"),
        "dashscope": ("DASHSCOPE_API_KEY", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "openai": ("OPENAI_API_KEY", "https://api.openai.com/v1"),
        "moonshot": ("MOONSHOT_API_KEY", "https://api.moonshot.cn/v1"),
    }
    for p in settings.model.providers:
        key_env, default_base = provider_env_map.get(p.name, ("", ""))
        if key_env:
            env_val = os.getenv(key_env, "")
            if env_val and not p.api_key:
                p.api_key = env_val
        if not p.base_url and default_base:
            p.base_url = default_base
    # 若未配置任何providers，用旧字段兜底生成一个
    if not settings.model.providers:
        settings.model.providers = [LLMProviderConfig(
            name="default",
            model=settings.model.llm_model,
            api_key=settings.model.llm_api_key,
            base_url=settings.model.llm_base_url,
            priority=1,
        )]

    # JWT_SECRET / DATABASE_URL 环境变量覆盖（生产安全）
    env_jwt_secret = os.getenv("JWT_SECRET", "")
    if env_jwt_secret:
        settings.auth.jwt_secret = env_jwt_secret
    env_db_url = os.getenv("DATABASE_URL", "")
    if env_db_url:
        settings.database.url = env_db_url

    # ---------- 路径绝对化（后端已整合到 backend/ 目录） ----------
    # 将配置中的相对路径统一解析为基于本文件（backend/）所在目录的绝对路径，
    # 保证无论从哪个工作目录启动（backend/ 或项目根），数据都写入 backend/ 下。
    base_dir = Path(__file__).parent.resolve()

    def _abs(path: str) -> str:
        p = Path(path)
        return str(p if p.is_absolute() else (base_dir / p))

    settings.vector_store.persist_directory = _abs(settings.vector_store.persist_directory)
    settings.storage.upload_dir = _abs(settings.storage.upload_dir)
    settings.logging.file = _abs(settings.logging.file)
    settings.database.sqlite_path = _abs(settings.database.sqlite_path)
    settings.fts5.db_path = _abs(settings.fts5.db_path)

    # ---------- 自动创建必要目录 ----------
    _ensure_directories(settings, base_dir)  # 确保必要的目录存在

    return settings  # 返回配置对象


def _ensure_directories(settings: Settings, base_dir: Path) -> None:
    """
    自动创建项目运行所需的目录

    Args:
        settings: 配置对象，包含项目运行所需的各种配置信息
        base_dir: 项目根目录的Path对象，作为其他目录的基准路径
    """
    # 需要创建的目录列表（将相对路径转为基于项目根目录的绝对路径）
    dirs_to_create = [
        base_dir / settings.vector_store.persist_directory,  # 向量库持久化目录
        base_dir / settings.storage.upload_dir,              # 上传文档目录
        base_dir / Path(settings.logging.file).parent,       # 日志目录
        base_dir / "data",                                   # 数据目录（FTS5/回退SQLite）
        base_dir / "evaluations",                            # 评估结果目录
    ]

    # 遍历所有需要创建的目录路径
    for dir_path in dirs_to_create:
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
    print(f"关键词检索: {settings.retrieval.keyword_retriever}")
    print(f"数据库: {settings.database.url}")
    print(f"认证: {'启用' if settings.auth.enabled else '禁用'}")
    print(f"Provider数量: {len(settings.model.providers)}")
    for p in settings.model.providers:
        print(f"  - {p.name}: {p.model} @ {p.base_url} (priority={p.priority}, key={'已配置' if p.api_key else '未配置'})")
    print(f"服务端口: {settings.server.port}")
    print("=" * 60)
