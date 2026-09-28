# -*- coding: utf-8 -*-
"""
text_splitter.py - 文本切分模块
=============================================
功能：
  1. 递归字符切分（RecursiveCharacterTextSplitter）
     分隔符优先级：\n\n > \n > 。！？； > ， > 空格
  2. 支持父子块策略：
     - 子块（小块，默认200字）用于检索，提高召回精度
     - 父块（大块，默认1000字）用于生成，提供更完整上下文
  3. 每个chunk添加唯一ID、位置索引、来源文件名、页码等元数据
  4. 保留原始Document的元数据并追加切分相关字段

切分后元数据结构：
  metadata = {
      "source": str,           # 文件路径
      "filename": str,         # 文件名
      "page": int,             # 页码
      "file_type": str,        # 文件类型
      "upload_time": str,      # 上传时间
      "chunk_id": str,         # 块唯一ID（格式: filename_page_index）
      "chunk_index": int,      # 块在文档中的索引
      "parent_id": str,        # 父块ID（父子块策略时使用）
      "is_parent": bool,       # 是否为父块
      "total_chunks": int      # 文档总块数
  }
"""

import hashlib
from typing import List, Optional, Dict, Any, Union, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from config import settings


# ============================================================
# 中文友好的递归字符切分器
# ============================================================

class ChineseTextSplitter:
    """
    中文优化的递归字符切分器
    在LangChain RecursiveCharacterTextSplitter基础上，
    添加中文标点符号作为分隔符，提高中文文本切分质量
    """

    # 分隔符优先级列表（从高到低）
    # 优先在段落、句子边界切分，避免切断完整语义
    SEPARATORS = [
        "\n\n",           # 段落分隔（最高优先级）
        "\n",             # 换行
        "。",             # 中文句号
        "！",             # 中文感叹号
        "？",             # 中文问号
        "；",             # 中文分号
        ". ",             # 英文句号+空格
        "! ",             # 英文感叹号+空格
        "? ",             # 英文问号+空格
        "，",             # 中文逗号
        ", ",             # 英文逗号+空格
        " ",              # 空格
        "",               # 字符级（最后兜底）
    ]

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        """
        初始化切分器

        Args:
            chunk_size: 每个块的最大字符数，默认从配置读取
            chunk_overlap: 块之间的重叠字符数，默认从配置读取
        """
        self.chunk_size = chunk_size or settings.chunking.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunking.chunk_overlap

        # 创建LangChain的递归字符切分器
        self.splitter = RecursiveCharacterTextSplitter(
            separators=self.SEPARATORS,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,           # 按字符数计算长度
            is_separator_regex=False,      # 分隔符不作为正则表达式
            keep_separator=True,           # 保留分隔符在块末尾
            strip_whitespace=True,         # 去除块首尾空白
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        对文档列表进行切分

        Args:
            documents: 原始文档列表（来自document_loader）

        Returns:
            List[Document]: 切分后的文档块列表，每个块带有完整元数据
        """
        all_chunks = []

        for doc in documents:
            # 对单个文档进行切分
            chunks = self._split_single_document(doc)
            all_chunks.extend(chunks)

        logger.info(f"文本切分完成: {len(documents)} 个文档 -> {len(all_chunks)} 个块")
        return all_chunks

    def _split_single_document(self, document: Document) -> List[Document]:
        """
        切分单个文档，并为每个块添加元数据

        Args:
            document: 原始Document对象

        Returns:
            List[Document]: 切分后的块列表
        """
        # 使用LangChain切分器切分文本
        raw_chunks = self.splitter.split_text(document.page_content)

        chunks = []
        total = len(raw_chunks)

        for idx, text in enumerate(raw_chunks):
            # 跳过空块
            if not text.strip():
                continue

            # 生成唯一chunk_id
            chunk_id = self._generate_chunk_id(document.metadata, idx)

            # 构建元数据（复制原始元数据 + 添加切分字段）
            metadata = dict(document.metadata)
            metadata.update({
                "chunk_id": chunk_id,
                "chunk_index": idx,
                "total_chunks": total,
                "parent_id": chunk_id,   # 普通模式下父块就是自己
                "is_parent": True,
            })

            chunks.append(Document(
                page_content=text,
                metadata=metadata,
            ))

        return chunks

    @staticmethod
    def _generate_chunk_id(metadata: Dict[str, Any], index: int) -> str:
        """
        生成唯一的chunk ID

        格式: {filename}_p{page}_{index}
        如果文件名过长，使用哈希缩短

        Args:
            metadata: 文档元数据
            index: 块索引

        Returns:
            str: 唯一chunk ID
        """
        filename = metadata.get("filename", "unknown")
        page = metadata.get("page", 0)

        # 清理文件名中的特殊字符
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)

        # 如果文件名太长，使用哈希
        if len(safe_name) > 50:
            hash_part = hashlib.md5(filename.encode()).hexdigest()[:8]
            safe_name = f"doc_{hash_part}"

        return f"{safe_name}_p{page}_{index}"


# ============================================================
# 父子块切分器（Parent-Child Chunking）
# ============================================================

class ParentChildTextSplitter:
    """
    父子块策略切分器

    原理：
      - 父块（Parent）：较大的文本块（默认1000字），用于LLM生成时提供完整上下文
      - 子块（Child）：较小的文本块（默认200字），用于向量检索，提高召回精度
      - 检索时用子块匹配，命中后返回对应的父块给LLM

    优势：
      1. 检索更精准（小块语义更单一）
      2. 生成上下文更完整（大块信息更丰富）
      3. 缓解"Lost in the Middle"问题
    """

    def __init__(
        self,
        child_chunk_size: Optional[int] = None,
        parent_chunk_size: Optional[int] = None,
        child_chunk_overlap: Optional[int] = None,
    ):
        """
        初始化父子块切分器

        Args:
            child_chunk_size: 子块大小（用于检索），默认从配置读取
            parent_chunk_size: 父块大小（用于生成），默认从配置读取
            child_chunk_overlap: 子块重叠大小
        """
        self.child_size = child_chunk_size or settings.chunking.child_chunk_size
        self.parent_size = parent_chunk_size or settings.chunking.parent_chunk_size
        self.child_overlap = child_chunk_overlap or 50

        # 子块切分器（小块，用于检索）
        self.child_splitter = RecursiveCharacterTextSplitter(
            separators=ChineseTextSplitter.SEPARATORS,
            chunk_size=self.child_size,
            chunk_overlap=self.child_overlap,
            length_function=len,
            keep_separator=True,
            strip_whitespace=True,
        )

        # 父块切分器（大块，用于生成）
        self.parent_splitter = RecursiveCharacterTextSplitter(
            separators=ChineseTextSplitter.SEPARATORS,
            chunk_size=self.parent_size,
            chunk_overlap=self.parent_size // 5,  # 父块重叠20%
            length_function=len,
            keep_separator=True,
            strip_whitespace=True,
        )

    def split_documents(self, documents: List[Document]) -> Tuple[List[Document], List[Document]]:
        """
        对文档列表进行父子块切分

        Args:
            documents: 原始文档列表

        Returns:
            tuple: (父块列表, 子块列表)
            - 父块：用于生成，metadata中 is_parent=True
            - 子块：用于检索，metadata中 is_parent=False，parent_id 指向对应父块
        """
        all_parent_chunks = []
        all_child_chunks = []

        for doc in documents:
            parents, children = self._split_single_document(doc)
            all_parent_chunks.extend(parents)
            all_child_chunks.extend(children)

        logger.info(
            f"父子块切分完成: {len(documents)} 个文档 -> "
            f"{len(all_parent_chunks)} 个父块, {len(all_child_chunks)} 个子块"
        )
        return all_parent_chunks, all_child_chunks

    def _split_single_document(self, document: Document) -> Tuple[List[Document], List[Document]]:
        """
        切分单个文档为父块和子块

        Args:
            document: 原始Document

        Returns:
            tuple: (父块列表, 子块列表)
        """
        text = document.page_content

        # 第一步：切分父块
        parent_texts = self.parent_splitter.split_text(text)
        parent_chunks = []
        child_chunks = []

        for p_idx, p_text in enumerate(parent_texts):
            if not p_text.strip():
                continue

            # 生成父块ID
            parent_id = ChineseTextSplitter._generate_chunk_id(document.metadata, p_idx)
            parent_id = f"{parent_id}_parent"

            # 创建父块Document
            parent_meta = dict(document.metadata)
            parent_meta.update({
                "chunk_id": parent_id,
                "chunk_index": p_idx,
                "total_chunks": len(parent_texts),
                "parent_id": parent_id,
                "is_parent": True,
            })
            parent_doc = Document(page_content=p_text, metadata=parent_meta)
            parent_chunks.append(parent_doc)

            # 第二步：在每个父块内切分子块
            child_texts = self.child_splitter.split_text(p_text)
            for c_idx, c_text in enumerate(child_texts):
                if not c_text.strip():
                    continue

                child_id = f"{parent_id}_child_{c_idx}"
                child_meta = dict(document.metadata)
                child_meta.update({
                    "chunk_id": child_id,
                    "chunk_index": c_idx,
                    "parent_id": parent_id,       # 指向父块
                    "is_parent": False,
                    "parent_chunk_index": p_idx,
                })
                child_doc = Document(page_content=c_text, metadata=child_meta)
                child_chunks.append(child_doc)

        return parent_chunks, child_chunks


# ============================================================
# 统一切分入口函数
# ============================================================

def split_documents(
    documents: List[Document],
    use_parent_child: Optional[bool] = None,
) -> Union[List[Document], Tuple[List[Document], List[Document]]]:
    """
    统一文本切分入口函数

    Args:
        documents: 原始文档列表
        use_parent_child: 是否使用父子块策略，None则从配置读取

    Returns:
        如果 use_parent_child=False: 返回普通块列表 List[Document]
        如果 use_parent_child=True: 返回 (父块列表, 子块列表)
    """
    if use_parent_child is None:
        use_parent_child = settings.chunking.enable_parent_child

    if use_parent_child:
        splitter = ParentChildTextSplitter()
        return splitter.split_documents(documents)
    else:
        splitter = ChineseTextSplitter()
        return splitter.split_documents(documents)


# ============================================================
# 模块自测
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("文本切分模块自测")
    print("=" * 60)

    # 构造测试文档
    test_text = """
    第一章 系统概述

    本系统是一个基于RAG技术的智能文档问答系统。RAG全称检索增强生成（Retrieval-Augmented Generation），
    是一种结合了信息检索和大语言模型的技术架构。它通过从外部知识库中检索相关文档，
    然后将检索到的内容作为上下文提供给大模型，从而生成更加准确、有依据的回答。

    系统的主要特点包括：
    1. 支持多种文档格式的解析，包括PDF、Word、TXT、HTML等。
    2. 采用向量检索和BM25关键词检索相结合的混合召回策略。
    3. 使用Reranker模型对检索结果进行精排，提高相关性。
    4. 支持流式输出，提供更好的用户体验。

    第二章 技术架构

    系统采用模块化设计，主要包括文档加载、文本切分、向量化、检索引擎、问答生成等模块。
    每个模块都有明确的接口定义，可以独立替换和升级。向量数据库默认使用Chroma，
    生产环境可以切换到Milvus以获得更好的性能和扩展性。
    """

    test_doc = Document(
        page_content=test_text,
        metadata={
            "source": "/test/example.txt",
            "filename": "example.txt",
            "page": 1,
            "file_type": ".txt",
            "upload_time": "2026-08-17 10:00:00",
        }
    )

    # 测试普通切分
    print("\n【普通切分模式】")
    splitter = ChineseTextSplitter(chunk_size=150, chunk_overlap=30)
    chunks = splitter.split_documents([test_doc])
    print(f"切分结果: {len(chunks)} 个块")
    for i, chunk in enumerate(chunks):
        print(f"\n--- 块 {i+1} (ID: {chunk.metadata['chunk_id']}) ---")
        print(f"内容: {chunk.page_content[:80]}...")

    # 测试父子块切分
    print("\n\n【父子块切分模式】")
    pc_splitter = ParentChildTextSplitter(
        child_chunk_size=80,
        parent_chunk_size=200,
    )
    parents, children = pc_splitter.split_documents([test_doc])
    print(f"父块: {len(parents)} 个, 子块: {len(children)} 个")
    for i, parent in enumerate(parents):
        print(f"\n父块 {i+1}: {parent.page_content[:60]}...")
        # 找到对应的子块
        related_children = [c for c in children if c.metadata["parent_id"] == parent.metadata["chunk_id"]]
        print(f"  对应子块数: {len(related_children)}")

    print("\n" + "=" * 60)
