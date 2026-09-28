# -*- coding: utf-8 -*-
"""
document_loader.py - 文档加载与解析模块
=============================================
功能：
  1. 支持 PDF / DOCX / TXT / HTML 四种格式文档的解析
  2. 自动识别文件格式并选择对应解析器
  3. 保留页码、文件名等元数据
  4. 清洗多余空白、特殊字符，统一编码
  5. 支持单文件加载和批量目录加载
  6. 返回标准 Document 对象列表（与LangChain兼容）

数据结构：
  Document(
      page_content: str,          # 文本内容
      metadata: {
          "source": str,          # 文件路径
          "filename": str,        # 文件名
          "page": int,            # 页码（PDF有，其他默认0）
          "file_type": str,       # 文件类型
          "upload_time": str      # 上传时间
      }
  )
"""

import os
import re
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from langchain_core.documents import Document
from loguru import logger

from config import settings


# ============================================================
# 文档加载器基类（定义统一接口）
# ============================================================

class BaseDocumentLoader:
    """文档加载器基类，所有具体加载器继承此类"""

    # 这是文档加载器的基类，提供了一个基本的框架结构
    # 具体的文档加载器需要继承这个类并实现其方法
    def load(self, file_path: str) -> List[Document]:
        """
        加载文档并返回Document列表

        Args:
            file_path: 文档文件路径

                # 这是一个字符串参数，表示要加载的文档的完整路径
        Returns:
            List[Document]: 解析后的文档块列表（每页/每段一个Document）
                # 返回一个Document对象的列表，每个Document代表文档的一个块（可以是页或段）
        """
        raise NotImplementedError("子类必须实现load方法")

        # 这是一个抽象方法，强制子类必须实现自己的文档加载逻辑
        # 如果子类没有实现此方法，调用时会抛出NotImplementedError异常

    def _make_metadata(self, file_path: str, page: int = 0) -> Dict[str, Any]:
        """
        构建标准元数据字典

        Args:
            file_path: 文件路径
                # 字符串参数，表示文档的完整路径
            page: 页码

                # 整数参数，表示文档的页码，默认值为0
        Returns:
            元数据字典
                # 返回一个包含文档元数据的字典，包括文件路径、文件名、页码等信息
        """
        return {
            "source": str(file_path),  # 文档的完整路径，转换为字符串类型
            "filename": os.path.basename(file_path),  # 文档的文件名
            "page": page,  # 文档的页码
            "file_type": Path(file_path).suffix.lower(),  # 文件扩展名，转换为小写
            "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 当前时间，格式为年-月-日 时:分:秒
        }


# ============================================================
# PDF 文档加载器（使用 PyMuPDF）
# ============================================================

class PDFLoader(BaseDocumentLoader):
    """
    PDF文档加载器
    使用 PyMuPDF (fitz) 解析，支持按页提取文本
    """

    def load(self, file_path: str) -> List[Document]:
        """
        加载PDF文档，每页生成一个Document

        Args:
            file_path: PDF文件路径

        Returns:
            List[Document]: 每页一个Document对象
        """
        try:
            import fitz  # PyMuPDF  # 导入PyMuPDF库，用于PDF文档处理
        except ImportError:
            raise ImportError("请安装 PyMuPDF: pip install PyMuPDF")  # 如果未安装PyMuPDF，抛出导入错误

        documents = []  # 初始化空列表，用于存储处理后的文档
        logger.info(f"开始解析PDF: {file_path}")  # 记录开始解析PDF的日志

        try:
            doc = fitz.open(file_path)  # 打开PDF文件
            total_pages = len(doc)  # 获取PDF总页数

            for page_num in range(total_pages):  # 遍历每一页
                page = doc[page_num]  # 获取当前页
                # 提取页面文本
                text = page.get_text("text")  # 从当前页提取文本内容
                # 清洗文本
                text = self._clean_text(text)  # 调用文本清洗方法处理提取的文本

                if text.strip():  # 跳过空白页
                    documents.append(Document(  # 将处理后的文本添加到文档列表
                        page_content=text,
                        metadata=self._make_metadata(file_path, page=page_num + 1)  # 创建文档元数据
                    ))

            doc.close()  # 关闭PDF文档
            logger.info(f"PDF解析完成: {file_path}, 共 {total_pages} 页, 有效页 {len(documents)}")  # 记录解析完成的日志

        except Exception as e:  # 捕获并处理可能出现的异常
            logger.error(f"PDF解析失败 {file_path}: {str(e)}")  # 记录错误日志
            raise  # 重新抛出异常

        return documents  # 返回处理后的文档列表

    @staticmethod
    def _clean_text(text: str) -> str:
        """清洗PDF提取的文本：去除多余空白、页眉页脚特征等"""
        # 替换多种空白字符为普通空格
        text = re.sub(r'[\t\r\f\v]+', ' ', text)  # 将制表符、回车、换页、垂直制表符等替换为空格
        # 多个连续空行合并为一个
        text = re.sub(r'\n{3,}', '\n\n', text)  # 将3个或更多连续换行符替换为2个换行符
        # 行尾多余空格去除
        text = re.sub(r'[ \t]+\n', '\n', text)  # 去除行尾的空格和制表符
        # 去除零宽字符和BOM
        text = text.replace('\ufeff', '').replace('\u200b', '')  # 移除零宽字符和BOM标记
        return text.strip()  # 返回处理后的文本，去除首尾空白字符


# ============================================================
# Word (DOCX) 文档加载器（使用 python-docx）
# ============================================================

class DOCXLoader(BaseDocumentLoader):
    """
    Word文档加载器
    使用 python-docx 解析，按段落提取文本
    """

    def load(self, file_path: str) -> List[Document]:
        """
        加载DOCX文档，整个文档作为一个Document（页码统一为0）

        Args:
            file_path: DOCX文件路径

        Returns:
            List[Document]: 包含一个Document的列表
        """
        try:
            # 动态导入 python-docx 库
            from docx import Document as DocxDocument
        except ImportError:
            # 如果导入失败，抛出异常提示用户安装
            raise ImportError("请安装 python-docx: pip install python-docx")

        # 记录开始解析文档的日志
        logger.info(f"开始解析Word: {file_path}")
        # 用于存储文档所有文本的列表
        full_text = []

        try:
            # 使用python-docx加载文档
            doc = DocxDocument(file_path)

            # 遍历所有段落
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    full_text.append(text)

            # 遍历所有表格（提取表格内容）
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        cell_text = cell.text.strip()
                        if cell_text:
                            row_text.append(cell_text)
                    if row_text:
                        full_text.append(" | ".join(row_text))

            # 合并所有文本
            combined_text = "\n\n".join(full_text)
            combined_text = self._clean_text(combined_text)

            if not combined_text:
                logger.warning(f"Word文档内容为空: {file_path}")
                return []

            document = Document(
                page_content=combined_text,
                metadata=self._make_metadata(file_path, page=0)
            )
            logger.info(f"Word解析完成: {file_path}, 字符数: {len(combined_text)}")
            return [document]

        except Exception as e:
            logger.error(f"Word解析失败 {file_path}: {str(e)}")
            raise

    @staticmethod
    def _clean_text(text: str) -> str:
        """清洗Word文本"""
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.replace('\ufeff', '')
        return text.strip()


# ============================================================
# TXT 纯文本加载器
# ============================================================

class TXTLoader(BaseDocumentLoader):
    """
    纯文本加载器
    自动检测编码（UTF-8 / GBK / GB2312）
    """

    # 尝试的编码列表（按优先级）
    ENCODINGS = ["utf-8", "utf-8-sig", "gbk", "gb2312", "gb18030", "latin-1"]

    def load(self, file_path: str) -> List[Document]:
        """
        加载TXT文件，自动检测编码

        Args:
            file_path: TXT文件路径

        Returns:
            List[Document]: 包含一个Document的列表
        """
        logger.info(f"开始解析TXT: {file_path}")

        text = None
        used_encoding = None

        # 依次尝试不同编码
        for encoding in self.ENCODINGS:
            try:
                # 尝试用当前编码打开文件
                with open(file_path, "r", encoding=encoding) as f:
                    text = f.read()
                # 记录成功使用的编码
                used_encoding = encoding
                # 成功读取后跳出循环
                break
            except (UnicodeDecodeError, UnicodeError):
                # 如果当前编码失败，继续尝试下一种编码
                continue

        # 如果所有编码都尝试失败，抛出异常
        if text is None:
            raise ValueError(f"无法识别文件编码: {file_path}")

        # 清洗文本内容
        text = self._clean_text(text)

        # 如果清洗后文本为空，返回空列表
        if not text:
            logger.warning(f"TXT文件内容为空: {file_path}")
            return []

        # 创建并返回Document对象
        document = Document(
            page_content=text,
            metadata=self._make_metadata(file_path, page=0)
        )
        logger.info(f"TXT解析完成: {file_path}, 编码: {used_encoding}, 字符数: {len(text)}")
        return [document]

    @staticmethod
    def _clean_text(text: str) -> str:
        """清洗TXT文本"""
        # 统一换行符为\n
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # 去除BOM（字节顺序标记）字符
        text = text.replace('\ufeff', '')
        # 合并多余空行，保留最多两个连续换行符
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 去除首尾空白字符
        return text.strip()


# ============================================================
# HTML 文档加载器（使用 BeautifulSoup）
# ============================================================

class HTMLLoader(BaseDocumentLoader):
    """
    HTML文档加载器
    使用 BeautifulSoup 提取正文文本，去除标签、脚本、样式
    """

    def load(self, file_path: str) -> List[Document]:
        """
        加载HTML文件，提取纯文本内容

        Args:
            file_path: HTML文件路径

        Returns:
            List[Document]: 包含一个Document的列表
        """
        try:
            # 尝试导入BeautifulSoup库
            from bs4 import BeautifulSoup
        except ImportError:
            # 如果导入失败，提示用户安装
            raise ImportError("请安装 beautifulsoup4: pip install beautifulsoup4")

        # 记录开始解析HTML文件的信息
        logger.info(f"开始解析HTML: {file_path}")

        # 读取HTML文件（尝试多种编码）
        html_content = None
        # 尝试使用不同编码读取文件
        for encoding in ["utf-8", "gbk", "gb2312"]:
            try:
                # 以指定编码打开文件并读取内容
                with open(file_path, "r", encoding=encoding) as f:
                    html_content = f.read()
                break  # 成功读取则退出循环
            except (UnicodeDecodeError, UnicodeError):
                # 如果当前编码失败，继续尝试下一种编码
                continue

        # 如果所有编码尝试都失败，抛出异常
        if html_content is None:
            raise ValueError(f"无法识别HTML文件编码: {file_path}")

        # 使用BeautifulSoup解析HTML内容
        soup = BeautifulSoup(html_content, "lxml")

        # 移除不需要的标签（脚本、样式等）
        for tag in soup(["script", "style", "noscript", "meta", "link", "head", "nav", "footer"]):
            tag.decompose()

        # 提取标题
        title = ""
        # 检查是否存在title标签及其内容
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        # 提取正文文本并清理
        text = soup.get_text(separator="\n", strip=True)
        text = self._clean_text(text)

        # 如果有标题，放在最前面
        if title and not text.startswith(title):
            text = f"{title}\n\n{text}"

        if not text:
            logger.warning(f"HTML文件内容为空: {file_path}")
            return []

        document = Document(
            page_content=text,
            metadata=self._make_metadata(file_path, page=0)
        )
        logger.info(f"HTML解析完成: {file_path}, 字符数: {len(text)}")
        return [document]

    @staticmethod
    def _clean_text(text: str) -> str:
        """清洗HTML提取的文本"""
        # 合并多余空行和空格
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        # 去除HTML特殊字符
        text = text.replace('&nbsp;', ' ').replace('&amp;', '&')
        text = text.replace('&lt;', '<').replace('&gt;', '>')
        return text.strip()


# ============================================================
# 文档加载器工厂（统一入口，自动选择加载器）
# ============================================================

class DocumentLoaderFactory:
    """
    文档加载器工厂类
    根据文件扩展名自动选择对应的加载器
    """

    # 扩展名 -> 加载器类 的映射
    LOADER_MAP = {
        ".pdf": PDFLoader,  # PDF文件加载器
        ".docx": DOCXLoader,  # Word文档加载器
        ".doc": DOCXLoader,   # 注意：.doc 旧格式可能不完全支持
        ".txt": TXTLoader,  # 纯文本文件加载器
        ".html": HTMLLoader,  # HTML文件加载器
        ".htm": HTMLLoader,  # HTML文件加载器(短扩展名)
    }

    @classmethod
    def get_loader(cls, file_path: str) -> BaseDocumentLoader:
        """
        根据文件路径获取对应的加载器实例

        Args:
            file_path: 文件路径，用于确定文件类型并返回相应的加载器实例

        Returns:
            BaseDocumentLoader: 对应的加载器实例，可用于加载指定类型的文档

        Raises:
            ValueError: 不支持的文件格式
        """
        # 获取文件扩展名并转换为小写
        ext = Path(file_path).suffix.lower()
        # 从映射字典中获取对应的加载器类
        loader_class = cls.LOADER_MAP.get(ext)

        # 如果没有找到对应的加载器类
        if loader_class is None:
            # 获取所有支持的文件扩展名列表
            supported = ", ".join(cls.LOADER_MAP.keys())
            # 抛出不支持的文件格式异常
            raise ValueError(f"不支持的文件格式: {ext}，支持的格式: {supported}")

        # 返回加载器实例
        return loader_class()

    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        """检查文件格式是否受支持"""
        # 获取文件扩展名并转换为小写
        ext = Path(file_path).suffix.lower()
        # 检查扩展名是否在支持的映射字典中
        return ext in cls.LOADER_MAP


# ============================================================
# 统一文档加载入口（对外主要使用这个函数）
# ============================================================

def load_document(file_path: str) -> List[Document]:
    """
    加载单个文档（统一入口函数）

    这是一个统一的文档加载函数，它根据文件路径自动选择合适的加载器来处理不同类型的文档。
    Args:
        file_path: 文档文件路径，可以是PDF、Word、TXT等格式的文件路径

    Returns:
        List[Document]: 解析后的文档列表，每个Document对象包含文档内容和元数据

    Example:
        >>> docs = load_document("example.pdf")
        >>> print(f"加载了 {len(docs)} 个文档块")
    Raises:
        FileNotFoundError: 当指定的文件路径不存在时抛出此异常
    """
    # 检查文件是否存在，如果不存在则抛出异常
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # 根据文件类型获取合适的加载器
    loader = DocumentLoaderFactory.get_loader(file_path)
    # 使用加载器加载文档内容
    return loader.load(file_path)


def load_directory(dir_path: str, recursive: bool = True) -> List[Document]:
    """
    批量加载目录下所有支持的文档

    Args:
        dir_path: 目录路径
        recursive: 是否递归遍历子目录

    Returns:
        List[Document]: 所有文档解析后的Document列表

    Example:
        >>> docs = load_directory("./data/docs")
        >>> print(f"共加载 {len(docs)} 个文档块")
    """
    # 检查目录是否存在，如果不存在则抛出异常
    if not os.path.isdir(dir_path):
        raise NotADirectoryError(f"目录不存在: {dir_path}")

    # 用于存储所有加载的文档
    all_documents = []
    # 获取所有支持的文件扩展名
    supported_exts = set(DocumentLoaderFactory.LOADER_MAP.keys())

    # 遍历目录，根据recursive参数决定是否递归遍历子目录
    glob_pattern = "**/*" if recursive else "*"
    for file_path in Path(dir_path).glob(glob_pattern):
        # 确保是文件且扩展名在支持的扩展名列表中
        if file_path.is_file() and file_path.suffix.lower() in supported_exts:
            try:
                # 加载单个文档并添加到文档列表中
                docs = load_document(str(file_path))
                all_documents.extend(docs)
            except Exception as e:
                # 记录加载失败的文件
                logger.error(f"加载文件失败 {file_path}: {str(e)}")
                continue

    # 记录加载完成的文档数量
    logger.info(f"目录加载完成: {dir_path}, 共 {len(all_documents)} 个文档块")
    return all_documents


# ============================================================
# 模块自测
# ============================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("文档加载器模块自测")
    print("=" * 60)

    # 显示支持的格式
    print(f"支持的文件格式: {list(DocumentLoaderFactory.LOADER_MAP.keys())}")

    # 如果有命令行参数，尝试加载指定文件
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        if os.path.isfile(test_path):
            docs = load_document(test_path)
            print(f"\n加载结果: {len(docs)} 个文档块")
            for i, doc in enumerate(docs[:3]):  # 只显示前3个
                print(f"\n--- 文档块 {i+1} ---")
                print(f"元数据: {doc.metadata}")
                print(f"内容预览: {doc.page_content[:200]}...")
        elif os.path.isdir(test_path):
            docs = load_directory(test_path)
            print(f"\n目录加载结果: {len(docs)} 个文档块")
    else:
        # ==================================================
        # 【需手动填写】将下方路径替换为你自己的测试文档路径
        # ==================================================
        test_file = "./data/example.pdf"  # ← 请替换为你的测试文档路径
        print(f"\n提示: 可运行 python document_loader.py <文件路径> 测试加载")
        print(f"示例测试路径（需自行替换）: {test_file}")

    print("=" * 60)
