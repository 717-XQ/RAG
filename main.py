# -*- coding: utf-8 -*-
"""
main.py - 项目主入口（命令行工具）
=============================================
功能：
  1. 提供命令行接口，方便快速操作
  2. 支持子命令：
     - build:   从目录构建知识库
     - chat:    交互式问答（命令行）
     - ask:     单次问答
     - list:    列出知识库文档
     - delete:  删除文档
     - serve:   启动API服务
     - web:     启动Gradio前端（备份版本，推荐使用frontend-vue目录下的Vue3前端）
     - eval:    运行评估
     - init:    初始化项目目录

使用方式：
  python main.py build --dir ./data/docs
  python main.py chat
  python main.py ask "什么是RAG？"
  python main.py list
  python main.py delete example.pdf
  python main.py serve
  python main.py web
  python main.py init
"""

import os
import sys
import argparse
from pathlib import Path

from loguru import logger

from config import settings
from document_loader import load_document, load_directory
from text_splitter import split_documents
from vector_store import get_vector_store, BGEEmbedding
from retriever import HybridRetriever, Reranker
from rag_chain import RAGChain


# ============================================================
# 系统组件管理
# ============================================================

class RAGCLI:
    """RAG命令行工具"""

    def __init__(self):
        self.embedding_model = None
        self.vector_store = None
        self.retriever = None
        self.rag_chain = None

    def _init_components(self, load_rag_chain: bool = True):
        """延迟初始化组件"""
        if self.embedding_model is None:
            logger.info("正在加载模型和组件...")
            self.embedding_model = BGEEmbedding()
            self.vector_store = get_vector_store(embedding_model=self.embedding_model)

            if load_rag_chain:
                self.reranker = Reranker()
                self.retriever = HybridRetriever(
                    vector_store=self.vector_store,
                    embedding_model=self.embedding_model,
                    reranker=self.reranker,
                )
                self.rag_chain = RAGChain(retriever=self.retriever)

            logger.info("组件加载完成！")

    # ---------- 构建知识库 ----------
    def build(self, dir_path: str, recursive: bool = True):
        """
        从目录构建知识库

        Args:
            dir_path: 文档目录路径
            recursive: 是否递归子目录
        """
        self._init_components(load_rag_chain=False)

        if not os.path.isdir(dir_path):
            print(f"❌ 目录不存在: {dir_path}")
            return

        print(f"📂 开始扫描目录: {dir_path}")

        # 加载所有文档
        documents = load_directory(dir_path, recursive=recursive)
        if not documents:
            print("❌ 未找到任何支持的文档")
            return

        print(f"📄 共加载 {len(documents)} 个文档块")

        # 文本切分
        print("✂️  正在进行文本切分...")
        chunks = split_documents(documents)
        if isinstance(chunks, tuple):
            parent_chunks, child_chunks = chunks
            chunks_to_add = parent_chunks
            print(f"   父块: {len(parent_chunks)}, 子块: {len(child_chunks)}")
        else:
            chunks_to_add = chunks

        print(f"   切分完成: {len(chunks_to_add)} 个文本块")

        # 向量化并存储
        print("🔢 正在向量化并存储...")
        self.vector_store.add_documents(chunks_to_add)

        print(f"✅ 知识库构建完成！")
        print(f"   当前向量总数: {self.vector_store.count()}")
        print(f"   文档数量: {len(self.vector_store.list_documents())}")

    # ---------- 添加单个文件 ----------
    def add_file(self, file_path: str):
        """添加单个文件到知识库"""
        self._init_components(load_rag_chain=False)

        if not os.path.isfile(file_path):
            print(f"❌ 文件不存在: {file_path}")
            return

        print(f"📄 加载文件: {file_path}")
        documents = load_document(file_path)

        if not documents:
            print("❌ 文件内容为空")
            return

        chunks = split_documents(documents)
        if isinstance(chunks, tuple):
            chunks_to_add = chunks[0]
        else:
            chunks_to_add = chunks

        self.vector_store.add_documents(chunks_to_add)
        print(f"✅ 添加完成，共 {len(chunks_to_add)} 个文本块")

    # ---------- 交互式问答 ----------
    def chat(self):
        """启动交互式问答"""
        self._init_components(load_rag_chain=True)

        print("=" * 60)
        print("🤖 RAG智能问答系统（交互式）")
        print("=" * 60)
        print("输入问题开始问答，输入 'quit' 或 'exit' 退出")
        print("输入 'sources' 切换显示引用来源")
        print("=" * 60)

        show_sources = True

        while True:
            try:
                question = input("\n👤 你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 再见！")
                break

            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                print("👋 再见！")
                break
            if question.lower() == "sources":
                show_sources = not show_sources
                print(f"📝 引用来源显示: {'开启' if show_sources else '关闭'}")
                continue

            print("🤖 正在思考...")
            result = self.rag_chain.query(question)

            print(f"\n💬 回答: {result.answer}")

            if show_sources and result.sources:
                print("\n📚 引用来源:")
                for i, source in enumerate(result.sources, 1):
                    page_info = f"第{source.page}页" if source.page > 0 else "全文"
                    print(f"  {i}. {source.filename} - {page_info} (相关度: {source.score:.2%})")

            print(f"\n⏱️  检索: {result.retrieve_time:.0f}ms | 生成: {result.generate_time:.0f}ms | 总计: {result.total_time:.0f}ms")

    # ---------- 单次问答 ----------
    def ask(self, question: str):
        """单次问答并输出结果"""
        self._init_components(load_rag_chain=True)

        print(f"👤 问题: {question}")
        print("🤖 正在思考...")

        result = self.rag_chain.query(question)

        print(f"\n💬 回答:\n{result.answer}")

        if result.sources:
            print("\n📚 引用来源:")
            for i, source in enumerate(result.sources, 1):
                page_info = f"第{source.page}页" if source.page > 0 else "全文"
                print(f"  {i}. {source.filename} - {page_info} (相关度: {source.score:.2%})")

        print(f"\n⏱️  检索: {result.retrieve_time:.0f}ms | 生成: {result.generate_time:.0f}ms | 总计: {result.total_time:.0f}ms")

    # ---------- 列出文档 ----------
    def list_docs(self):
        """列出知识库中的文档"""
        self._init_components(load_rag_chain=False)

        docs = self.vector_store.list_documents()
        print(f"📚 知识库文档（共 {len(docs)} 个，{self.vector_store.count()} 个向量块）:")
        print("-" * 60)

        if not docs:
            print("  （知识库为空）")
            return

        for i, doc in enumerate(docs, 1):
            print(f"  {i}. {doc['filename']}")
            print(f"     块数: {doc['chunk_count']} | 类型: {doc.get('file_type', '?')} | 上传: {doc.get('upload_time', '?')}")

    # ---------- 删除文档 ----------
    def delete(self, filename: str):
        """删除指定文档"""
        self._init_components(load_rag_chain=False)

        confirm = input(f"⚠️  确认删除文档 '{filename}' 及其所有向量？(y/N): ")
        if confirm.lower() != "y":
            print("已取消")
            return

        deleted = self.vector_store.delete_by_filename(filename)
        if deleted > 0:
            print(f"✅ 已删除 {deleted} 个向量块")
        else:
            print(f"❌ 未找到文档: {filename}")

    # ---------- 初始化项目 ----------
    def init(self):
        """初始化项目目录结构"""
        dirs = [
            "./data/docs",           # 示例文档目录
            "./data/test",           # 测试文档目录
            "./uploaded_docs",       # 上传文档目录
            "./chroma_db",           # 向量库目录
            "./logs",                # 日志目录
            "./evaluations",         # 评估结果目录
        ]

        print("📁 初始化项目目录...")
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
            print(f"  ✓ {d}")

        # 创建 .env 模板
        env_path = Path("./.env")
        if not env_path.exists():
            env_path.write_text(
                "# OpenAI API Key（必填）\n"
                "OPENAI_API_KEY=your_api_key_here\n\n"
                "# 可选：自定义API Base URL（使用第三方兼容接口时）\n"
                "# LLM_BASE_URL=https://api.openai.com/v1\n",
                encoding="utf-8",
            )
            print("  ✓ .env 模板已创建")

        print("\n✅ 项目初始化完成！")
        print("下一步:")
        print("  1. 编辑 .env 文件，填入你的 API Key")
        print("  2. 将文档放入 data/docs 目录")
        print("  3. 运行 python main.py build --dir ./data/docs 构建知识库")
        print("  4. 运行 python main.py chat 开始问答")

    # ---------- 启动API服务 ----------
    def serve(self):
        """启动FastAPI服务"""
        import uvicorn
        print(f"🚀 启动API服务: http://{settings.server.host}:{settings.server.port}")
        uvicorn.run(
            "api:app",
            host=settings.server.host,
            port=settings.server.port,
            reload=False,
        )

    # ---------- 启动Web前端 ----------
    def web(self):
        """启动Gradio前端（备份版本，推荐使用frontend-vue目录下的Vue3前端）"""
        import frontend
        frontend.demo = frontend.build_interface()
        frontend.demo.queue()
        frontend.demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
        )


# ============================================================
# 命令行参数解析
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="RAG智能文档问答系统 - 命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py init                    初始化项目
  python main.py build --dir ./data      从目录构建知识库
  python main.py add file.pdf            添加单个文件
  python main.py chat                    交互式问答
  python main.py ask "什么是RAG？"        单次问答
  python main.py list                    列出知识库文档
  python main.py delete example.pdf      删除文档
  python main.py serve                   启动API服务
  python main.py web                     启动Web前端
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # init
    subparsers.add_parser("init", help="初始化项目目录")

    # build
    build_parser = subparsers.add_parser("build", help="从目录构建知识库")
    build_parser.add_argument("--dir", required=True, help="文档目录路径")
    build_parser.add_argument("--no-recursive", action="store_true", help="不递归子目录")

    # add
    add_parser = subparsers.add_parser("add", help="添加单个文件到知识库")
    add_parser.add_argument("file", help="文件路径")

    # chat
    subparsers.add_parser("chat", help="交互式问答")

    # ask
    ask_parser = subparsers.add_parser("ask", help="单次问答")
    ask_parser.add_argument("question", help="问题")

    # list
    subparsers.add_parser("list", help="列出知识库文档")

    # delete
    delete_parser = subparsers.add_parser("delete", help="删除文档")
    delete_parser.add_argument("filename", help="要删除的文件名")

    # serve
    subparsers.add_parser("serve", help="启动API服务")

    # web
    subparsers.add_parser("web", help="启动Web前端")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    cli = RAGCLI()

    if args.command == "init":
        cli.init()
    elif args.command == "build":
        cli.build(args.dir, recursive=not args.no_recursive)
    elif args.command == "add":
        cli.add_file(args.file)
    elif args.command == "chat":
        cli.chat()
    elif args.command == "ask":
        cli.ask(args.question)
    elif args.command == "list":
        cli.list_docs()
    elif args.command == "delete":
        cli.delete(args.filename)
    elif args.command == "serve":
        cli.serve()
    elif args.command == "web":
        cli.web()


if __name__ == "__main__":
    main()
