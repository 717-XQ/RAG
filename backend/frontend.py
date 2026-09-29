# -*- coding: utf-8 -*-
"""
frontend.py - Gradio 前端界面模块
=============================================
功能：
  1. 基于Gradio的ChatInterface聊天界面
  2. 支持文档上传区域（上传后自动加入知识库）
  3. 回答展示引用来源及相关度分数
  4. 提供示例问题快捷按钮
  5. 显示检索耗时、生成耗时等信息

启动方式：
  python frontend.py
  或: gradio frontend.py
"""

import os
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import gradio as gr
from loguru import logger

# ============================================================
# Gradio版本兼容性检查
# gradio < 4.44.0 与 pydantic 2.8+ 存在兼容性问题，
# 会导致 "Unable to generate pydantic-core schema for starlette.requests.Request" 错误
# ============================================================
def _check_gradio_version():
    """检查Gradio版本，过低则给出升级提示"""
    try:
        from packaging import version
        current = gr.__version__
        if version.parse(current) < version.parse("4.44.0"):
            print("=" * 60)
            print(f"[警告] 当前 Gradio 版本 {current} 过低，可能导致兼容性问题！")
            print(f"[警告] 请运行以下命令升级：")
            print(f"       pip install --upgrade gradio")
            print("=" * 60)
    except Exception:
        pass

_check_gradio_version()

from config import settings
from document_loader import load_document, DocumentLoaderFactory
from text_splitter import split_documents
from vector_store import get_vector_store, BGEEmbedding
from retriever import HybridRetriever, Reranker
from rag_chain import RAGChain, AnswerResult


# ============================================================
# 全局系统组件（与api.py共享逻辑）
# ============================================================

class RAGFrontendSystem:
    """前端使用的RAG系统组件"""

    def __init__(self):
        logger.info("初始化前端RAG系统...")

        # 核心组件
        self.embedding_model = BGEEmbedding()
        self.vector_store = get_vector_store(embedding_model=self.embedding_model)
        self.reranker = Reranker()
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedding_model=self.embedding_model,
            reranker=self.reranker,
        )
        self.rag_chain = RAGChain(retriever=self.retriever)

        # 上传目录
        self.upload_dir = Path(settings.storage.upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        logger.info("前端RAG系统初始化完成！")

    def add_file(self, file_path: str) -> str:
        """
        添加上传的文件到知识库

        Args:
            file_path: 上传的临时文件路径

        Returns:
            str: 处理结果消息
        """
        if not file_path:
            return "请先选择文件"

        filename = os.path.basename(file_path)

        # 检查格式
        if not DocumentLoaderFactory.is_supported(file_path):
            return f"不支持的文件格式: {filename}"

        # 复制到上传目录
        dest_path = self.upload_dir / filename
        try:
            import shutil
            shutil.copy2(file_path, dest_path)
        except Exception as e:
            return f"文件保存失败: {str(e)}"

        # 处理文档
        try:
            documents = load_document(str(dest_path))
            if not documents:
                return f"文档 {filename} 内容为空"

            chunks = split_documents(documents)
            if isinstance(chunks, tuple):
                chunks_to_add = chunks[0]  # 父块
            else:
                chunks_to_add = chunks

            self.vector_store.add_documents(chunks_to_add)
            self.retriever.refresh_index()

            return f"✅ 文档 {filename} 上传成功！已添加 {len(chunks_to_add)} 个文本块到知识库。"

        except Exception as e:
            return f"❌ 文档处理失败: {str(e)}"

    def chat(
        self,
        message: str,
        history: List[Tuple[str, str]],
        use_reranker: bool = True,
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """
        处理聊天消息

        Args:
            message: 用户消息
            history: 对话历史
            use_reranker: 是否使用Reranker

        Returns:
            Tuple: (空字符串（清空输入框）, 更新后的对话历史)
        """
        if not message.strip():
            return "", history

        # 调用RAG问答
        result: AnswerResult = self.rag_chain.query(
            question=message,
            use_reranker=use_reranker,
        )

        # 构建回答（包含来源信息）
        answer_text = result.answer

        # 添加引用来源
        if result.sources:
            answer_text += "\n\n---\n📚 **引用来源：**\n"
            for i, source in enumerate(result.sources, 1):
                page_info = f"第{source.page}页" if source.page > 0 else "全文"
                answer_text += (
                    f"{i}. {source.filename} - {page_info} "
                    f"(相关度: {source.score:.2%})\n"
                )

        # 添加耗时信息
        answer_text += (
            f"\n⏱️ 检索: {result.retrieve_time:.0f}ms | "
            f"生成: {result.generate_time:.0f}ms | "
            f"总计: {result.total_time:.0f}ms"
        )

        # 更新历史
        history.append((message, answer_text))
        return "", history

    def get_doc_list(self) -> str:
        """获取知识库文档列表"""
        docs = self.vector_store.list_documents()
        if not docs:
            return "📭 知识库为空，请先上传文档。"

        text = f"📚 知识库文档（共 {len(docs)} 个，{self.vector_store.count()} 个向量块）：\n\n"
        for doc in docs:
            text += (
                f"• **{doc['filename']}** - "
                f"{doc['chunk_count']} 个块 - "
                f"上传于 {doc.get('upload_time', '未知')}\n"
            )
        return text


# 全局系统实例
system = RAGFrontendSystem()


# ============================================================
# Gradio 界面构建
# ============================================================

def build_interface() -> gr.Blocks:
    """
    构建Gradio界面

    Returns:
        gr.Blocks: Gradio界面对象
    """
    with gr.Blocks(
        title="RAG智能文档问答系统",
        theme=gr.themes.Soft(),
        css="""
        .chatbot { height: 500px; }
        .upload-box { background: #f8f9fa; border-radius: 8px; padding: 10px; }
        """,
    ) as demo:

        # ---------- 标题区 ----------
        gr.Markdown("""
        # 🤖 RAG智能文档问答系统

        基于检索增强生成（RAG）技术的智能文档问答系统，支持PDF、Word、TXT、HTML等多种格式文档。
        上传文档后即可用自然语言提问，系统会自动检索相关内容并生成带引用来源的准确回答。
        """)

        # ---------- 主体区：左侧聊天 + 右侧侧边栏 ----------
        with gr.Row():

            # 左侧：聊天界面
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="对话",
                    elem_classes=["chatbot"],
                    show_label=False,
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="请输入您的问题...",
                        show_label=False,
                        scale=4,
                        lines=1,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)

                # 示例问题
                gr.Markdown("**💡 示例问题（点击快速填入）：**")
                example_questions = [
                    "这个文档主要讲了什么内容？",
                    "系统的技术架构是怎样的？",
                    "如何部署这个系统？",
                    "支持哪些文档格式？",
                ]
                with gr.Row():
                    for q in example_questions:
                        gr.Button(q, size="sm").click(
                            lambda x=q: x,
                            outputs=msg_input,
                        )

            # 右侧：侧边栏（上传 + 设置 + 文档列表）
            with gr.Column(scale=1):

                # 文档上传
                gr.Markdown("### 📄 文档上传")
                with gr.Group(elem_classes=["upload-box"]):
                    file_input = gr.File(
                        label="上传文档",
                        file_types=[".pdf", ".docx", ".txt", ".html", ".htm"],
                    )
                    upload_btn = gr.Button("上传到知识库", variant="secondary")
                    upload_status = gr.Markdown("等待上传...")

                    upload_btn.click(
                        fn=system.add_file,
                        inputs=file_input,
                        outputs=upload_status,
                    )

                # 设置选项
                gr.Markdown("### ⚙️ 设置")
                use_reranker_checkbox = gr.Checkbox(
                    label="启用 Reranker 精排（推荐）",
                    value=True,
                )

                # 知识库状态
                gr.Markdown("### 📚 知识库")
                doc_list = gr.Markdown(system.get_doc_list())
                refresh_btn = gr.Button("刷新文档列表", size="sm")
                refresh_btn.click(
                    fn=system.get_doc_list,
                    outputs=doc_list,
                )

        # ---------- 底部信息 ----------
        gr.Markdown(f"""
        ---
        **系统配置：**
        Embedding: `{settings.model.embedding_model}` |
        Reranker: `{settings.model.reranker_model}` |
        LLM: `{settings.model.llm_model}` |
        向量库: `{settings.vector_store.type}`
        """)

        # ---------- 事件绑定 ----------
        # 发送按钮
        send_btn.click(
            fn=system.chat,
            inputs=[msg_input, chatbot, use_reranker_checkbox],
            outputs=[msg_input, chatbot],
        )

        # 回车发送
        msg_input.submit(
            fn=system.chat,
            inputs=[msg_input, chatbot, use_reranker_checkbox],
            outputs=[msg_input, chatbot],
        )

        # 上传后刷新文档列表
        upload_btn.click(
            fn=system.get_doc_list,
            outputs=doc_list,
        )

    return demo


# ============================================================
# 启动入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RAG智能文档问答系统 - Gradio前端启动")
    print("=" * 60)
    print(f"服务地址: http://localhost:7860")
    print(f"向量库: {settings.vector_store.type}")
    print(f"Embedding模型: {settings.model.embedding_model}")
    print(f"LLM模型: {settings.model.llm_model}")
    print("=" * 60)

    demo = build_interface()
    demo.queue()  # 启用队列，支持并发
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # 设为True可生成公网分享链接
        show_error=True,
    )
