# ============================================================
# RAG智能文档问答系统 - Dockerfile
# ============================================================
# 基础镜像：Python 3.11 slim（轻量级）
FROM python:3.11-slim

# 设置工作目录（后端整合在 backend/ 目录下）
WORKDIR /app/backend

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Shanghai

# 安装系统依赖（PyMuPDF等需要的编译工具）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件（利用Docker缓存）
COPY backend/requirements.txt .

# 安装Python依赖
# 注意：torch较大，单独安装以利用缓存
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# 复制后端代码
COPY backend/ .

# 创建必要目录
RUN mkdir -p /app/backend/uploaded_docs /app/backend/chroma_db /app/backend/logs /app/backend/evaluations /app/backend/data/docs

# 暴露端口
# 8000: FastAPI API
# 7860: Gradio前端
EXPOSE 8000 7860

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# 默认启动命令（启动API服务）
# 可通过 docker run 覆盖启动前端或其他命令
CMD ["python", "api.py"]
