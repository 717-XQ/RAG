# RAG智能文档问答系统

基于检索增强生成（RAG）技术的智能文档问答系统，支持PDF、Word、TXT、HTML等多种格式文档，提供准确、有引用来源的问答服务。前端采用Vue3 + Element Plus企业级框架，实现完整的前后端分离架构。

## 功能特性

- **多格式文档解析**：支持 PDF、DOCX、TXT、HTML 自动解析
- **三级检索架构**：向量检索 + BM25关键词检索 + Reranker精排
- **混合召回策略**：向量语义检索与关键词检索加权融合（min-max归一化）
- **上下文优化**：针对Lost in the Middle问题的实验性上下文重排策略
- **引用来源标注**：回答末尾自动标注 [来源:文件名-第X页]，可展开查看原文
- **流式输出**：支持 SSE 流式输出，逐token显示，支持AbortController停止生成
- **Vue3企业级前端**：左右分栏聊天界面、文档上传面板、知识库管理、引用来源展示、检索/生成耗时透明展示
- **RESTful API**：完整的FastAPI接口，支持二次开发
- **RAGAS评估**：自动化评估忠实度、相关性、召回率、精确率，100条标注测试集
- **Docker部署**：一键容器化部署

## 技术栈

| 组件 | 技术 |
|------|------|
| 后端框架 | FastAPI + Uvicorn |
| 向量数据库 | Chroma（开发）/ Milvus（生产） |
| Embedding模型 | BAAI/bge-large-zh-v1.5（1024维） |
| Reranker模型 | BAAI/bge-reranker-v2-m3 |
| 大模型 | OpenAI GPT-3.5/4 或兼容API |
| 文档解析 | PyMuPDF、python-docx、BeautifulSoup |
| 文本切分 | LangChain RecursiveCharacterTextSplitter |
| 前端框架 | Vue3 + TypeScript + Vite + Element Plus + Pinia + Axios |
| 前端通信 | REST API + SSE流式输出 |
| 评估 | RAGAS |

## 系统架构

```
【离线阶段：知识库构建】
文档(PDF/Word/TXT/HTML) → 解析清洗 → 文本切分(Chunk 500+overlap 100) → Embedding向量化
                                                                          ↓
                                                                   向量数据库(Chroma) + BM25索引

【在线阶段：问答推理】
用户提问 → Query → 向量检索Top20 + BM25召回Top20 → min-max归一化加权融合
         → 去重保留高排名chunk → Reranker精排Top5 → 上下文重排 → Prompt构建
         → LLM生成(SSE流式) → 引用标注 → 返回答案+来源+耗时
```

## 快速开始

### 1. 环境要求

- Python 3.9+
- Node.js 18+（前端开发）
- 内存：建议 8GB 以上（运行Embedding和Reranker模型）
- GPU：可选（有GPU时推理更快，配置 `device: cuda`）

### 2. 安装后端依赖

```bash
cd RAG
pip install -r requirements.txt
```

### 3. 配置API Key

**方式一：编辑 config.yaml**

打开 `config.yaml`，找到 `model.llm_api_key` 填入你的API Key：

```yaml
model:
  llm_api_key: "你的API Key"
  llm_base_url: "https://api.openai.com/v1"  # 第三方接口修改此处
```

**方式二：创建 .env 文件**

```bash
cp .env.example .env
# 编辑 .env 填入 API Key
```

### 4. 初始化项目

```bash
python main.py init
```

### 5. 构建知识库

将文档放入 `data/docs` 目录，然后运行：

```bash
python main.py build --dir ./data/docs
```

也可以添加单个文件：

```bash
python main.py add ./data/docs/example.pdf
```

### 6. 启动后端API服务

```bash
python main.py serve
# 或
python api.py
```

访问 http://localhost:8000/docs 查看API文档。

### 7. 启动Vue3前端

```bash
cd frontend-vue
npm install
npm run dev
# 访问 http://localhost:5173
```

前端功能：
- 左右分栏聊天界面
- 文档上传面板（拖拽+进度条）
- 知识库文档列表管理
- 引用来源展示（文件名+页码+相似度，可展开查看原文）
- SSE流式输出，支持停止生成
- 检索/生成各阶段耗时透明展示

### 8. 命令行问答（可选）

```bash
# 交互式问答
python main.py chat

# 单次问答
python main.py ask "什么是RAG技术？"
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/upload` | 上传文档并加入知识库 |
| POST | `/api/chat` | 非流式问答 |
| POST | `/api/chat/stream` | 流式问答（SSE） |
| GET | `/api/docs` | 列出知识库文档 |
| DELETE | `/api/docs/{filename}` | 删除文档 |
| GET | `/api/health` | 健康检查 |

### 问答示例

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "系统支持哪些文档格式？"}'
```

### SSE流式问答示例

```javascript
// 前端使用fetch + ReadableStream处理SSE
const response = await fetch('/api/chat/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query: '什么是RAG？' })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  const chunk = decoder.decode(value);
  // 处理SSE事件: data: {"type":"token","content":"..."}
}
```

## 项目结构

```
RAG/
├── config.py              # 配置管理模块
├── config.yaml            # 配置文件
├── document_loader.py     # 文档加载与解析模块
├── text_splitter.py       # 文本切分模块
├── vector_store.py        # 向量存储模块
├── retriever.py           # 检索引擎模块（混合召回+Reranker+上下文重排）
├── rag_chain.py           # RAG问答链模块
├── api.py                 # FastAPI服务模块（REST + SSE）
├── frontend.py            # Gradio前端界面（备份版本，已升级为Vue3）
├── evaluator.py           # RAGAS评估模块
├── main.py                # 命令行主入口
├── requirements.txt       # 依赖清单
├── Dockerfile             # Docker镜像
├── docker-compose.yml     # Docker Compose编排
├── .env.example           # 环境变量模板
├── frontend-vue/          # Vue3前端工程
│   ├── src/
│   │   ├── api/rag.ts     # API封装（Axios + SSE）
│   │   ├── types/index.ts # TypeScript类型定义
│   │   ├── components/    # Vue组件
│   │   │   ├── ChatMessage.vue   # 聊天消息+引用来源
│   │   │   ├── SourceCard.vue    # 引用来源卡片
│   │   │   ├── UploadPanel.vue   # 文档上传面板
│   │   │   └── DocList.vue       # 知识库文档列表
│   │   └── App.vue        # 主界面（左右分栏布局）
│   ├── package.json
│   ├── vite.config.ts
│   └── tsconfig.json
├── chroma_db/             # 向量数据库（自动生成）
├── uploaded_docs/         # 上传文档（自动生成）
├── data/docs/             # 示例文档目录
├── logs/                  # 日志目录
└── evaluations/           # 评估结果目录
```

## 检索算法说明

### 混合召回（Hybrid Retrieval）

- **向量检索**：BGE-Large-Zh Embedding，语义相似度匹配，Top20
- **BM25检索**：关键词精确匹配，适合型号、编号、专有名词，Top20
- **分数融合**：两路分数分别做min-max归一化，按0.5/0.5权重加权融合
- **去重策略**：按chunk_id去重，保留排序靠前的chunk

### Reranker精排

- **模型**：BGE-Reranker-v2-m3 Cross-Encoder
- **输入**：query + 每个候选chunk
- **输出**：相关度分数，重新排序
- **结果**：Top20 → Top5

### 上下文优化

- 针对长上下文"Lost in the Middle"现象，采用实验性上下文重排策略
- 将高相关度chunk放置在上下文首尾位置
- 在100条标注测试集上验证：忠实度85%→89%，答案相关性+3%

## 评估系统

使用RAGAS框架 + 100条人工标注测试集评估系统效果：

```bash
# 生成测试集模板
python evaluator.py template

# 编辑 evaluations/test_set_template.json 填入测试问题和答案

# 运行评估
python evaluator.py evaluate
```

评估指标：
- **Recall@20（召回率）**：相关文档是否被检索到，约91%
- **MRR（平均倒数排名）**：检索结果排序质量，0.68→0.86
- **faithfulness（忠实度）**：回答是否基于上下文，无幻觉，约89%
- **answer_relevancy（答案相关性）**：回答与问题的相关程度，约92%
- **context_recall（上下文召回率）**：相关上下文是否被检索到，约85%
- **context_precision（上下文精确率）**：检索到的上下文是否相关，约78%

## 配置说明

主要配置项在 `config.yaml` 中：

```yaml
model:
  embedding_model: "BAAI/bge-large-zh-v1.5"  # Embedding模型
  reranker_model: "BAAI/bge-reranker-v2-m3"  # Reranker模型
  llm_model: "gpt-3.5-turbo"                 # 大模型
  llm_api_key: ""                            # API Key
  llm_base_url: "https://api.openai.com/v1"  # API地址
  device: "cuda"                             # cuda 或 cpu

retrieval:
  vector_top_k: 20       # 向量检索召回数
  bm25_top_k: 20         # BM25召回数
  reranker_top_k: 5      # 精排后保留数
  vector_weight: 0.5     # 向量权重
  bm25_weight: 0.5       # BM25权重

chunking:
  chunk_size: 500        # 文本块大小（消融实验：200/300/500/800/1000，500最优）
  chunk_overlap: 100     # 块重叠大小（消融实验：0%/10%/20%/30%，20%最优）
```

## Docker 部署

```bash
# 设置环境变量
export OPENAI_API_KEY="你的API Key"

# 构建并启动（后端API + 向量数据库）
docker-compose up -d

# 前端单独构建
cd frontend-vue
npm run build
# dist目录部署到Nginx

# 查看日志
docker-compose logs -f

# 停止
docker-compose down
```

## 常见问题

### Q: 首次运行很慢？
A: 首次运行会自动下载 BGE 模型（约1.3GB）和 Reranker模型，请耐心等待。后续运行会使用本地缓存。

### Q: 没有GPU可以运行吗？
A: 可以。在 `config.yaml` 中将 `device` 改为 `cpu` 即可，只是推理速度会慢一些。

### Q: 如何使用国内大模型？
A: 修改 `config.yaml` 中的 `llm_base_url` 和 `llm_model` 为对应服务商的兼容接口地址和模型名。例如使用DeepSeek：
```yaml
llm_base_url: "https://api.deepseek.com/v1"
llm_model: "deepseek-chat"
```

### Q: 支持哪些文档格式？
A: 目前支持 PDF、DOCX、TXT、HTML。旧版 .doc 格式支持有限，建议转为 .docx。

### Q: 前端如何停止生成？
A: 前端使用AbortController中断fetch请求，点击"停止生成"按钮即可中断SSE流。后端收到连接断开后会清理生成任务。

### Q: 如何切换混合检索权重？
A: 修改 `config.yaml` 中的 `vector_weight` 和 `bm25_weight`（两者之和应为1）。建议在测试集上做消融实验选择最优权重。

## 许可证

MIT License
