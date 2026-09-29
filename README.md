# RAG智能文档问答系统

基于检索增强生成（RAG）技术的智能文档问答系统，支持PDF、Word、TXT、HTML等多种格式文档，提供准确、有引用来源的问答服务。前端采用Vue3 + Element Plus企业级框架，实现完整的前后端分离架构。

## 功能特性

- **多格式文档解析**：支持 PDF、DOCX、TXT、HTML 自动解析
- **三级检索架构**：向量检索 + 关键词检索（BM25 / SQLite FTS5）+ Reranker精排
- **SQLite FTS5 全文检索**：trigram 分词器，中文关键词检索（技术栈2.1）
- **混合召回策略**：向量语义检索与关键词检索加权融合（min-max归一化）
- **多Provider自动降级**：DeepSeek / DashScope(通义千问) / OpenAI 等按优先级自动降级（技术栈2.1）
- **JWT认证**：python-jose + bcrypt，注册/登录/刷新Token，前端Token自动续期（技术栈2.1/2.2）
- **关系数据库**：SQLAlchemy 2.0 + Alembic 迁移，MySQL（开发）/ PostgreSQL（生产）（技术栈2.1）
- **报告导出**：WeasyPrint PDF 生成 / python-docx Word 导出（技术栈2.1）
- **Token用量统计**：按天聚合，ECharts 图表展示（技术栈2.2）
- **上下文优化**：针对Lost in the Middle问题的实验性上下文重排策略
- **引用来源标注**：回答末尾自动标注 [来源:文件名-第X页]，可展开查看原文
- **流式输出**：支持 SSE 流式输出，逐token显示，支持AbortController停止生成
- **Vue3企业级前端**：Vue Router 路由、Pinia 状态、markdown-it 渲染、ECharts 图表（技术栈2.2）
- **RESTful API**：完整的FastAPI接口，支持二次开发
- **RAGAS评估**：自动化评估忠实度、相关性、召回率、精确率，100条标注测试集
- **Docker部署**：一键容器化部署

## 技术栈

| 组件 | 技术 |
|------|------|
| 后端框架 | FastAPI + Uvicorn |
| Agent编排 | LangGraph + LangChain 0.2+ |
| 向量数据库 | Chroma（开发）/ Milvus（生产） |
| 全文检索 | SQLite FTS5（trigram分词器，技术栈2.1） |
| Embedding模型 | BAAI/bge-large-zh-v1.5（1024维） |
| Reranker模型 | BAAI/bge-reranker-v2-m3 |
| 大模型 | DeepSeek / DashScope / OpenAI 多Provider自动降级（OpenAI兼容API） |
| 文档解析 | PyMuPDF、python-docx、BeautifulSoup |
| 文本切分 | LangChain RecursiveCharacterTextSplitter |
| 关系数据库 | MySQL（开发）/ PostgreSQL（生产）+ SQLAlchemy 2.0 + Alembic |
| 认证 | python-jose（JWT）+ bcrypt |
| 文档导出 | WeasyPrint（PDF）/ python-docx（Word） |
| 前端框架 | Vue3 + TypeScript 5.4 + Vite + Element Plus + Pinia + Vue Router + Axios |
| Markdown渲染 | markdown-it + highlight.js |
| 图表 | ECharts 5.5+（Token用量统计） |
| 前端通信 | REST API + SSE流式输出（Token自动刷新） |
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
cd RAG/backend        # 后端整合在 backend/ 目录（前端为 frontend-vue/）
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
cd RAG/backend
python main.py init
```

### 4.1 初始化关系数据库（MySQL / PostgreSQL，技术栈2.1）

本地已安装 MySQL 时，先修改 `config.yaml` 中的 `database.url` 为你的连接串：

```yaml
database:
  url: "mysql+pymysql://用户名:密码@127.0.0.1:3306/rag_kb?charset=utf8mb4"
```

然后执行（自动建库+建表，也可生成Alembic迁移）：

```bash
cd RAG/backend
python main.py db-init            # 建库建表
python main.py db-init --alembic  # 同时生成Alembic迁移
```

> 未配置MySQL时系统会自动回退本地SQLite，便于快速开发。

**SQLite 旧数据迁移到 MySQL**：若之前使用过 SQLite（`backend/data/rag.db`），希望把用户/会话/文档/消息/Token用量迁移到 MySQL：

```bash
cd RAG/backend
python migrate_sqlite_to_mysql.py
```

脚本自动从 `config.yaml` 读取 MySQL 连接并建表，目标表已有相同主键时自动跳过，**可安全重复运行**（迁移 5 张表：users / documents / chat_sessions / chat_messages / token_usage）。

### 4.2 配置多Provider（DeepSeek / DashScope，技术栈2.1）

优先通过环境变量注入 API Key（推荐，避免明文密钥入库）：

```bash
# Windows PowerShell
$env:DEEPSEEK_API_KEY="sk-..."
$env:DASHSCOPE_API_KEY="sk-..."
```

或在 `.env` 文件中配置（`config.yaml` 的 `model.providers` 中留空 `api_key` 即可）：

```
DEEPSEEK_API_KEY=sk-...
DASHSCOPE_API_KEY=sk-...
```

系统按 `priority` 顺序尝试各Provider，调用失败自动降级到下一个。

### 5. 构建知识库

将文档放入 `backend/data/docs` 目录，然后运行：

```bash
cd RAG/backend
python main.py build --dir ./data/docs
```

也可以添加单个文件：

```bash
python main.py add ./data/docs/example.pdf
```

### 6. 启动后端API服务

```bash
cd RAG/backend
python main.py serve
# 或
python api.py
```

访问 http://localhost:8000/swagger 查看API文档。

### 6.1 登录获取Token

```bash
# 注册
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","email":"admin@example.com","password":"your_password"}'

# 登录（返回 access_token / refresh_token）
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"your_password"}'
```

所有业务API（上传/问答/文档/统计/导出）需要携带 `Authorization: Bearer <access_token>`；
访问Token 30分钟过期，前端Axios会自动用刷新Token续期。

> **本地测试账号**：`test_user` / `test123456`（MySQL 初始化时已预置；也可自行注册新账号）。

### 7. 启动Vue3前端

```bash
cd frontend-vue
npm install
npm run dev
# 访问 http://localhost:5173
```

前端功能：
- 左右分栏聊天界面（登录后使用）
- 文档上传面板（拖拽+进度条）
- 知识库文档列表管理
- 引用来源展示（文件名+页码+相似度，可展开查看原文）
- Markdown渲染（markdown-it + highlight.js 代码高亮）
- SSE流式输出，支持停止生成
- 检索/生成各阶段耗时与Token用量透明展示
- 问答报告导出（PDF / DOCX 一键下载）
- Token用量统计图表（ECharts，按天查看7/14/30天）
- 登录/注册页面与Token自动刷新

### 7.1 前端访问

1. 打开 http://localhost:5173 自动跳转登录页
2. 注册新账号或使用已注册账号登录
3. 登录后进入聊天主界面，上传文档后开始问答

### 7.2 后端冒烟测试（可选）

```bash
cd RAG/backend
python smoke_test_api.py    # 快速API连通性测试
python smoke_test_full.py   # 全流程测试（注册→登录→上传→问答→Token统计）
```

### 8. 命令行问答（可选）

```bash
cd RAG/backend
# 交互式问答
python main.py chat

# 单次问答
python main.py ask "什么是RAG技术？"
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/auth/register` | 用户注册（返回Token） |
| POST | `/auth/login` | 用户登录（返回Token） |
| POST | `/auth/refresh` | 刷新Token |
| GET | `/auth/me` | 当前用户信息 |
| POST | `/api/upload` | 上传文档并加入知识库 |
| POST | `/api/chat` | 非流式问答 |
| POST | `/api/chat/stream` | 流式问答（SSE） |
| GET | `/api/docs` | 列出知识库文档 |
| DELETE | `/api/docs/{filename}` | 删除文档 |
| POST | `/api/export/pdf` | 导出PDF问答报告（WeasyPrint） |
| POST | `/api/export/docx` | 导出Word问答报告（python-docx） |
| GET | `/api/stats/token-usage` | Token用量统计（ECharts数据源） |
| GET | `/api/stats/sessions` | 会话列表 |
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
├── backend/                # 后端（对齐技术栈2.1，与 frontend-vue/ 平行区分）
│   ├── config.py           # 配置管理模块（pydantic-settings，多Provider/数据库/认证/导出）
│   ├── config.yaml         # 配置文件（含敏感Key，已 .gitignore 排除）
│   ├── config.yaml.example # 配置文件模板（占位值，可入库）
│   ├── database.py         # SQLAlchemy 引擎与会话（MySQL优先/SQLite回退）
│   ├── models.py           # ORM模型（用户/文档记录/会话/消息/Token用量）
│   ├── auth.py             # JWT认证（python-jose + bcrypt）
│   ├── fts5.py             # SQLite FTS5 全文检索（trigram分词器）
│   ├── llm_client.py       # 多Provider LLM客户端（自动降级）
│   ├── export.py           # 报告导出（WeasyPrint PDF / python-docx Word）
│   ├── document_loader.py  # 文档加载与解析模块
│   ├── text_splitter.py    # 文本切分模块
│   ├── vector_store.py     # 向量存储模块（Chroma / Milvus）
│   ├── retriever.py        # 检索引擎模块（混合召回+Reranker+上下文重排）
│   ├── rag_chain.py        # RAG问答链模块
│   ├── api.py              # FastAPI服务模块（REST + SSE + 认证 + 导出 + 统计）
│   ├── evaluator.py        # RAGAS评估模块
│   ├── main.py             # 命令行主入口（含 db-init 数据库初始化）
│   ├── migrate_sqlite_to_mysql.py  # SQLite→MySQL 数据迁移（可重复运行）
│   ├── smoke_test_api.py   # API冒烟测试（连通性）
│   ├── smoke_test_full.py  # 全流程冒烟测试（注册→登录→上传→问答→统计）
│   ├── frontend.py         # Gradio 轻量聊天界面（可选，替代/补充Vue前端）
│   ├── requirements.txt    # 依赖清单
│   ├── alembic.ini         # Alembic迁移配置
│   ├── migrations/         # Alembic迁移目录（初始迁移 init_tables）
│   ├── .env / .env.example # 环境变量
│   ├── chroma_db/          # 向量数据库（自动生成）
│   ├── uploaded_docs/      # 上传文档（自动生成）
│   ├── data/               # 数据目录（docs示例/rag.db/fts5.db）
│   ├── logs/               # 日志目录
│   └── evaluations/        # 评估结果目录
├── frontend-vue/           # Vue3前端工程（对齐技术栈2.2）
│   ├── src/
│   │   ├── api/rag.ts      # API封装（Axios + Token自动刷新 + SSE）
│   │   ├── types/index.ts  # TypeScript类型定义
│   │   ├── stores/auth.ts  # Pinia认证状态
│   │   ├── router/         # Vue Router路由（登录/主页）
│   │   ├── views/          # 页面（LoginView / HomeView）
│   │   ├── components/     # Vue组件（ChatMessage/MarkdownContent/TokenChart/SourceCard/UploadPanel/DocList）
│   │   └── App.vue         # 应用入口（路由出口）
│   ├── package.json
│   ├── vite.config.ts
│   └── tsconfig.json
├── Dockerfile              # Docker镜像（构建 backend/）
├── docker-compose.yml      # Docker Compose编排
├── README.md               # 项目说明
├── LICENSE                 # MIT许可证
└── api_screenshot.png      # API截图
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

# 构建并启动（后端API + 向量数据库，镜像内使用 backend/ 目录）
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

### Q: 如何使用国内大模型 / 多Provider降级？
A: 在 `config.yaml` 的 `model.providers` 中配置多个Provider（DeepSeek/DashScope/OpenAI），
系统按 `priority` 顺序调用，失败自动降级到下一个。API Key 通过环境变量注入：

```yaml
model:
  providers:
    - name: "deepseek"    # DEEPSEEK_API_KEY
      model: "deepseek-chat"
      base_url: "https://api.deepseek.com/v1"
      priority: 1
    - name: "dashscope"   # DASHSCOPE_API_KEY（阿里云百炼通义千问）
      model: "qwen-plus"
      base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
      priority: 2
```

### Q: 如何切换到 Milvus 向量库（生产环境）？
A: 修改 `config.yaml` 的 `vector_store.type` 为 `milvus` 并配置 Milvus 地址，系统已内置 pymilvus 支持：
```yaml
vector_store:
  type: "milvus"          # chroma（开发）/ milvus（生产）
  milvus_host: "localhost"
  milvus_port: "19530"
```

### Q: PDF导出报错 `cannot load library 'libgobject-2.0-0'`？
A: WeasyPrint 在 Windows 上需要 GTK3 运行时。请安装
[GTK3 Runtime for Windows](https://github.com/nicedash/gtk3-runtime/releases)（或
[GTK-for-Windows-Runtime-Environment-Installer](https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer)），
安装后重启终端即可。Word（DOCX）导出无需额外依赖。

### Q: 数据库连接失败？
A: 检查 `config.yaml` 的 `database.url` 是否正确；系统默认在连接失败时回退本地SQLite。
生产环境建议配置 MySQL/PostgreSQL 并使用 `python main.py db-init` 初始化。

### Q: 支持哪些文档格式？
A: 目前支持 PDF、DOCX、TXT、HTML。旧版 .doc 格式支持有限，建议转为 .docx。

### Q: 前端如何停止生成？
A: 前端使用AbortController中断fetch请求，点击"停止生成"按钮即可中断SSE流。后端收到连接断开后会清理生成任务。

### Q: 如何切换混合检索权重 / 关键词检索实现？
A: 修改 `config.yaml` 中的 `vector_weight` 和 `bm25_weight`（两者之和应为1）；
关键词检索实现通过 `retrieval.keyword_retriever` 切换：`fts5`（SQLite全文索引，默认）或 `bm25`（内存索引）。

## 许可证

MIT License
