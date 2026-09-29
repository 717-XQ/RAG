// ============================================================
// 前端类型定义（已按《技术栈规范》2.2 扩展：认证/统计/导出）
// ============================================================

// 聊天消息类型
export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: number
  sources?: SourceInfo[]
  retrieveTime?: number
  generateTime?: number
  totalTime?: number
  provider?: string
  tokenUsage?: TokenUsage
  loading?: boolean
}

// 引用来源信息
export interface SourceInfo {
  doc_id?: string
  filename?: string
  content?: string
  content_preview?: string
  score?: number
  chunk_index?: number
  page?: number
  source?: string
}

// 文档信息
export interface DocInfo {
  filename: string
  chunk_count: number
  upload_time: string
  file_type: string
  source: string
}

// 健康检查响应
export interface HealthResponse {
  status: string
  vector_count: number
  doc_count: number
  model_settings: Record<string, any>
}

// 上传响应
export interface UploadResponse {
  status: string
  filename: string
  chunks_added: number
  doc_id: string
  message: string
}

// 聊天请求
export interface ChatRequest {
  query: string
  use_reranker: boolean
  session_id?: string
}

// 聊天响应
export interface ChatResponse {
  answer: string
  sources: SourceInfo[]
  retrieve_time: number
  generate_time: number
  total_time: number
  session_id?: string
  provider?: string
  token_usage?: TokenUsage
}

// Token用量
export interface TokenUsage {
  prompt_tokens: number
  completion_tokens: number
}

// ---------- 认证（技术栈2.2：JWT + Token自动刷新） ----------
export interface TokenResponse {
  access_token: string
  refresh_token: string
  token_type: string
  expires_in: number
}

export interface UserInfo {
  id: number
  username: string
  email: string
  is_active: boolean
  created_at?: string
}

export interface LoginRequest {
  username: string
  password: string
}

export interface RegisterRequest {
  username: string
  email: string
  password: string
}

// ---------- 报告导出（技术栈2.2：PDF/DOCX） ----------
export interface ExportRequest {
  question: string
  answer: string
  sources: SourceInfo[]
  title?: string
}

// ---------- Token用量统计（技术栈2.2：ECharts图表） ----------
export interface TokenUsagePoint {
  date: string
  input_tokens: number
  output_tokens: number
  total_tokens: number
  calls: number
}

export interface TokenUsageStats {
  days: number
  data: TokenUsagePoint[]
}

// 会话信息
export interface SessionInfo {
  session_id: string
  title: string
  created_at: string
  updated_at: string
}
