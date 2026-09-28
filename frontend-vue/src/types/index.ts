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
  loading?: boolean
}

// 引用来源信息
export interface SourceInfo {
  doc_id?: string
  filename?: string
  content?: string
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
}

// 聊天响应
export interface ChatResponse {
  answer: string
  sources: SourceInfo[]
  retrieve_time: number
  generate_time: number
  total_time: number
}
