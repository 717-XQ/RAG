import axios, { AxiosError, type InternalAxiosRequestConfig } from 'axios'
import type {
  ChatRequest,
  ChatResponse,
  DocInfo,
  HealthResponse,
  UploadResponse,
  TokenResponse,
  LoginRequest,
  RegisterRequest,
  UserInfo,
  ExportRequest,
  TokenUsageStats
} from '@/types'

const API_BASE = '/api'

const http = axios.create({
  baseURL: API_BASE,
  timeout: 60000
})

// ---------- Token 管理（技术栈2.2：Axios Token自动刷新） ----------
const ACCESS_TOKEN_KEY = 'rag_access_token'
const REFRESH_TOKEN_KEY = 'rag_refresh_token'

export function getAccessToken(): string {
  return localStorage.getItem(ACCESS_TOKEN_KEY) || ''
}

export function getRefreshToken(): string {
  return localStorage.getItem(REFRESH_TOKEN_KEY) || ''
}

export function setTokens(access: string, refresh: string) {
  localStorage.setItem(ACCESS_TOKEN_KEY, access)
  localStorage.setItem(REFRESH_TOKEN_KEY, refresh)
}

export function clearTokens() {
  localStorage.removeItem(ACCESS_TOKEN_KEY)
  localStorage.removeItem(REFRESH_TOKEN_KEY)
}

// 请求拦截器：自动附加 Bearer Token
http.interceptors.request.use((config: InternalAxiosRequestConfig) => {
  const token = getAccessToken()
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// 响应拦截器：401 时用刷新Token自动续期并重试（只重试一次）
let refreshing: Promise<string | null> | null = null

async function doRefresh(): Promise<string | null> {
  const refreshToken = getRefreshToken()
  if (!refreshToken) return null
  try {
    const { data } = await axios.post<TokenResponse>(`${API_BASE}/auth/refresh`, {
      refresh_token: refreshToken
    })
    setTokens(data.access_token, data.refresh_token)
    return data.access_token
  } catch {
    clearTokens()
    return null
  }
}

http.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const original = error.config as InternalAxiosRequestConfig & { _retried?: boolean }
    const status = error.response?.status

    // 401 且未重试过，且不是认证接口本身
    if (
      status === 401 &&
      original &&
      !original._retried &&
      !original.url?.includes('/auth/login') &&
      !original.url?.includes('/auth/register') &&
      !original.url?.includes('/auth/refresh')
    ) {
      original._retried = true
      if (!refreshing) {
        refreshing = doRefresh()
      }
      const newToken = await refreshing
      refreshing = null

      if (newToken) {
        original.headers.Authorization = `Bearer ${newToken}`
        return http(original)
      }
      // 刷新失败：跳转登录页
      window.location.hash = '#/login'
      clearTokens()
    }
    return Promise.reject(error)
  }
)

// ---------- 认证 API（技术栈2.2：JWT） ----------
export async function login(request: LoginRequest): Promise<TokenResponse> {
  const { data } = await http.post<TokenResponse>('/auth/login', request)
  return data
}

export async function register(request: RegisterRequest): Promise<TokenResponse> {
  const { data } = await http.post<TokenResponse>('/auth/register', request)
  return data
}

export async function getMe(): Promise<UserInfo> {
  const { data } = await http.get<UserInfo>('/auth/me')
  return data
}

// 健康检查
export async function healthCheck(): Promise<HealthResponse> {
  const { data } = await http.get<HealthResponse>('/health')
  return data
}

// 上传文档
export async function uploadDocument(file: File): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append('file', file)
  const { data } = await http.post<UploadResponse>('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
  return data
}

// 非流式问答
export async function chat(request: ChatRequest): Promise<ChatResponse> {
  const { data } = await http.post<ChatResponse>('/chat', request)
  return data
}

// 流式问答（SSE，携带Token）
export function chatStream(
  request: ChatRequest,
  onToken: (token: string) => void,
  onDone: (response: ChatResponse) => void,
  onError: (error: Error) => void
): () => void {
  const controller = new AbortController()

  fetch(`${API_BASE}/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${getAccessToken()}`
    },
    body: JSON.stringify(request),
    signal: controller.signal
  })
    .then(async (response) => {
      if (response.status === 401) {
        // Token过期：尝试刷新后重试一次
        const newToken = await doRefresh()
        if (newToken) {
          const retry = await fetch(`${API_BASE}/chat/stream`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              Authorization: `Bearer ${newToken}`
            },
            body: JSON.stringify(request),
            signal: controller.signal
          })
          if (!retry.ok) throw new Error(`HTTP ${retry.status}`)
          return retry
        }
        window.location.hash = '#/login'
        throw new Error('登录已过期，请重新登录')
      }
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
      return response
    })
    .then(async (response) => {
      const reader = response.body?.getReader()
      if (!reader) {
        throw new Error('No response body')
      }
      const decoder = new TextDecoder()
      let fullAnswer = ''
      let buffer = ''
      let provider = ''
      let tokenUsage = {}

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6)
            if (data === '[DONE]') {
              onDone({
                answer: fullAnswer,
                sources: [],
                retrieve_time: 0,
                generate_time: 0,
                total_time: 0,
                provider
              })
              return
            }
            try {
              const parsed = JSON.parse(data)
              if (parsed.type === 'content' && parsed.content) {
                fullAnswer += parsed.content
                onToken(parsed.content)
              } else if (parsed.type === 'provider') {
                provider = parsed.name || ''
              } else if (parsed.type === 'error') {
                throw new Error(parsed.message || '流式输出错误')
              } else if (parsed.token) {
                // 兼容旧协议
                fullAnswer += parsed.token
                onToken(parsed.token)
              } else if (parsed.answer) {
                fullAnswer = parsed.answer
                onDone({ ...parsed, provider, token_usage: tokenUsage })
                return
              }
            } catch (e) {
              if (e instanceof Error) throw e
              // 忽略其他解析错误
            }
          }
        }
      }
      onDone({
        answer: fullAnswer,
        sources: [],
        retrieve_time: 0,
        generate_time: 0,
        total_time: 0,
        provider
      })
    })
    .catch((error) => {
      if (error.name !== 'AbortError') {
        onError(error)
      }
    })

  return () => controller.abort()
}

// 获取文档列表
export async function listDocuments(): Promise<DocInfo[]> {
  const { data } = await http.get<DocInfo[]>('/docs')
  return data
}

// 删除文档
export async function deleteDocument(filename: string): Promise<void> {
  await http.delete(`/docs/${encodeURIComponent(filename)}`)
}

// ---------- 报告导出（技术栈2.2：PDF / DOCX） ----------
export async function exportReport(request: ExportRequest, format: 'pdf' | 'docx') {
  const response = await http.post(`/export/${format}`, request, { responseType: 'blob' })
  return response.data as Blob
}

// ---------- Token用量统计（技术栈2.2：ECharts） ----------
export async function getTokenUsage(days = 7): Promise<TokenUsageStats> {
  const { data } = await http.get<TokenUsageStats>('/stats/token-usage', { params: { days } })
  return data
}
