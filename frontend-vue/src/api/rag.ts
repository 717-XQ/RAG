import axios from 'axios'
import type {
  ChatRequest,
  ChatResponse,
  DocInfo,
  HealthResponse,
  UploadResponse
} from '@/types'

const API_BASE = '/api'

const http = axios.create({
  baseURL: API_BASE,
  timeout: 60000
})

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

// 流式问答（SSE）
export function chatStream(
  request: ChatRequest,
  onToken: (token: string) => void,
  onDone: (response: ChatResponse) => void,
  onError: (error: Error) => void
): () => void {
  const controller = new AbortController()

  fetch(`${API_BASE}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
    signal: controller.signal
  })
    .then(async (response) => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
      const reader = response.body?.getReader()
      if (!reader) {
        throw new Error('No response body')
      }
      const decoder = new TextDecoder()
      let fullAnswer = ''
      let buffer = ''

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
                total_time: 0
              })
              return
            }
            try {
              const parsed = JSON.parse(data)
              if (parsed.token) {
                fullAnswer += parsed.token
                onToken(parsed.token)
              } else if (parsed.answer) {
                fullAnswer = parsed.answer
                onDone(parsed)
                return
              }
            } catch {
              // 忽略解析错误
            }
          }
        }
      }
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
