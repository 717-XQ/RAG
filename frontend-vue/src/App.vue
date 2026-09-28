<template>
  <div class="app-container">
    <!-- 顶部导航 -->
    <header class="app-header">
      <div class="header-left">
        <el-icon class="logo-icon"><Reading /></el-icon>
        <h1 class="app-title">RAG 智能文档问答系统</h1>
      </div>
      <div class="header-right">
        <el-tag v-if="healthStatus" :type="healthStatus.status === 'healthy' ? 'success' : 'danger'" size="small">
          {{ healthStatus.status === 'healthy' ? '服务正常' : '服务异常' }}
        </el-tag>
        <el-tag v-if="healthStatus" size="small" type="info">
          {{ healthStatus.vector_count }} 向量 / {{ healthStatus.doc_count }} 文档
        </el-tag>
      </div>
    </header>

    <!-- 主内容区 -->
    <div class="main-content">
      <!-- 左侧边栏：文档管理 -->
      <aside class="sidebar">
        <el-tabs v-model="activeTab" class="sidebar-tabs">
          <el-tab-pane label="文档上传" name="upload">
            <UploadPanel @upload-success="handleUploadSuccess" />
          </el-tab-pane>
          <el-tab-pane label="知识库" name="docs">
            <DocList ref="docListRef" />
          </el-tab-pane>
        </el-tabs>
      </aside>

      <!-- 右侧：聊天区 -->
      <main class="chat-area">
        <!-- 消息列表 -->
        <el-scrollbar ref="scrollRef" class="message-list" @scroll="handleScroll">
          <div v-if="messages.length === 0" class="welcome">
            <el-icon class="welcome-icon"><ChatDotRound /></el-icon>
            <h2>欢迎使用 RAG 智能文档问答</h2>
            <p>上传文档后，我可以基于知识库内容回答您的问题</p>
            <div class="example-questions">
              <el-button
                v-for="q in exampleQuestions"
                :key="q"
                class="example-btn"
                @click="sendQuestion(q)"
              >{{ q }}</el-button>
            </div>
          </div>
          <transition-group name="message">
            <ChatMessage
              v-for="msg in messages"
              :key="msg.id"
              :message="msg"
            />
          </transition-group>
        </el-scrollbar>

        <!-- 输入区 -->
        <div class="input-area">
          <div class="input-options">
            <el-checkbox v-model="useReranker" size="small">使用 Reranker 精排</el-checkbox>
            <el-checkbox v-model="useStream" size="small">流式输出</el-checkbox>
          </div>
          <div class="input-box">
            <el-input
              v-model="inputText"
              type="textarea"
              :rows="2"
              placeholder="输入您的问题，按 Enter 发送，Shift+Enter 换行"
              @keydown.enter.exact.prevent="handleSend"
              :disabled="isLoading"
              resize="none"
            />
            <el-button
              type="primary"
              :icon="Promotion"
              class="send-btn"
              @click="handleSend"
              :loading="isLoading"
              :disabled="!inputText.trim()"
            >发送</el-button>
          </div>
        </div>
      </main>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { Reading, ChatDotRound, Promotion } from '@element-plus/icons-vue'
import ChatMessage from '@/components/ChatMessage.vue'
import UploadPanel from '@/components/UploadPanel.vue'
import DocList from '@/components/DocList.vue'
import { chat, chatStream, healthCheck } from '@/api/rag'
import type { ChatMessage as ChatMessageType, HealthResponse } from '@/types'

const messages = ref<ChatMessageType[]>([])
const inputText = ref('')
const isLoading = ref(false)
const useReranker = ref(true)
const useStream = ref(true)
const activeTab = ref('upload')
const healthStatus = ref<HealthResponse | null>(null)
const scrollRef = ref()
const docListRef = ref()

const exampleQuestions = [
  '这个文档主要讲了什么内容？',
  '请总结文档的核心要点',
  '文档中提到了哪些关键技术？',
  '如何理解文档中的核心概念？'
]

function generateId(): string {
  return Date.now().toString(36) + Math.random().toString(36).substr(2)
}

async function scrollToBottom() {
  await nextTick()
  if (scrollRef.value) {
    const scrollbar = scrollRef.value
    scrollbar.scrollTo({ top: scrollbar.scrollHeight, behavior: 'smooth' })
  }
}

function handleScroll() {
  // 滚动处理
}

async function handleSend() {
  const query = inputText.value.trim()
  if (!query || isLoading.value) return
  await sendQuestion(query)
}

async function sendQuestion(query: string) {
  if (isLoading.value) return

  // 添加用户消息
  const userMsg: ChatMessageType = {
    id: generateId(),
    role: 'user',
    content: query,
    timestamp: Date.now()
  }
  messages.value.push(userMsg)

  // 添加AI加载消息
  const aiMsg: ChatMessageType = {
    id: generateId(),
    role: 'assistant',
    content: '',
    timestamp: Date.now(),
    loading: true
  }
  messages.value.push(aiMsg)

  inputText.value = ''
  isLoading.value = true
  scrollToBottom()

  try {
    if (useStream.value) {
      // 流式输出
      chatStream(
        { query, use_reranker: useReranker.value },
        (token) => {
          aiMsg.content += token
          scrollToBottom()
        },
        (response) => {
          aiMsg.loading = false
          aiMsg.sources = response.sources
          aiMsg.retrieveTime = response.retrieve_time
          aiMsg.generateTime = response.generate_time
          aiMsg.totalTime = response.total_time
          isLoading.value = false
          scrollToBottom()
        },
        (error) => {
          aiMsg.loading = false
          aiMsg.content = `抱歉，发生错误: ${error.message}`
          isLoading.value = false
        }
      )
    } else {
      // 非流式
      const response = await chat({ query, use_reranker: useReranker.value })
      aiMsg.content = response.answer
      aiMsg.loading = false
      aiMsg.sources = response.sources
      aiMsg.retrieveTime = response.retrieve_time
      aiMsg.generateTime = response.generate_time
      aiMsg.totalTime = response.total_time
      isLoading.value = false
      scrollToBottom()
    }
  } catch (error: any) {
    aiMsg.loading = false
    aiMsg.content = `抱歉，发生错误: ${error.message || '未知错误'}`
    isLoading.value = false
    ElMessage.error('请求失败，请检查后端服务是否启动')
  }
}

function handleUploadSuccess() {
  activeTab.value = 'docs'
  nextTick(() => {
    docListRef.value?.loadDocs()
  })
}

async function loadHealth() {
  try {
    healthStatus.value = await healthCheck()
  } catch {
    healthStatus.value = { status: 'unhealthy', vector_count: 0, doc_count: 0, model_settings: {} }
  }
}

onMounted(() => {
  loadHealth()
  setInterval(loadHealth, 30000)
})
</script>

<style scoped>
.app-container {
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: #f5f7fa;
}

.app-header {
  height: 56px;
  background: #fff;
  border-bottom: 1px solid #ebeef5;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  flex-shrink: 0;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 10px;
}

.logo-icon {
  font-size: 24px;
  color: #409eff;
}

.app-title {
  font-size: 18px;
  font-weight: 600;
  color: #303133;
  margin: 0;
}

.header-right {
  display: flex;
  gap: 8px;
}

.main-content {
  flex: 1;
  display: flex;
  overflow: hidden;
}

.sidebar {
  width: 320px;
  background: #fff;
  border-right: 1px solid #ebeef5;
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
}

.sidebar-tabs {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.sidebar-tabs :deep(.el-tabs__content) {
  flex: 1;
  overflow: hidden;
}

.sidebar-tabs :deep(.el-tab-pane) {
  height: 100%;
  overflow: hidden;
}

.chat-area {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.message-list {
  flex: 1;
  overflow: hidden;
}

.welcome {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  padding: 40px;
  text-align: center;
}

.welcome-icon {
  font-size: 64px;
  color: #409eff;
  margin-bottom: 16px;
}

.welcome h2 {
  font-size: 22px;
  color: #303133;
  margin: 0 0 8px 0;
}

.welcome p {
  font-size: 14px;
  color: #909399;
  margin: 0 0 24px 0;
}

.example-questions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  justify-content: center;
  max-width: 600px;
}

.example-btn {
  margin: 0;
}

.input-area {
  background: #fff;
  border-top: 1px solid #ebeef5;
  padding: 12px 20px;
  flex-shrink: 0;
}

.input-options {
  display: flex;
  gap: 16px;
  margin-bottom: 8px;
}

.input-box {
  display: flex;
  gap: 12px;
  align-items: flex-end;
}

.send-btn {
  height: 40px;
  padding: 0 20px;
}
</style>
