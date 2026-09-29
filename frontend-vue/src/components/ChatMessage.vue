<template>
  <div class="chat-message" :class="message.role">
    <div class="avatar">
      <el-icon v-if="message.role === 'user'" size="20"><User /></el-icon>
      <el-icon v-else size="20"><ChatDotRound /></el-icon>
    </div>
    <div class="message-body">
      <div class="message-header">
        <span class="role-name">{{ message.role === 'user' ? '我' : 'RAG 助手' }}</span>
        <el-tag v-if="message.provider" size="small" type="info" effect="plain">{{ message.provider }}</el-tag>
        <span class="time">{{ formatTime(message.timestamp) }}</span>
      </div>
      <div class="message-content">
        <p v-if="message.loading" class="loading-text">
          <el-icon class="is-loading"><Loading /></el-icon>
          正在检索文档并生成回答...
        </p>
        <template v-else>
          <!-- 技术栈2.2：markdown-it + highlight.js 渲染回答 -->
          <MarkdownContent v-if="message.content" :content="message.content" />
          <div v-if="message.sources && message.sources.length > 0" class="sources-section">
            <div class="sources-title">
              <el-icon><Document /></el-icon>
              <span>引用来源 ({{ message.sources.length }})</span>
            </div>
            <SourceCard
              v-for="(source, index) in message.sources"
              :key="index"
              :source="source"
            />
          </div>
          <div v-if="message.totalTime" class="meta-info">
            <el-tag size="small" type="success">检索 {{ message.retrieveTime?.toFixed(2) }}s</el-tag>
            <el-tag size="small" type="primary">生成 {{ message.generateTime?.toFixed(2) }}s</el-tag>
            <el-tag size="small" type="info">总计 {{ message.totalTime.toFixed(2) }}s</el-tag>
            <el-tag v-if="message.tokenUsage" size="small" type="warning">
              输入 {{ message.tokenUsage.prompt_tokens }} / 输出 {{ message.tokenUsage.completion_tokens }}
            </el-tag>
            <!-- 技术栈2.2：WeasyPrint PDF / python-docx 报告导出 -->
            <el-tooltip content="导出本次问答为报告">
              <el-button-group size="small" class="export-group">
                <el-button size="small" type="primary" plain :icon="Document" @click="handleExport('pdf')">PDF</el-button>
                <el-button size="small" type="success" plain :icon="DocumentCopy" @click="handleExport('docx')">DOCX</el-button>
              </el-button-group>
            </el-tooltip>
          </div>
        </template>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { ChatMessage } from '@/types'
import SourceCard from './SourceCard.vue'
import MarkdownContent from './MarkdownContent.vue'
import { exportReport } from '@/api/rag'
import { ElMessage } from 'element-plus'
import { Document, DocumentCopy } from '@element-plus/icons-vue'

const props = defineProps<{
  message: ChatMessage
  question?: string
}>()

function formatTime(timestamp: number): string {
  const date = new Date(timestamp)
  return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
}

async function handleExport(format: 'pdf' | 'docx') {
  if (!props.message.content) {
    ElMessage.warning('回答为空，无法导出')
    return
  }
  try {
    const blob = await exportReport(
      {
        question: props.question || 'RAG 问答',
        answer: props.message.content,
        sources: (props.message.sources || []).map((s) => ({
          filename: s.filename,
          page: s.page,
          score: s.score,
          content_preview: s.content_preview || s.content
        })),
        title: `RAG 问答报告 - ${new Date(props.message.timestamp).toLocaleString('zh-CN')}`
      },
      format
    )
    // 触发浏览器下载
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `rag_report_${Date.now()}.${format}`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
    ElMessage.success(`报告已导出（${format.toUpperCase()}）`)
  } catch (e: any) {
    ElMessage.error(e.response?.data?.detail || `${format.toUpperCase()} 导出失败`)
  }
}
</script>

<style scoped>
.chat-message {
  display: flex;
  padding: 16px;
  gap: 12px;
}

.chat-message.user {
  flex-direction: row-reverse;
}

.avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.chat-message.user .avatar {
  background: #409eff;
  color: #fff;
}

.chat-message.assistant .avatar {
  background: #67c23a;
  color: #fff;
}

.message-body {
  max-width: 75%;
  min-width: 0;
}

.chat-message.user .message-body {
  text-align: right;
}

.message-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}

.chat-message.user .message-header {
  justify-content: flex-end;
}

.role-name {
  font-size: 13px;
  font-weight: 600;
  color: #303133;
}

.time {
  font-size: 12px;
  color: #909399;
}

.message-content {
  background: #fff;
  border-radius: 12px;
  padding: 12px 16px;
  text-align: left;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}

.chat-message.user .message-content {
  background: #ecf5ff;
}

.loading-text {
  margin: 0;
  font-size: 14px;
  color: #909399;
  display: flex;
  align-items: center;
  gap: 6px;
}

.sources-section {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid #ebeef5;
}

.sources-title {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  font-weight: 600;
  color: #606266;
  margin-bottom: 8px;
}

.meta-info {
  margin-top: 10px;
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  align-items: center;
}

.export-group {
  margin-left: 8px;
}
</style>
