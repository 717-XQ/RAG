<template>
  <div class="chat-message" :class="message.role">
    <div class="avatar">
      <el-icon v-if="message.role === 'user'" size="20"><User /></el-icon>
      <el-icon v-else size="20"><ChatDotRound /></el-icon>
    </div>
    <div class="message-body">
      <div class="message-header">
        <span class="role-name">{{ message.role === 'user' ? '我' : 'RAG 助手' }}</span>
        <span class="time">{{ formatTime(message.timestamp) }}</span>
      </div>
      <div class="message-content">
        <p v-if="message.loading" class="loading-text">
          <el-icon class="is-loading"><Loading /></el-icon>
          正在检索文档并生成回答...
        </p>
        <template v-else>
          <p class="answer-text">{{ message.content }}</p>
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
          </div>
        </template>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { ChatMessage } from '@/types'
import SourceCard from './SourceCard.vue'

defineProps<{
  message: ChatMessage
}>()

function formatTime(timestamp: number): string {
  const date = new Date(timestamp)
  return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
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

.answer-text {
  margin: 0;
  font-size: 14px;
  line-height: 1.7;
  color: #303133;
  white-space: pre-wrap;
  word-break: break-word;
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
}
</style>
