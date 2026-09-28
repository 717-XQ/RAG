<template>
  <div class="source-card">
    <div class="source-header" @click="expanded = !expanded">
      <el-icon class="expand-icon">
        <CaretBottom v-if="expanded" />
        <CaretRight v-else />
      </el-icon>
      <span class="source-title">{{ source.filename || source.doc_id || '未知文档' }}</span>
      <el-tag v-if="source.score !== undefined" size="small" type="info" class="score-tag">
        相似度 {{ (source.score * 100).toFixed(1) }}%
      </el-tag>
      <el-tag v-if="source.page !== undefined" size="small" class="page-tag">
        第 {{ source.page }} 页
      </el-tag>
    </div>
    <div v-show="expanded" class="source-content">
      <p>{{ source.content }}</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import type { SourceInfo } from '@/types'

defineProps<{
  source: SourceInfo
}>()

const expanded = ref(false)
</script>

<style scoped>
.source-card {
  border: 1px solid #e4e7ed;
  border-radius: 6px;
  margin-bottom: 8px;
  overflow: hidden;
  background: #fff;
}

.source-header {
  display: flex;
  align-items: center;
  padding: 8px 12px;
  background: #f5f7fa;
  cursor: pointer;
  user-select: none;
}

.expand-icon {
  margin-right: 6px;
  color: #909399;
}

.source-title {
  flex: 1;
  font-size: 13px;
  color: #303133;
  font-weight: 500;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.score-tag {
  margin-left: 8px;
}

.page-tag {
  margin-left: 4px;
}

.source-content {
  padding: 12px;
  font-size: 13px;
  color: #606266;
  line-height: 1.6;
  max-height: 200px;
  overflow-y: auto;
}

.source-content p {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
}
</style>
