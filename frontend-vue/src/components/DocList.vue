<template>
  <div class="doc-list">
    <div class="list-header">
      <span class="title">知识库文档 ({{ docs.length }})</span>
      <el-button size="small" :icon="Refresh" @click="loadDocs" :loading="loading">刷新</el-button>
    </div>
    <el-scrollbar class="doc-scroll">
      <div v-if="loading" class="loading">
        <el-icon class="is-loading"><Loading /></el-icon>
        <span>加载中...</span>
      </div>
      <div v-else-if="docs.length === 0" class="empty">
        <el-icon><FolderOpened /></el-icon>
        <span>暂无文档，请先上传</span>
      </div>
      <div v-else class="doc-items">
        <div v-for="doc in docs" :key="doc.filename" class="doc-item">
          <div class="doc-info">
            <el-icon class="doc-icon"><Document /></el-icon>
            <div class="doc-meta">
              <span class="doc-name" :title="doc.filename">{{ doc.filename }}</span>
              <span class="doc-detail">{{ doc.chunk_count }} 块 · {{ doc.file_type }} · {{ formatTime(doc.upload_time) }}</span>
            </div>
          </div>
          <el-button
            type="danger"
            size="small"
            text
            :icon="Delete"
            @click="handleDelete(doc.filename)"
          >删除</el-button>
        </div>
      </div>
    </el-scrollbar>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Refresh, Delete, Document, FolderOpened, Loading } from '@element-plus/icons-vue'
import { listDocuments, deleteDocument } from '@/api/rag'
import type { DocInfo } from '@/types'

const docs = ref<DocInfo[]>([])
const loading = ref(false)

async function loadDocs() {
  loading.value = true
  try {
    docs.value = await listDocuments()
  } catch (error: any) {
    ElMessage.error(`加载文档列表失败: ${error.message}`)
  } finally {
    loading.value = false
  }
}

async function handleDelete(filename: string) {
  try {
    await ElMessageBox.confirm(`确定要删除文档 "${filename}" 吗？`, '确认删除', {
      type: 'warning'
    })
    await deleteDocument(filename)
    ElMessage.success('删除成功')
    loadDocs()
  } catch {
    // 用户取消
  }
}

function formatTime(timeStr: string): string {
  if (!timeStr) return ''
  const date = new Date(timeStr)
  if (isNaN(date.getTime())) return timeStr
  return date.toLocaleDateString('zh-CN')
}

onMounted(loadDocs)
defineExpose({ loadDocs })
</script>

<style scoped>
.doc-list {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.list-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  border-bottom: 1px solid #ebeef5;
}

.title {
  font-size: 14px;
  font-weight: 600;
  color: #303133;
}

.doc-scroll {
  flex: 1;
  overflow: hidden;
}

.loading, .empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px 20px;
  color: #909399;
  font-size: 13px;
  gap: 8px;
}

.doc-items {
  padding: 8px;
}

.doc-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 12px;
  border-radius: 6px;
  margin-bottom: 4px;
  transition: background 0.2s;
}

.doc-item:hover {
  background: #f5f7fa;
}

.doc-info {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
  flex: 1;
}

.doc-icon {
  color: #409eff;
  font-size: 20px;
  flex-shrink: 0;
}

.doc-meta {
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.doc-name {
  font-size: 13px;
  color: #303133;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.doc-detail {
  font-size: 11px;
  color: #909399;
  margin-top: 2px;
}
</style>
