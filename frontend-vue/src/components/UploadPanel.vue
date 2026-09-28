<template>
  <div class="upload-panel">
    <el-upload
      drag
      :auto-upload="false"
      :on-change="handleFileChange"
      :file-list="fileList"
      accept=".pdf,.txt,.md,.docx,.doc"
      multiple
    >
      <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
      <div class="el-upload__text">
        拖拽文件到此处，或<em>点击上传</em>
      </div>
      <template #tip>
        <div class="el-upload__tip">
          支持 PDF / TXT / Markdown / Word 文档，单个文件不超过 50MB
        </div>
      </template>
    </el-upload>
    <div v-if="uploading" class="upload-progress">
      <el-progress :percentage="progress" :status="progress === 100 ? 'success' : ''" />
      <span class="upload-text">{{ currentFile }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import { uploadDocument } from '@/api/rag'
import type { UploadFile } from 'element-plus'

const emit = defineEmits<{
  (e: 'upload-success'): void
}>()

const fileList = ref<UploadFile[]>([])
const uploading = ref(false)
const progress = ref(0)
const currentFile = ref('')

async function handleFileChange(file: UploadFile) {
  if (!file.raw) return
  uploading.value = true
  progress.value = 0
  currentFile.value = file.name

  // 模拟进度
  const progressTimer = setInterval(() => {
    if (progress.value < 90) {
      progress.value += Math.random() * 15
    }
  }, 200)

  try {
    const result = await uploadDocument(file.raw)
    progress.value = 100
    clearInterval(progressTimer)
    ElMessage.success(`${result.filename} 上传成功，新增 ${result.chunks_added} 个文档块`)
    emit('upload-success')
  } catch (error: any) {
    clearInterval(progressTimer)
    ElMessage.error(`上传失败: ${error.message || '未知错误'}`)
  } finally {
    setTimeout(() => {
      uploading.value = false
      progress.value = 0
      currentFile.value = ''
      fileList.value = []
    }, 1500)
  }
}
</script>

<style scoped>
.upload-panel {
  padding: 16px;
}

.upload-progress {
  margin-top: 16px;
}

.upload-text {
  display: block;
  margin-top: 8px;
  font-size: 13px;
  color: #606266;
}
</style>
