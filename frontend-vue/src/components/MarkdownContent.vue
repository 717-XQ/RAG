<template>
  <div class="markdown-content" v-html="rendered"></div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import MarkdownIt from 'markdown-it'
import hljs from 'highlight.js'
import 'highlight.js/styles/github.css'

/**
 * Markdown渲染组件（技术栈2.2：markdown-it + highlight.js）
 * 用于渲染AI回答与报告内容，支持代码高亮
 */
const props = defineProps<{
  content: string
}>()

const md: MarkdownIt = new MarkdownIt({
  html: false, // 不解析原始HTML，防XSS
  linkify: true,
  typographer: true,
  highlight(str: string, lang: string): string {
    if (lang && hljs.getLanguage(lang)) {
      try {
        return `<pre class="hljs"><code>${hljs.highlight(str, { language: lang, ignoreIllegals: true }).value}</code></pre>`
      } catch {
        /* fallthrough */
      }
    }
    return `<pre class="hljs"><code>${md.utils.escapeHtml(str)}</code></pre>`
  }
})

const rendered = computed(() => {
  if (!props.content) return ''
  return md.render(props.content)
})
</script>

<style scoped>
.markdown-content {
  line-height: 1.7;
  word-break: break-word;
  font-size: 14px;
  color: #303133;
}

.markdown-content :deep(h1),
.markdown-content :deep(h2),
.markdown-content :deep(h3) {
  margin: 12px 0 8px;
  color: #1f2937;
}

.markdown-content :deep(p) {
  margin: 6px 0;
}

.markdown-content :deep(ul),
.markdown-content :deep(ol) {
  padding-left: 22px;
  margin: 6px 0;
}

.markdown-content :deep(li) {
  margin: 2px 0;
}

.markdown-content :deep(code) {
  background: #f3f4f6;
  padding: 2px 5px;
  border-radius: 4px;
  font-size: 13px;
  color: #c7254e;
}

.markdown-content :deep(pre) {
  background: #f6f8fa;
  padding: 12px 14px;
  border-radius: 6px;
  overflow-x: auto;
  margin: 8px 0;
}

.markdown-content :deep(pre code) {
  background: transparent;
  padding: 0;
  color: #24292e;
}

.markdown-content :deep(table) {
  border-collapse: collapse;
  margin: 8px 0;
  width: 100%;
}

.markdown-content :deep(th),
.markdown-content :deep(td) {
  border: 1px solid #e5e7eb;
  padding: 6px 10px;
  font-size: 13px;
}

.markdown-content :deep(th) {
  background: #f9fafb;
  font-weight: 600;
}

.markdown-content :deep(blockquote) {
  border-left: 3px solid #d1d5db;
  padding-left: 12px;
  color: #6b7280;
  margin: 8px 0;
}

.markdown-content :deep(a) {
  color: #409eff;
}
</style>
