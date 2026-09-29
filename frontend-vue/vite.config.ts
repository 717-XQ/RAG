import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath, URL } from 'node:url'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  },
  server: {
    host: '127.0.0.1',      // 强制 IPv4：避免 Vite 只监听 ::1 导致与后端(IPv4)无法互通
    port: 5173,
    proxy: {
      // 业务API（前端以 /api 开头，代理时去掉前缀）
      '/api': {
        target: 'http://127.0.0.1:8000',   // 用 IPv4 避免 localhost 解析为 ::1 导致 ECONNREFUSED
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      },
      // 认证API（/auth/login 等直接转发，无需 rewrite）
      '/auth': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true
      }
    }
  }
})
