import { defineStore } from 'pinia'
import { ref } from 'vue'
import { login as apiLogin, register as apiRegister, getMe, setTokens, clearTokens, getAccessToken } from '@/api/rag'
import type { UserInfo, LoginRequest, RegisterRequest } from '@/types'

/**
 * 认证Store（技术栈2.2：Pinia 状态管理 + JWT）
 */
export const useAuthStore = defineStore('auth', () => {
  const user = ref<UserInfo | null>(null)
  const token = ref<string>(getAccessToken())
  const loading = ref(false)

  async function login(request: LoginRequest) {
    loading.value = true
    try {
      const resp = await apiLogin(request)
      setTokens(resp.access_token, resp.refresh_token)
      token.value = resp.access_token
      await fetchMe()
    } finally {
      loading.value = false
    }
  }

  async function register(request: RegisterRequest) {
    loading.value = true
    try {
      const resp = await apiRegister(request)
      setTokens(resp.access_token, resp.refresh_token)
      token.value = resp.access_token
      await fetchMe()
    } finally {
      loading.value = false
    }
  }

  async function fetchMe() {
    try {
      user.value = await getMe()
    } catch {
      // Token无效时静默（拦截器会跳登录）
      user.value = null
    }
  }

  function logout() {
    clearTokens()
    token.value = ''
    user.value = null
  }

  return { user, token, loading, login, register, fetchMe, logout }
})
