import { createRouter, createWebHashHistory } from 'vue-router'
import { getAccessToken } from '@/api/rag'

/**
 * 路由配置（技术栈2.2：Vue Router 4.3+）
 */
const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    {
      path: '/login',
      name: 'login',
      component: () => import('@/views/LoginView.vue'),
      meta: { title: '登录', public: true }
    },
    {
      path: '/',
      name: 'home',
      component: () => import('@/views/HomeView.vue'),
      meta: { title: 'RAG 智能文档问答系统', requiresAuth: true }
    },
    {
      path: '/:pathMatch(.*)*',
      redirect: '/'
    }
  ]
})

// 路由守卫：未登录跳转登录页
router.beforeEach((to) => {
  const hasToken = !!getAccessToken()
  if (to.meta.requiresAuth && !hasToken) {
    return { name: 'login', query: { redirect: to.fullPath } }
  }
  if (to.name === 'login' && hasToken) {
    return { name: 'home' }
  }
  return true
})

export default router
