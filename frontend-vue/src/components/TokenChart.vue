<template>
  <div class="token-chart">
    <div class="chart-header">
      <div class="chart-title">
        <el-icon><DataLine /></el-icon>
        <span>Token 用量统计（近 {{ days }} 天）</span>
      </div>
      <el-radio-group v-model="days" size="small" @change="loadData">
        <el-radio-button :value="7">7天</el-radio-button>
        <el-radio-button :value="14">14天</el-radio-button>
        <el-radio-button :value="30">30天</el-radio-button>
      </el-radio-group>
    </div>
    <div ref="chartRef" class="chart-body"></div>
    <div v-if="!loading && !hasData" class="chart-empty">暂无 Token 用量数据，先发起一次问答试试</div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount, nextTick } from 'vue'
import * as echarts from 'echarts'
import { DataLine } from '@element-plus/icons-vue'
import { getTokenUsage } from '@/api/rag'
import type { TokenUsageStats } from '@/types'

/**
 * Token用量统计图表（技术栈2.2：ECharts 5.5+）
 * 数据源：GET /stats/token-usage
 */
const days = ref(7)
const chartRef = ref<HTMLDivElement>()
const loading = ref(false)
const hasData = ref(false)
let chart: echarts.ECharts | null = null
let resizeObserver: ResizeObserver | null = null

async function loadData() {
  loading.value = true
  try {
    const stats: TokenUsageStats = await getTokenUsage(days.value)
    hasData.value = stats.data.some((p) => p.total_tokens > 0)
    await nextTick()
    renderChart(stats)
  } catch (e) {
    hasData.value = false
  } finally {
    loading.value = false
  }
}

function renderChart(stats: TokenUsageStats) {
  if (!chartRef.value) return
  if (!chart) {
    chart = echarts.init(chartRef.value)
    window.addEventListener('resize', handleResize)

    // 关键修复：容器尺寸变化（tab切换/窗口缩放/布局完成）时自动resize，
    // 避免刷新时容器宽度为0导致图表错乱
    if (typeof ResizeObserver !== 'undefined') {
      resizeObserver = new ResizeObserver(() => {
        chart?.resize()
      })
      resizeObserver.observe(chartRef.value)
    }
    // 延迟resize：确保DOM布局完成后按真实尺寸重算（刷新场景的关键）
    setTimeout(() => chart?.resize(), 120)
  }

  const dates = stats.data.map((p) => p.date)
  const input = stats.data.map((p) => p.input_tokens)
  const output = stats.data.map((p) => p.output_tokens)
  const total = stats.data.map((p) => p.total_tokens)
  const calls = stats.data.map((p) => p.calls)

  // 30天时标签密集，间隔显示（每3个1个），7/14天全部显示；统一45°倾斜避免遮挡
  const labelInterval = days.value >= 30 ? 2 : 0

  chart.setOption({
    color: ['#409eff', '#67c23a', '#e6a23c', '#909399'],
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      backgroundColor: 'rgba(255,255,255,0.97)',
      borderColor: '#e4e7ed',
      borderWidth: 1,
      padding: [10, 14],
      textStyle: { fontSize: 12, color: '#303133' },
      formatter: (params: any[]) => {
        const date = params[0]?.axisValue ?? ''
        let html = `<div style="font-weight:600;margin-bottom:6px;font-size:12px">${date}</div>`
        params.forEach((p) => {
          html += `<div style="display:flex;align-items:center;gap:6px;margin:3px 0;font-size:12px">
            <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${p.color}"></span>
            <span style="color:#606266">${p.seriesName}</span>
            <b style="margin-left:auto;padding-left:12px">${p.value}</b>
          </div>`
        })
        return html
      }
    },
    legend: {
      data: ['输入Token', '输出Token', '总Token', '调用次数'],
      top: 0,
      right: 4,
      icon: 'circle',
      itemWidth: 10,
      itemHeight: 10,
      itemGap: 18,
      textStyle: { fontSize: 12, color: '#606266' }
    },
    grid: { left: 58, right: 62, top: 48, bottom: 88 },
    xAxis: {
      type: 'category',
      data: dates,
      axisTick: { alignWithLabel: true, lineStyle: { color: '#dcdfe6' } },
      axisLine: { lineStyle: { color: '#dcdfe6' } },
      axisLabel: {
        fontSize: 11,
        color: '#606266',
        rotate: 45,                    // 统一45°倾斜，避免互相遮挡
        interval: labelInterval,
        margin: 14,
        hideOverlap: true
      }
    },
    yAxis: [
      {
        type: 'value',
        name: 'Token',
        nameTextStyle: { fontSize: 12, color: '#909399', padding: [0, 0, 0, -4] },
        axisLabel: { fontSize: 11, color: '#909399' },
        axisLine: { show: false },
        axisTick: { show: false },
        splitLine: { lineStyle: { color: '#f0f2f5', type: 'dashed' } }
      },
      {
        type: 'value',
        name: '调用次数',
        min: 0,
        minInterval: 1,                // 刻度取整，避免 0.5/2.5 小数刻度
        nameTextStyle: { fontSize: 12, color: '#909399', padding: [0, 0, 0, 4] },
        axisLabel: { fontSize: 11, color: '#909399' },
        axisLine: { show: false },
        axisTick: { show: false },
        splitLine: { show: false }
      }
    ],
    series: [
      {
        name: '输入Token',
        type: 'line',
        smooth: true,
        symbol: 'circle',
        symbolSize: 7,
        lineStyle: { width: 2.5 },
        data: input,
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(64,158,255,0.28)' },
            { offset: 1, color: 'rgba(64,158,255,0.02)' }
          ])
        },
        itemStyle: { color: '#409eff' }
      },
      {
        name: '输出Token',
        type: 'line',
        smooth: true,
        symbol: 'circle',
        symbolSize: 7,
        lineStyle: { width: 2.5 },
        data: output,
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(103,194,58,0.28)' },
            { offset: 1, color: 'rgba(103,194,58,0.02)' }
          ])
        },
        itemStyle: { color: '#67c23a' }
      },
      {
        name: '总Token',
        type: 'line',
        smooth: true,
        symbol: 'circle',
        symbolSize: 7,
        lineStyle: { type: 'dashed', width: 2 },
        data: total,
        itemStyle: { color: '#e6a23c' }
      },
      {
        name: '调用次数',
        type: 'bar',
        yAxisIndex: 1,
        data: calls,
        barWidth: '36%',
        itemStyle: {
          color: '#909399',
          opacity: 0.55,
          borderRadius: [4, 4, 0, 0]
        }
      }
    ]
  })
}

function handleResize() {
  chart?.resize()
}

onMounted(loadData)
onBeforeUnmount(() => {
  window.removeEventListener('resize', handleResize)
  resizeObserver?.disconnect()
  resizeObserver = null
  chart?.dispose()
  chart = null
})
</script>

<style scoped>
.token-chart {
  padding: 6px 10px;
}

.chart-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 14px;
  flex-wrap: wrap;
  gap: 8px;
}

.chart-title {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 14px;
  font-weight: 600;
  color: #303133;
}

.chart-title :deep(.el-icon) {
  color: #409eff;
  font-size: 16px;
}

.chart-body {
  height: 300px;
  width: 100%;
}

.chart-empty {
  text-align: center;
  color: #909399;
  font-size: 13px;
  padding: 30px 0;
}
</style>
