import type { Comparison } from './comparison'

export type AppState = {
  currentComparison: Comparison | null
  history: Comparison[]   // persisted to localStorage
  sliderValue: number
  loading: boolean
  api_error: string | null
}
