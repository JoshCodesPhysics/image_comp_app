import type { Comparison } from './comparison'

export type HistoryListProps = {
  history: Comparison[]
  onSelect: (c: Comparison) => void
  onClear: () => void
}