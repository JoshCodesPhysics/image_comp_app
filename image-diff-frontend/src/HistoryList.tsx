import type { HistoryListProps } from './types/history_list'

export default function HistoryList({ history, onSelect, onClear }: HistoryListProps) {
  if (history.length === 0) return null

  return (
    <div className="mb-6">
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-lg font-semibold">History</h2>
        <button
          onClick={onClear}
          className="text-sm text-red-600 hover:underline"
        >
          Clear History
        </button>
      </div>

      <ul className="space-y-2">
        {history.map(c => (
          <li key={c.id}>
            <button
              onClick={() => onSelect(c)}
              className="text-blue-600 hover:underline"
            >
              {new Date(c.createdAt).toLocaleString()}
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}