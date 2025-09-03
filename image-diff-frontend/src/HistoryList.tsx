import type { HistoryListProps } from './types/history_list'

export default function HistoryList({ history, onSelect }: HistoryListProps) {
  if (history.length === 0) return null

  // Text-lg sets font size to 1.125 rem, space-y-2 adds 0.5 rem top margin to all child elements except the first, hover:underline adds a blue underline on hover
  return (
    <div className="mb-6">
      <h2 className="text-lg font-semibold mb-2">History</h2>
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