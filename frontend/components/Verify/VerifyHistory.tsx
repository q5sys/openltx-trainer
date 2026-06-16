import type { VerificationHistoryEntry } from '../../hooks/useVerification'

interface VerifyHistoryProps {
  entries: VerificationHistoryEntry[]
  onSelect: (entry: VerificationHistoryEntry) => void
}

export function VerifyHistory({ entries, onSelect }: VerifyHistoryProps) {
  if (entries.length === 0) {
    return (
      <div className="text-xs text-zinc-600 py-2">
        No verification history yet.
      </div>
    )
  }

  return (
    <div className="space-y-1">
      <label className="text-xs font-medium text-zinc-400 uppercase tracking-wide">History</label>
      <div className="flex gap-2 overflow-x-auto pb-2">
        {entries.map(entry => (
          <button
            key={entry.generation_id}
            onClick={() => onSelect(entry)}
            className="flex-shrink-0 w-24 bg-zinc-900 rounded border border-zinc-800 hover:border-zinc-600 overflow-hidden transition-colors"
          >
            <div className="aspect-video bg-zinc-950 flex items-center justify-center">
              {entry.output_path ? (
                <video
                  src={`file://${entry.output_path}`}
                  muted
                  className="w-full h-full object-cover"
                />
              ) : (
                <span className="text-zinc-700 text-xs">--</span>
              )}
            </div>
            <div className="p-1">
              <p className="text-[10px] text-zinc-500 truncate">{entry.prompt}</p>
              <p className="text-[10px] text-zinc-600">seed {entry.seed}</p>
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}
