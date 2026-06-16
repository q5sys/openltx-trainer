/**
 * Raw training log viewer.
 *
 * Displays progress records as formatted log lines with auto-scroll.
 */

import { useEffect, useRef, useState } from 'react'
import { ChevronDown, ChevronUp } from 'lucide-react'
import { Button } from '../ui/button'
import type { ProgressRecord } from '../../hooks/useTrainingProgress'

interface LogTailProps {
  records: ProgressRecord[]
}

function formatLogLine(record: ProgressRecord): string {
  const time = new Date(record.ts * 1000).toLocaleTimeString()
  const parts = [
    `[${time}]`,
    `step=${record.step}`,
    `epoch=${record.epoch}`,
    `loss=${record.loss.toFixed(4)}`,
    `lr=${record.lr.toExponential(2)}`,
    `grad_norm=${record.grad_norm.toFixed(3)}`,
    `ips=${record.ips.toFixed(1)}`,
  ]
  if (record.phase) parts.push(`phase=${record.phase}`)
  if (record.paused) parts.push('PAUSED')
  if (record.cancelled) parts.push('CANCELLED')
  return parts.join(' | ')
}

export function LogTail({ records }: LogTailProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [autoScroll, setAutoScroll] = useState(true)

  useEffect(() => {
    if (autoScroll && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight
    }
  }, [records, autoScroll])

  return (
    <div className="bg-zinc-900 rounded-lg border border-zinc-800 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-zinc-800">
        <span className="text-xs font-medium text-zinc-400">Training Log</span>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setAutoScroll(!autoScroll)}
          className={`h-6 w-6 p-0 ${autoScroll ? 'text-blue-400' : 'text-zinc-500'}`}
          title={autoScroll ? 'Auto-scroll enabled' : 'Auto-scroll disabled'}
        >
          {autoScroll ? <ChevronDown className="h-3 w-3" /> : <ChevronUp className="h-3 w-3" />}
        </Button>
      </div>

      {/* Log content */}
      <div ref={containerRef} className="flex-1 overflow-auto p-3 font-mono text-xs max-h-48 bg-black/50">
        {records.length === 0 ? (
          <span className="text-zinc-600">Waiting for training output...</span>
        ) : (
          records.map(record => (
            <div
              key={record.step}
              className={`whitespace-pre-wrap ${
                record.cancelled
                  ? 'text-red-400'
                  : record.paused
                    ? 'text-yellow-400'
                    : 'text-zinc-400'
              }`}
            >
              {formatLogLine(record)}
            </div>
          ))
        )}
      </div>

      {/* Footer */}
      <div className="px-4 py-1.5 border-t border-zinc-800 text-xs text-zinc-600">
        {records.length} entries
      </div>
    </div>
  )
}
