/**
 * TriggerWordPanel: set, validate, and bulk-apply the project trigger word.
 *
 * Displays inline in the Dataset tab header area. Shows trigger input,
 * validation status, caption audit count, and bulk-prepend action.
 */

import { useState, useEffect } from 'react'
import type { TriggerValidationResult } from '../../../types/dataset'

interface TriggerWordPanelProps {
  trigger: string | null
  datasetDir: string
  clipCount: number
  triggerPresentCount: number
  captionedCount: number
  onTriggerChange: (trigger: string) => void
  onValidateTrigger: (trigger: string) => Promise<TriggerValidationResult | null>
  onPrependTrigger: (datasetDir: string, trigger: string) => Promise<number>
  onRecaptionMissing?: () => void
  onRefresh: () => void
}

export function TriggerWordPanel({
  trigger,
  datasetDir,
  clipCount,
  triggerPresentCount,
  captionedCount,
  onTriggerChange,
  onValidateTrigger,
  onPrependTrigger,
  onRecaptionMissing,
  onRefresh,
}: TriggerWordPanelProps) {
  const [inputValue, setInputValue] = useState(trigger ?? '')
  const [validationResult, setValidationResult] = useState<TriggerValidationResult | null>(null)
  const [isPrepending, setIsPrepending] = useState(false)
  const [lastPrependCount, setLastPrependCount] = useState<number | null>(null)

  // Sync input with prop changes.
  useEffect(() => {
    setInputValue(trigger ?? '')
  }, [trigger])

  const handleApply = async () => {
    const trimmed = inputValue.trim()
    if (!trimmed) return

    const result = await onValidateTrigger(trimmed)
    setValidationResult(result)

    if (result?.valid) {
      onTriggerChange(trimmed)
      setLastPrependCount(null)
      onRefresh()
    }
  }

  const handlePrepend = async () => {
    const trimmed = inputValue.trim()
    if (!trimmed || !datasetDir) return

    setIsPrepending(true)
    try {
      const count = await onPrependTrigger(datasetDir, trimmed)
      setLastPrependCount(count)
      onRefresh()
    } finally {
      setIsPrepending(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      void handleApply()
    }
  }

  const missingCount = captionedCount - triggerPresentCount
  const hasTrigger = Boolean(trigger)

  return (
    <div className="px-4 py-3 border-b border-zinc-800 bg-zinc-900/50">
      <div className="flex items-center gap-3 flex-wrap">
        {/* Label + input */}
        <label className="text-xs font-medium text-zinc-400 whitespace-nowrap">
          Trigger Word:
        </label>
        <input
          type="text"
          value={inputValue}
          onChange={e => {
            setInputValue(e.target.value)
            setValidationResult(null)
            setLastPrependCount(null)
          }}
          onKeyDown={handleKeyDown}
          placeholder="e.g. Enid, my_subject"
          className="px-2 py-1 text-sm bg-zinc-800 border border-zinc-700 rounded text-zinc-200 w-48 focus:outline-none focus:border-blue-500"
        />
        <button
          onClick={() => void handleApply()}
          disabled={!inputValue.trim()}
          className="px-3 py-1 text-xs font-medium bg-blue-600 hover:bg-blue-700 disabled:bg-zinc-700 disabled:text-zinc-500 text-white rounded transition-colors"
        >
          Apply
        </button>

        {/* Caption audit count */}
        {hasTrigger && clipCount > 0 && (
          <span className="text-xs text-zinc-500">
            captions: {triggerPresentCount}/{captionedCount} contain trigger
          </span>
        )}

        {/* Bulk prepend button */}
        {hasTrigger && missingCount > 0 && (
          <button
            onClick={() => void handlePrepend()}
            disabled={isPrepending}
            className="px-3 py-1 text-xs font-medium bg-amber-600 hover:bg-amber-700 disabled:bg-zinc-700 disabled:text-zinc-500 text-white rounded transition-colors"
          >
            {isPrepending ? 'Prepending...' : `Prepend to ${missingCount} missing`}
          </button>
        )}

        {/* Re-caption clips missing trigger */}
        {hasTrigger && missingCount > 0 && onRecaptionMissing && (
          <button
            onClick={onRecaptionMissing}
            className="px-3 py-1 text-xs font-medium bg-purple-600 hover:bg-purple-700 text-white rounded transition-colors"
          >
            Re-caption {missingCount} missing
          </button>
        )}

        {/* Prepend result */}
        {lastPrependCount !== null && (
          <span className="text-xs text-green-400">
            Updated {lastPrependCount} caption{lastPrependCount !== 1 ? 's' : ''}
          </span>
        )}
      </div>

      {/* Validation feedback */}
      {validationResult && (
        <div className="mt-2">
          {validationResult.error && (
            <p className="text-xs text-red-400">{validationResult.error}</p>
          )}
          {validationResult.warning && (
            <p className="text-xs text-amber-400">{validationResult.warning}</p>
          )}
        </div>
      )}
    </div>
  )
}
