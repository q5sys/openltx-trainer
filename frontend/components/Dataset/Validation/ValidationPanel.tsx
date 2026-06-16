/**
 * ValidationPanel: displays dataset validation errors, warnings, and stats.
 *
 * Shown inline in the Dataset tab. Errors block training. Warnings are
 * informational. Stats provide a quick summary of dataset health.
 */

import type { DatasetValidationResult } from '../../../types/dataset'

interface ValidationPanelProps {
  validation: DatasetValidationResult | null
  isLoading: boolean
  onValidate: () => void
}

export function ValidationPanel({ validation, isLoading, onValidate }: ValidationPanelProps) {
  return (
    <div className="px-4 py-3 border-b border-zinc-800 bg-zinc-900/30">
      <div className="flex items-center gap-3 mb-2">
        <h4 className="text-xs font-medium text-zinc-400">Dataset Validation</h4>
        <button
          onClick={onValidate}
          disabled={isLoading}
          className="px-3 py-1 text-xs font-medium bg-zinc-700 hover:bg-zinc-600 disabled:bg-zinc-800 disabled:text-zinc-600 text-zinc-200 rounded transition-colors"
        >
          {isLoading ? 'Validating...' : 'Validate'}
        </button>

        {validation && (
          <span className={`text-xs font-medium ${validation.valid ? 'text-green-400' : 'text-red-400'}`}>
            {validation.valid ? 'Ready for training' : 'Not ready'}
          </span>
        )}
      </div>

      {validation && (
        <div className="space-y-2">
          {/* Stats summary */}
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-zinc-500">
            <span>Clips: {validation.stats.clip_count}</span>
            {validation.stats.image_count > 0 && (
              <span>Images: {validation.stats.image_count}</span>
            )}
            <span>Captioned: {validation.stats.captioned}/{validation.stats.clip_count}</span>
            <span>Trigger: {validation.stats.trigger_present}/{validation.stats.captioned}</span>
            <span>Audio: {validation.stats.with_audio}/{validation.stats.clip_count - validation.stats.image_count}</span>
            <span>Duration: {validation.stats.total_duration_s.toFixed(1)}s</span>
          </div>

          {/* Errors */}
          {validation.errors.length > 0 && (
            <div className="space-y-1">
              <h5 className="text-xs font-medium text-red-400">
                Errors ({validation.errors.length})
              </h5>
              <ul className="space-y-0.5">
                {validation.errors.map((issue, i) => (
                  <li key={`err-${i}`} className="text-xs text-red-300 pl-2 border-l-2 border-red-800">
                    <span className="font-mono text-red-500 mr-1">{issue.code}</span>
                    {issue.msg}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Warnings */}
          {validation.warnings.length > 0 && (
            <div className="space-y-1">
              <h5 className="text-xs font-medium text-amber-400">
                Warnings ({validation.warnings.length})
              </h5>
              <ul className="space-y-0.5">
                {validation.warnings.map((issue, i) => (
                  <li key={`warn-${i}`} className="text-xs text-amber-300 pl-2 border-l-2 border-amber-800">
                    <span className="font-mono text-amber-500 mr-1">{issue.code}</span>
                    {issue.msg}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* All clear */}
          {validation.errors.length === 0 && validation.warnings.length === 0 && (
            <p className="text-xs text-green-400">No issues found. Dataset is ready for training.</p>
          )}
        </div>
      )}
    </div>
  )
}
