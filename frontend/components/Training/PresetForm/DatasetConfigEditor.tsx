/**
 * DatasetConfigEditor: controls for dataset-related training parameters.
 *
 * Edits target_frames, target_resolution, and auto_repeats.
 * Character mode defaults differ from concept mode defaults.
 */

import { type ChangeEvent } from 'react'

export interface DatasetConfigValues {
  target_frames: number
  target_resolution: number
  auto_repeats: boolean
}

interface DatasetConfigEditorProps {
  config: DatasetConfigValues
  onChange: (field: keyof DatasetConfigValues, value: number | boolean) => void
  disabled?: boolean
}

export function DatasetConfigEditor({ config, onChange, disabled }: DatasetConfigEditorProps) {
  function handleNumber(field: 'target_frames' | 'target_resolution') {
    return (e: ChangeEvent<HTMLInputElement>) => {
      const parsed = parseInt(e.target.value, 10)
      if (!isNaN(parsed)) {
        onChange(field, parsed)
      }
    }
  }

  return (
    <div className="rounded-lg border border-zinc-700 bg-zinc-900 p-4">
      <h4 className="text-sm font-medium text-zinc-200 mb-3">Dataset Settings</h4>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        <label className="flex flex-col gap-1">
          <span className="text-xs text-zinc-400">Target Frames</span>
          <input
            type="number"
            value={config.target_frames}
            onChange={handleNumber('target_frames')}
            min={1}
            max={121}
            disabled={disabled}
            className="w-full rounded bg-zinc-800 border border-zinc-700 px-2 py-1 text-sm
                       text-zinc-200 focus:border-blue-500 focus:outline-none disabled:opacity-50"
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs text-zinc-400">Resolution</span>
          <select
            value={config.target_resolution}
            onChange={(e) => onChange('target_resolution', parseInt(e.target.value, 10))}
            disabled={disabled}
            className="rounded bg-zinc-800 border border-zinc-700 px-2 py-1 text-sm
                       text-zinc-200 focus:border-blue-500 focus:outline-none disabled:opacity-50"
          >
            <option value={256}>256px</option>
            <option value={384}>384px</option>
            <option value={512}>512px</option>
            <option value={768}>768px</option>
          </select>
        </label>

        <label className="flex items-center gap-2 self-end pb-1">
          <input
            type="checkbox"
            checked={config.auto_repeats}
            onChange={(e) => onChange('auto_repeats', e.target.checked)}
            disabled={disabled}
            className="rounded border-zinc-600"
          />
          <span className="text-xs text-zinc-400">Auto Repeats</span>
        </label>
      </div>
    </div>
  )
}
