/**
 * PhaseEditor: inline editor for a single training phase.
 *
 * Renders numeric fields for lora_rank, learning_rate, steps,
 * gradient_accumulation, differential_guidance, timestep_bias,
 * and save_every. All values are controlled via the parent
 * PresetForm. Sample cadence is no longer per-phase; it is a single
 * global knob in SamplingConfigEditor.
 */


import { type ChangeEvent } from 'react'

export interface PhaseValues {
  name: string
  start_step: number
  end_step: number
  lora_rank: number
  learning_rate: number
  gradient_accumulation: number
  differential_guidance: number
  timestep_bias: string
  save_every: number
}


interface PhaseEditorProps {
  phase: PhaseValues
  index: number
  onChange: (index: number, field: keyof PhaseValues, value: string | number) => void
  onRemove: (index: number) => void
  removable: boolean
  disabled?: boolean
}

function NumericField({
  label,
  value,
  onChange,
  min,
  max,
  step,
  disabled,
}: {
  label: string
  value: number
  onChange: (v: number) => void
  min?: number
  max?: number
  step?: number
  disabled?: boolean
}) {
  function handleChange(e: ChangeEvent<HTMLInputElement>) {
    const parsed = parseFloat(e.target.value)
    if (!isNaN(parsed)) {
      onChange(parsed)
    }
  }

  return (
    <label className="flex flex-col gap-1">
      <span className="text-xs text-zinc-400">{label}</span>
      <input
        type="number"
        value={value}
        onChange={handleChange}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        className="w-full rounded bg-zinc-800 border border-zinc-700 px-2 py-1 text-sm
                   text-zinc-200 focus:border-blue-500 focus:outline-none disabled:opacity-50"
      />
    </label>
  )
}

export function PhaseEditor({ phase, index, onChange, onRemove, removable, disabled }: PhaseEditorProps) {
  function handleNumeric(field: keyof PhaseValues, value: number) {
    onChange(index, field, value)
  }

  return (
    <div className="rounded-lg border border-zinc-700 bg-zinc-900 p-4">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-medium text-zinc-200">
          Phase {index + 1}: {phase.name}
        </h4>
        {removable && (
          <button
            type="button"
            onClick={() => onRemove(index)}
            disabled={disabled}
            className="text-xs text-red-400 hover:text-red-300 disabled:opacity-50"
          >
            Remove
          </button>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <NumericField
          label="Start Step"
          value={phase.start_step}
          onChange={(v) => handleNumeric('start_step', v)}
          min={0}
          disabled={disabled}
        />
        <NumericField
          label="End Step"
          value={phase.end_step}
          onChange={(v) => handleNumeric('end_step', v)}
          min={1}
          disabled={disabled}
        />
        <NumericField
          label="LoRA Rank"
          value={phase.lora_rank}
          onChange={(v) => handleNumeric('lora_rank', v)}
          min={1}
          max={128}
          disabled={disabled}
        />
        <NumericField
          label="Learning Rate"
          value={phase.learning_rate}
          onChange={(v) => handleNumeric('learning_rate', v)}
          min={0}
          step={0.00001}
          disabled={disabled}
        />
        <NumericField
          label="Grad Accumulation"
          value={phase.gradient_accumulation}
          onChange={(v) => handleNumeric('gradient_accumulation', v)}
          min={1}
          max={16}
          disabled={disabled}
        />
        <NumericField
          label="Diff Guidance"
          value={phase.differential_guidance}
          onChange={(v) => handleNumeric('differential_guidance', v)}
          min={0}
          step={0.5}
          disabled={disabled}
        />
        <NumericField
          label="Save Every"
          value={phase.save_every}
          onChange={(v) => handleNumeric('save_every', v)}
          min={1}
          disabled={disabled}
        />
      </div>


      <div className="mt-3">
        <label className="flex flex-col gap-1">
          <span className="text-xs text-zinc-400">Timestep Bias</span>
          <select
            value={phase.timestep_bias}
            onChange={(e) => onChange(index, 'timestep_bias', e.target.value)}
            disabled={disabled}
            className="rounded bg-zinc-800 border border-zinc-700 px-2 py-1 text-sm
                       text-zinc-200 focus:border-blue-500 focus:outline-none disabled:opacity-50"
          >
            <option value="none">None</option>
            <option value="high_noise">High Noise</option>
            <option value="low_noise">Low Noise</option>
          </select>
        </label>
      </div>
    </div>
  )
}
