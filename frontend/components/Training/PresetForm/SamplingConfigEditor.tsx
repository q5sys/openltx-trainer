/**
 * SamplingConfigEditor: controls for sample generation during training.
 *
 * Edits up to MAX_SAMPLE_SPECS per-sample specs (each with its own
 * prompt, width, and height so portrait and landscape previews can be
 * mixed in one cycle), plus the shared num_inference_steps, num_frames,
 * and guidance_scale knobs and a single global sample_every_n_steps
 * cadence. Mirrors backend SamplingConfig in
 * backend/training_worker/config.py.
 */

import { type ChangeEvent } from 'react'

/** Server-side cap on sample specs (backend MAX_SAMPLE_SPECS). */
export const MAX_SAMPLE_SPECS = 4

export interface SampleSpecValues {
  prompt: string
  width: number
  height: number
}

export interface SamplingConfigValues {
  samples: SampleSpecValues[]
  num_inference_steps: number
  num_frames: number
  guidance_scale: number
  sample_every_n_steps: number
}

type SharedNumericField =
  | 'num_inference_steps'
  | 'num_frames'
  | 'guidance_scale'
  | 'sample_every_n_steps'

interface SamplingConfigEditorProps {
  config: SamplingConfigValues
  onChange: (field: keyof SamplingConfigValues, value: number | SampleSpecValues[]) => void
  disabled?: boolean
  /**
   * The project's trigger word. Used only to show the user a live preview
   * of what each prompt becomes after substitution. When empty the prompt
   * is shown verbatim and the "{trigger}" placeholder is dropped at
   * generation time.
   */
  triggerWord?: string
}

export function SamplingConfigEditor({ config, onChange, disabled, triggerWord }: SamplingConfigEditorProps) {
  // Mirror the backend _render_prompt helper so the preview the user sees
  // here matches exactly what the worker will generate.
  function renderPreview(prompt: string): string {
    if (!prompt.includes('{trigger}')) return prompt
    if (triggerWord) return prompt.replace(/\{trigger\}/g, triggerWord)
    return prompt.replace(/\{trigger\}/g, '').replace(/\s{2,}/g, ' ').trim()
  }

  function handleNumber(field: SharedNumericField) {
    return (e: ChangeEvent<HTMLInputElement>) => {
      const parsed = parseFloat(e.target.value)
      if (!isNaN(parsed)) {
        onChange(field, parsed)
      }
    }
  }

  function handleSampleField(index: number, field: keyof SampleSpecValues, value: string | number) {
    const updated = config.samples.map((spec, i) => {
      if (i !== index) return spec
      return { ...spec, [field]: value }
    })
    onChange('samples', updated)
  }

  function addSample() {
    if (config.samples.length >= MAX_SAMPLE_SPECS) return
    onChange('samples', [...config.samples, { prompt: '', width: 512, height: 512 }])
  }

  function removeSample(index: number) {
    const updated = config.samples.filter((_, i) => i !== index)
    onChange('samples', updated)
  }

  const atMax = config.samples.length >= MAX_SAMPLE_SPECS

  return (
    <div className="rounded-lg border border-zinc-700 bg-zinc-900 p-4">
      <h4 className="text-sm font-medium text-zinc-200 mb-1">Sample Generation</h4>
      <p className="text-xs text-zinc-500 mb-3">
        Use <code className="text-zinc-300">{'{trigger}'}</code> in a prompt to insert this
        project&apos;s trigger word
        {triggerWord
          ? <> (<span className="text-zinc-300">{triggerWord}</span>)</>
          : <> (no trigger word set; the placeholder is removed)</>}
        .
      </p>

      <div className="grid grid-cols-2 gap-3 mb-4 sm:grid-cols-4">
        <label className="flex flex-col gap-1">
          <span className="text-xs text-zinc-400">Inference Steps</span>
          <input
            type="number"
            value={config.num_inference_steps}
            onChange={handleNumber('num_inference_steps')}
            min={1}
            max={100}
            disabled={disabled}
            className="w-full rounded bg-zinc-800 border border-zinc-700 px-2 py-1 text-sm
                       text-zinc-200 focus:border-blue-500 focus:outline-none disabled:opacity-50"
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs text-zinc-400">Frames</span>
          <input
            type="number"
            value={config.num_frames}
            onChange={handleNumber('num_frames')}
            min={1}
            max={121}
            disabled={disabled}
            className="w-full rounded bg-zinc-800 border border-zinc-700 px-2 py-1 text-sm
                       text-zinc-200 focus:border-blue-500 focus:outline-none disabled:opacity-50"
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs text-zinc-400">Guidance Scale</span>
          <input
            type="number"
            value={config.guidance_scale}
            onChange={handleNumber('guidance_scale')}
            min={1}
            max={30}
            step={0.5}
            disabled={disabled}
            className="w-full rounded bg-zinc-800 border border-zinc-700 px-2 py-1 text-sm
                       text-zinc-200 focus:border-blue-500 focus:outline-none disabled:opacity-50"
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs text-zinc-400">Sample Every</span>
          <input
            type="number"
            value={config.sample_every_n_steps}
            onChange={handleNumber('sample_every_n_steps')}
            min={1}
            disabled={disabled}
            className="w-full rounded bg-zinc-800 border border-zinc-700 px-2 py-1 text-sm
                       text-zinc-200 focus:border-blue-500 focus:outline-none disabled:opacity-50"
          />
        </label>
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs text-zinc-400">
            Sample Prompts ({config.samples.length}/{MAX_SAMPLE_SPECS})
          </span>
          <button
            type="button"
            onClick={addSample}
            disabled={disabled || atMax}
            className="text-xs text-blue-400 hover:text-blue-300 disabled:opacity-50"
          >
            + Add Sample
          </button>
        </div>
        {config.samples.map((spec, i) => (
          <div key={i} className="flex items-start gap-2">
            <label className="flex flex-1 flex-col gap-1">
              <span className="text-xs text-zinc-500">Prompt</span>
              <input
                type="text"
                value={spec.prompt}
                onChange={(e) => handleSampleField(i, 'prompt', e.target.value)}
                placeholder="A video of {trigger} ..."
                disabled={disabled}
                className="w-full rounded bg-zinc-800 border border-zinc-700 px-2 py-1 text-sm
                           text-zinc-200 placeholder:text-zinc-600 focus:border-blue-500
                           focus:outline-none disabled:opacity-50"
              />
              {spec.prompt.includes('{trigger}') && (
                <span className="text-xs text-zinc-600 truncate" title={renderPreview(spec.prompt)}>
                  Preview: {renderPreview(spec.prompt)}
                </span>
              )}
            </label>
            <label className="flex w-20 flex-col gap-1">
              <span className="text-xs text-zinc-500">Width</span>
              <input
                type="number"
                value={spec.width}
                onChange={(e) => {
                  const parsed = parseInt(e.target.value, 10)
                  if (!isNaN(parsed)) handleSampleField(i, 'width', parsed)
                }}
                min={256}
                max={1024}
                step={64}
                disabled={disabled}
                className="w-full rounded bg-zinc-800 border border-zinc-700 px-2 py-1 text-sm
                           text-zinc-200 focus:border-blue-500 focus:outline-none disabled:opacity-50"
              />
            </label>
            <label className="flex w-20 flex-col gap-1">
              <span className="text-xs text-zinc-500">Height</span>
              <input
                type="number"
                value={spec.height}
                onChange={(e) => {
                  const parsed = parseInt(e.target.value, 10)
                  if (!isNaN(parsed)) handleSampleField(i, 'height', parsed)
                }}
                min={256}
                max={1024}
                step={64}
                disabled={disabled}
                className="w-full rounded bg-zinc-800 border border-zinc-700 px-2 py-1 text-sm
                           text-zinc-200 focus:border-blue-500 focus:outline-none disabled:opacity-50"
              />
            </label>
            <button
              type="button"
              onClick={() => removeSample(i)}
              disabled={disabled}
              className="h-8 text-xs text-red-400 hover:text-red-300 disabled:opacity-50 px-1"
            >
              x
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}
