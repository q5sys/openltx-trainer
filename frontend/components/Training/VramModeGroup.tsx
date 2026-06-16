/**
 * Stage F: low-VRAM mode picker for the Training tab.
 *
 * Renders three inputs (mode select, blocks-resident-on-GPU number,
 * gradient-checkpointing checkbox) plus an "Auto-tune for my GPU"
 * button that calls the backend's ``/api/training/auto-tune-vram``
 * endpoint and fills the inputs from the response.
 *
 * The component is fully controlled so the parent ``TrainingTab``
 * owns the values and passes them through ``config_overrides`` on
 * ``POST /api/training/jobs``.
 */

import { useState } from 'react'
import { Button } from '../ui/button'
import { Sparkles } from 'lucide-react'
import {
  useVramAutoTune,
  type AutoTuneVramResult,
  type LowVramMode,
} from '../../hooks/useVramAutoTune'
import { VramBenchmarkTable } from './VramBenchmarkTable'


export interface VramModeState {
  low_vram_mode: LowVramMode
  blocks_resident_on_gpu: number
  gradient_checkpointing: boolean
}

export const VRAM_MODE_DEFAULTS: VramModeState = {
  low_vram_mode: 'off',
  blocks_resident_on_gpu: 0,
  gradient_checkpointing: false,
}

const ONE_GIB = 1024 * 1024 * 1024

// Plain-language explanation shown as a hover tooltip on the gradient
// checkpointing control. A general user has no reason to know the term,
// so this spells out the trade-off in everyday words.
const GRADIENT_CHECKPOINTING_HELP =
  'Gradient checkpointing trades speed for memory. Normally training ' +
  'keeps every intermediate result in GPU memory so it can reuse them; ' +
  'this option throws most of them away and recomputes them when needed. ' +
  'That roughly halves the VRAM the model uses during training, at the ' +
  'cost of about 30% slower steps. Turn it on if training fails with an ' +
  'out-of-memory error; leave it off if your card has memory to spare and ' +
  'you want maximum speed.'


interface VramModeGroupProps {
  value: VramModeState
  onChange: (next: VramModeState) => void
  disabled?: boolean
}

export function VramModeGroup({ value, onChange, disabled = false }: VramModeGroupProps) {
  const { result, loading, error, autoTune } = useVramAutoTune()

  // Optional override: simulate a smaller card on the 5090 for Stage F
  // verification. ``0`` (the default) means "use the real GPU"; any
  // positive value is passed as ``vram_bytes`` to the backend.
  const [simulatedVramGb, setSimulatedVramGb] = useState<number>(0)

  const applyAutoTune = async () => {
    const request = simulatedVramGb > 0
      ? { vram_bytes: simulatedVramGb * ONE_GIB }
      : {}
    const tuned = await autoTune(request)
    if (tuned) {
      onChange({
        low_vram_mode: tuned.low_vram_mode,
        blocks_resident_on_gpu: tuned.blocks_resident_on_gpu,
        gradient_checkpointing: tuned.gradient_checkpointing,
      })
    }
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium text-zinc-300">VRAM Mode</h4>
        <div className="flex items-center gap-2">
          <label className="text-xs text-zinc-500">
            Simulate VRAM (GB):
            <input
              type="number"
              min={0}
              max={96}
              step={1}
              value={simulatedVramGb}
              onChange={(e) => setSimulatedVramGb(parseInt(e.target.value, 10) || 0)}
              disabled={disabled || loading}
              className="ml-2 w-20 px-2 py-1 bg-zinc-900 border border-zinc-700 rounded text-zinc-200 disabled:opacity-50"
              title="Stage F harness: pretend the card is this size. 0 = use real GPU."
            />
          </label>
          <Button
            size="sm"
            variant="outline"
            onClick={applyAutoTune}
            disabled={disabled || loading}
            className="gap-2"
          >
            <Sparkles className="h-3.5 w-3.5" />
            {loading ? 'Tuning...' : 'Auto-tune for my GPU'}
          </Button>
        </div>
      </div>

      {result && <RecommendationBanner result={result} />}
      {error && <p className="text-xs text-red-400">{error}</p>}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <div className="space-y-1">
          <label className="text-xs text-zinc-500">Low-VRAM mode</label>
          <select
            value={value.low_vram_mode}
            onChange={(e) =>
              onChange({ ...value, low_vram_mode: e.target.value as LowVramMode })
            }
            disabled={disabled}
            className="w-full px-3 py-2 bg-zinc-900 border border-zinc-700 rounded-lg text-sm text-zinc-200 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
          >
            <option value="off">off (BF16, 32+ GB cards)</option>
            <option value="fp8">fp8 (TorchAO, ~22 GB)</option>
            <option value="nf4">nf4 (bitsandbytes, ~11 GB, experimental)</option>
          </select>
        </div>

        <div className="space-y-1">
          <label className="text-xs text-zinc-500">
            Blocks resident on GPU
          </label>
          <input
            type="number"
            min={0}
            max={48}
            step={1}
            value={value.blocks_resident_on_gpu}
            onChange={(e) =>
              onChange({
                ...value,
                blocks_resident_on_gpu: Math.max(0, parseInt(e.target.value, 10) || 0),
              })
            }
            disabled={disabled}
            className="w-full px-3 py-2 bg-zinc-900 border border-zinc-700 rounded-lg text-sm text-zinc-200 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
          />
          <p className="text-xs text-zinc-600">0 = block swap off</p>
        </div>

        <div className="space-y-1">
          <label
            className="text-xs text-zinc-500"
            title={GRADIENT_CHECKPOINTING_HELP}
          >
            Gradient checkpointing
          </label>
          <div className="flex items-center gap-2 h-[38px]">
            <input
              type="checkbox"
              id="vram-gc"
              checked={value.gradient_checkpointing}
              onChange={(e) =>
                onChange({ ...value, gradient_checkpointing: e.target.checked })
              }
              disabled={disabled}
              title={GRADIENT_CHECKPOINTING_HELP}
              className="h-4 w-4 rounded border-zinc-700 bg-zinc-900 disabled:opacity-50"
            />
            <label
              htmlFor="vram-gc"
              className="text-sm text-zinc-300"
              title={GRADIENT_CHECKPOINTING_HELP}
            >
              Enable (~0.5x VRAM, ~1.3x time)
            </label>
          </div>
          <p className="text-xs text-zinc-600">
            Saves VRAM by recomputing data instead of storing it. Turn on
            if you run out of memory.
          </p>
        </div>

      </div>

      <VramBenchmarkTable
        onApply={(mode, blocksResident) =>
          onChange({
            ...value,
            low_vram_mode: mode,
            blocks_resident_on_gpu: blocksResident,
          })
        }
        selectedMode={value.low_vram_mode}
        selectedBlocks={value.blocks_resident_on_gpu}
        disabled={disabled}
      />
    </div>
  )
}


function RecommendationBanner({ result }: { result: AutoTuneVramResult }) {
  const tone = bannerTone(result.confidence)
  return (
    <div className={`rounded-lg border px-3 py-2 ${tone.container}`}>
      <div className="flex items-baseline justify-between gap-3">
        <span className={`text-sm font-medium ${tone.title}`}>
          {result.tier_label}
        </span>
        <span className="text-xs text-zinc-400">
          Est. peak: {result.estimated_peak_vram_gb.toFixed(1)} GB,{' '}
          throughput: {result.estimated_throughput_multiplier.toFixed(2)}x,{' '}
          host RAM: {result.required_host_ram_gb} GB
        </span>
      </div>
      {result.warning && (
        <p className={`text-xs mt-1 ${tone.warning}`}>{result.warning}</p>
      )}
    </div>
  )
}

function bannerTone(confidence: AutoTuneVramResult['confidence']) {
  switch (confidence) {
    case 'baseline':
      return {
        container: 'border-emerald-700 bg-emerald-900/20',
        title: 'text-emerald-300',
        warning: 'text-emerald-200',
      }
    case 'supported':
      return {
        container: 'border-blue-700 bg-blue-900/20',
        title: 'text-blue-300',
        warning: 'text-blue-200',
      }
    case 'plausible':
      return {
        container: 'border-yellow-700 bg-yellow-900/20',
        title: 'text-yellow-300',
        warning: 'text-yellow-200',
      }
    case 'unsupported':
      return {
        container: 'border-red-700 bg-red-900/20',
        title: 'text-red-300',
        warning: 'text-red-200',
      }
  }
}
