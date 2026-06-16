/**
 * CostTimeEstimate: displays estimated training time and VRAM cost
 * based on the selected preset and GPU.
 *
 * Estimates are rough heuristics based on step count, resolution,
 * LoRA rank, and known GPU throughput benchmarks.
 */

import { Clock, Cpu } from 'lucide-react'

interface EstimateProps {
  /** Total training steps across all phases. */
  totalSteps: number
  /** Training resolution in pixels. */
  resolution: number
  /** Maximum LoRA rank used across phases. */
  maxLoraRank: number
  /** GPU name from health endpoint. */
  gpuName: string | null
  /** VRAM in bytes from health endpoint. */
  vramBytes: number | null
  /** Optional CSS class. */
  className?: string
}

/** Known GPU throughput benchmarks: steps per minute at 512px, rank 32. */
const GPU_BENCHMARKS: Record<string, number> = {
  '4090': 8.5,
  '4080': 6.2,
  '3090': 5.0,
  '5090': 12.0,
  '5080': 9.0,
  'a6000': 7.0,
  'a100': 14.0,
  'h100': 22.0,
}

function estimateStepsPerMinute(gpuName: string | null, resolution: number, maxRank: number): number {
  // Find a matching benchmark.
  let baseRate = 5.0 // Conservative default.
  if (gpuName) {
    const lower = gpuName.toLowerCase()
    for (const [key, rate] of Object.entries(GPU_BENCHMARKS)) {
      if (lower.includes(key)) {
        baseRate = rate
        break
      }
    }
  }

  // Scale by resolution (quadratic cost).
  const resolutionFactor = (512 * 512) / (resolution * resolution)
  // Scale by rank (linear cost).
  const rankFactor = 32 / Math.max(maxRank, 1)

  return baseRate * resolutionFactor * rankFactor
}

function formatDuration(minutes: number): string {
  if (minutes < 1) return 'less than 1 minute'
  if (minutes < 60) return `${Math.round(minutes)} min`
  const hours = Math.floor(minutes / 60)
  const remainingMinutes = Math.round(minutes % 60)
  if (remainingMinutes === 0) return `${hours}h`
  return `${hours}h ${remainingMinutes}m`
}

function formatVram(bytes: number): string {
  const gb = bytes / (1024 * 1024 * 1024)
  return `${gb.toFixed(1)} GB`
}

/** Rough VRAM estimate in GB for training. */
function estimateVramGb(resolution: number, maxRank: number): number {
  // Base VRAM for model + optimizer at 512px rank 32.
  const baseVramGb = 6.5
  const resolutionScale = (resolution * resolution) / (512 * 512)
  const rankScale = maxRank / 32
  return baseVramGb * resolutionScale * rankScale
}

export function CostTimeEstimate({
  totalSteps,
  resolution,
  maxLoraRank,
  gpuName,
  vramBytes,
  className,
}: EstimateProps) {
  const stepsPerMinute = estimateStepsPerMinute(gpuName, resolution, maxLoraRank)
  const estimatedMinutes = totalSteps / stepsPerMinute
  const estimatedVramGb = estimateVramGb(resolution, maxLoraRank)
  const availableVramGb = vramBytes ? vramBytes / (1024 * 1024 * 1024) : null

  const vramSufficient = availableVramGb === null || estimatedVramGb <= availableVramGb * 0.9

  return (
    <div className={`rounded-lg border border-zinc-700 bg-zinc-900 p-4 ${className ?? ''}`}>
      <h4 className="text-sm font-medium text-zinc-200 mb-3">Training Estimate</h4>

      <div className="grid grid-cols-2 gap-4">
        {/* Time estimate */}
        <div className="flex items-start gap-2">
          <Clock className="h-4 w-4 text-zinc-500 mt-0.5 shrink-0" />
          <div>
            <p className="text-sm text-zinc-200">{formatDuration(estimatedMinutes)}</p>
            <p className="text-xs text-zinc-500">
              ~{stepsPerMinute.toFixed(1)} steps/min
              {gpuName ? ` on ${gpuName}` : ''}
            </p>
          </div>
        </div>

        {/* VRAM estimate */}
        <div className="flex items-start gap-2">
          <Cpu className="h-4 w-4 text-zinc-500 mt-0.5 shrink-0" />
          <div>
            <p className={`text-sm ${vramSufficient ? 'text-zinc-200' : 'text-amber-400'}`}>
              ~{estimatedVramGb.toFixed(1)} GB VRAM
            </p>
            <p className="text-xs text-zinc-500">
              {availableVramGb !== null
                ? `${formatVram(vramBytes!)} available`
                : 'GPU info unavailable'}
            </p>
            {!vramSufficient && (
              <p className="text-xs text-amber-400 mt-1">
                May exceed available VRAM. Consider reducing resolution or LoRA rank.
              </p>
            )}
          </div>
        </div>
      </div>

      <p className="text-xs text-zinc-600 mt-3">
        Estimates are approximate and vary based on GPU load, batch size, and system configuration.
      </p>
    </div>
  )
}
