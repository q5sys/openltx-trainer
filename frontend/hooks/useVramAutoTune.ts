/**
 * Hook for calling the Stage F low-VRAM auto-tune endpoint.
 *
 * Wraps ``POST /api/training/auto-tune-vram`` so the Training tab
 * can ask the backend to pick a ``low_vram_mode`` /
 * ``blocks_resident_on_gpu`` / ``gradient_checkpointing`` tuple
 * that fits the user's GPU (or a simulated override).
 *
 * The response mirrors ``backend/training_worker/engine/gpu_budget.py``
 * one-to-one. The frontend treats every field as opaque and renders
 * ``tier_label`` plus ``warning`` verbatim.
 */

import { useState, useCallback } from 'react'
import { backendFetch } from '../lib/backend'

export type LowVramMode = 'off' | 'fp8' | 'nf4'
export type RecommendationConfidence =
  | 'baseline'
  | 'supported'
  | 'plausible'
  | 'unsupported'

export interface AutoTuneVramResult {
  tier_label: string
  low_vram_mode: LowVramMode
  blocks_resident_on_gpu: number
  gradient_checkpointing: boolean
  estimated_peak_vram_gb: number
  estimated_throughput_multiplier: number
  required_host_ram_gb: number
  confidence: RecommendationConfidence
  warning: string
  detected_vram_bytes: number
  detected_system_ram_bytes: number
}

export interface AutoTuneVramRequest {
  vram_bytes?: number
  system_ram_bytes?: number
}

export function useVramAutoTune() {
  const [result, setResult] = useState<AutoTuneVramResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const autoTune = useCallback(
    async (request: AutoTuneVramRequest = {}): Promise<AutoTuneVramResult | null> => {
      setLoading(true)
      setError(null)
      try {
        const response = await backendFetch('/api/training/auto-tune-vram', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(request),
        })
        if (!response.ok) {
          const detail = (await response.json().catch(() => ({}))) as { detail?: string }
          throw new Error(detail.detail ?? `Auto-tune failed (${response.status})`)
        }
        const data = (await response.json()) as AutoTuneVramResult
        setResult(data)
        return data
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Auto-tune failed'
        setError(message)
        return null
      } finally {
        setLoading(false)
      }
    },
    [],
  )

  const reset = useCallback(() => {
    setResult(null)
    setError(null)
  }, [])

  return { result, loading, error, autoTune, reset }
}
