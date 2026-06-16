/**
 * Hook for fetching the full measured VRAM benchmark sweep.
 *
 * Wraps ``GET /api/training/vram-sweep`` so the Training tab can show
 * every measured (profile, quant, blocks_resident) cell as a table.
 * The operator can then pick any combination directly instead of
 * relying only on the auto-tune recommendation.
 *
 * The response mirrors ``backend/training_worker/engine/vram_sweep_data.py``.
 * The data is static, so the hook fetches once and caches in state.
 */

import { useState, useCallback } from 'react'
import { backendFetch } from '../lib/backend'

export type SweepProfile = 'image' | 'video'
export type SweepQuant = 'nf4' | 'fp8' | 'bf16'

export interface VramSweepCell {
  profile: SweepProfile
  quant: SweepQuant
  blocks_resident_on_gpu: number
  peak_vram_gb: number
  runtime_s: number
}

export interface VramSweepResult {
  source: string
  total_blocks: number
  cells: VramSweepCell[]
}

export function useVramSweep() {
  const [result, setResult] = useState<VramSweepResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchSweep = useCallback(async (): Promise<VramSweepResult | null> => {
    // Static data: fetch once and reuse.
    setLoading(true)
    setError(null)
    try {
      const response = await backendFetch('/api/training/vram-sweep', {
        method: 'GET',
      })
      if (!response.ok) {
        const detail = (await response.json().catch(() => ({}))) as { detail?: string }
        throw new Error(detail.detail ?? `Failed to load benchmark data (${response.status})`)
      }
      const data = (await response.json()) as VramSweepResult
      setResult(data)
      return data
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load benchmark data'
      setError(message)
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  return { result, loading, error, fetchSweep }
}
