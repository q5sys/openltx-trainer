/**
 * Hook for polling live GPU VRAM usage for a single device index.
 *
 * Backed by the backend `/api/gpu-memory` endpoint, which reads
 * device-wide used/total memory via NVML. Because NVML reports usage
 * across all processes, this reflects the training worker subprocess's
 * allocation even though the backend itself holds no CUDA context.
 *
 * Polling only runs while `active` is true (a running/paused job), so an
 * idle Monitor view does not hammer the backend.
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import { backendFetch } from '../lib/backend'

export interface GpuMemory {
  available: boolean
  total_mb: number
  used_mb: number
}

const POLL_INTERVAL_MS = 2000

export function useGpuMemory(gpuIndex: number | null, active: boolean) {
  const [memory, setMemory] = useState<GpuMemory | null>(null)
  // Avoid a state update (and re-render) when nothing changed.
  const lastRef = useRef<string>('')

  const fetchMemory = useCallback(async () => {
    if (gpuIndex === null) return
    try {
      const res = await backendFetch(`/api/gpu-memory?index=${gpuIndex}`)
      if (!res.ok) return
      const data = (await res.json()) as GpuMemory
      const key = `${data.available}:${data.used_mb}:${data.total_mb}`
      if (key !== lastRef.current) {
        lastRef.current = key
        setMemory(data)
      }
    } catch {
      // ignore: a failed poll just leaves the previous reading in place
    }
  }, [gpuIndex])

  useEffect(() => {
    if (gpuIndex === null || !active) return
    // Fetch immediately, then on interval.
    fetchMemory()
    const interval = setInterval(fetchMemory, POLL_INTERVAL_MS)
    return () => clearInterval(interval)
  }, [gpuIndex, active, fetchMemory])

  return memory
}
