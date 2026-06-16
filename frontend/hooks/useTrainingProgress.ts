/**
 * Hook for polling training progress data.
 *
 * Fetches progress records from the backend at a regular interval,
 * only requesting records since the last known step to minimize payload.
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import { backendFetch } from '../lib/backend'

export interface ProgressRecord {
  ts: number
  step: number
  epoch: number
  loss: number
  lr: number
  grad_norm: number
  ips: number
  phase: string | null
  cancelled: boolean
  paused: boolean
}

export interface ProgressSlice {
  job_id: string
  records: ProgressRecord[]
  latest_step: number
}

const POLL_INTERVAL_MS = 2000

export function useTrainingProgress(jobId: string | null) {
  const [records, setRecords] = useState<ProgressRecord[]>([])
  const [latestStep, setLatestStep] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const sinceStepRef = useRef(0)

  const fetchProgress = useCallback(async () => {
    if (!jobId) return
    try {
      const res = await backendFetch(
        `/api/training/jobs/${jobId}/progress?since_step=${sinceStepRef.current}`
      )
      if (!res.ok) return
      const data = (await res.json()) as ProgressSlice
      if (data.records.length > 0) {
        setRecords(prev => {
          const combined = [...prev, ...data.records]
          // Deduplicate by step (keep latest)
          const seen = new Map<number, ProgressRecord>()
          for (const r of combined) {
            seen.set(r.step, r)
          }
          const sorted = Array.from(seen.values()).sort((a, b) => a.step - b.step)
          return sorted
        })
        setLatestStep(data.latest_step)
        sinceStepRef.current = data.latest_step + 1
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch progress')
    }
  }, [jobId])

  // Reset when job changes
  useEffect(() => {
    setRecords([])
    setLatestStep(0)
    sinceStepRef.current = 0
    setError(null)
    if (jobId) {
      setLoading(true)
      fetchProgress().finally(() => setLoading(false))
    }
  }, [jobId, fetchProgress])

  // Poll on interval
  useEffect(() => {
    if (!jobId) return
    const interval = setInterval(fetchProgress, POLL_INTERVAL_MS)
    return () => clearInterval(interval)
  }, [jobId, fetchProgress])

  return { records, latestStep, loading, error }
}
