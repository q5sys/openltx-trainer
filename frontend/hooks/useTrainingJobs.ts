/**
 * Hook for polling the training job list.
 *
 * Fetches job summaries at a regular interval so the Monitor tab
 * can show all jobs across projects.
 */

import { useState, useEffect, useCallback } from 'react'
import { backendFetch } from '../lib/backend'
import type { TrainingJobSummary } from './useTraining'

const POLL_INTERVAL_MS = 3000

export function useTrainingJobs() {
  const [jobs, setJobs] = useState<TrainingJobSummary[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchJobs = useCallback(async () => {
    try {
      const res = await backendFetch('/api/training/jobs')
      if (!res.ok) return
      const data = (await res.json()) as TrainingJobSummary[]
      setJobs(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch jobs')
    }
  }, [])

  useEffect(() => {
    setLoading(true)
    fetchJobs().finally(() => setLoading(false))
    const interval = setInterval(fetchJobs, POLL_INTERVAL_MS)
    return () => clearInterval(interval)
  }, [fetchJobs])

  return { jobs, loading, error, refetch: fetchJobs }
}
