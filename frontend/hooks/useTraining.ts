/**
 * Hook for training job management.
 *
 * Provides methods to start, pause, resume, cancel training jobs
 * and fetch job lists, presets, and progress.
 */

import { useState, useCallback } from 'react'
import { backendFetch } from '../lib/backend'

// ---- Types ----

export type TrainingJobState =
  | 'created'
  | 'starting'
  | 'running'
  | 'paused'
  | 'completed'
  | 'cancelled'
  | 'errored'

export type TrainingPresetId = 'character_v1' | 'concept_v1' | 'character_image_v1'

export interface TrainingJobRecord {
  job_id: string
  project_id: string
  preset_id: TrainingPresetId
  gpu_index: number
  name: string
  state: TrainingJobState
  pid: number | null
  current_step: number
  total_steps: number
  current_phase: string | null
  current_loss: number | null
  eta_seconds: number | null
  error_message: string | null
  // Coarse lifecycle stage and a human-readable message sourced from
  // the worker's stage.json. Drives the Monitor status line during the
  // long windows (model load, dataset precache, sample generation) when
  // no per-step progress record is landing.
  stage: string | null
  stage_message: string | null
  created_at: number

  dataset_dir: string
  trigger_word: string
  job_dir: string
  config_path: string
}

export interface TrainingJobSummary {
  job_id: string
  project_id: string
  name: string
  state: TrainingJobState
  current_step: number
  total_steps: number
  current_loss: number | null
  gpu_index: number
  created_at: number
}

export interface TrainingPreset {
  id: string
  name: string
  description: string
}

export interface StartTrainingRequest {
  project_id: string
  preset_id: TrainingPresetId
  gpu_index: number
  dataset_dir: string
  trigger_word?: string
  model_path?: string
  name?: string
  config_overrides?: Record<string, unknown>
}


export interface TrainingProgressSlice {
  job_id: string
  records: Record<string, unknown>[]
  latest_step: number
}

// ---- Hook ----

export function useTraining() {
  const [jobs, setJobs] = useState<TrainingJobSummary[]>([])
  const [activeJob, setActiveJob] = useState<TrainingJobRecord | null>(null)
  const [presets, setPresets] = useState<TrainingPreset[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchPresets = useCallback(async () => {
    try {
      const res = await backendFetch('/api/training/presets')
      const data = (await res.json()) as TrainingPreset[]
      setPresets(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch presets')
    }
  }, [])

  const fetchJobs = useCallback(async () => {
    try {
      const res = await backendFetch('/api/training/jobs')
      const data = (await res.json()) as TrainingJobSummary[]
      setJobs(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch jobs')
    }
  }, [])

  const startJob = useCallback(async (request: StartTrainingRequest) => {
    setLoading(true)
    setError(null)
    try {
      const res = await backendFetch('/api/training/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      })
      const data = (await res.json()) as TrainingJobRecord
      setActiveJob(data)
      await fetchJobs()
      return data
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start job')
      return null
    } finally {
      setLoading(false)
    }
  }, [fetchJobs])

  const pauseJob = useCallback(async (jobId: string) => {
    try {
      const res = await backendFetch(`/api/training/jobs/${jobId}/pause`, { method: 'POST' })
      const data = (await res.json()) as TrainingJobRecord
      setActiveJob(data)
      await fetchJobs()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to pause job')
    }
  }, [fetchJobs])

  const resumeJob = useCallback(async (jobId: string) => {
    try {
      const res = await backendFetch(`/api/training/jobs/${jobId}/resume`, { method: 'POST' })
      const data = (await res.json()) as TrainingJobRecord
      setActiveJob(data)
      await fetchJobs()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to resume job')
    }
  }, [fetchJobs])

  const cancelJob = useCallback(async (jobId: string) => {
    try {
      const res = await backendFetch(`/api/training/jobs/${jobId}/cancel`, { method: 'POST' })
      const data = (await res.json()) as TrainingJobRecord
      setActiveJob(data)
      await fetchJobs()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to cancel job')
    }
  }, [fetchJobs])

  const fetchJob = useCallback(async (jobId: string) => {
    try {
      const res = await backendFetch(`/api/training/jobs/${jobId}`)
      const data = (await res.json()) as TrainingJobRecord
      setActiveJob(data)
      return data
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch job')
      return null
    }
  }, [])

  const deleteJob = useCallback(async (jobId: string) => {
    try {
      const res = await backendFetch(`/api/training/jobs/${jobId}`, { method: 'DELETE' })
      if (!res.ok) {
        const detail = await res.json().catch(() => ({})) as { detail?: string }
        throw new Error(detail.detail ?? `Failed to delete job (${res.status})`)
      }
      // If the deleted job was the active one, clear it so the form re-enables.
      setActiveJob(prev => (prev && prev.job_id === jobId ? null : prev))
      await fetchJobs()
      return true
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete job')
      return false
    }
  }, [fetchJobs])

  const restartJob = useCallback(async (jobId: string, name?: string) => {
    try {
      const res = await backendFetch(`/api/training/jobs/${jobId}/restart`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name ?? null }),
      })
      if (!res.ok) {
        const detail = await res.json().catch(() => ({})) as { detail?: string }
        throw new Error(detail.detail ?? `Failed to restart job (${res.status})`)
      }
      const data = (await res.json()) as TrainingJobRecord
      setActiveJob(data)
      await fetchJobs()
      return data
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to restart job')
      return null
    }
  }, [fetchJobs])

  return {
    jobs,
    activeJob,
    presets,
    loading,
    error,
    fetchPresets,
    fetchJobs,
    startJob,
    pauseJob,
    resumeJob,
    cancelJob,
    fetchJob,
    deleteJob,
    restartJob,
  }
}

