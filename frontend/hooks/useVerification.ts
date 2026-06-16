import { useState, useCallback } from 'react'
import { backendFetch } from '../lib/backend'

// ============================================================
// Types
// ============================================================

export interface LoraStackEntry {
  checkpoint_path: string
  weight: number
}

export interface LoraDescriptor {
  checkpoint_path: string
  project_id: string
  project_name: string
  job_id: string
  step: number
  phase: string | null
  rank: number | null
  weight: number
}

export interface VerifyGenerateRequest {
  project_id: string
  prompt: string
  negative_prompt: string
  width: number
  height: number
  num_frames: number
  guidance_scale: number
  seed: number
  gpu_index: number
  lora_stack: LoraStackEntry[]
  num_inference_steps: number
}

export type VerificationJobState =
  | 'queued'
  | 'loading_model'
  | 'loading_lora'
  | 'generating'
  | 'completed'
  | 'errored'
  | 'cancelled'

export interface VerificationJobStatus {
  generation_id: string
  status: VerificationJobState
  progress: number
  output_path: string | null
  error_message: string | null
  prompt: string
  seed: number
  lora_stack: LoraStackEntry[]
}

export interface VerificationHistoryEntry {
  generation_id: string
  project_id: string
  prompt: string
  seed: number
  output_path: string
  lora_stack: LoraStackEntry[]
  created_at: number
}

export interface ExportLoraRequest {
  checkpoint_path: string
  destination_dir: string
  include_config: boolean
  include_preview: boolean
  preview_generation_id: string | null
}

export interface ExportLoraResponse {
  exported_path: string
  config_path: string | null
  preview_path: string | null
}

// ============================================================
// Hook
// ============================================================

export function useVerification() {
  const [loras, setLoras] = useState<LoraDescriptor[]>([])
  const [activeJob, setActiveJob] = useState<VerificationJobStatus | null>(null)
  const [history, setHistory] = useState<VerificationHistoryEntry[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchLoras = useCallback(async (projectId?: string) => {
    try {
      const params = projectId ? `?project_id=${projectId}` : ''
      const res = await backendFetch(`/api/verification/loras${params}`)
      if (!res.ok) throw new Error(`Failed to fetch LORAs: ${res.status}`)
      const data: LoraDescriptor[] = await res.json()
      setLoras(data)
      return data
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      return []
    }
  }, [])

  const generate = useCallback(async (request: VerifyGenerateRequest) => {
    setLoading(true)
    setError(null)
    try {
      const res = await backendFetch('/api/verification/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      })
      if (!res.ok) throw new Error(`Generation failed: ${res.status}`)
      const data = await res.json()
      // Immediately poll for status
      await pollJob(data.generation_id)
      return data
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  const pollJob = useCallback(async (generationId: string) => {
    try {
      const res = await backendFetch(`/api/verification/jobs/${generationId}`)
      if (!res.ok) throw new Error(`Failed to poll job: ${res.status}`)
      const data: VerificationJobStatus | null = await res.json()
      setActiveJob(data)
      return data
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      return null
    }
  }, [])

  const cancelJob = useCallback(async (generationId: string) => {
    try {
      const res = await backendFetch(`/api/verification/jobs/${generationId}/cancel`, {
        method: 'POST',
      })
      if (!res.ok) throw new Error(`Cancel failed: ${res.status}`)
      await pollJob(generationId)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }, [pollJob])

  const fetchHistory = useCallback(async (projectId: string) => {
    try {
      const res = await backendFetch(`/api/verification/history/${projectId}`)
      if (!res.ok) throw new Error(`Failed to fetch history: ${res.status}`)
      const data: VerificationHistoryEntry[] = await res.json()
      setHistory(data)
      return data
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      return []
    }
  }, [])

  const exportLora = useCallback(async (request: ExportLoraRequest) => {
    setLoading(true)
    setError(null)
    try {
      const res = await backendFetch('/api/verification/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      })
      if (!res.ok) throw new Error(`Export failed: ${res.status}`)
      const data: ExportLoraResponse = await res.json()
      return data
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  return {
    loras,
    activeJob,
    history,
    loading,
    error,
    fetchLoras,
    generate,
    pollJob,
    cancelJob,
    fetchHistory,
    exportLora,
  }
}
