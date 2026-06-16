/**
 * Hook for captioning operations: backend listing, model selection,
 * single-clip captioning, batch captioning, and API key management.
 */

import { useState, useCallback } from 'react'
import { backendFetch } from '../lib/backend'
import type {
  BackendDescriptor,
  CaptionBackendId,
  CaptionBatchStatus,
  CaptionResult,
  LocalModelChoice,
  ModelSetupStatus,
  PromptTemplate,
  ApiKeyTestResult,
} from '../types/caption'
import { DEFAULT_PROMPT_TEMPLATE } from '../types/caption'

interface UseCaptioningReturn {
  // State
  backends: BackendDescriptor[]
  modelStatus: ModelSetupStatus | null
  batchStatus: CaptionBatchStatus | null
  isLoading: boolean
  error: string | null

  // Actions
  listBackends: () => Promise<void>
  listModelChoices: () => Promise<LocalModelChoice[]>
  getModelStatus: () => Promise<void>
  selectModel: (choice: LocalModelChoice, gpuIndex?: number) => Promise<void>
  unloadModel: () => Promise<void>
  captionClip: (datasetDir: string, clipId: string, backendId?: CaptionBackendId, template?: PromptTemplate) => Promise<CaptionResult | null>
  captionBatch: (datasetDir: string, clipIds: string[], backendId?: CaptionBackendId, template?: PromptTemplate) => Promise<CaptionBatchStatus | null>
  getBatchStatus: (jobId: string) => Promise<void>
  cancelBatch: (jobId: string) => Promise<void>
  saveApiKey: (provider: CaptionBackendId, key: string) => Promise<void>
  deleteApiKey: (provider: CaptionBackendId) => Promise<void>
  testApiKey: (provider: CaptionBackendId) => Promise<ApiKeyTestResult | null>
  clearError: () => void
}

export function useCaptioning(): UseCaptioningReturn {
  const [backends, setBackends] = useState<BackendDescriptor[]>([])
  const [modelStatus, setModelStatus] = useState<ModelSetupStatus | null>(null)
  const [batchStatus, setBatchStatus] = useState<CaptionBatchStatus | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const clearError = useCallback(() => setError(null), [])

  const listBackends = useCallback(async (): Promise<void> => {
    try {
      const resp = await backendFetch('/api/caption/backends')
      if (!resp.ok) throw new Error(`List backends failed: ${resp.status}`)
      setBackends(await resp.json())
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }, [])

  const listModelChoices = useCallback(async (): Promise<LocalModelChoice[]> => {
    try {
      const resp = await backendFetch('/api/caption/local-model/choices')
      if (!resp.ok) throw new Error(`List model choices failed: ${resp.status}`)
      return await resp.json()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      return []
    }
  }, [])

  const getModelStatus = useCallback(async (): Promise<void> => {
    try {
      const resp = await backendFetch('/api/caption/local-model/setup-status')
      if (!resp.ok) throw new Error(`Get model status failed: ${resp.status}`)
      setModelStatus(await resp.json())
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }, [])

  const selectModel = useCallback(async (choice: LocalModelChoice, gpuIndex?: number): Promise<void> => {
    setIsLoading(true)
    setError(null)
    try {
      const resp = await backendFetch('/api/caption/local-model/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ choice, gpu_index: gpuIndex ?? null }),
      })
      if (!resp.ok) throw new Error(`Select model failed: ${resp.status}`)
      setModelStatus(await resp.json())
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setIsLoading(false)
    }
  }, [])

  const unloadModel = useCallback(async (): Promise<void> => {
    setIsLoading(true)
    setError(null)
    try {
      const resp = await backendFetch('/api/caption/local-model/unload', { method: 'POST' })
      if (!resp.ok) throw new Error(`Unload model failed: ${resp.status}`)
      setModelStatus(await resp.json())
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setIsLoading(false)
    }
  }, [])

  const captionClip = useCallback(async (
    datasetDir: string,
    clipId: string,
    backendId: CaptionBackendId = 'local',
    template: PromptTemplate = DEFAULT_PROMPT_TEMPLATE,
  ): Promise<CaptionResult | null> => {
    setIsLoading(true)
    setError(null)
    try {
      const resp = await backendFetch('/api/caption/clip', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_dir: datasetDir,
          clip_id: clipId,
          backend_id: backendId,
          prompt_template: template,
        }),
      })
      if (!resp.ok) throw new Error(`Caption clip failed: ${resp.status}`)
      return await resp.json()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      return null
    } finally {
      setIsLoading(false)
    }
  }, [])

  const captionBatch = useCallback(async (
    datasetDir: string,
    clipIds: string[],
    backendId: CaptionBackendId = 'local',
    template: PromptTemplate = DEFAULT_PROMPT_TEMPLATE,
  ): Promise<CaptionBatchStatus | null> => {
    setIsLoading(true)
    setError(null)
    try {
      const resp = await backendFetch('/api/caption/batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_dir: datasetDir,
          clip_ids: clipIds,
          backend_id: backendId,
          prompt_template: template,
        }),
      })
      if (!resp.ok) throw new Error(`Batch caption failed: ${resp.status}`)
      const status: CaptionBatchStatus = await resp.json()
      setBatchStatus(status)
      return status
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      return null
    } finally {
      setIsLoading(false)
    }
  }, [])

  const getBatchStatus = useCallback(async (jobId: string): Promise<void> => {
    try {
      const resp = await backendFetch(`/api/caption/jobs/${jobId}`)
      if (!resp.ok) throw new Error(`Get batch status failed: ${resp.status}`)
      setBatchStatus(await resp.json())
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }, [])

  const cancelBatch = useCallback(async (jobId: string): Promise<void> => {
    try {
      const resp = await backendFetch(`/api/caption/jobs/${jobId}/cancel`, { method: 'POST' })
      if (!resp.ok) throw new Error(`Cancel batch failed: ${resp.status}`)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }, [])

  const saveApiKey = useCallback(async (provider: CaptionBackendId, key: string): Promise<void> => {
    try {
      const resp = await backendFetch(`/api/caption/api-keys/${provider}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key }),
      })
      if (!resp.ok) throw new Error(`Save API key failed: ${resp.status}`)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }, [])

  const deleteApiKey = useCallback(async (provider: CaptionBackendId): Promise<void> => {
    try {
      const resp = await backendFetch(`/api/caption/api-keys/${provider}`, { method: 'DELETE' })
      if (!resp.ok) throw new Error(`Delete API key failed: ${resp.status}`)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }, [])

  const testApiKey = useCallback(async (provider: CaptionBackendId): Promise<ApiKeyTestResult | null> => {
    try {
      const resp = await backendFetch(`/api/caption/api-keys/${provider}/test`, { method: 'POST' })
      if (!resp.ok) throw new Error(`Test API key failed: ${resp.status}`)
      return await resp.json()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      return null
    }
  }, [])

  return {
    backends,
    modelStatus,
    batchStatus,
    isLoading,
    error,
    listBackends,
    listModelChoices,
    getModelStatus,
    selectModel,
    unloadModel,
    captionClip,
    captionBatch,
    getBatchStatus,
    cancelBatch,
    saveApiKey,
    deleteApiKey,
    testApiKey,
    clearError,
  }
}
