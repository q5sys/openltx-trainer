/**
 * Hook for dataset operations: probing sources, scene detection,
 * clip creation, listing, deletion, and caption editing.
 */

import { useState, useCallback, useRef } from 'react'
import { backendFetch } from '../lib/backend'
import type {
  SourceMediaInfo,
  SceneProposal,
  ClipResult,
  ClipRecord,
  TriggerValidationResult,
  DatasetValidationResult,
} from '../types/dataset'

export interface ImportProgress {
  total: number
  current: number
  currentFile: string
}

interface UseDatasetReturn {
  // State
  sources: SourceMediaInfo[]
  scenes: SceneProposal[]
  clips: ClipRecord[]
  isLoading: boolean
  error: string | null
  validation: DatasetValidationResult | null
  importProgress: ImportProgress | null

  // Actions
  probeSource: (sourcePath: string) => Promise<SourceMediaInfo | null>
  detectScenes: (sourcePath: string, threshold?: number, targetLength?: number) => Promise<void>
  createClip: (sourcePath: string, datasetDir: string, startS: number, endS: number) => Promise<ClipResult | null>
  createClipsBatch: (sourcePath: string, datasetDir: string, segments: { start_s: number; end_s: number }[]) => Promise<ClipResult[]>
  importImage: (sourcePath: string, datasetDir: string) => Promise<ClipResult | null>
  listClips: (datasetDir: string) => Promise<void>
  deleteClip: (datasetDir: string, clipId: string) => Promise<void>
  updateCaption: (datasetDir: string, clipId: string, caption: string) => Promise<void>
  validateTrigger: (trigger: string) => Promise<TriggerValidationResult | null>
  validateDataset: (datasetDir: string, trigger: string | null) => Promise<DatasetValidationResult | null>
  prependTrigger: (datasetDir: string, trigger: string, clipIds?: string[]) => Promise<number>
  scanAndImport: (datasetDir: string) => Promise<void>
  cancelImport: () => void
  deleteAllClips: (datasetDir: string) => Promise<number>
  clearError: () => void
  clearScenes: () => void
  clearValidation: () => void
}

export function useDataset(): UseDatasetReturn {
  const [sources, setSources] = useState<SourceMediaInfo[]>([])
  const [scenes, setScenes] = useState<SceneProposal[]>([])
  const [clips, setClips] = useState<ClipRecord[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [validation, setValidation] = useState<DatasetValidationResult | null>(null)
  const [importProgress, setImportProgress] = useState<ImportProgress | null>(null)
  const importAbortRef = useRef(false)

  const clearError = useCallback(() => setError(null), [])
  const clearScenes = useCallback(() => setScenes([]), [])
  const clearValidation = useCallback(() => setValidation(null), [])

  const probeSource = useCallback(async (sourcePath: string): Promise<SourceMediaInfo | null> => {
    setIsLoading(true)
    setError(null)
    try {
      const resp = await backendFetch('/api/dataset/probe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source_path: sourcePath }),
      })
      if (!resp.ok) throw new Error(`Probe failed: ${resp.status}`)
      const info: SourceMediaInfo = await resp.json()
      setSources(prev => {
        const exists = prev.some(s => s.path === info.path)
        return exists ? prev : [...prev, info]
      })
      return info
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      return null
    } finally {
      setIsLoading(false)
    }
  }, [])

  const detectScenes = useCallback(async (
    sourcePath: string,
    threshold = 27.0,
    targetLength = 5.0,
  ): Promise<void> => {
    setIsLoading(true)
    setError(null)
    try {
      const resp = await backendFetch('/api/dataset/scenes/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source_path: sourcePath,
          threshold,
          target_clip_length_s: targetLength,
        }),
      })
      if (!resp.ok) throw new Error(`Scene detection failed: ${resp.status}`)
      const proposals: SceneProposal[] = await resp.json()
      setScenes(proposals)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setIsLoading(false)
    }
  }, [])

  const createClip = useCallback(async (
    sourcePath: string,
    datasetDir: string,
    startS: number,
    endS: number,
  ): Promise<ClipResult | null> => {
    setIsLoading(true)
    setError(null)
    try {
      const resp = await backendFetch('/api/dataset/clips', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source_path: sourcePath,
          dataset_dir: datasetDir,
          start_s: startS,
          end_s: endS,
        }),
      })
      if (!resp.ok) throw new Error(`Clip creation failed: ${resp.status}`)
      return await resp.json()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      return null
    } finally {
      setIsLoading(false)
    }
  }, [])

  const createClipsBatch = useCallback(async (
    sourcePath: string,
    datasetDir: string,
    segments: { start_s: number; end_s: number }[],
  ): Promise<ClipResult[]> => {
    setIsLoading(true)
    setError(null)
    try {
      const resp = await backendFetch('/api/dataset/clips/batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source_path: sourcePath,
          dataset_dir: datasetDir,
          segments,
        }),
      })
      if (!resp.ok) throw new Error(`Batch clip creation failed: ${resp.status}`)
      return await resp.json()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      return []
    } finally {
      setIsLoading(false)
    }
  }, [])

  const importImage = useCallback(async (
    sourcePath: string,
    datasetDir: string,
  ): Promise<ClipResult | null> => {
    setIsLoading(true)
    setError(null)
    try {
      const resp = await backendFetch('/api/dataset/images', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source_path: sourcePath,
          dataset_dir: datasetDir,
        }),
      })
      if (!resp.ok) throw new Error(`Image import failed: ${resp.status}`)
      return await resp.json()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      return null
    } finally {
      setIsLoading(false)
    }
  }, [])

  const listClips = useCallback(async (datasetDir: string): Promise<void> => {
    setIsLoading(true)
    setError(null)
    try {
      const resp = await backendFetch('/api/dataset/clips/list', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_dir: datasetDir }),
      })
      if (!resp.ok) throw new Error(`List clips failed: ${resp.status}`)
      const records: ClipRecord[] = await resp.json()
      setClips(records)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setIsLoading(false)
    }
  }, [])

  const deleteClip = useCallback(async (datasetDir: string, clipId: string): Promise<void> => {
    setError(null)
    try {
      const resp = await backendFetch('/api/dataset/clips/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_dir: datasetDir, clip_id: clipId }),
      })
      if (!resp.ok) throw new Error(`Delete clip failed: ${resp.status}`)
      setClips(prev => prev.filter(c => c.clip_id !== clipId))
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }, [])

  const updateCaption = useCallback(async (
    datasetDir: string,
    clipId: string,
    caption: string,
  ): Promise<void> => {
    setError(null)
    try {
      const resp = await backendFetch('/api/dataset/clips/caption', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_dir: datasetDir, clip_id: clipId, caption }),
      })
      if (!resp.ok) throw new Error(`Caption update failed: ${resp.status}`)
      setClips(prev => prev.map(c =>
        c.clip_id === clipId ? { ...c, caption } : c
      ))
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }, [])

  const validateTrigger = useCallback(async (trigger: string): Promise<TriggerValidationResult | null> => {
    try {
      const resp = await backendFetch('/api/dataset/trigger/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ trigger }),
      })
      if (!resp.ok) throw new Error(`Trigger validation failed: ${resp.status}`)
      return await resp.json()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      return null
    }
  }, [])

  const validateDataset = useCallback(async (
    datasetDir: string,
    trigger: string | null,
  ): Promise<DatasetValidationResult | null> => {
    setIsLoading(true)
    setError(null)
    try {
      const resp = await backendFetch('/api/dataset/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_dir: datasetDir, trigger }),
      })
      if (!resp.ok) throw new Error(`Dataset validation failed: ${resp.status}`)
      const result: DatasetValidationResult = await resp.json()
      setValidation(result)
      return result
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      return null
    } finally {
      setIsLoading(false)
    }
  }, [])

  const prependTrigger = useCallback(async (
    datasetDir: string,
    trigger: string,
    clipIds?: string[],
  ): Promise<number> => {
    setError(null)
    try {
      const resp = await backendFetch('/api/dataset/trigger/prepend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_dir: datasetDir,
          trigger,
          clip_ids: clipIds ?? null,
        }),
      })
      if (!resp.ok) throw new Error(`Prepend trigger failed: ${resp.status}`)
      const data = await resp.json()
      return data.modified_count as number
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      return 0
    }
  }, [])

  const deleteAllClips = useCallback(async (datasetDir: string): Promise<number> => {
    setIsLoading(true)
    setError(null)
    try {
      const resp = await backendFetch('/api/dataset/clips/delete-all', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_dir: datasetDir }),
      })
      if (!resp.ok) throw new Error(`Delete all clips failed: ${resp.status}`)
      const data = await resp.json()
      setClips([])
      setSources([])
      setValidation(null)
      return data.deleted_count as number
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      return 0
    } finally {
      setIsLoading(false)
    }
  }, [])

  const cancelImport = useCallback(() => {
    importAbortRef.current = true
  }, [])

  const scanAndImport = useCallback(async (datasetDir: string): Promise<void> => {
    importAbortRef.current = false
    setIsLoading(true)
    setError(null)
    try {
      // Scan the directory for media files.
      const scanResp = await backendFetch('/api/dataset/scan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ directory: datasetDir }),
      })
      if (!scanResp.ok) throw new Error(`Scan failed: ${scanResp.status}`)
      const { files } = await scanResp.json() as { files: string[] }

      // First, get existing clips to avoid re-importing.
      const existingClipsResp = await backendFetch('/api/dataset/clips/list', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_dir: datasetDir }),
      })
      const existingClips: ClipRecord[] = existingClipsResp.ok ? await existingClipsResp.json() : []
      const existingClipNames = new Set(existingClips.map(c => c.clip_id))

      setImportProgress({ total: files.length, current: 0, currentFile: 'Scanning...' })

      // Probe each file to add it to sources, and auto-import images that are not yet clips.
      const probed: SourceMediaInfo[] = []
      for (let i = 0; i < files.length; i++) {
        if (importAbortRef.current) break
        const filePath = files[i]
        const fileName = filePath.split('/').pop() ?? filePath
        setImportProgress({ total: files.length, current: i + 1, currentFile: fileName })
        try {
          const resp = await backendFetch('/api/dataset/probe', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source_path: filePath }),
          })
          if (!resp.ok) continue
          const info: SourceMediaInfo = await resp.json()
          probed.push(info)

          // Auto-import images that are not already in the dataset.
          if (info.is_image) {
            const stem = filePath.split('/').pop()?.replace(/\.[^.]+$/, '') ?? ''
            if (!existingClipNames.has(stem)) {
              await backendFetch('/api/dataset/images', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ source_path: filePath, dataset_dir: datasetDir }),
              })
            }
          }
        } catch {
          // Skip individual files that fail.
        }
      }

      setSources(prev => {
        const existingPaths = new Set(prev.map(s => s.path))
        const newSources = probed.filter(s => !existingPaths.has(s.path))
        return [...prev, ...newSources]
      })

      // Refresh the clip list after importing.
      const listResp = await backendFetch('/api/dataset/clips/list', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_dir: datasetDir }),
      })
      if (listResp.ok) {
        const records: ClipRecord[] = await listResp.json()
        setClips(records)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setIsLoading(false)
      setImportProgress(null)
    }
  }, [])

  return {
    sources,
    scenes,
    clips,
    isLoading,
    error,
    validation,
    probeSource,
    detectScenes,
    createClip,
    createClipsBatch,
    importImage,
    listClips,
    deleteClip,
    updateCaption,
    validateTrigger,
    validateDataset,
    prependTrigger,
    importProgress,
    scanAndImport,
    cancelImport,
    deleteAllClips,
    clearError,
    clearScenes,
    clearValidation,
  }
}
