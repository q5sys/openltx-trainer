/**
 * CaptionPanel: captioning sub-view within the Dataset tab.
 * Provides backend selection, local model configuration overrides,
 * batch captioning with per-clip progress, and prompt template editing.
 */

import { useState, useEffect, useRef, useCallback, memo } from 'react'
import { useCaptioning } from '../../../hooks/useCaptioning'
import { useAppSettings } from '../../../contexts/AppSettingsContext'
import { backendFetch } from '../../../lib/backend'
import type { ClipRecord } from '../../../types/dataset'
import type { CaptionBackendId, LocalModelChoice, CaptionQuantization, CaptionModelSize } from '../../../types/caption'

interface CaptionPanelProps {
  clips: ClipRecord[]
  datasetDir: string
  onCaptionsUpdated: () => void
}

/**
 * The catalog of local model variants the user can pick in the Caption panel.
 *
 * Every entry is an *Instruct* variant. We never list a "Thinking" Qwen3-VL
 * model: those emit chain-of-thought reasoning that leaks into captions.
 * The backend's `_build_hf_model_id` maps each `(size, abliterated=True)`
 * pair to the corresponding `huihui-ai/Huihui-Qwen3-VL-<size>-Instruct-abliterated`
 * repo, never to a Thinking-abliterated repo.
 */
interface LocalModelVariant {
  id: string
  label: string
  size: CaptionModelSize
  abliterated: boolean
}

const LOCAL_MODEL_VARIANTS: LocalModelVariant[] = [
  { id: '2B-instruct', label: 'Qwen3-VL-2B-Instruct', size: '2B', abliterated: false },
  { id: '2B-abliterated', label: 'Qwen3-VL-2B-Instruct (abliterated)', size: '2B', abliterated: true },
  { id: '4B-instruct', label: 'Qwen3-VL-4B-Instruct', size: '4B', abliterated: false },
  { id: '4B-abliterated', label: 'Qwen3-VL-4B-Instruct (abliterated)', size: '4B', abliterated: true },
  { id: '8B-instruct', label: 'Qwen3-VL-8B-Instruct', size: '8B', abliterated: false },
  { id: '8B-abliterated', label: 'Qwen3-VL-8B-Instruct (abliterated)', size: '8B', abliterated: true },
  { id: '32B-instruct', label: 'Qwen3-VL-32B-Instruct', size: '32B', abliterated: false },
  { id: '32B-abliterated', label: 'Qwen3-VL-32B-Instruct (abliterated)', size: '32B', abliterated: true },
]

function findVariantId(size: CaptionModelSize, abliterated: boolean): string {
  const found = LOCAL_MODEL_VARIANTS.find(v => v.size === size && v.abliterated === abliterated)
  return found ? found.id : LOCAL_MODEL_VARIANTS[0].id
}

/**
 * Small thumbnail beside each row in the caption list.
 *
 * Fetches the clip's thumbnail from `/api/dataset/clips/thumbnail` (the same
 * endpoint the Browser tab uses) and renders a fixed-size preview. Wrapped
 * in `React.memo` so re-rendering the list when a caption updates or a row
 * gets selected does not re-fetch every thumbnail.
 */
const ClipRowThumbnail = memo(function ClipRowThumbnail({
  datasetDir,
  clipId,
  filename,
  isImage,
  durationSeconds,
}: {
  datasetDir: string
  clipId: string
  filename: string
  isImage: boolean
  durationSeconds: number
}) {
  const [src, setSrc] = useState<string | null>(null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    let cancelled = false
    setSrc(null)
    setFailed(false)
    backendFetch('/api/dataset/clips/thumbnail', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset_dir: datasetDir, clip_id: clipId }),
    })
      .then(async resp => {
        if (!resp.ok || cancelled) {
          if (!cancelled) setFailed(true)
          return
        }
        const data = await resp.json()
        if (!cancelled) {
          if (data.thumbnail_b64) setSrc(`data:image/png;base64,${data.thumbnail_b64}`)
          else setFailed(true)
        }
      })
      .catch(() => { if (!cancelled) setFailed(true) })
    return () => { cancelled = true }
  }, [datasetDir, clipId])

  return (
    <div
      className="w-16 h-10 shrink-0 rounded overflow-hidden bg-zinc-800 border border-zinc-700 relative flex items-center justify-center"
      title={filename}
    >
      {src ? (
        <img src={src} alt={filename} className="w-full h-full object-cover" />
      ) : failed ? (
        <span className="text-[10px] text-zinc-500">
          {isImage ? 'IMG' : 'VID'}
        </span>
      ) : (
        <div className="w-3 h-3 border border-zinc-500 border-t-transparent rounded-full animate-spin" />
      )}
      {/* Play overlay so the user can visually tell a thumbnail comes from a
          video clip, not a still image. */}
      {src && !isImage && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <svg className="w-4 h-4 text-white drop-shadow opacity-80" viewBox="0 0 24 24" fill="currentColor">
            <path d="M8 5v14l11-7z" />
          </svg>
        </div>
      )}
      {/* Duration badge in the corner for videos. Cheap visual cue. */}
      {src && !isImage && durationSeconds > 0 && (
        <span className="absolute bottom-0 right-0 px-1 text-[9px] leading-tight text-white bg-black/60 rounded-tl">
          {durationSeconds.toFixed(1)}s
        </span>
      )}
    </div>
  )
})

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  const kb = bytes / 1024
  if (kb < 1024) return `${kb.toFixed(1)} KB`
  const mb = kb / 1024
  if (mb < 1024) return `${mb.toFixed(1)} MB`
  const gb = mb / 1024
  return `${gb.toFixed(2)} GB`
}

export function CaptionPanel({ clips, datasetDir, onCaptionsUpdated }: CaptionPanelProps) {
  const captioning = useCaptioning()
  const [selectedBackend, setSelectedBackend] = useState<CaptionBackendId>('local')
  const [selectedClipIds, setSelectedClipIds] = useState<Set<string>>(new Set())

  const { settings } = useAppSettings()

  // Local model override state, initialized from settings.
  const [modelSize, setModelSize] = useState<CaptionModelSize>(
    (settings.captioningDefaults.modelSize as CaptionModelSize) || '4B'
  )
  const [abliterated, setAbliterated] = useState(settings.captioningDefaults.abliterated)
  const [quantization, setQuantization] = useState<CaptionQuantization>(
    (settings.captioningDefaults.quantization as CaptionQuantization) || 'fp16'
  )
  const [modelLoading, setModelLoading] = useState(false)
  // The choice the user most recently asked the backend to load. We hold a
  // ref to it so the "close overlay" effect can verify that
  // `modelStatus.model_choice` actually reflects this new request before
  // declaring the load complete. Without this, clicking "Load Model" while
  // the previous status is still `ready` triggers a stale-state race: the
  // transition effect sees `state === 'ready'` from the OLD load and closes
  // the overlay before the backend returns the new "downloading" state.
  const pendingChoiceRef = useRef<LocalModelChoice | null>(null)

  // GPU selection state.
  const [gpuDevices, setGpuDevices] = useState<{ index: number; name: string }[]>([])
  const [gpuIndex, setGpuIndex] = useState<number>(settings.defaultGpuIndex ?? 0)

  // Captioning progress state.
  const [captionProgress, setCaptionProgress] = useState<{
    running: boolean
    current: number
    total: number
    currentFile: string
    failed: number
    cancelRequested: boolean
  }>({ running: false, current: 0, total: 0, currentFile: '', failed: 0, cancelRequested: false })
  const cancelRef = useRef(false)

  // Build a LocalModelChoice from the current local overrides.
  const buildChoice = useCallback((): LocalModelChoice => ({
    family: 'qwen3-vl',
    size: modelSize,
    abliterated,
    quantization,
  }), [modelSize, abliterated, quantization])

  // Load backends and GPU list on mount, auto-select local model from settings if needed.
  useEffect(() => {
    const init = async () => {
      await captioning.listBackends()
      await captioning.getModelStatus()
      // Fetch GPU list for the GPU selector.
      try {
        const resp = await backendFetch('/api/gpu-list')
        const data: { devices: { index: number; name: string }[] } = await resp.json()
        setGpuDevices(data.devices)
      } catch { /* ignore */ }
    }
    void init()
  }, [])

  // Note: the captioning model is NOT auto-loaded when this panel opens.
  // Loading a multi-GB VL model on every visit to the Caption tab wastes
  // VRAM and download time when the user only wants to review captions.
  // The user explicitly loads a model via the "Load Model" button and can
  // free it again with "Unload Model".

  // Poll model status while loading so the UI can show live download/load progress.
  useEffect(() => {
    if (!modelLoading) return
    let cancelled = false
    const tick = async () => {
      if (cancelled) return
      await captioning.getModelStatus()
    }
    const interval = window.setInterval(() => { void tick() }, 1000)
    return () => {
      cancelled = true
      window.clearInterval(interval)
    }
  }, [modelLoading, captioning.getModelStatus])

  // When status transitions to ready or error, stop the loading overlay.
  //
  // We must not close the overlay on a `ready` state left over from the
  // PREVIOUS load. After clicking "Load Model", `modelStatus` is still
  // `ready` for a moment until the backend's selectModel response returns
  // with the new downloading state. We guard against that by comparing the
  // reported `model_choice` against `pendingChoiceRef`: a stale `ready`
  // from the prior load has the OLD model_choice, so we ignore it. We do
  // accept `error` immediately because errors are always for the most
  // recent request - if the prior load was already `ready` the backend
  // will never spontaneously emit `error` afterwards.
  useEffect(() => {
    if (!modelLoading) return
    const status = captioning.modelStatus
    if (!status) return
    const s = status.state
    if (s === 'error') {
      pendingChoiceRef.current = null
      setModelLoading(false)
      void captioning.listBackends()
      return
    }
    if (s === 'ready') {
      const pending = pendingChoiceRef.current
      const loaded = status.model_choice
      if (pending && loaded) {
        const matches =
          loaded.size === pending.size &&
          loaded.abliterated === pending.abliterated &&
          loaded.quantization === pending.quantization
        if (!matches) return
      }
      pendingChoiceRef.current = null
      setModelLoading(false)
      void captioning.listBackends()
    }
  }, [captioning.modelStatus, modelLoading, captioning.listBackends])

  // Check if current overrides differ from the loaded model.
  //
  // We compare against the *loaded* model_choice unless a load is in flight,
  // in which case we compare against the pending choice. Without this,
  // immediately after clicking "Load Model" the displayed loadedChoice can
  // still be the previous model for a render or two, making the button flip
  // to "Loaded" while the new download is silently running underneath.
  const loadedChoice = captioning.modelStatus?.model_choice
  const effectiveLoadedChoice = pendingChoiceRef.current ?? loadedChoice
  const modelDiffers = effectiveLoadedChoice
    ? effectiveLoadedChoice.size !== modelSize ||
      effectiveLoadedChoice.abliterated !== abliterated ||
      effectiveLoadedChoice.quantization !== quantization
    : false

  const handleLoadModel = async () => {
    const choice = buildChoice()
    pendingChoiceRef.current = choice
    setModelLoading(true)
    // Backend returns once subprocess has spawned; polling effect tracks progress
    // and the transition effect closes the overlay when state becomes ready/error
    // AND the reported model_choice matches `pendingChoiceRef`.
    await captioning.selectModel(choice, gpuIndex)
  }

  const uncaptionedClips = clips.filter(c => !c.caption || c.caption.trim() === '')
  const allClipIds = clips.map(c => c.clip_id)

  const handleSelectAll = () => setSelectedClipIds(new Set(allClipIds))
  const handleSelectUncaptioned = () => setSelectedClipIds(new Set(uncaptionedClips.map(c => c.clip_id)))
  const handleClearSelection = () => setSelectedClipIds(new Set())

  const toggleClip = (clipId: string) => {
    setSelectedClipIds(prev => {
      const next = new Set(prev)
      if (next.has(clipId)) next.delete(clipId)
      else next.add(clipId)
      return next
    })
  }

  // Caption clips one by one for real-time progress.
  const runCaptioning = async (clipIds: string[]) => {
    cancelRef.current = false
    setCaptionProgress({ running: true, current: 0, total: clipIds.length, currentFile: '', failed: 0, cancelRequested: false })

    let failed = 0
    for (let i = 0; i < clipIds.length; i++) {
      if (cancelRef.current) break

      const clipId = clipIds[i]
      const clip = clips.find(c => c.clip_id === clipId)
      const filename = clip?.filename || clipId

      setCaptionProgress(prev => ({ ...prev, current: i, currentFile: filename }))

      const result = await captioning.captionClip(datasetDir, clipId, selectedBackend)
      if (!result || !result.success) failed++

      // Refresh after each clip so captions appear in real time.
      onCaptionsUpdated()
    }

    setCaptionProgress(prev => ({
      ...prev,
      running: false,
      current: cancelRef.current ? prev.current : clipIds.length,
      failed,
    }))
  }

  const handleCaptionAll = () => void runCaptioning(allClipIds)
  const handleCaptionSelected = () => void runCaptioning(Array.from(selectedClipIds))

  const handleCancelCaptioning = () => {
    cancelRef.current = true
    setCaptionProgress(prev => ({ ...prev, cancelRequested: true }))
  }

  return (
    <div className="flex flex-col h-full relative">
      {/* Model loading overlay */}
      {modelLoading && (() => {
        const status = captioning.modelStatus
        const state = status?.state ?? 'downloading'
        const downloaded = status?.downloaded_bytes ?? null
        const total = status?.total_bytes ?? null
        const currentFile = status?.current_file ?? null
        const message = status?.message ?? null
        const hasProgress = downloaded !== null && total !== null && total > 0
        const percent = hasProgress ? Math.min(100, Math.round((downloaded! / total!) * 100)) : 0

        const title = state === 'loading'
          ? 'Loading Captioning Model'
          : state === 'downloading'
            ? 'Downloading Captioning Model'
            : 'Preparing Captioning Model'

        return (
          <div className="absolute inset-0 z-20 bg-black/70 flex items-center justify-center">
            <div className="bg-zinc-800 border border-zinc-700 rounded-lg p-6 max-w-md w-full mx-4 space-y-4">
              <div className="flex items-center gap-3">
                <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin shrink-0" />
                <h3 className="text-sm font-medium text-white">{title}</h3>
              </div>

              <p className="text-xs text-zinc-400">
                Qwen3-VL-{modelSize}{abliterated ? ' (abliterated)' : ''} {quantization}
              </p>

              {/* Download progress bar - only when downloading with known total */}
              {state === 'downloading' && hasProgress && (
                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs text-zinc-400">
                    <span>{formatBytes(downloaded!)} / {formatBytes(total!)}</span>
                    <span>{percent}%</span>
                  </div>
                  <div className="h-2 bg-zinc-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-blue-500 rounded-full transition-all duration-300"
                      style={{ width: `${percent}%` }}
                    />
                  </div>
                </div>
              )}

              {/* Indeterminate bar when downloading without size info or loading */}
              {(state === 'loading' || (state === 'downloading' && !hasProgress)) && (
                <div className="h-2 bg-zinc-700 rounded-full overflow-hidden relative">
                  <div className="absolute inset-y-0 left-0 w-1/3 bg-blue-500 animate-pulse rounded-full" />
                </div>
              )}

              {/* Current file or status message */}
              {currentFile && (
                <p className="text-xs text-zinc-500 truncate font-mono" title={currentFile}>
                  {currentFile}
                </p>
              )}
              {!currentFile && message && (
                <p className="text-xs text-zinc-500 truncate">{message}</p>
              )}

              <p className="text-xs text-zinc-600">
                First download may take several minutes depending on model size and connection speed.
                Models are cached in ~/.cache/huggingface/hub/
              </p>
            </div>
          </div>
        )
      })()}

      {/* Captioning progress overlay */}
      {captionProgress.running && (
        <div className="absolute inset-0 z-20 bg-black/70 flex items-center justify-center">
          <div className="bg-zinc-800 border border-zinc-700 rounded-lg p-6 max-w-md w-full mx-4 space-y-4">
            <h3 className="text-sm font-medium text-white">Captioning in Progress</h3>

            {/* Progress bar */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs text-zinc-400">
                <span>{captionProgress.current} / {captionProgress.total}</span>
                <span>{Math.round((captionProgress.current / captionProgress.total) * 100)}%</span>
              </div>
              <div className="h-2 bg-zinc-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 rounded-full transition-all duration-300"
                  style={{ width: `${(captionProgress.current / captionProgress.total) * 100}%` }}
                />
              </div>
            </div>

            {/* Current file */}
            <p className="text-xs text-zinc-500 truncate">
              Processing: {captionProgress.currentFile}
            </p>

            {captionProgress.failed > 0 && (
              <p className="text-xs text-red-400">{captionProgress.failed} failed</p>
            )}

            <div className="flex justify-end">
              <button
                onClick={handleCancelCaptioning}
                disabled={captionProgress.cancelRequested}
                className="px-3 py-1.5 bg-red-600 hover:bg-red-700 disabled:bg-zinc-700 disabled:text-zinc-500 text-white text-xs rounded transition-colors"
              >
                {captionProgress.cancelRequested ? 'Cancelling...' : 'Cancel'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Header controls */}
      <div className="px-4 py-3 border-b border-zinc-800 space-y-3">
        {/* Backend selector */}
        <div className="flex items-center gap-3">
          <label className="text-xs text-zinc-400 whitespace-nowrap">Backend:</label>
          <select
            value={selectedBackend}
            onChange={e => setSelectedBackend(e.target.value as CaptionBackendId)}
            className="flex-1 max-w-xs px-2 py-1 bg-zinc-800 border border-zinc-700 rounded text-xs text-zinc-200 focus:outline-none focus:border-blue-500"
          >
            {captioning.backends.map(b => (
              <option key={b.backend_id} value={b.backend_id} disabled={!b.is_configured}>
                {b.display_name}{!b.is_configured ? ' (not configured)' : ''}
              </option>
            ))}
            {captioning.backends.length === 0 && (
              <option value="local">Local: Qwen3-VL</option>
            )}
          </select>
        </div>

        {/* Local model configuration overrides.
            The size + abliterated combination is presented as a single
            "Model" dropdown so the user never has an intermediate invalid
            state (e.g. flipping the abliterated checkbox while the backend
            is mid-load). The Load Model button is always rendered and is
            only disabled when the current selection already matches the
            loaded model. */}
        {selectedBackend === 'local' && (
          <div className="flex items-center gap-3 flex-wrap">
            <label className="text-xs text-zinc-400">Model:</label>
            <select
              value={findVariantId(modelSize, abliterated)}
              onChange={e => {
                const variant = LOCAL_MODEL_VARIANTS.find(v => v.id === e.target.value)
                if (!variant) return
                setModelSize(variant.size)
                setAbliterated(variant.abliterated)
              }}
              className="px-2 py-1 bg-zinc-800 border border-zinc-700 rounded text-xs text-zinc-200 focus:outline-none focus:border-blue-500 min-w-[16rem]"
            >
              {LOCAL_MODEL_VARIANTS.map(v => (
                <option key={v.id} value={v.id}>{v.label}</option>
              ))}
            </select>

            <label className="text-xs text-zinc-400">Quant:</label>
            <select
              value={quantization}
              onChange={e => setQuantization(e.target.value as CaptionQuantization)}
              className="px-2 py-1 bg-zinc-800 border border-zinc-700 rounded text-xs text-zinc-200 focus:outline-none focus:border-blue-500"
            >
              <option value="fp16">FP16</option>
              <option value="8bit">8-bit</option>
              <option value="4bit">4-bit</option>
            </select>

            {gpuDevices.length > 1 && (
              <>
                <label className="text-xs text-zinc-400">GPU:</label>
                <select
                  value={gpuIndex}
                  onChange={e => setGpuIndex(parseInt(e.target.value))}
                  className="px-2 py-1 bg-zinc-800 border border-zinc-700 rounded text-xs text-zinc-200 focus:outline-none focus:border-blue-500"
                >
                  {gpuDevices.map(gpu => (
                    <option key={gpu.index} value={gpu.index}>
                      {gpu.index}: {gpu.name}
                    </option>
                  ))}
                </select>
              </>
            )}

            <button
              onClick={handleLoadModel}
              disabled={modelLoading || !modelDiffers}
              title={modelDiffers
                ? 'Load the selected model'
                : 'Selected model is already loaded'}
              className="px-2 py-1 bg-blue-600 hover:bg-blue-700 disabled:bg-zinc-700 disabled:text-zinc-500 text-white text-xs rounded transition-colors"
            >
              {modelLoading ? 'Loading...' : modelDiffers ? 'Load Model' : 'Loaded'}
            </button>

            {/* Unload Model: only meaningful when a model is currently
                resident. Frees the VL model's VRAM without leaving the
                Caption tab (useful before a training run on the same GPU). */}
            {captioning.modelStatus?.state === 'ready' && (
              <button
                onClick={() => void captioning.unloadModel()}
                disabled={modelLoading}
                title="Unload the model and free its GPU memory"
                className="px-2 py-1 bg-zinc-700 hover:bg-zinc-600 disabled:bg-zinc-800 disabled:text-zinc-600 text-zinc-200 text-xs rounded transition-colors"
              >
                Unload Model
              </button>
            )}
          </div>
        )}

        {/* Model status for local backend */}
        {selectedBackend === 'local' && captioning.modelStatus && !modelLoading && (
          <div className="text-xs text-zinc-500">
            {captioning.modelStatus.state === 'ready' && captioning.modelStatus.model_choice && (
              <span className="text-green-400">
                Active: Qwen3-VL-{captioning.modelStatus.model_choice.size}
                {captioning.modelStatus.model_choice.abliterated ? ' (abliterated)' : ''}
                {' '}{captioning.modelStatus.model_choice.quantization}
              </span>
            )}
            {captioning.modelStatus.state === 'not_started' && (
              <span className="text-amber-500">No local model selected. Select one in Settings.</span>
            )}
            {captioning.modelStatus.state === 'error' && (
              <span className="text-red-400">{captioning.modelStatus.error_message}</span>
            )}
          </div>
        )}

        {/* Batch actions */}
        <div className="flex items-center gap-2 flex-wrap">
          <button
            onClick={handleCaptionAll}
            disabled={captionProgress.running || modelLoading || clips.length === 0}
            className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 disabled:bg-zinc-700 disabled:text-zinc-500 text-white text-xs rounded transition-colors"
          >
            Caption All ({clips.length})
          </button>
          <button
            onClick={handleCaptionSelected}
            disabled={captionProgress.running || modelLoading || selectedClipIds.size === 0}
            className="px-3 py-1.5 bg-zinc-700 hover:bg-zinc-600 disabled:bg-zinc-800 disabled:text-zinc-600 text-zinc-200 text-xs rounded transition-colors"
          >
            Caption Selected ({selectedClipIds.size})
          </button>
          <div className="flex-1" />
          <button onClick={handleSelectAll} className="text-xs text-zinc-400 hover:text-zinc-200">
            Select all
          </button>
          <button onClick={handleSelectUncaptioned} className="text-xs text-zinc-400 hover:text-zinc-200">
            Select uncaptioned ({uncaptionedClips.length})
          </button>
          <button onClick={handleClearSelection} className="text-xs text-zinc-400 hover:text-zinc-200">
            Clear
          </button>
        </div>
      </div>

      {/* Error banner */}
      {captioning.error && (
        <div className="px-4 py-2 bg-red-900/30 border-b border-red-800 text-red-400 text-xs flex items-center justify-between">
          <span>{captioning.error}</span>
          <button onClick={captioning.clearError} className="text-red-300 hover:text-red-100 ml-2">
            Dismiss
          </button>
        </div>
      )}

      {/* Clip list with selection + caption preview */}
      <div className="flex-1 overflow-auto p-4">
        {clips.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <p className="text-sm text-zinc-500">No clips in the dataset. Import sources and cut clips first.</p>
          </div>
        ) : (
          <div className="space-y-1">
            {clips.map(clip => {
              const isSelected = selectedClipIds.has(clip.clip_id)
              const hasCaption = clip.caption && clip.caption.trim() !== ''
              return (
                <div
                  key={clip.clip_id}
                  onClick={() => toggleClip(clip.clip_id)}
                  className={`flex items-start gap-3 px-3 py-2 rounded cursor-pointer transition-colors ${
                    isSelected ? 'bg-blue-900/30 border border-blue-700' : 'bg-zinc-900 border border-zinc-800 hover:border-zinc-700'
                  }`}
                >
                  <div className="pt-0.5">
                    <div className={`w-4 h-4 rounded border flex items-center justify-center ${
                      isSelected ? 'bg-blue-600 border-blue-500' : 'border-zinc-600'
                    }`}>
                      {isSelected && <span className="text-white text-xs">v</span>}
                    </div>
                  </div>
                  {/* Small thumbnail so the user can identify the clip at a
                      glance without opening the Browser tab. */}
                  <ClipRowThumbnail
                    datasetDir={datasetDir}
                    clipId={clip.clip_id}
                    filename={clip.filename}
                    isImage={clip.duration_s === 0}
                    durationSeconds={clip.duration_s}
                  />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono text-zinc-400">{clip.filename}</span>
                      <span className="text-xs text-zinc-600">
                        {clip.duration_s > 0 ? `${clip.duration_s.toFixed(1)}s` : 'Image'}
                      </span>
                      {!hasCaption && (
                        <span className="text-xs text-amber-600 font-medium">No caption</span>
                      )}
                    </div>
                    {hasCaption && (
                      <p className="text-xs text-zinc-400 mt-1 line-clamp-2">{clip.caption}</p>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
