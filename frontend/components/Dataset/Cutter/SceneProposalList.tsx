/**
 * SceneProposalList: shows scene detection proposals and manual clip creation.
 * User can run scene detection, review proposals, and accept them as clips.
 */

import { useState, useEffect, useRef, useCallback } from 'react'
import type { SourceMediaInfo, SceneProposal, ClipResult } from '../../../types/dataset'
import { backendFetch } from '../../../lib/backend'

interface SceneProposalListProps {
  source: SourceMediaInfo
  scenes: SceneProposal[]
  isLoading: boolean
  datasetDir: string
  onDetectScenes: (sourcePath: string, threshold?: number, targetLength?: number) => Promise<void>
  onCreateClip: (sourcePath: string, datasetDir: string, startS: number, endS: number) => Promise<ClipResult | null>
  onCreateBatch: (sourcePath: string, datasetDir: string, segments: { start_s: number; end_s: number }[]) => Promise<ClipResult[]>
  onDone: () => void
  onClearScenes: () => void
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = (seconds % 60).toFixed(1)
  return `${m}:${s.padStart(4, '0')}`
}

const STATUS_COLORS: Record<string, string> = {
  short: 'bg-yellow-900/50 text-yellow-400 border-yellow-700',
  on_target: 'bg-green-900/50 text-green-400 border-green-700',
  long: 'bg-red-900/50 text-red-400 border-red-700',
}

export function SceneProposalList({
  source,
  scenes,
  isLoading,
  datasetDir,
  onDetectScenes,
  onCreateClip,
  onCreateBatch,
  onDone,
  onClearScenes,
}: SceneProposalListProps) {
  const [selectedScenes, setSelectedScenes] = useState<Set<number>>(new Set())
  const [manualStart, setManualStart] = useState('')
  const [manualEnd, setManualEnd] = useState('')
  const [previewScene, setPreviewScene] = useState<SceneProposal | null>(null)
  const [videoUrl, setVideoUrl] = useState<string | null>(null)
  const [videoLoading, setVideoLoading] = useState(false)
  const videoRef = useRef<HTMLVideoElement | null>(null)

  const [detectPhase, setDetectPhase] = useState<string | null>(null)
  const elapsedRef = useRef(0)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const [elapsedDisplay, setElapsedDisplay] = useState(0)

  // Elapsed timer for the progress dialog.
  useEffect(() => {
    if (detectPhase !== null) {
      elapsedRef.current = 0
      setElapsedDisplay(0)
      timerRef.current = setInterval(() => {
        elapsedRef.current += 1
        setElapsedDisplay(elapsedRef.current)
      }, 1000)
    } else {
      if (timerRef.current) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [detectPhase])

  // Fetch video as blob with auth headers, then create object URL.
  const openPreview = useCallback(async (scene: SceneProposal) => {
    setPreviewScene(scene)
    setVideoLoading(true)
    setVideoUrl(null)
    try {
      const resp = await backendFetch(`/api/dataset/stream-video?path=${encodeURIComponent(source.path)}`)
      if (!resp.ok) {
        setVideoLoading(false)
        return
      }
      const blob = await resp.blob()
      const blobUrl = URL.createObjectURL(blob)
      setVideoUrl(blobUrl)
    } catch {
      // Silently fail preview.
    } finally {
      setVideoLoading(false)
    }
  }, [source.path])

  const closePreview = useCallback(() => {
    const video = videoRef.current
    if (video) {
      video.pause()
    }
    if (videoUrl) URL.revokeObjectURL(videoUrl)
    videoRef.current = null
    setPreviewScene(null)
    setVideoUrl(null)
  }, [videoUrl])

  // When the video element loads metadata, seek to the clip start time.
  const handleVideoLoaded = useCallback(() => {
    const video = videoRef.current
    if (!video || !previewScene) return
    video.currentTime = previewScene.start_s
  }, [previewScene])

  // Stop playback at the clip end time.
  const handleTimeUpdate = useCallback(() => {
    const video = videoRef.current
    if (!video || !previewScene) return
    if (video.currentTime >= previewScene.end_s) {
      video.pause()
      video.currentTime = previewScene.start_s
    }
  }, [previewScene])

  const handleDetect = async () => {
    setDetectPhase('Analyzing video frames for scene boundaries...')
    try {
      await onDetectScenes(source.path)
      setDetectPhase(null)
    } catch {
      setDetectPhase(null)
    }
  }

  const toggleScene = (index: number) => {
    setSelectedScenes(prev => {
      const next = new Set(prev)
      if (next.has(index)) {
        next.delete(index)
      } else {
        next.add(index)
      }
      return next
    })
  }

  const selectAll = () => {
    setSelectedScenes(new Set(scenes.map((_, i) => i)))
  }

  const selectNone = () => {
    setSelectedScenes(new Set())
  }

  const handleAcceptSelected = async () => {
    const segments = scenes
      .filter((_, i) => selectedScenes.has(i))
      .map(s => ({ start_s: s.start_s, end_s: s.end_s }))

    if (segments.length === 0) return
    await onCreateBatch(source.path, datasetDir, segments)
    onClearScenes()
    onDone()
  }

  const handleManualClip = async () => {
    const start = parseFloat(manualStart)
    const end = parseFloat(manualEnd)
    if (isNaN(start) || isNaN(end) || end <= start) return

    await onCreateClip(source.path, datasetDir, start, end)
    setManualStart('')
    setManualEnd('')
  }

  return (
    <div className="p-4">
      {/* Scene detection progress dialog */}
      {detectPhase && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="bg-zinc-900 border border-zinc-700 rounded-lg p-6 max-w-md w-full mx-4 shadow-xl">
            <h3 className="text-sm font-medium text-zinc-200 mb-3">Detecting Scenes</h3>
            <div className="flex items-center gap-3 mb-3">
              <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
              <p className="text-xs text-zinc-400">{detectPhase}</p>
            </div>
            <div className="w-full bg-zinc-800 rounded-full h-1.5 mb-2">
              <div className="bg-blue-600 h-1.5 rounded-full animate-pulse" style={{ width: '100%' }} />
            </div>
            <p className="text-xs text-zinc-500">
              Elapsed: {Math.floor(elapsedDisplay / 60)}:{String(elapsedDisplay % 60).padStart(2, '0')}
              {' '} | This may take several minutes for long videos.
            </p>
          </div>
        </div>
      )}

      {/* Video preview modal */}
      {previewScene && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70" onClick={closePreview}>
          <div className="bg-zinc-900 border border-zinc-700 rounded-lg p-4 max-w-2xl w-full mx-4 shadow-xl" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-medium text-zinc-200">
                Preview: {formatTime(previewScene.start_s)} - {formatTime(previewScene.end_s)}
                <span className="text-zinc-500 ml-2">({previewScene.duration_s.toFixed(1)}s)</span>
              </h3>
              <button onClick={closePreview} className="text-zinc-400 hover:text-zinc-200 text-lg px-2">&times;</button>
            </div>
            {videoLoading && (
              <div className="flex items-center justify-center h-48 bg-black rounded">
                <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                <span className="text-xs text-zinc-400 ml-2">Loading video...</span>
              </div>
            )}
            {videoUrl && (
              <video
                ref={videoRef}
                src={videoUrl}
                controls
                onLoadedMetadata={handleVideoLoaded}
                onTimeUpdate={handleTimeUpdate}
                className="w-full rounded bg-black"
                style={{ maxHeight: '400px' }}
              />
            )}
            <p className="text-xs text-zinc-500 mt-2">
              Press play to preview. Playback stops at the clip boundary.
              Verify audio does not cut off mid-word.
            </p>
          </div>
        </div>
      )}

      {/* Source info header */}
      <div className="mb-4 p-3 bg-zinc-900 rounded border border-zinc-800">
        <p className="text-sm text-zinc-200">{source.filename}</p>
        <p className="text-xs text-zinc-500 mt-1">
          {source.width}x{source.height} | {source.fps} fps | {formatTime(source.duration_s)}
        </p>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-2 mb-4">
        <button
          onClick={handleDetect}
          disabled={isLoading}
          className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white text-xs rounded transition-colors"
        >
          {isLoading ? 'Detecting...' : 'Detect Scenes'}
        </button>

        {scenes.length > 0 && (
          <>
            <button onClick={selectAll} className="px-2 py-1 text-xs text-zinc-400 hover:text-zinc-200">
              Select All
            </button>
            <button onClick={selectNone} className="px-2 py-1 text-xs text-zinc-400 hover:text-zinc-200">
              Select None
            </button>
            <div className="flex-1" />
            <button
              onClick={handleAcceptSelected}
              disabled={selectedScenes.size === 0 || isLoading}
              className="px-3 py-1.5 bg-green-700 hover:bg-green-600 disabled:opacity-50 text-white text-xs rounded transition-colors"
            >
              Accept {selectedScenes.size} Clip{selectedScenes.size !== 1 ? 's' : ''}
            </button>
          </>
        )}
      </div>

      {/* Scene proposals */}
      {scenes.length > 0 && (
        <div className="space-y-1 mb-6">
          {scenes.map((scene, idx) => (
            <div
              key={scene.scene_index}
              onClick={() => toggleScene(idx)}
              className={`flex items-center gap-3 p-2 rounded border cursor-pointer transition-colors ${
                selectedScenes.has(idx)
                  ? 'bg-blue-900/30 border-blue-700'
                  : 'bg-zinc-900 border-zinc-800 hover:border-zinc-700'
              }`}
            >
              <input
                type="checkbox"
                checked={selectedScenes.has(idx)}
                onChange={() => toggleScene(idx)}
                className="accent-blue-500"
              />

              {scene.thumbnail_b64 && (
                <div className="relative group">
                  <img
                    src={`data:image/png;base64,${scene.thumbnail_b64}`}
                    alt={`Scene ${scene.scene_index}`}
                    className="w-16 h-9 object-cover rounded"
                  />
                  <button
                    onClick={(e) => { e.stopPropagation(); void openPreview(scene) }}
                    className="absolute inset-0 flex items-center justify-center bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity rounded"
                    title="Preview clip"
                  >
                    <svg className="w-5 h-5 text-white" viewBox="0 0 24 24" fill="currentColor">
                      <path d="M8 5v14l11-7z" />
                    </svg>
                  </button>
                </div>
              )}

              <div className="flex-1 min-w-0">
                <span className="text-xs text-zinc-300">
                  {formatTime(scene.start_s)} - {formatTime(scene.end_s)}
                </span>
                <span className="text-xs text-zinc-500 ml-2">
                  ({scene.duration_s.toFixed(1)}s)
                </span>
              </div>

              <span className={`text-xs px-2 py-0.5 rounded border ${STATUS_COLORS[scene.length_status] || ''}`}>
                {scene.length_status}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Manual clip creation */}
      <div className="border-t border-zinc-800 pt-4">
        <h4 className="text-xs font-medium text-zinc-400 mb-2">Manual Clip</h4>
        <div className="flex items-center gap-2">
          <input
            type="number"
            placeholder="Start (s)"
            value={manualStart}
            onChange={e => setManualStart(e.target.value)}
            className="w-24 px-2 py-1 bg-zinc-900 border border-zinc-700 rounded text-xs text-zinc-200 focus:outline-none focus:border-blue-500"
            step="0.1"
            min="0"
          />
          <span className="text-xs text-zinc-500">to</span>
          <input
            type="number"
            placeholder="End (s)"
            value={manualEnd}
            onChange={e => setManualEnd(e.target.value)}
            className="w-24 px-2 py-1 bg-zinc-900 border border-zinc-700 rounded text-xs text-zinc-200 focus:outline-none focus:border-blue-500"
            step="0.1"
            min="0"
          />
          <button
            onClick={handleManualClip}
            disabled={isLoading}
            className="px-3 py-1 bg-zinc-700 hover:bg-zinc-600 disabled:opacity-50 text-white text-xs rounded transition-colors"
          >
            Add Clip
          </button>
        </div>
      </div>
    </div>
  )
}
