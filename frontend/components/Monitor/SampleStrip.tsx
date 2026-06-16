/**
 * Horizontal strip of training sample previews.
 *
 * Polls the backend for sample videos generated during training and
 * displays them as a scrollable strip with step labels. Clicking a tile
 * opens a centered player modal so the operator can watch the clip at a
 * usable size with playback and volume controls, the same way the
 * Dataset page's Browser tab opens a clip preview.
 *
 * Playback note: the sample files live on disk outside the renderer's
 * sandbox, and the app's CSP does not allow loading raw ``file://`` URLs
 * in <video>/<img> tags. We therefore stream each file through the
 * backend (``/api/dataset/stream-video``, which serves any absolute path
 * via FileResponse), read it as a Blob, and bind an object URL to the
 * element. This is the exact pattern the Dataset page's clip preview
 * uses, so training samples play the same way dataset clips do.
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import { backendFetch } from '../../lib/backend'

interface SampleInfo {
  step: number
  path: string
}

interface SampleStripProps {
  jobId: string | null
}

const POLL_INTERVAL_MS = 5000

export function SampleStrip({ jobId }: SampleStripProps) {
  const [samples, setSamples] = useState<SampleInfo[]>([])
  // The sample currently open in the player modal, or null when closed.
  // Lifted to the strip level so only one modal exists at a time and the
  // tiles stay lightweight thumbnails.
  const [previewSample, setPreviewSample] = useState<SampleInfo | null>(null)

  const fetchSamples = useCallback(async () => {
    if (!jobId) return
    try {
      const res = await backendFetch(`/api/training/jobs/${jobId}/samples`)
      if (!res.ok) return
      const data = (await res.json()) as SampleInfo[]
      setSamples(data)
    } catch {
      // Silently ignore sample fetch errors
    }
  }, [jobId])

  useEffect(() => {
    setSamples([])
    setPreviewSample(null)
    if (jobId) fetchSamples()
  }, [jobId, fetchSamples])

  useEffect(() => {
    if (!jobId) return
    const interval = setInterval(fetchSamples, POLL_INTERVAL_MS)
    return () => clearInterval(interval)
  }, [jobId, fetchSamples])

  if (samples.length === 0) {
    return (
      <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-4">
        <span className="text-xs font-medium text-zinc-400">Samples</span>
        <div className="flex items-center justify-center h-24 mt-2">
          <span className="text-xs text-zinc-600">
            No samples generated yet. Samples appear periodically during training.
          </span>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-4">
      <span className="text-xs font-medium text-zinc-400 mb-2 block">
        Samples ({samples.length})
      </span>
      <div className="flex gap-3 overflow-x-auto pb-2">
        {samples.map(sample => (
          // Key on the full path, not the step: a single step emits one
          // sample per prompt spec (step_<NNNNNN>_prompt_<NN>.mp4), so
          // multiple samples share the same step. Keying on step would
          // collide and React would drop all but the first preview.
          <SamplePreview
            key={sample.path}
            sample={sample}
            onOpen={() => setPreviewSample(sample)}
          />
        ))}
      </div>
      {previewSample && (
        <SamplePreviewModal
          sample={previewSample}
          onClose={() => setPreviewSample(null)}
        />
      )}
    </div>
  )
}

/**
 * One sample tile. Streams the file from the backend into an object URL
 * so it plays inside the sandboxed renderer (a raw file:// src is blocked
 * by the app CSP). The object URL is revoked on unmount / path change.
 *
 * Tile sizing honors the sample's real aspect ratio (issue 9). The tile
 * keeps a fixed WIDTH so the strip stays tidy and derives its HEIGHT from
 * the loaded video's intrinsic dimensions, so a square preview renders
 * square and a portrait preview renders tall. ``object-contain`` shows
 * the whole frame (letterboxed at worst) instead of center-cropping it.
 *
 * Playback is thumbnail-first to avoid the autoplay race that left some
 * tiles stuck on a black first frame: the tile loads paused, seeks to a
 * mid-clip frame for a stable poster, and plays on hover. This makes the
 * strip render consistently instead of "two images + one black video".
 * Clicking the tile opens the full player modal via ``onOpen``.
 */
function SamplePreview({ sample, onOpen }: { sample: SampleInfo; onOpen: () => void }) {
  const [mediaUrl, setMediaUrl] = useState<string | null>(null)
  // Default to a square tile until the real dimensions load; 512x512 is
  // the preset sample resolution so this is the common case.
  const [aspectRatio, setAspectRatio] = useState<number>(1)
  const isVideo = sample.path.toLowerCase().endsWith('.mp4')
  const videoRef = useRef<HTMLVideoElement | null>(null)
  // Keep the latest object URL in a ref so the cleanup effect can revoke
  // it without re-running every render.
  const urlRef = useRef<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setMediaUrl(null)
    backendFetch(`/api/dataset/stream-video?path=${encodeURIComponent(sample.path)}`)
      .then(async resp => {
        if (cancelled || !resp.ok) return
        const blob = await resp.blob()
        if (cancelled) return
        const url = URL.createObjectURL(blob)
        urlRef.current = url
        setMediaUrl(url)
      })
      .catch(() => {
        // Sample may not be flushed to disk yet; the poll will retry.
      })
    return () => {
      cancelled = true
      if (urlRef.current) {
        URL.revokeObjectURL(urlRef.current)
        urlRef.current = null
      }
    }
  }, [sample.path])

  // Record the intrinsic aspect ratio and seek to a mid-clip frame so the
  // paused tile shows a meaningful poster instead of a (often black) first
  // frame.
  const handleLoadedMetadata = () => {
    const video = videoRef.current
    if (!video) return
    if (video.videoWidth > 0 && video.videoHeight > 0) {
      setAspectRatio(video.videoWidth / video.videoHeight)
    }
    if (Number.isFinite(video.duration) && video.duration > 0) {
      try {
        video.currentTime = video.duration / 2
      } catch {
        // Seeking can throw if metadata is incomplete; ignore and keep
        // the first frame as the poster.
      }
    }
  }

  const handleMouseEnter = () => {
    void videoRef.current?.play().catch(() => {
      // Autoplay-on-hover can be rejected; ignore.
    })
  }

  const handleMouseLeave = () => {
    const video = videoRef.current
    if (!video) return
    video.pause()
  }

  return (
    <div className="flex-shrink-0 text-center">
      <button
        type="button"
        onClick={onOpen}
        title="Click to play with sound"
        className="w-40 bg-zinc-800 rounded border border-zinc-700 overflow-hidden flex items-center justify-center cursor-pointer relative group"
        style={{ aspectRatio: String(aspectRatio) }}
      >
        {!mediaUrl ? (
          <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
        ) : isVideo ? (
          <>
            <video
              ref={videoRef}
              src={mediaUrl}
              className="w-full h-full object-contain"
              muted
              loop
              playsInline
              preload="metadata"
              onLoadedMetadata={handleLoadedMetadata}
              onMouseEnter={handleMouseEnter}
              onMouseLeave={handleMouseLeave}
            />
            {/* Play overlay so the tile reads as clickable. */}
            <span className="absolute inset-0 flex items-center justify-center bg-black/0 group-hover:bg-black/30 transition-colors pointer-events-none">
              <svg className="w-8 h-8 text-white opacity-0 group-hover:opacity-80 drop-shadow-lg transition-opacity" viewBox="0 0 24 24" fill="currentColor">
                <path d="M8 5v14l11-7z" />
              </svg>
            </span>
          </>
        ) : (
          <img
            src={mediaUrl}
            alt={`Sample at step ${sample.step}`}
            className="w-full h-full object-contain"
          />
        )}
      </button>
      <span className="text-xs text-zinc-500 mt-1 block">Step {sample.step}</span>
    </div>
  )
}

/**
 * SamplePreviewModal: streams a training sample into a centered player modal.
 *
 * Mirrors the Dataset Browser's ``VideoPreview``: fetch the file as a blob
 * through ``backendFetch`` so auth headers are attached, build an object
 * URL, then bind it to a <video> element with native ``controls`` (which
 * include the volume slider). The clip is NOT muted so the audio the joint
 * LTX-2 model generated is audible; native controls let the operator set
 * volume or mute. The blob URL is revoked on close to avoid leaking memory.
 */
function SamplePreviewModal({ sample, onClose }: { sample: SampleInfo; onClose: () => void }) {
  const [videoUrl, setVideoUrl] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const urlRef = useRef<string | null>(null)
  const isVideo = sample.path.toLowerCase().endsWith('.mp4')

  const handleClose = useCallback(() => {
    const video = videoRef.current
    if (video) video.pause()
    if (urlRef.current) {
      URL.revokeObjectURL(urlRef.current)
      urlRef.current = null
    }
    onClose()
  }, [onClose])

  // Close on Escape key.
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') handleClose()
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [handleClose])

  // Fetch the sample file as a blob and turn it into an object URL.
  useEffect(() => {
    let cancelled = false
    setIsLoading(true)
    setErrorMessage(null)
    backendFetch(`/api/dataset/stream-video?path=${encodeURIComponent(sample.path)}`)
      .then(async resp => {
        if (cancelled) return
        if (!resp.ok) {
          setErrorMessage(`Failed to load sample (status ${resp.status}).`)
          setIsLoading(false)
          return
        }
        const blob = await resp.blob()
        if (cancelled) return
        const url = URL.createObjectURL(blob)
        urlRef.current = url
        setVideoUrl(url)
        setIsLoading(false)
      })
      .catch(() => {
        if (!cancelled) {
          setErrorMessage('Failed to load sample.')
          setIsLoading(false)
        }
      })
    return () => {
      cancelled = true
      if (urlRef.current) {
        URL.revokeObjectURL(urlRef.current)
        urlRef.current = null
      }
    }
  }, [sample.path])

  return (
    <div
      className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-8"
      onClick={handleClose}
    >
      <div
        className="relative bg-zinc-900 border border-zinc-700 rounded-lg p-4 max-w-3xl w-full"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-3">
          <p className="text-sm font-medium text-zinc-200 truncate">
            Sample at step {sample.step}
          </p>
          <button
            onClick={handleClose}
            className="ml-3 w-8 h-8 bg-zinc-800 hover:bg-zinc-700 border border-zinc-600 rounded-full flex items-center justify-center text-zinc-300 text-sm transition-colors shrink-0"
            title="Close (Esc)"
          >
            X
          </button>
        </div>

        {isLoading && (
          <div className="flex items-center justify-center h-64 bg-black rounded">
            <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <span className="text-xs text-zinc-400 ml-2">Loading sample...</span>
          </div>
        )}

        {errorMessage && (
          <div className="flex items-center justify-center h-32 bg-black rounded">
            <p className="text-xs text-red-400">{errorMessage}</p>
          </div>
        )}

        {videoUrl && !errorMessage && isVideo && (
          <video
            ref={videoRef}
            src={videoUrl}
            controls
            autoPlay
            loop
            className="w-full rounded bg-black"
            style={{ maxHeight: '70vh' }}
          />
        )}

        {videoUrl && !errorMessage && !isVideo && (
          <img
            src={videoUrl}
            alt={`Sample at step ${sample.step}`}
            className="w-full rounded bg-black object-contain"
            style={{ maxHeight: '70vh' }}
          />
        )}
      </div>
    </div>
  )
}
