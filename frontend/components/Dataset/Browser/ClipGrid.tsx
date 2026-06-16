/**
 * ClipGrid: displays the dataset clips as a grid of thumbnail cards.
 * Each card shows clip info and an editable caption.
 *
 * Clicking a thumbnail opens a preview:
 *   - Image clips open in an image lightbox.
 *   - Video clips open in a video preview modal that streams the clip's mp4
 *     from the backend (same approach used by the Cutter scene preview).
 */

import { useState, useEffect, useRef, useCallback } from 'react'
import { backendFetch } from '../../../lib/backend'
import type { ClipRecord } from '../../../types/dataset'

interface ClipGridProps {
  clips: ClipRecord[]
  datasetDir: string
  isLoading: boolean
  onDeleteClip: (datasetDir: string, clipId: string) => Promise<void>
  onUpdateCaption: (datasetDir: string, clipId: string, caption: string) => Promise<void>
}

// Preview state for the modal at the grid level. Either nothing, an image,
// or a video clip (the clip record is enough to build the stream URL).
type PreviewState =
  | { kind: 'image'; src: string; alt: string }
  | { kind: 'video'; clip: ClipRecord }
  | null

function ClipCard({
  clip,
  datasetDir,
  onDelete,
  onUpdateCaption,
  onPreview,
}: {
  clip: ClipRecord
  datasetDir: string
  onDelete: (datasetDir: string, clipId: string) => Promise<void>
  onUpdateCaption: (datasetDir: string, clipId: string, caption: string) => Promise<void>
  onPreview: (state: PreviewState) => void
}) {
  const [caption, setCaption] = useState(clip.caption)
  const [editing, setEditing] = useState(false)
  const [thumbnailSrc, setThumbnailSrc] = useState<string | null>(null)

  const isImage = clip.duration_s === 0
  const isPortrait = clip.width && clip.height ? clip.width < clip.height : false

  // Fetch thumbnail from backend as base64.
  useEffect(() => {
    let cancelled = false
    backendFetch('/api/dataset/clips/thumbnail', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset_dir: datasetDir, clip_id: clip.clip_id }),
    })
      .then(async resp => {
        if (!resp.ok || cancelled) return
        const data = await resp.json()
        if (!cancelled && data.thumbnail_b64) {
          setThumbnailSrc(`data:image/png;base64,${data.thumbnail_b64}`)
        }
      })
      .catch(() => { /* thumbnail load failed, show fallback */ })
    return () => { cancelled = true }
  }, [datasetDir, clip.clip_id])

  const handleSaveCaption = async () => {
    if (caption !== clip.caption) {
      await onUpdateCaption(datasetDir, clip.clip_id, caption)
    }
    setEditing(false)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      void handleSaveCaption()
    }
    if (e.key === 'Escape') {
      setCaption(clip.caption)
      setEditing(false)
    }
  }

  // Open the right kind of preview for this clip's media type.
  const handleThumbnailClick = () => {
    if (!thumbnailSrc) return
    if (isImage) {
      onPreview({ kind: 'image', src: thumbnailSrc, alt: clip.filename })
    } else {
      onPreview({ kind: 'video', clip })
    }
  }

  return (
    <div className="bg-zinc-900 rounded border border-zinc-800 overflow-hidden">
      {/* Thumbnail with correct aspect ratio */}
      <div
        className="bg-zinc-800 flex items-center justify-center overflow-hidden relative"
        style={{
          aspectRatio: `${clip.width || 16} / ${clip.height || 9}`,
          maxHeight: isPortrait ? '280px' : undefined,
        }}
      >
        {thumbnailSrc ? (
          <>
            <img
              src={thumbnailSrc}
              alt={clip.filename}
              className="w-full h-full object-cover cursor-pointer"
              onClick={handleThumbnailClick}
            />
            {/* Play overlay for video clips so the user knows the thumbnail is clickable. */}
            {!isImage && (
              <button
                onClick={handleThumbnailClick}
                className="absolute inset-0 flex items-center justify-center bg-black/0 hover:bg-black/40 transition-colors"
                title="Preview clip"
              >
                <svg className="w-10 h-10 text-white opacity-70 drop-shadow-lg" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M8 5v14l11-7z" />
                </svg>
              </button>
            )}
          </>
        ) : (
          <div className="text-center">
            <span className="text-xs text-zinc-500">
              {isImage ? 'Loading...' : `${clip.duration_s.toFixed(1)}s`}
            </span>
            <p className="text-xs text-zinc-600 mt-1">{clip.filename}</p>
          </div>
        )}
      </div>

      {/* Info row */}
      <div className="px-2 py-1.5 border-t border-zinc-800">
        <div className="flex items-center gap-2 text-xs text-zinc-500">
          <span>{clip.width}x{clip.height}</span>
          {!isImage && <span>{clip.fps}fps</span>}
          {clip.has_audio && <span>Audio</span>}
          <div className="flex-1" />
          <button
            onClick={() => void onDelete(datasetDir, clip.clip_id)}
            className="text-red-500 hover:text-red-400 transition-colors"
            title="Delete clip"
          >
            Delete
          </button>
        </div>
      </div>

      {/* Caption */}
      <div className="px-2 pb-2">
        {editing ? (
          <textarea
            value={caption}
            onChange={e => setCaption(e.target.value)}
            onBlur={() => void handleSaveCaption()}
            onKeyDown={handleKeyDown}
            autoFocus
            rows={2}
            className="w-full px-1.5 py-1 bg-zinc-800 border border-zinc-600 rounded text-xs text-zinc-200 resize-none focus:outline-none focus:border-blue-500"
            placeholder="Enter caption..."
          />
        ) : (
          <button
            onClick={() => setEditing(true)}
            className="w-full text-left px-1.5 py-1 text-xs rounded hover:bg-zinc-800 transition-colors min-h-[2rem]"
          >
            {clip.caption ? (
              <span className="text-zinc-300">{clip.caption}</span>
            ) : (
              <span className="text-zinc-600 italic">Click to add caption</span>
            )}
          </button>
        )}
      </div>
    </div>
  )
}

function ImageLightbox({
  src,
  alt,
  onClose,
}: {
  src: string
  alt: string
  onClose: () => void
}) {
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [onClose])

  return (
    <div
      className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-8"
      onClick={onClose}
    >
      <div
        className="relative max-w-[90vw] max-h-[90vh]"
        onClick={e => e.stopPropagation()}
      >
        <img
          src={src}
          alt={alt}
          className="max-w-full max-h-[85vh] object-contain rounded"
        />
        <button
          onClick={onClose}
          className="absolute -top-3 -right-3 w-8 h-8 bg-zinc-800 hover:bg-zinc-700 border border-zinc-600 rounded-full flex items-center justify-center text-zinc-300 text-sm transition-colors"
        >
          X
        </button>
      </div>
    </div>
  )
}

/**
 * VideoPreview: streams a dataset video clip into a centered player modal.
 *
 * Implementation mirrors the Cutter scene preview: fetch the file as a blob
 * through `backendFetch` so auth headers are attached, build an object URL,
 * then bind it to a <video> element with native controls. The blob URL is
 * revoked on close to avoid leaking memory.
 */
function VideoPreview({
  clip,
  datasetDir,
  onClose,
}: {
  clip: ClipRecord
  datasetDir: string
  onClose: () => void
}) {
  const [videoUrl, setVideoUrl] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const videoRef = useRef<HTMLVideoElement | null>(null)

  const handleClose = useCallback(() => {
    const video = videoRef.current
    if (video) video.pause()
    if (videoUrl) URL.revokeObjectURL(videoUrl)
    onClose()
  }, [videoUrl, onClose])

  // Close on Escape key.
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') handleClose()
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [handleClose])

  // Fetch the clip mp4 as a blob and turn it into an object URL.
  useEffect(() => {
    let cancelled = false
    setIsLoading(true)
    setErrorMessage(null)
    // The clip file lives at {datasetDir}/clips/{filename}. The stream-video
    // endpoint takes an absolute path query param.
    const clipPath = `${datasetDir}/clips/${clip.filename}`
    backendFetch(`/api/dataset/stream-video?path=${encodeURIComponent(clipPath)}`)
      .then(async resp => {
        if (cancelled) return
        if (!resp.ok) {
          setErrorMessage(`Failed to load clip (status ${resp.status}).`)
          setIsLoading(false)
          return
        }
        const blob = await resp.blob()
        if (cancelled) return
        const url = URL.createObjectURL(blob)
        setVideoUrl(url)
        setIsLoading(false)
      })
      .catch(() => {
        if (!cancelled) {
          setErrorMessage('Failed to load clip.')
          setIsLoading(false)
        }
      })
    return () => { cancelled = true }
  }, [datasetDir, clip.filename])

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
          <div className="min-w-0">
            <p className="text-sm font-medium text-zinc-200 truncate">{clip.filename}</p>
            <p className="text-xs text-zinc-500 mt-0.5">
              {clip.width}x{clip.height}
              {clip.fps ? ` | ${clip.fps}fps` : ''}
              {clip.duration_s ? ` | ${clip.duration_s.toFixed(1)}s` : ''}
              {clip.has_audio ? ' | Audio' : ''}
            </p>
          </div>
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
            <span className="text-xs text-zinc-400 ml-2">Loading clip...</span>
          </div>
        )}

        {errorMessage && (
          <div className="flex items-center justify-center h-32 bg-black rounded">
            <p className="text-xs text-red-400">{errorMessage}</p>
          </div>
        )}

        {videoUrl && !errorMessage && (
          <video
            ref={videoRef}
            src={videoUrl}
            controls
            autoPlay
            className="w-full rounded bg-black"
            style={{ maxHeight: '70vh' }}
          />
        )}
      </div>
    </div>
  )
}

export function ClipGrid({
  clips,
  datasetDir,
  isLoading,
  onDeleteClip,
  onUpdateCaption,
}: ClipGridProps) {
  const [preview, setPreview] = useState<PreviewState>(null)

  if (isLoading && clips.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-sm text-zinc-500">Loading clips...</p>
      </div>
    )
  }

  if (clips.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <p className="text-sm text-zinc-500 mb-1">No clips in the dataset yet.</p>
          <p className="text-xs text-zinc-600">Import sources and cut clips to populate the dataset.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="p-4">
      {preview?.kind === 'image' && (
        <ImageLightbox
          src={preview.src}
          alt={preview.alt}
          onClose={() => setPreview(null)}
        />
      )}
      {preview?.kind === 'video' && (
        <VideoPreview
          clip={preview.clip}
          datasetDir={datasetDir}
          onClose={() => setPreview(null)}
        />
      )}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
        {clips.map(clip => (
          <ClipCard
            key={clip.clip_id}
            clip={clip}
            datasetDir={datasetDir}
            onDelete={onDeleteClip}
            onUpdateCaption={onUpdateCaption}
            onPreview={setPreview}
          />
        ))}
      </div>
    </div>
  )
}
