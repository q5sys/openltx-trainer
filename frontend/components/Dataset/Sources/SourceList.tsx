/**
 * SourceList: displays imported sources and lets the user add new ones.
 *
 * Videos offer two paths:
 *  - "Cut Clips" opens the cutter to detect scenes and create multiple clips.
 *  - "Import as Clip" takes the entire source video and imports it as a single
 *    clip without going through scene detection. This is useful when the user
 *    has already pre-cut their footage externally and just wants every source
 *    file copied into the dataset as-is.
 */

import { useState } from 'react'
import type { SourceMediaInfo, ClipResult, ClipRecord } from '../../../types/dataset'

interface SourceListProps {
  sources: SourceMediaInfo[]
  clips: ClipRecord[]
  isLoading: boolean
  datasetDir: string
  onProbe: (path: string) => Promise<SourceMediaInfo | null>
  onSourceSelected: (source: SourceMediaInfo) => void
  onImportImage: (sourcePath: string, datasetDir: string) => Promise<ClipResult | null>
  onCreateBatch: (
    sourcePath: string,
    datasetDir: string,
    segments: { start_s: number; end_s: number }[],
  ) => Promise<ClipResult[]>
  onClipsImported?: () => void
}

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

function formatResolution(w: number, h: number): string {
  return `${w}x${h}`
}

export function SourceList({
  sources,
  clips,
  isLoading,
  datasetDir,
  onProbe,
  onSourceSelected,
  onImportImage,
  onCreateBatch,
  onClipsImported,
}: SourceListProps) {

  // Tracks which sources are currently being imported as a full-length clip
  // so the button can show a per-row pending state.
  const [importingSources, setImportingSources] = useState<Set<string>>(new Set())

  // Set of source filenames that already have at least one clip in the dataset.
  // Used to flag a source as "Imported" so the user does not import the same
  // file twice. Matches by `source_filename` which the backend stores when a
  // clip is created from a video source.
  const importedSourceFilenames = new Set(
    clips.map(c => c.source_filename).filter(name => name.length > 0)
  )


  const handleAddSource = async () => {
    try {
      const paths = await window.electronAPI.showOpenFileDialog({})
      if (!paths || paths.length === 0) return

      for (const filePath of paths) {
        const info = await onProbe(filePath)
        if (info && info.is_image) {
          // Auto-import images directly.
          await onImportImage(filePath, datasetDir)
        }
      }
    } catch {
      // User cancelled.
    }
  }

  const handleImportAsClip = async (source: SourceMediaInfo) => {
    if (source.duration_s <= 0) return
    setImportingSources(prev => new Set(prev).add(source.path))
    try {
      // Use the existing clips/batch endpoint with a single segment spanning
      // the whole source. This re-uses the same transcode pipeline as the
      // cutter so the resulting clip is LTX-Video compatible.
      await onCreateBatch(source.path, datasetDir, [
        { start_s: 0, end_s: source.duration_s },
      ])
      if (onClipsImported) onClipsImported()
    } finally {
      setImportingSources(prev => {
        const next = new Set(prev)
        next.delete(source.path)
        return next
      })
    }
  }

  return (
    <div className="p-4 pb-8">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-zinc-300">Sources</h3>
        <button
          onClick={handleAddSource}
          disabled={isLoading}
          className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white text-xs rounded transition-colors"
        >
          {isLoading ? 'Loading...' : 'Add Source'}
        </button>
      </div>

      {sources.length === 0 ? (
        <div className="text-center py-12">
          <p className="text-sm text-zinc-500 mb-2">No sources imported yet.</p>
          <p className="text-xs text-zinc-600">
            Click "Add Source" to import video files or images for your dataset.
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          {sources.map((source, idx) => {
            const isImporting = importingSources.has(source.path)
            const isAlreadyImported = importedSourceFilenames.has(source.filename)
            return (
              <div
                key={`${source.path}-${idx}`}
                className="flex items-center gap-3 p-3 bg-zinc-900 rounded border border-zinc-800 hover:border-zinc-700 transition-colors"
              >
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-zinc-200 truncate">{source.filename}</p>
                  <div className="flex gap-3 mt-1 text-xs text-zinc-500">
                    {source.is_image ? (
                      <span>Image</span>
                    ) : (
                      <span>{formatDuration(source.duration_s)}</span>
                    )}
                    <span>{formatResolution(source.width, source.height)}</span>
                    {!source.is_image && <span>{source.fps} fps</span>}
                    {source.has_audio && <span>Audio</span>}
                    <span>{source.codec}</span>
                  </div>
                </div>

                {!source.is_image && (
                  <div className="flex items-center gap-2">
                    {isAlreadyImported && (
                      <span className="text-xs text-green-500 mr-2">Imported</span>
                    )}
                    <button
                      onClick={() => void handleImportAsClip(source)}
                      disabled={isImporting || isAlreadyImported}
                      title={
                        isAlreadyImported
                          ? 'This source has already been imported as a clip'
                          : 'Import the entire source video as a single clip without cutting'
                      }
                      className="px-3 py-1 text-xs bg-zinc-800 hover:bg-zinc-700 disabled:bg-zinc-800 disabled:opacity-50 text-zinc-300 rounded transition-colors"
                    >
                      {isImporting ? 'Importing...' : 'Import as Clip'}
                    </button>
                    <button
                      onClick={() => onSourceSelected(source)}
                      disabled={isImporting}
                      className="px-3 py-1 text-xs bg-zinc-800 hover:bg-zinc-700 disabled:opacity-50 text-zinc-300 rounded transition-colors"
                    >
                      Cut Clips
                    </button>
                  </div>
                )}

                {source.is_image && (
                  <span className="text-xs text-green-500">Imported</span>
                )}
              </div>
            )
          })}
        </div>
      )}

      {sources.length > 0 && (
        <p className="text-xs text-zinc-600 mt-4 text-center">
          To remove individual items from the dataset, use the Browser tab.
        </p>
      )}
    </div>
  )
}
