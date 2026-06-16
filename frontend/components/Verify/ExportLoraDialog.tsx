import { useState } from 'react'
import { Download } from 'lucide-react'
import { Button } from '../ui/button'
import type { ExportLoraRequest } from '../../hooks/useVerification'

interface ExportLoraDialogProps {
  checkpointPath: string
  previewGenerationId: string | null
  onExport: (request: ExportLoraRequest) => void
  loading: boolean
}

export function ExportLoraDialog({
  checkpointPath,
  previewGenerationId,
  onExport,
  loading,
}: ExportLoraDialogProps) {
  const [destinationDir, setDestinationDir] = useState('')
  const [includeConfig, setIncludeConfig] = useState(true)
  const [includePreview, setIncludePreview] = useState(true)

  const handleBrowse = async () => {
    if (!window.electronAPI?.showOpenDirectoryDialog) return
    const result = await window.electronAPI.showOpenDirectoryDialog({ title: 'Select export destination' })
    if (result) {
      setDestinationDir(result)
    }
  }

  const handleExport = () => {
    if (!destinationDir) return
    onExport({
      checkpoint_path: checkpointPath,
      destination_dir: destinationDir,
      include_config: includeConfig,
      include_preview: includePreview,
      preview_generation_id: includePreview ? previewGenerationId : null,
    })
  }

  const filename = checkpointPath.split('/').pop() || 'checkpoint.safetensors'

  return (
    <div className="bg-zinc-900 border border-zinc-700 rounded-lg p-4 space-y-3">
      <h4 className="text-sm font-medium text-zinc-200">Export LORA</h4>
      <p className="text-xs text-zinc-500">
        Export <span className="text-zinc-300">{filename}</span> and sidecar files.
      </p>

      {/* Destination */}
      <div>
        <label className="text-xs font-medium text-zinc-400">Destination</label>
        <div className="flex gap-2 mt-1">
          <input
            type="text"
            value={destinationDir}
            onChange={e => setDestinationDir(e.target.value)}
            placeholder="/path/to/export"
            className="flex-1 bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-sm text-zinc-200"
          />
          <Button variant="outline" size="sm" onClick={handleBrowse}>
            Browse
          </Button>
        </div>
      </div>

      {/* Options */}
      <div className="space-y-1">
        <label className="flex items-center gap-2 text-xs text-zinc-400">
          <input
            type="checkbox"
            checked={includeConfig}
            onChange={e => setIncludeConfig(e.target.checked)}
            className="rounded"
          />
          Include training config (.json)
        </label>
        <label className="flex items-center gap-2 text-xs text-zinc-400">
          <input
            type="checkbox"
            checked={includePreview}
            onChange={e => setIncludePreview(e.target.checked)}
            className="rounded"
            disabled={!previewGenerationId}
          />
          Include preview video (.mp4)
        </label>
      </div>

      <Button
        onClick={handleExport}
        disabled={loading || !destinationDir}
        className="w-full"
      >
        <Download className="h-4 w-4 mr-2" />
        {loading ? 'Exporting...' : 'Export'}
      </Button>
    </div>
  )
}
