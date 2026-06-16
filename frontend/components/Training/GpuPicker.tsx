/**
 * GpuPicker: GPU selection dropdown for training jobs.
 *
 * Displays available GPUs and allows the user to select which one
 * to use for a training job. Shows GPU name and VRAM if available.
 */

import { useState, useEffect } from 'react'
import { Cpu } from 'lucide-react'

interface GpuInfo {
  index: number
  name: string
  vramMb: number
}

interface GpuPickerProps {
  /** Currently selected GPU index. */
  value: number
  /** Called when the user selects a GPU. */
  onChange: (gpuIndex: number) => void
  /** Optional CSS class. */
  className?: string
}

export function GpuPicker({ value, onChange, className = '' }: GpuPickerProps) {
  const [gpus, setGpus] = useState<GpuInfo[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Try to detect available GPUs via the backend.
    async function detectGpus() {
      try {
        const info = await window.electronAPI.checkGpu()
        if (info.available) {
          // For now, we get a single GPU from the Electron API.
          // Multi-GPU detection will come from the backend /api/health endpoint.
          setGpus([{
            index: 0,
            name: info.name || 'GPU 0',
            vramMb: info.vram || 0,
          }])
        } else {
          setGpus([])
        }
      } catch {
        // Fallback: assume at least one GPU.
        setGpus([{ index: 0, name: 'GPU 0', vramMb: 0 }])
      }
      setLoading(false)
    }
    detectGpus()
  }, [])

  if (loading) {
    return (
      <div className={`text-xs text-muted-foreground ${className}`}>
        Detecting GPUs...
      </div>
    )
  }

  if (gpus.length === 0) {
    return (
      <div className={`flex items-center gap-2 text-xs text-destructive ${className}`}>
        <Cpu className="h-4 w-4" />
        No GPU detected
      </div>
    )
  }

  if (gpus.length === 1) {
    return (
      <div className={`flex items-center gap-2 text-xs text-muted-foreground ${className}`}>
        <Cpu className="h-4 w-4" />
        <span>{gpus[0].name}</span>
        {gpus[0].vramMb > 0 && (
          <span className="text-muted-foreground/60">
            ({Math.round(gpus[0].vramMb / 1024)} GB)
          </span>
        )}
      </div>
    )
  }

  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <Cpu className="h-4 w-4 text-muted-foreground" />
      <select
        value={value}
        onChange={(e) => onChange(parseInt(e.target.value, 10))}
        className="rounded border border-border bg-background px-2 py-1 text-xs"
      >
        {gpus.map((gpu) => (
          <option key={gpu.index} value={gpu.index}>
            {gpu.name}
            {gpu.vramMb > 0 ? ` (${Math.round(gpu.vramMb / 1024)} GB)` : ''}
          </option>
        ))}
      </select>
    </div>
  )
}
