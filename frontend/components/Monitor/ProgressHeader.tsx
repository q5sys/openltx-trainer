/**
 * Progress header showing current job status, step count, loss, ETA, and phase.
 */

import { Progress } from '../ui/progress'
import type { TrainingJobRecord } from '../../hooks/useTraining'
import type { ProgressRecord } from '../../hooks/useTrainingProgress'
import type { GpuMemory } from '../../hooks/useGpuMemory'

interface ProgressHeaderProps {
  job: TrainingJobRecord
  latestRecord: ProgressRecord | null
  gpuMemory: GpuMemory | null
}

function formatGpuMemory(memory: GpuMemory | null): string | null {
  if (memory === null || !memory.available || memory.total_mb <= 0) return null
  const usedGb = memory.used_mb / 1024
  const totalGb = memory.total_mb / 1024
  const percent = Math.round((memory.used_mb / memory.total_mb) * 100)
  return `${usedGb.toFixed(1)} / ${totalGb.toFixed(1)} GB (${percent}%)`
}


function formatEta(seconds: number | null): string {
  if (seconds === null || seconds <= 0) return '--'
  const hours = Math.floor(seconds / 3600)
  const mins = Math.floor((seconds % 3600) / 60)
  if (hours > 0) return `${hours}h ${mins}m`
  return `${mins}m`
}

function stateLabel(state: string): { text: string; color: string } {
  switch (state) {
    case 'running':
      return { text: 'Running', color: 'text-green-400' }
    case 'paused':
      return { text: 'Paused', color: 'text-yellow-400' }
    case 'completed':
      return { text: 'Completed', color: 'text-blue-400' }
    case 'cancelled':
      return { text: 'Cancelled', color: 'text-zinc-400' }
    case 'errored':
      return { text: 'Error', color: 'text-red-400' }
    case 'starting':
      return { text: 'Starting', color: 'text-blue-300' }
    default:
      return { text: state, color: 'text-zinc-400' }
  }
}

export function ProgressHeader({ job, latestRecord, gpuMemory }: ProgressHeaderProps) {
  const percentage = job.total_steps > 0 ? (job.current_step / job.total_steps) * 100 : 0
  const { text: statusText, color: statusColor } = stateLabel(job.state)
  const loss = latestRecord?.loss ?? job.current_loss
  const phase = latestRecord?.phase ?? job.current_phase
  const ips = latestRecord?.ips ?? null
  const vramText = formatGpuMemory(gpuMemory)
  // Warn (amber) once the card is nearly full. Sample generation is the
  // usual point where a run that trained fine tips over into an OOM.
  const vramNearFull =
    gpuMemory !== null &&
    gpuMemory.available &&
    gpuMemory.total_mb > 0 &&
    gpuMemory.used_mb / gpuMemory.total_mb >= 0.9


  // Show the coarse-stage line whenever the worker is busy doing
  // something other than the plain training loop (loading models,
  // encoding the dataset, generating samples). During those windows no
  // per-step progress record lands, so without this the UI looks frozen.
  // We hide it for the steady-state "training" stage because the
  // step counter and loss already convey that.
  const isActive = job.state === 'running' || job.state === 'starting'
  const showStage =
    isActive && job.stage !== null && job.stage !== 'training' && !!job.stage_message


  return (
    <div className="space-y-3 p-4 bg-zinc-900 rounded-lg border border-zinc-800">
      {/* Job name + short id */}
      <div className="flex items-baseline gap-2 min-w-0">
        <span className="text-sm font-medium text-zinc-200 truncate">
          {job.name || `Job ${job.job_id.slice(0, 8)}`}
        </span>
        {job.name && (
          <span className="text-xs text-zinc-500 font-mono flex-shrink-0">
            {job.job_id.slice(0, 8)}
          </span>
        )}
      </div>

      {/* Top row: status and step count */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className={`text-sm font-medium ${statusColor}`}>{statusText}</span>
          {phase && (
            <span className="text-xs text-zinc-500 bg-zinc-800 px-2 py-0.5 rounded">
              {phase}
            </span>
          )}
        </div>
        <span className="text-sm text-zinc-400 font-mono">
          {job.current_step} / {job.total_steps} steps
        </span>
      </div>


      {/* Coarse stage line: what the worker is busy doing during the
          non-stepping windows (model load, dataset encode, sampling). */}
      {showStage && (
        <div className="flex items-center gap-2 text-xs text-blue-300">
          <span className="inline-block w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
          <span>{job.stage_message}</span>
        </div>
      )}

      {/* Progress bar */}
      <Progress value={percentage} />


      {/* Stats row */}
      <div className="flex items-center gap-6 text-xs text-zinc-500">
        <span>
          Loss: <span className="text-zinc-300 font-mono">{loss !== null ? loss.toFixed(4) : '--'}</span>
        </span>
        <span>
          ETA: <span className="text-zinc-300 font-mono">{formatEta(job.eta_seconds)}</span>
        </span>
        {ips !== null && (
          <span>
            Speed: <span className="text-zinc-300 font-mono">{ips.toFixed(1)} it/s</span>
          </span>
        )}
        <span>
          GPU: <span className="text-zinc-300 font-mono">{job.gpu_index}</span>
        </span>
        {vramText && (
          <span>
            VRAM:{' '}
            <span className={`font-mono ${vramNearFull ? 'text-amber-400' : 'text-zinc-300'}`}>
              {vramText}
            </span>
          </span>
        )}
      </div>


      {/* Error message */}
      {job.error_message && (
        <div className="text-xs text-red-400 bg-red-950/30 border border-red-900/50 rounded px-3 py-2">
          {job.error_message}
        </div>
      )}
    </div>
  )
}
