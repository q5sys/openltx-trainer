/**
 * Left rail job list for the Monitor tab.
 *
 * Shows all training jobs across projects with state indicators.
 * The selected job drives the main monitor panel. Terminal-state
 * jobs expose Restart and Delete buttons on hover.
 */

import { RotateCcw, Trash2 } from 'lucide-react'
import type { TrainingJobSummary } from '../../hooks/useTraining'

interface JobListProps {
  jobs: TrainingJobSummary[]
  selectedJobId: string | null
  onSelectJob: (jobId: string) => void
  onRestartJob?: (jobId: string) => void
  onDeleteJob?: (jobId: string) => void
}

// A job is in a "terminal" state when training is no longer in flight.
// Restart and Delete are only meaningful for those.
const TERMINAL_STATES = new Set(['completed', 'cancelled', 'errored'])

function stateIndicator(state: string): { dot: string; label: string } {

  switch (state) {
    case 'running':
      return { dot: 'bg-green-400', label: 'Running' }
    case 'paused':
      return { dot: 'bg-yellow-400', label: 'Paused' }
    case 'completed':
      return { dot: 'bg-blue-400', label: 'Done' }
    case 'cancelled':
      return { dot: 'bg-zinc-500', label: 'Cancelled' }
    case 'errored':
      return { dot: 'bg-red-400', label: 'Error' }
    case 'starting':
      return { dot: 'bg-blue-300 animate-pulse', label: 'Starting' }
    default:
      return { dot: 'bg-zinc-600', label: state }
  }
}

export function JobList({
  jobs,
  selectedJobId,
  onSelectJob,
  onRestartJob,
  onDeleteJob,
}: JobListProps) {
  if (jobs.length === 0) {
    return (
      <div className="p-4 text-xs text-zinc-600 text-center">
        No training jobs yet. Start a job in the Training tab.
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-1 p-2">
      {jobs.map(job => {
        const selected = job.job_id === selectedJobId
        const { dot, label } = stateIndicator(job.state)
        const progress = job.total_steps > 0
          ? Math.round((job.current_step / job.total_steps) * 100)
          : 0
        const terminal = TERMINAL_STATES.has(job.state)
        const displayName = job.name || `Job ${job.job_id.slice(0, 8)}`

        return (
          <div
            key={job.job_id}
            role="button"
            tabIndex={0}
            onClick={() => onSelectJob(job.job_id)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
                onSelectJob(job.job_id)
              }
            }}
            className={`group w-full text-left px-3 py-2 rounded text-sm transition-colors cursor-pointer ${
              selected
                ? 'bg-zinc-800 border border-zinc-700'
                : 'hover:bg-zinc-800/50 border border-transparent'
            }`}
          >
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full flex-shrink-0 ${dot}`} />
              <span className="text-zinc-200 truncate text-xs flex-1" title={displayName}>
                {displayName}
              </span>
              <span className="text-zinc-600 text-xs flex-shrink-0">{label}</span>
            </div>
            <div className="flex items-center gap-2 mt-1 ml-4">
              <div className="flex-1 h-1 bg-zinc-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 transition-all"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <span className="text-xs text-zinc-600 font-mono w-8 text-right">{progress}%</span>
            </div>
            <div className="flex items-center justify-between mt-1 ml-4 min-h-[14px]">
              <div className="text-xs text-zinc-600 font-mono">
                {job.job_id.slice(0, 8)}
                {job.current_loss !== null && (
                  <span className="ml-2">loss: {job.current_loss.toFixed(4)}</span>
                )}
              </div>
              {terminal && (onRestartJob || onDeleteJob) && (
                <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  {onRestartJob && (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation()
                        onRestartJob(job.job_id)
                      }}
                      title="Restart job with same configuration"
                      className="p-1 rounded text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700"
                    >
                      <RotateCcw className="h-3 w-3" />
                    </button>
                  )}
                  {onDeleteJob && (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation()
                        onDeleteJob(job.job_id)
                      }}
                      title="Delete job and its files"
                      className="p-1 rounded text-zinc-400 hover:text-red-400 hover:bg-zinc-700"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  )}
                </div>
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}

