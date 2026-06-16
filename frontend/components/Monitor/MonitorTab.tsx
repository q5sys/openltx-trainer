/**
 * Monitor tab: live training progress, loss chart, sample previews, and log tail.
 *
 * Layout: left rail (job list) + main panel (progress header, loss chart, samples, log).
 */

import { useState, useEffect, useCallback } from 'react'
import { Pause, Play, XCircle, RotateCcw, Trash2 } from 'lucide-react'
import { backendFetch } from '../../lib/backend'
import { useTrainingJobs } from '../../hooks/useTrainingJobs'
import { useTrainingProgress } from '../../hooks/useTrainingProgress'
import { useGpuMemory } from '../../hooks/useGpuMemory'
import type { TrainingJobRecord } from '../../hooks/useTraining'

import { Button } from '../ui/button'
import { JobList } from './JobList'
import { ProgressHeader } from './ProgressHeader'
import { LossChart } from './LossChart'
import { SampleStrip } from './SampleStrip'
import { LogTail } from './LogTail'

const JOB_POLL_INTERVAL_MS = 3000
const TERMINAL_STATES = new Set(['completed', 'cancelled', 'errored'])

export function MonitorTab() {
  const { jobs, refetch: refetchJobs } = useTrainingJobs()
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null)
  const [jobDetail, setJobDetail] = useState<TrainingJobRecord | null>(null)
  const { records, latestStep } = useTrainingProgress(selectedJobId)

  // Live VRAM for the job's GPU. Poll only while the job is active (not
  // in a terminal state) so an idle Monitor view does not poll forever.
  // This keeps updating during sample generation, the window where a run
  // that trained fine often tips into an OOM.
  const gpuMemoryActive =
    jobDetail !== null && !TERMINAL_STATES.has(jobDetail.state)
  const gpuMemory = useGpuMemory(jobDetail?.gpu_index ?? null, gpuMemoryActive)


  // Auto-select the first running job, or first job overall

  useEffect(() => {
    if (selectedJobId) return
    const running = jobs.find(j => j.state === 'running')
    const first = running ?? jobs[0]
    if (first) setSelectedJobId(first.job_id)
  }, [jobs, selectedJobId])

  // Fetch full job detail for the selected job
  const fetchJobDetail = useCallback(async () => {
    if (!selectedJobId) {
      setJobDetail(null)
      return
    }
    try {
      const res = await backendFetch(`/api/training/jobs/${selectedJobId}`)
      if (!res.ok) return
      const data = (await res.json()) as TrainingJobRecord
      setJobDetail(data)
    } catch {
      // ignore
    }
  }, [selectedJobId])

  useEffect(() => {
    fetchJobDetail()
  }, [fetchJobDetail, latestStep])

  // Poll job detail on interval for state changes
  useEffect(() => {
    if (!selectedJobId) return
    const interval = setInterval(fetchJobDetail, JOB_POLL_INTERVAL_MS)
    return () => clearInterval(interval)
  }, [selectedJobId, fetchJobDetail])

  // Job control actions
  const pauseJob = useCallback(async () => {
    if (!selectedJobId) return
    await backendFetch(`/api/training/jobs/${selectedJobId}/pause`, { method: 'POST' })
    await fetchJobDetail()
  }, [selectedJobId, fetchJobDetail])

  const resumeJob = useCallback(async () => {
    if (!selectedJobId) return
    await backendFetch(`/api/training/jobs/${selectedJobId}/resume`, { method: 'POST' })
    await fetchJobDetail()
  }, [selectedJobId, fetchJobDetail])

  const cancelJob = useCallback(async () => {
    if (!selectedJobId) return
    await backendFetch(`/api/training/jobs/${selectedJobId}/cancel`, { method: 'POST' })
    await fetchJobDetail()
  }, [selectedJobId, fetchJobDetail])

  const restartJob = useCallback(async (jobId: string) => {
    try {
      const res = await backendFetch(`/api/training/jobs/${jobId}/restart`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: null }),
      })
      if (!res.ok) return
      const data = (await res.json()) as TrainingJobRecord
      // Switch the monitor selection to the freshly-spawned job
      setSelectedJobId(data.job_id)
      setJobDetail(data)
      await refetchJobs()
    } catch {
      // ignore
    }
  }, [refetchJobs])

  const deleteJob = useCallback(async (jobId: string) => {
    const target = jobs.find(j => j.job_id === jobId)
    const label = target?.name?.trim() || `Job ${jobId.slice(0, 8)}`
    if (!window.confirm(`Delete "${label}" and its files? This cannot be undone.`)) return
    try {
      const res = await backendFetch(`/api/training/jobs/${jobId}`, { method: 'DELETE' })
      if (!res.ok) {
        const body = await res.json().catch(() => ({})) as { detail?: string }
        window.alert(body.detail ?? `Failed to delete job (${res.status})`)
        return
      }
      if (selectedJobId === jobId) {
        setSelectedJobId(null)
        setJobDetail(null)
      }
      await refetchJobs()
    } catch {
      // ignore
    }
  }, [jobs, selectedJobId, refetchJobs])

  const isRunning = jobDetail?.state === 'running'
  const isPaused = jobDetail?.state === 'paused'
  const isActive = isRunning || isPaused
  const isTerminal = jobDetail !== null && TERMINAL_STATES.has(jobDetail.state)


  return (
    <div className="flex flex-1 overflow-hidden">
      {/* Left rail: job list */}
      <div className="w-56 flex-shrink-0 border-r border-zinc-800 overflow-y-auto">
        <div className="px-3 py-2 border-b border-zinc-800">
          <span className="text-xs font-medium text-zinc-400">Training Jobs</span>
        </div>
        <JobList
          jobs={jobs}
          selectedJobId={selectedJobId}
          onSelectJob={setSelectedJobId}
          onRestartJob={restartJob}
          onDeleteJob={deleteJob}
        />

      </div>

      {/* Main panel */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {!jobDetail ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <p className="text-sm text-zinc-500">
                {jobs.length === 0
                  ? 'No training jobs found. Start a job from the Training tab.'
                  : 'Select a job from the list to view its progress.'}
              </p>
            </div>
          </div>
        ) : (
          <>
            {/* Progress header */}
            <ProgressHeader
              job={jobDetail}
              latestRecord={records.length > 0 ? records[records.length - 1] : null}
              gpuMemory={gpuMemory}
            />


            {/* Controls */}
            {isActive && (
              <div className="flex items-center gap-2">
                {isRunning && (
                  <Button variant="outline" size="sm" onClick={pauseJob}>
                    <Pause className="h-3.5 w-3.5 mr-1.5" />
                    Pause
                  </Button>
                )}
                {isPaused && (
                  <Button variant="outline" size="sm" onClick={resumeJob}>
                    <Play className="h-3.5 w-3.5 mr-1.5" />
                    Resume
                  </Button>
                )}
                <Button variant="outline" size="sm" onClick={cancelJob} className="text-red-400 hover:text-red-300">
                  <XCircle className="h-3.5 w-3.5 mr-1.5" />
                  Cancel
                </Button>
              </div>
            )}

            {isTerminal && (
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => restartJob(jobDetail.job_id)}
                  title="Spawn a new job with the same configuration"
                >
                  <RotateCcw className="h-3.5 w-3.5 mr-1.5" />
                  Restart
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => deleteJob(jobDetail.job_id)}
                  className="text-red-400 hover:text-red-300"
                  title="Delete this job and its files"
                >
                  <Trash2 className="h-3.5 w-3.5 mr-1.5" />
                  Delete
                </Button>
              </div>
            )}


            {/* Loss chart */}
            <LossChart records={records} />

            {/* Sample strip */}
            <SampleStrip jobId={selectedJobId} />

            {/* Log tail */}
            <LogTail records={records} />
          </>
        )}
      </div>
    </div>
  )
}
