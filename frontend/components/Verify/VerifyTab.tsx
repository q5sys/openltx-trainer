import { useEffect, useState } from 'react'
import { useProjects } from '../../contexts/ProjectContext'
import { useVerification } from '../../hooks/useVerification'
import type { LoraStackEntry, VerificationJobStatus } from '../../hooks/useVerification'
import type { GenerationParams } from './GenerationForm'
import { LoraSelector } from './LoraSelector'
import { GenerationForm } from './GenerationForm'
import { ComparisonGrid } from './ComparisonGrid'
import { VerifyHistory } from './VerifyHistory'
import { ExportLoraDialog } from './ExportLoraDialog'

export function VerifyTab() {
  const { activeProject } = useProjects()
  const {
    loras,
    activeJob,
    history,
    loading,
    error,
    fetchLoras,
    generate,
    fetchHistory,
    exportLora,
  } = useVerification()

  const [loraStack, setLoraStack] = useState<LoraStackEntry[]>([])
  const [comparisonJobs, setComparisonJobs] = useState<VerificationJobStatus[]>([])
  const [showExport, setShowExport] = useState(false)

  const projectId = activeProject?.id || ''
  const triggerWord = activeProject?.trigger || ''

  // Fetch LORAs and history on mount
  useEffect(() => {
    if (!projectId) return
    fetchLoras(projectId)
    fetchHistory(projectId)
  }, [projectId, fetchLoras, fetchHistory])

  // Track completed jobs for comparison
  useEffect(() => {
    if (activeJob?.status === 'completed') {
      setComparisonJobs(prev => {
        const exists = prev.some(j => j.generation_id === activeJob.generation_id)
        if (exists) return prev
        const updated = [...prev, activeJob]
        // Keep at most 4 for the comparison grid
        return updated.slice(-4)
      })
      // Refresh history
      if (projectId) fetchHistory(projectId)
    }
  }, [activeJob, projectId, fetchHistory])

  const handleGenerate = async (params: GenerationParams) => {
    if (!projectId) return
    await generate({
      project_id: projectId,
      prompt: params.prompt,
      negative_prompt: params.negative_prompt,
      width: params.width,
      height: params.height,
      num_frames: params.num_frames,
      guidance_scale: params.guidance_scale,
      seed: params.seed,
      gpu_index: 0,
      lora_stack: loraStack,
      num_inference_steps: params.num_inference_steps,
    })
  }

  const selectedCheckpointPath = loraStack.length > 0 ? loraStack[0].checkpoint_path : ''

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {error && (
          <div className="bg-red-950 border border-red-800 rounded px-3 py-2 text-sm text-red-300">
            {error}
          </div>
        )}

        {/* LORA selector */}
        <LoraSelector
          availableLoras={loras}
          loraStack={loraStack}
          onStackChange={setLoraStack}
        />

        {/* Generation form */}
        <GenerationForm
          onGenerate={handleGenerate}
          loading={loading}
          triggerWord={triggerWord}
        />

        {/* Active job status */}
        {activeJob && activeJob.status !== 'completed' && (
          <div className="bg-zinc-900 border border-zinc-800 rounded p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-zinc-300 capitalize">{activeJob.status}</span>
              <span className="text-xs text-zinc-500">
                {Math.round(activeJob.progress * 100)}%
              </span>
            </div>
            <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500 rounded-full transition-all"
                style={{ width: `${activeJob.progress * 100}%` }}
              />
            </div>
          </div>
        )}

        {/* Preview area */}
        {activeJob?.status === 'completed' && activeJob.output_path && (
          <div className="bg-zinc-900 border border-zinc-800 rounded overflow-hidden">
            <video
              src={`file://${activeJob.output_path}`}
              controls
              loop
              autoPlay
              muted
              className="w-full aspect-video object-contain bg-black"
            />
            <div className="p-3 flex items-center justify-between">
              <div>
                <p className="text-xs text-zinc-400 truncate">{activeJob.prompt}</p>
                <p className="text-xs text-zinc-600">Seed: {activeJob.seed}</p>
              </div>
              {selectedCheckpointPath && (
                <button
                  onClick={() => setShowExport(!showExport)}
                  className="text-xs text-blue-400 hover:text-blue-300"
                >
                  Export LORA
                </button>
              )}
            </div>
          </div>
        )}

        {/* Export dialog */}
        {showExport && selectedCheckpointPath && (
          <ExportLoraDialog
            checkpointPath={selectedCheckpointPath}
            previewGenerationId={activeJob?.generation_id || null}
            onExport={exportLora}
            loading={loading}
          />
        )}

        {/* Comparison grid */}
        {comparisonJobs.length > 1 && (
          <div>
            <label className="text-xs font-medium text-zinc-400 uppercase tracking-wide mb-2 block">
              Comparison
            </label>
            <ComparisonGrid jobs={comparisonJobs} />
          </div>
        )}

        {/* History strip */}
        <VerifyHistory
          entries={history}
          onSelect={() => {
            // History selection could pre-fill the form; for now just visual
          }}
        />
      </div>
    </div>
  )
}
