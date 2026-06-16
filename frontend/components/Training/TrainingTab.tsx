/**
 * Training tab: preset selection, GPU picker, and job controls.
 *
 * Allows the user to select a training preset, pick a GPU,
 * configure overrides, and start/pause/resume/cancel training jobs.
 */

import { useEffect, useState, useCallback } from 'react'
import { useProjects } from '../../contexts/ProjectContext'
import { useTraining, type TrainingPresetId, type StartTrainingRequest } from '../../hooks/useTraining'
import type { TrainingProfile } from '../../types/project'
import { Button } from '../ui/button'
import { Play, Pause, Square, RotateCcw, Download, Upload } from 'lucide-react'
import { VramModeGroup, VRAM_MODE_DEFAULTS, type VramModeState } from './VramModeGroup'
import {
  SamplingConfigEditor,
  type SamplingConfigValues,
  type SampleSpecValues,
} from './PresetForm/SamplingConfigEditor'
import {
  type TrainingFormState,
  loadTrainingForm,
  saveTrainingForm,
  serializeTrainingConfig,
  parseTrainingConfig,
} from '../../lib/trainingConfig'


/**
 * Default sampling values seeded into the optional in-training sampling
 * panel. They mirror the backend SamplingConfig defaults; when the user
 * leaves "Customize sampling" off we send nothing and the preset's own
 * [sampling] table is used unchanged.
 */
const SAMPLING_DEFAULTS: SamplingConfigValues = {
  samples: [{ prompt: '{trigger} talking to the camera', width: 512, height: 512 }],
  // Mirror the backend SamplingConfig defaults. Preview CFG = 3.0 matches
  // the LTX-2 reference inference paths; the prior 10.0 "deep-fried" the
  // in-app preview (oversaturated, crushed detail) so good LoRAs looked
  // broken even though the same weights render cleanly in ComfyUI at cfg 3.
  // 30 steps is the reference inference floor (24 left previews soft).
  num_inference_steps: 30,
  num_frames: 49,
  guidance_scale: 3.0,
  sample_every_n_steps: 100,
}




function PresetCard({
  preset,
  selected,
  onSelect,
}: {
  preset: { id: string; name: string; description: string }
  selected: boolean
  onSelect: () => void
}) {
  return (
    <button
      onClick={onSelect}
      className={`text-left p-4 rounded-lg border transition-colors ${
        selected
          ? 'border-blue-500 bg-blue-500/10'
          : 'border-zinc-700 bg-zinc-900 hover:border-zinc-600'
      }`}
    >
      <h4 className="text-sm font-medium text-zinc-200">{preset.name}</h4>
      <p className="text-xs text-zinc-500 mt-1">{preset.description}</p>
    </button>
  )
}

function JobStatus({
  activeJob,
  onPause,
  onResume,
  onCancel,
}: {
  activeJob: NonNullable<ReturnType<typeof useTraining>['activeJob']>
  onPause: () => void
  onResume: () => void
  onCancel: () => void
}) {
  const progress = activeJob.total_steps > 0
    ? Math.round((activeJob.current_step / activeJob.total_steps) * 100)
    : 0

  return (
    <div className="border border-zinc-700 rounded-lg p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-zinc-200 truncate">
              {activeJob.name || `Job ${activeJob.job_id.slice(0, 8)}`}
            </span>
            <span className={`text-xs px-2 py-0.5 rounded flex-shrink-0 ${
              activeJob.state === 'running' ? 'bg-green-900 text-green-300' :
              activeJob.state === 'paused' ? 'bg-yellow-900 text-yellow-300' :
              activeJob.state === 'completed' ? 'bg-blue-900 text-blue-300' :
              activeJob.state === 'errored' ? 'bg-red-900 text-red-300' :
              'bg-zinc-800 text-zinc-400'
            }`}>
              {activeJob.state}
            </span>
          </div>
          {activeJob.name && (
            <p className="text-xs text-zinc-500 font-mono mt-0.5">{activeJob.job_id.slice(0, 8)}</p>
          )}
        </div>
        <div className="flex gap-2">

          {activeJob.state === 'running' && (
            <>
              <Button variant="ghost" size="sm" onClick={onPause} title="Pause">
                <Pause className="h-4 w-4" />
              </Button>
              <Button variant="ghost" size="sm" onClick={onCancel} title="Cancel">
                <Square className="h-4 w-4" />
              </Button>
            </>
          )}
          {activeJob.state === 'paused' && (
            <>
              <Button variant="ghost" size="sm" onClick={onResume} title="Resume">
                <RotateCcw className="h-4 w-4" />
              </Button>
              <Button variant="ghost" size="sm" onClick={onCancel} title="Cancel">
                <Square className="h-4 w-4" />
              </Button>
            </>
          )}
        </div>
      </div>

      {/* Progress bar */}
      <div className="space-y-1">
        <div className="flex justify-between text-xs text-zinc-500">
          <span>Step {activeJob.current_step} / {activeJob.total_steps}</span>
          <span>{progress}%</span>
        </div>
        <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-blue-500 rounded-full transition-all"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Loss and phase */}
      <div className="flex gap-4 text-xs text-zinc-500">
        {activeJob.current_loss !== null && (
          <span>Loss: {activeJob.current_loss.toFixed(4)}</span>
        )}
        {activeJob.current_phase && (
          <span>Phase: {activeJob.current_phase}</span>
        )}
        <span>GPU: {activeJob.gpu_index}</span>
      </div>

      {activeJob.error_message && (
        <p className="text-xs text-red-400">{activeJob.error_message}</p>
      )}
    </div>
  )
}

/**
 * Canonical defaults for a fresh Training form. Used both as the seed for
 * a brand-new project and as the fallback the persistence layer merges a
 * saved (or imported) form over, so a missing or stale field can never
 * leave the form in an invalid state. ``selectedPreset`` here is the
 * video-profile default; ``formDefaultsForProfile`` overrides it for
 * image-profile projects.
 */
const FORM_DEFAULTS: TrainingFormState = {
  selectedPreset: 'character_v1',
  gpuIndex: 0,
  jobName: '',
  vramMode: VRAM_MODE_DEFAULTS,
  // Default ON: in-training preview samples are the main way an operator
  // tells whether the LoRA is learning the trigger, so we surface the
  // editor by default.
  customizeSampling: true,
  sampling: SAMPLING_DEFAULTS,
}

/**
 * The default preset id for a project profile. Image-profile projects use
 * the single-stage image preset; video (and concept) projects keep the
 * 4-phase character preset as the default.
 */
function defaultPresetForProfile(profile: TrainingProfile): TrainingPresetId {
  return profile === 'image' ? 'character_image_v1' : 'character_v1'
}

/**
 * The preset ids selectable for a project profile. An image dataset can
 * only train the image preset; a video dataset trains the video character
 * or concept presets. Keeping these disjoint stops the operator from
 * pairing a video preset (target_frames=25) with a still-image dataset,
 * which is the exact mismatch this feature exists to prevent.
 */
function allowedPresetsForProfile(profile: TrainingProfile): TrainingPresetId[] {
  return profile === 'image'
    ? ['character_image_v1']
    : ['character_v1', 'concept_v1']
}

/** Form defaults seeded with the right preset for the project profile. */
function formDefaultsForProfile(profile: TrainingProfile): TrainingFormState {
  return { ...FORM_DEFAULTS, selectedPreset: defaultPresetForProfile(profile) }
}

export function TrainingTab() {
  const { activeProject } = useProjects()
  const {
    presets,
    activeJob,
    loading,
    error,
    fetchPresets,
    fetchJobs,
    fetchJob,
    startJob,
    pauseJob,
    resumeJob,
    cancelJob,
  } = useTraining()

  // Seed every field from the per-project persisted form so switching
  // tabs (which unmounts this component) does not discard the operator's
  // configuration. activeProject can be null on the first render, in which
  // case we seed from defaults and the early return below renders nothing.
  const projectId = activeProject?.id ?? ''
  // Seed the form with the preset that matches this project's dataset
  // profile (image vs video) so a fresh image project does not start out
  // pointing at the video preset.
  const projectProfile: TrainingProfile = activeProject?.profile ?? 'video'
  const initialForm = loadTrainingForm(projectId, formDefaultsForProfile(projectProfile))

  const [selectedPreset, setSelectedPreset] = useState<TrainingPresetId>(initialForm.selectedPreset)
  const [gpuIndex, setGpuIndex] = useState(initialForm.gpuIndex)
  const [gpuDevices, setGpuDevices] = useState<{ index: number; name: string }[]>([])
  const [jobName, setJobName] = useState(initialForm.jobName)
  const [vramMode, setVramMode] = useState<VramModeState>(initialForm.vramMode)
  const [customizeSampling, setCustomizeSampling] = useState(initialForm.customizeSampling)
  const [sampling, setSampling] = useState<SamplingConfigValues>(initialForm.sampling)


  // Update one sampling field. The samples list arrives as an array; the
  // rest are scalar numbers, matching SamplingConfigEditor's onChange.
  const handleSamplingChange = useCallback(
    (field: keyof SamplingConfigValues, value: number | SampleSpecValues[]) => {
      setSampling(prev => ({ ...prev, [field]: value }))
    },
    [],
  )




  const fetchGpuDevices = useCallback(async () => {
    try {
      const { backendFetch } = await import('../../lib/backend')
      const r = await backendFetch('/api/gpu-list')
      const data: { devices: { index: number; name: string }[] } = await r.json()
      setGpuDevices(data.devices)
    } catch {
      // fallback: empty list
    }
  }, [])

  useEffect(() => {
    fetchGpuDevices()
  }, [fetchGpuDevices])

  useEffect(() => {
    fetchPresets()
    fetchJobs()
  }, [fetchPresets, fetchJobs])

  // Keep the selected preset valid for this project's profile. A persisted
  // form (or an imported config) could carry a preset that does not belong
  // to the current profile (e.g. a video preset on an image project after
  // the profile field was added); snap it back to the profile default so
  // the dataset and preset can never disagree.
  const allowedPresets = allowedPresetsForProfile(projectProfile)
  useEffect(() => {
    if (!allowedPresets.includes(selectedPreset)) {
      setSelectedPreset(defaultPresetForProfile(projectProfile))
    }
  }, [allowedPresets, selectedPreset, projectProfile])

  // While a job is in-flight, poll its status so the UI does not get stuck
  // on "running, step 0" when the worker dies or finishes silently.
  useEffect(() => {
    if (!activeJob) return
    if (!['running', 'paused', 'starting'].includes(activeJob.state)) return
    const jobId = activeJob.job_id
    const handle = window.setInterval(() => {
      fetchJob(jobId)
      fetchJobs()
    }, 2000)
    return () => window.clearInterval(handle)
  }, [activeJob, fetchJob, fetchJobs])

  // Persist the whole form on every change so a tab switch (which unmounts
  // this component) does not lose the operator's configuration. Keyed per
  // project so two projects do not share one form.
  useEffect(() => {
    if (!projectId) return
    saveTrainingForm(projectId, {
      selectedPreset,
      gpuIndex,
      jobName,
      vramMode,
      customizeSampling,
      sampling,
    })
  }, [projectId, selectedPreset, gpuIndex, jobName, vramMode, customizeSampling, sampling])

  // Apply a restored / imported form to every field at once.
  const applyForm = useCallback((form: TrainingFormState) => {
    setSelectedPreset(form.selectedPreset)
    setGpuIndex(form.gpuIndex)
    setJobName(form.jobName)
    setVramMode(form.vramMode)
    setCustomizeSampling(form.customizeSampling)
    setSampling(form.sampling)
  }, [])

  // Export the current form to a JSON file the user can keep or share.
  const handleExportConfig = useCallback(async () => {
    const form: TrainingFormState = {
      selectedPreset,
      gpuIndex,
      jobName,
      vramMode,
      customizeSampling,
      sampling,
    }
    const filePath = await window.electronAPI.showSaveDialog({
      title: 'Export Training Config',
      defaultPath: `${(jobName.trim() || selectedPreset)}.json`,
      filters: [{ name: 'Training Config', extensions: ['json'] }],
    })
    if (!filePath) return
    await window.electronAPI.saveFile({
      filePath,
      data: serializeTrainingConfig(form),
      encoding: 'utf-8',
    })
  }, [selectedPreset, gpuIndex, jobName, vramMode, customizeSampling, sampling])

  // Import a previously exported config and overwrite the form with it.
  const handleImportConfig = useCallback(async () => {
    const paths = await window.electronAPI.showOpenFileDialog({
      title: 'Import Training Config',
      filters: [{ name: 'Training Config', extensions: ['json'] }],
    })
    if (!paths || paths.length === 0) return
    const file = await window.electronAPI.readLocalFile({ filePath: paths[0] })
    // readLocalFile returns base64-encoded bytes; decode to UTF-8 text.
    const jsonText = atob(file.data)
    applyForm(parseTrainingConfig(jsonText, FORM_DEFAULTS))
  }, [applyForm])


  if (!activeProject) return null


  const handleStart = async () => {
    // Only ship low-VRAM overrides when they differ from the preset
    // defaults so the worker honors the preset's BF16 path for users
    // who never touched the VRAM panel.
    const overrides: Record<string, unknown> = {}
    if (vramMode.low_vram_mode !== VRAM_MODE_DEFAULTS.low_vram_mode) {
      overrides.low_vram_mode = vramMode.low_vram_mode
    }
    if (vramMode.blocks_resident_on_gpu !== VRAM_MODE_DEFAULTS.blocks_resident_on_gpu) {
      overrides.blocks_resident_on_gpu = vramMode.blocks_resident_on_gpu
    }
    if (vramMode.gradient_checkpointing !== VRAM_MODE_DEFAULTS.gradient_checkpointing) {
      overrides.gradient_checkpointing = vramMode.gradient_checkpointing
    }

    // Only send the nested [sampling] override when the operator opted
    // into customizing it; otherwise the preset's own sampling table is
    // used unchanged. The backend rewrites the whole [sampling] table
    // from this object (see TrainingSupervisorImpl._apply_sampling_override).
    if (customizeSampling) {
      overrides.sampling = sampling
    }

    const request: StartTrainingRequest = {

      project_id: activeProject.id,
      preset_id: selectedPreset,
      gpu_index: gpuIndex,
      dataset_dir: activeProject.datasetPath,
      trigger_word: activeProject.trigger || '',
      name: jobName.trim() || undefined,
      config_overrides: Object.keys(overrides).length > 0 ? overrides : undefined,
    }
    const result = await startJob(request)
    if (result) {
      setJobName('')
    }
  }



  const isJobActive = activeJob && ['running', 'paused', 'starting'].includes(activeJob.state)

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h3 className="text-lg font-medium text-zinc-200">Training Configuration</h3>
          <p className="text-sm text-zinc-500 mt-1">
            Select a training preset and GPU, then start training.
          </p>
        </div>
        {/* Export / import the whole form as a JSON config so a known-good
            setup can be saved and re-loaded (or shared) instead of being
            rebuilt by hand each run. */}
        <div className="flex gap-2 flex-shrink-0">
          <Button
            variant="outline"
            size="sm"
            onClick={handleExportConfig}
            disabled={!!isJobActive}
            className="gap-2"
            title="Export the current configuration to a JSON file"
          >
            <Download className="h-4 w-4" />
            Export Config
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleImportConfig}
            disabled={loading || !!isJobActive}
            className="gap-2"
            title="Load a configuration from a JSON file"
          >
            <Upload className="h-4 w-4" />
            Import Config
          </Button>
        </div>
      </div>

      {/* Active job status */}
      {activeJob && (
        <JobStatus
          activeJob={activeJob}
          onPause={() => pauseJob(activeJob.job_id)}
          onResume={() => resumeJob(activeJob.job_id)}
          onCancel={() => cancelJob(activeJob.job_id)}
        />
      )}

      {/* Job name */}
      <div className="space-y-2">
        <h4 className="text-sm font-medium text-zinc-300">
          Job Name <span className="text-zinc-500 font-normal">(optional)</span>
        </h4>
        <input
          type="text"
          value={jobName}
          onChange={e => setJobName(e.target.value)}
          placeholder={`e.g. ${activeProject.name} run 1`}
          disabled={loading || !!isJobActive}
          className="w-full max-w-md px-3 py-2 bg-zinc-900 border border-zinc-700 rounded-lg text-sm text-zinc-200 placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
        />
        <p className="text-xs text-zinc-500">
          A friendly label for this run. Leave empty to auto-generate one.
        </p>
      </div>

      {/* Preset selection */}
      <div className="space-y-3">
        <h4 className="text-sm font-medium text-zinc-300">Training Preset</h4>

        <div className="grid grid-cols-2 gap-3">
          {/* Only show presets that match this project's dataset profile.
              An image project sees the image preset only; a video project
              sees the video character / concept presets. */}
          {presets
            .filter(preset => allowedPresets.includes(preset.id as TrainingPresetId))
            .map(preset => (
              <PresetCard
                key={preset.id}
                preset={preset}
                selected={selectedPreset === preset.id}
                onSelect={() => setSelectedPreset(preset.id as TrainingPresetId)}
              />
            ))}
          {presets.length === 0 && projectProfile === 'image' && (
            <PresetCard
              preset={{ id: 'character_image_v1', name: 'Character (Images)', description: 'Single-stage character LORA training from still images' }}
              selected={selectedPreset === 'character_image_v1'}
              onSelect={() => setSelectedPreset('character_image_v1')}
            />
          )}
          {presets.length === 0 && projectProfile !== 'image' && (
            <>
              <PresetCard
                preset={{ id: 'character_v1', name: 'Character (4-phase)', description: '4-phase character LORA training' }}
                selected={selectedPreset === 'character_v1'}
                onSelect={() => setSelectedPreset('character_v1')}
              />
              <PresetCard
                preset={{ id: 'concept_v1', name: 'Concept', description: 'Concept/style LORA training' }}
                selected={selectedPreset === 'concept_v1'}
                onSelect={() => setSelectedPreset('concept_v1')}
              />
            </>
          )}
        </div>
      </div>

      {/* GPU selection */}
      <div className="space-y-3">
        <h4 className="text-sm font-medium text-zinc-300">GPU</h4>
        <select
          value={gpuIndex}
          onChange={e => setGpuIndex(parseInt(e.target.value, 10) || 0)}
          className="px-3 py-2 bg-zinc-900 border border-zinc-700 rounded-lg text-sm text-zinc-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          {gpuDevices.length > 0 ? (
            gpuDevices.map((gpu) => (
              <option key={gpu.index} value={gpu.index}>
                Device {gpu.index}: {gpu.name}
              </option>
            ))
          ) : (
            <option value={gpuIndex}>Device {gpuIndex}</option>
          )}
        </select>
      </div>

      {/* Stage F: low-VRAM mode group */}
      <VramModeGroup
        value={vramMode}
        onChange={setVramMode}
        disabled={loading || !!isJobActive}
      />

      {/* In-training sampling. On by default so preview videos are
          generated during training; unchecking falls back to the preset's
          own [sampling] table unchanged. When checked the whole block is
          sent as a config override. */}
      <div className="space-y-3">
        <label className="flex items-center gap-2 text-sm font-medium text-zinc-300">
          <input
            type="checkbox"
            checked={customizeSampling}
            onChange={e => setCustomizeSampling(e.target.checked)}
            disabled={loading || !!isJobActive}
            className="h-4 w-4 rounded border-zinc-600 bg-zinc-800 disabled:opacity-50"
          />
          Customize in-training sampling
        </label>
        <p className="text-xs text-zinc-500">
          Generate preview videos during training at a fixed step cadence.
          Leave off to use the preset defaults.
        </p>
        {customizeSampling && (
          <SamplingConfigEditor
            config={sampling}
            onChange={handleSamplingChange}
            disabled={loading || !!isJobActive}
            triggerWord={activeProject.trigger || ''}
          />
        )}
      </div>

      {/* Start button */}


      <div className="pt-2">
        <Button
          onClick={handleStart}
          disabled={loading || !!isJobActive || !activeProject.datasetPath}
          className="gap-2"
        >
          <Play className="h-4 w-4" />
          {loading ? 'Starting...' : 'Start Training'}
        </Button>
        {!activeProject.datasetPath && (
          <p className="text-xs text-yellow-500 mt-2">
            Set up your dataset in the Dataset tab before starting training.
          </p>
        )}
      </div>

      {/* Error display */}
      {error && (
        <p className="text-sm text-red-400">{error}</p>
      )}
    </div>
  )
}
