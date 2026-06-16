/**
 * Training form persistence and serialization.
 *
 * The Training tab is unmounted whenever the user switches to another
 * project tab (see Project.tsx, which conditionally renders one tab at a
 * time). That throws away the tab's local form state, so an operator who
 * fills in a job name, picks a preset, tunes VRAM, and edits the sampling
 * prompts loses all of it the moment they pop over to the Dataset tab to
 * add a trigger word. This module persists that state to localStorage,
 * keyed per project, so the form is restored on the next mount.
 *
 * The same shape is reused for the Export Config / Import Config buttons:
 * a config JSON file is just the serialized form state plus a version tag.
 */

import type { TrainingPresetId } from '../hooks/useTraining'
import type { VramModeState } from '../components/Training/VramModeGroup'
import type { SamplingConfigValues } from '../components/Training/PresetForm/SamplingConfigEditor'

/** Everything the Training tab form owns and needs to restore. */
export interface TrainingFormState {
  selectedPreset: TrainingPresetId
  gpuIndex: number
  jobName: string
  vramMode: VramModeState
  customizeSampling: boolean
  sampling: SamplingConfigValues
}

/** Bumped if the on-disk JSON shape ever changes incompatibly. */
const CONFIG_VERSION = 1

/** Wrapper written to exported .json files. */
interface TrainingConfigFile {
  version: number
  kind: 'openltx-training-config'
  form: TrainingFormState
}

function storageKey(projectId: string): string {
  return `openltx-training-form:${projectId}`
}

/**
 * Merge an untrusted partial object over a known-good fallback,
 * validating each field's type. Anything missing or the wrong type falls
 * back to the default so a hand-edited or stale file can never crash the
 * form. The fallback supplies the canonical defaults so this module does
 * not have to import them (and risk a circular dependency).
 */
function mergeForm(raw: unknown, fallback: TrainingFormState): TrainingFormState {
  if (raw === null || typeof raw !== 'object') return fallback
  const obj = raw as Record<string, unknown>

  const selectedPreset =
    obj.selectedPreset === 'character_v1' ||
    obj.selectedPreset === 'concept_v1' ||
    obj.selectedPreset === 'character_image_v1'
      ? (obj.selectedPreset as TrainingPresetId)
      : fallback.selectedPreset

  const gpuIndex =
    typeof obj.gpuIndex === 'number' && Number.isFinite(obj.gpuIndex)
      ? obj.gpuIndex
      : fallback.gpuIndex

  const jobName = typeof obj.jobName === 'string' ? obj.jobName : fallback.jobName

  const customizeSampling =
    typeof obj.customizeSampling === 'boolean'
      ? obj.customizeSampling
      : fallback.customizeSampling

  return {
    selectedPreset,
    gpuIndex,
    jobName,
    customizeSampling,
    vramMode: mergeVramMode(obj.vramMode, fallback.vramMode),
    sampling: mergeSampling(obj.sampling, fallback.sampling),
  }
}

function mergeVramMode(raw: unknown, fallback: VramModeState): VramModeState {
  if (raw === null || typeof raw !== 'object') return fallback
  const obj = raw as Record<string, unknown>
  const mode = obj.low_vram_mode
  return {
    low_vram_mode:
      mode === 'off' || mode === 'fp8' || mode === 'nf4'
        ? (mode as VramModeState['low_vram_mode'])
        : fallback.low_vram_mode,
    blocks_resident_on_gpu:
      typeof obj.blocks_resident_on_gpu === 'number' && Number.isFinite(obj.blocks_resident_on_gpu)
        ? obj.blocks_resident_on_gpu
        : fallback.blocks_resident_on_gpu,
    gradient_checkpointing:
      typeof obj.gradient_checkpointing === 'boolean'
        ? obj.gradient_checkpointing
        : fallback.gradient_checkpointing,
  }
}

function mergeSampling(raw: unknown, fallback: SamplingConfigValues): SamplingConfigValues {
  if (raw === null || typeof raw !== 'object') return fallback
  const obj = raw as Record<string, unknown>

  const samples = Array.isArray(obj.samples)
    ? obj.samples
        .filter((s): s is Record<string, unknown> => s !== null && typeof s === 'object')
        .map(s => ({
          prompt: typeof s.prompt === 'string' ? s.prompt : '',
          width: typeof s.width === 'number' && Number.isFinite(s.width) ? s.width : 512,
          height: typeof s.height === 'number' && Number.isFinite(s.height) ? s.height : 512,
        }))
    : fallback.samples

  const num = (value: unknown, fb: number): number =>
    typeof value === 'number' && Number.isFinite(value) ? value : fb

  return {
    samples: samples.length > 0 ? samples : fallback.samples,
    num_inference_steps: num(obj.num_inference_steps, fallback.num_inference_steps),
    num_frames: num(obj.num_frames, fallback.num_frames),
    guidance_scale: num(obj.guidance_scale, fallback.guidance_scale),
    sample_every_n_steps: num(obj.sample_every_n_steps, fallback.sample_every_n_steps),
  }
}

/** Read the persisted form for a project, merged over the given defaults. */
export function loadTrainingForm(projectId: string, fallback: TrainingFormState): TrainingFormState {
  try {
    const rawText = localStorage.getItem(storageKey(projectId))
    if (!rawText) return fallback
    return mergeForm(JSON.parse(rawText), fallback)
  } catch {
    return fallback
  }
}

/** Persist the form for a project. Silently ignores quota / serialization errors. */
export function saveTrainingForm(projectId: string, state: TrainingFormState): void {
  try {
    localStorage.setItem(storageKey(projectId), JSON.stringify(state))
  } catch {
    // Persistence is best-effort; a full quota must not break the form.
  }
}

/** Serialize the form to the on-disk config file format (pretty JSON). */
export function serializeTrainingConfig(state: TrainingFormState): string {
  const file: TrainingConfigFile = {
    version: CONFIG_VERSION,
    kind: 'openltx-training-config',
    form: state,
  }
  return JSON.stringify(file, null, 2)
}

/**
 * Parse an exported config file back into form state, merged over the
 * given defaults. Accepts both the wrapped file shape and a bare form
 * object so a user can paste either.
 */
export function parseTrainingConfig(jsonText: string, fallback: TrainingFormState): TrainingFormState {
  let parsed: unknown
  try {
    parsed = JSON.parse(jsonText)
  } catch {
    return fallback
  }
  if (parsed !== null && typeof parsed === 'object' && 'form' in (parsed as object)) {
    return mergeForm((parsed as TrainingConfigFile).form, fallback)
  }
  return mergeForm(parsed, fallback)
}
