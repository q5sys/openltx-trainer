/**
 * Training project types for OpenLTX Trainer.
 *
 * A TrainingProject represents a single LORA training effort:
 * a dataset, a trigger word, training job history, and verification results.
 */

export type TrainingMode = 'character' | 'concept'

/**
 * Dataset profile for a project (see
 * memory-bank/feature_two_profile_training.md). Character training has two
 * fundamentally different memory + hyperparameter profiles:
 * - "image": trains from a still-image dataset at a single latent frame.
 *   The Cutter sub-tab is not shown (images are imported directly), and the
 *   training preset defaults to the single-stage character_image_v1.
 * - "video": trains from a video dataset at a real temporal length and uses
 *   the 4-phase character_v1 preset.
 */
export type TrainingProfile = 'image' | 'video'

export type ProjectTab = 'dataset' | 'training' | 'monitor' | 'verify'

export interface TrainingProject {
  id: string
  name: string
  createdAt: string   // ISO 8601 timestamp
  updatedAt: string   // ISO 8601 timestamp
  trigger: string | null
  mode: TrainingMode
  profile: TrainingProfile
  datasetPath: string // relative to projects dir
  trainingJobIds: string[]
  verifyHistoryIds: string[]
  notes: string
  captioningBackendOverride: string | null // per-project override for captioning backend
}

/**
 * Create a new TrainingProject with sensible defaults.
 *
 * ``profile`` defaults to "video" so a caller that predates the
 * image/video split (or a concept project, where it is irrelevant) gets
 * the historical behaviour.
 */
export function createTrainingProject(
  name: string,
  mode: TrainingMode,
  profile: TrainingProfile = 'video',
): TrainingProject {
  const now = new Date().toISOString()
  return {
    id: crypto.randomUUID(),
    name,
    createdAt: now,
    updatedAt: now,
    trigger: null,
    mode,
    profile,
    datasetPath: '',
    trainingJobIds: [],
    verifyHistoryIds: [],
    notes: '',
    captioningBackendOverride: null,
  }
}

/**
 * Backfill the ``profile`` field on a project loaded from storage that
 * predates the image/video split. Older projects were all video-profile
 * character runs, so a missing profile defaults to "video".
 */
export function normalizeProject(project: TrainingProject): TrainingProject {
  if (project.profile === 'image' || project.profile === 'video') {
    return project
  }
  return { ...project, profile: 'video' }
}
