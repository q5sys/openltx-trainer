/**
 * Dataset types for the Dataset Tab.
 */

export interface SourceMediaInfo {
  path: string
  filename: string
  duration_s: number
  width: number
  height: number
  fps: number
  has_audio: boolean
  codec: string
  is_image: boolean
}

export interface SceneProposal {
  scene_index: number
  start_s: number
  end_s: number
  duration_s: number
  confidence: number
  thumbnail_b64: string
  length_status: 'short' | 'on_target' | 'long'
}

export interface ClipResult {
  clip_id: string
  filename: string
  duration_s: number
  width: number
  height: number
  fps: number
  has_audio: boolean
  thumbnail_path: string
}

export interface ClipRecord {
  clip_id: string
  filename: string
  duration_s: number
  width: number
  height: number
  fps: number
  has_audio: boolean
  caption: string
  thumbnail_path: string
  source_filename: string
  start_s: number
  end_s: number
}

// ---------------------------------------------------------------------------
// Trigger word + validation types
// ---------------------------------------------------------------------------

export interface TriggerValidationResult {
  valid: boolean
  error: string | null
  warning: string | null
}

export interface ValidationIssue {
  code: string
  msg: string
  clip_id: string | null
}

export interface DatasetStats {
  clip_count: number
  image_count: number
  captioned: number
  trigger_present: number
  with_audio: number
  without_audio: number
  total_duration_s: number
}

export interface DatasetValidationResult {
  valid: boolean
  errors: ValidationIssue[]
  warnings: ValidationIssue[]
  stats: DatasetStats
}
