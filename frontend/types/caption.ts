/**
 * Caption pipeline types mirroring the backend Pydantic models.
 */

export type CaptionBackendId = 'local' | 'gemini' | 'openai' | 'anthropic' | 'openai_compatible'

export type CaptionModelSize = '2B' | '4B' | '8B' | '32B'

export type CaptionQuantization = 'fp16' | '8bit' | '4bit'

export interface LocalModelChoice {
  family: 'qwen3-vl'
  size: CaptionModelSize
  abliterated: boolean
  quantization: CaptionQuantization
}

export interface BackendDescriptor {
  backend_id: CaptionBackendId
  display_name: string
  is_configured: boolean
  is_local: boolean
}

export interface ModelSetupStatus {
  state: 'not_started' | 'downloading' | 'loading' | 'ready' | 'error'
  progress: number
  current_file?: string | null
  downloaded_bytes?: number | null
  total_bytes?: number | null
  message?: string | null
  error_message: string | null
  model_choice: LocalModelChoice | null
}

export interface PromptTemplate {
  system_prompt: string
  user_prompt: string
  frame_count: number
}

export interface CaptionResult {
  clip_id: string
  caption: string
  backend_used: CaptionBackendId
  success: boolean
  error_message: string | null
}

export interface CaptionBatchStatus {
  job_id: string
  state: 'running' | 'complete' | 'cancelled' | 'error'
  total: number
  completed: number
  failed: number
  results: CaptionResult[]
}

export interface ApiKeyTestResult {
  valid: boolean
  error_message: string | null
}

export const DEFAULT_PROMPT_TEMPLATE: PromptTemplate = {
  system_prompt:
    'You are a video annotation assistant. Describe the subject, action, ' +
    'framing, expression, and setting in one to three sentences. Use video ' +
    'terminology (e.g., "shot", "footage"). Do not use words like "photograph" ' +
    'or "still". Do not invent details not visible in the frames.',
  user_prompt: 'Describe this short video clip.',
  frame_count: 8,
}
