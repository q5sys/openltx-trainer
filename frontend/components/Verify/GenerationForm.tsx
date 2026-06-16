import { useState } from 'react'
import { Play, Shuffle } from 'lucide-react'
import { Button } from '../ui/button'

interface GenerationFormProps {
  onGenerate: (params: GenerationParams) => void
  loading: boolean
  triggerWord: string
}

export interface GenerationParams {
  prompt: string
  negative_prompt: string
  width: number
  height: number
  num_frames: number
  guidance_scale: number
  seed: number
  num_inference_steps: number
}

const FRAME_PRESETS = [49, 65, 97, 129]
const SIZE_PRESETS = [
  { label: '512x512', w: 512, h: 512 },
  { label: '768x512', w: 768, h: 512 },
  { label: '512x768', w: 512, h: 768 },
]

export function GenerationForm({ onGenerate, loading, triggerWord }: GenerationFormProps) {
  const [prompt, setPrompt] = useState(triggerWord ? `${triggerWord}, ` : '')
  const [negativePrompt, setNegativePrompt] = useState('')
  const [width, setWidth] = useState(512)
  const [height, setHeight] = useState(512)
  const [numFrames, setNumFrames] = useState(49)
  // CFG 3.0 matches the LTX-2 reference inference paths (musubi
  // ltx2_defaults.py video_cfg_scale=3.0, LTX-Desktop cfg_scale=3.0). A
  // cfg_scale of 10 over-drives the cond-uncond delta and "deep-fries" the
  // result (oversaturated, crushed detail). The slider still lets the
  // operator raise it for a likeness check.
  const [guidanceScale, setGuidanceScale] = useState(3.0)

  const [seed, setSeed] = useState(-1)
  const [steps, setSteps] = useState(30)
  const [showAdvanced, setShowAdvanced] = useState(false)

  const randomizeSeed = () => {
    setSeed(Math.floor(Math.random() * 2147483647))
  }

  const handleSubmit = () => {
    onGenerate({
      prompt,
      negative_prompt: negativePrompt,
      width,
      height,
      num_frames: numFrames,
      guidance_scale: guidanceScale,
      seed,
      num_inference_steps: steps,
    })
  }

  return (
    <div className="space-y-3">
      {/* Prompt */}
      <div>
        <label className="text-xs font-medium text-zinc-400 uppercase tracking-wide">Prompt</label>
        <textarea
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          rows={3}
          className="w-full mt-1 bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm text-zinc-200 resize-none"
          placeholder={`${triggerWord}, medium shot, ...`}
        />
      </div>

      {/* Negative prompt */}
      <div>
        <label className="text-xs font-medium text-zinc-400 uppercase tracking-wide">Negative Prompt</label>
        <textarea
          value={negativePrompt}
          onChange={e => setNegativePrompt(e.target.value)}
          rows={1}
          className="w-full mt-1 bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm text-zinc-200 resize-none"
          placeholder="(optional)"
        />
      </div>

      {/* Size + Frames row */}
      <div className="flex gap-4 flex-wrap">
        <div>
          <label className="text-xs font-medium text-zinc-400">Size</label>
          <div className="flex gap-1 mt-1">
            {SIZE_PRESETS.map(preset => (
              <button
                key={preset.label}
                onClick={() => { setWidth(preset.w); setHeight(preset.h) }}
                className={`px-2 py-1 text-xs rounded ${
                  width === preset.w && height === preset.h
                    ? 'bg-blue-600 text-white'
                    : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'
                }`}
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="text-xs font-medium text-zinc-400">Frames</label>
          <div className="flex gap-1 mt-1">
            {FRAME_PRESETS.map(f => (
              <button
                key={f}
                onClick={() => setNumFrames(f)}
                className={`px-2 py-1 text-xs rounded ${
                  numFrames === f
                    ? 'bg-blue-600 text-white'
                    : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'
                }`}
              >
                {f}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* CFG + Seed row */}
      <div className="flex gap-4 items-end flex-wrap">
        <div>
          <label className="text-xs font-medium text-zinc-400">
            CFG: {guidanceScale.toFixed(1)}
          </label>
          <input
            type="range"
            min={1}
            max={15}
            step={0.5}
            value={guidanceScale}
            onChange={e => setGuidanceScale(parseFloat(e.target.value))}
            className="w-32 mt-1 block"
          />
        </div>

        <div className="flex items-center gap-1">
          <div>
            <label className="text-xs font-medium text-zinc-400">Seed</label>
            <input
              type="number"
              value={seed}
              onChange={e => setSeed(parseInt(e.target.value) || -1)}
              className="w-24 mt-1 bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-sm text-zinc-200 block"
            />
          </div>
          <Button variant="ghost" size="sm" onClick={randomizeSeed} className="mt-5 h-7 w-7 p-0">
            <Shuffle className="h-3 w-3" />
          </Button>
        </div>
      </div>

      {/* Advanced toggle */}
      <button
        onClick={() => setShowAdvanced(!showAdvanced)}
        className="text-xs text-zinc-500 hover:text-zinc-300"
      >
        {showAdvanced ? 'Hide advanced' : 'Show advanced'}
      </button>

      {showAdvanced && (
        <div className="flex gap-4">
          <div>
            <label className="text-xs font-medium text-zinc-400">Steps</label>
            <input
              type="number"
              value={steps}
              onChange={e => setSteps(parseInt(e.target.value) || 30)}
              min={1}
              max={100}
              className="w-20 mt-1 bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-sm text-zinc-200 block"
            />
          </div>
        </div>
      )}

      {/* Generate button */}
      <Button onClick={handleSubmit} disabled={loading || !prompt.trim()} className="w-full">
        <Play className="h-4 w-4 mr-2" />
        {loading ? 'Generating...' : 'Generate'}
      </Button>
    </div>
  )
}
