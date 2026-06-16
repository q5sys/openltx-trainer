import { Plus, X } from 'lucide-react'
import { Button } from '../ui/button'
import type { LoraDescriptor, LoraStackEntry } from '../../hooks/useVerification'

interface LoraSelectorProps {
  availableLoras: LoraDescriptor[]
  loraStack: LoraStackEntry[]
  onStackChange: (stack: LoraStackEntry[]) => void
}

function loraLabel(lora: LoraDescriptor): string {
  const parts: string[] = []
  if (lora.phase) parts.push(lora.phase)
  parts.push(`step ${lora.step}`)
  if (lora.rank) parts.push(`rank ${lora.rank}`)
  if (lora.project_name) parts.push(`(${lora.project_name})`)
  return parts.join(' - ')
}

export function LoraSelector({ availableLoras, loraStack, onStackChange }: LoraSelectorProps) {
  const addLora = () => {
    if (availableLoras.length === 0) return
    const first = availableLoras[0]
    onStackChange([...loraStack, { checkpoint_path: first.checkpoint_path, weight: 1.0 }])
  }

  const removeLora = (index: number) => {
    onStackChange(loraStack.filter((_, i) => i !== index))
  }

  const updateLoraPath = (index: number, path: string) => {
    const updated = [...loraStack]
    updated[index] = { ...updated[index], checkpoint_path: path }
    onStackChange(updated)
  }

  const updateLoraWeight = (index: number, weight: number) => {
    const updated = [...loraStack]
    updated[index] = { ...updated[index], weight }
    onStackChange(updated)
  }

  return (
    <div className="space-y-2">
      <label className="text-xs font-medium text-zinc-400 uppercase tracking-wide">LORA Stack</label>

      {loraStack.length === 0 && (
        <p className="text-xs text-zinc-600">No LORAs selected. Generate without LORA or add one below.</p>
      )}

      {loraStack.map((entry, index) => (
        <div key={index} className="flex items-center gap-2">
          <select
            value={entry.checkpoint_path}
            onChange={e => updateLoraPath(index, e.target.value)}
            className="flex-1 bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-sm text-zinc-200"
          >
            {availableLoras.map(lora => (
              <option key={lora.checkpoint_path} value={lora.checkpoint_path}>
                {loraLabel(lora)}
              </option>
            ))}
          </select>

          <div className="flex items-center gap-1">
            <input
              type="range"
              min={0}
              max={1.5}
              step={0.05}
              value={entry.weight}
              onChange={e => updateLoraWeight(index, parseFloat(e.target.value))}
              className="w-20"
            />
            <span className="text-xs text-zinc-400 w-8 text-right">{entry.weight.toFixed(2)}</span>
          </div>

          <Button variant="ghost" size="sm" onClick={() => removeLora(index)} className="h-7 w-7 p-0">
            <X className="h-3 w-3" />
          </Button>
        </div>
      ))}

      <Button variant="outline" size="sm" onClick={addLora} disabled={availableLoras.length === 0}>
        <Plus className="h-3 w-3 mr-1" />
        Add LORA
      </Button>
    </div>
  )
}
