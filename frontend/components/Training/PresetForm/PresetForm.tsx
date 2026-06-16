/**
 * PresetForm: advanced preset editing component.
 *
 * Loads a preset's phases, dataset config, and sampling config
 * and allows the user to edit all parameters before starting training.
 * Exposes the final config via an onChange callback.
 */

import { useState, useCallback } from 'react'
import { ChevronDown, ChevronUp, RotateCcw } from 'lucide-react'
import { PhaseEditor, type PhaseValues } from './PhaseEditor'
import { DatasetConfigEditor, type DatasetConfigValues } from './DatasetConfigEditor'
import {
  SamplingConfigEditor,
  type SamplingConfigValues,
  type SampleSpecValues,
} from './SamplingConfigEditor'

import { Button } from '../../ui/button'

export interface PresetFormValues {
  phases: PhaseValues[]
  dataset: DatasetConfigValues
  sampling: SamplingConfigValues
}

interface PresetFormProps {
  /** Initial values loaded from the selected preset. */
  initialValues: PresetFormValues
  /** Called whenever a field changes. */
  onChange: (values: PresetFormValues) => void
  /** Disable all editing (e.g., during active training). */
  disabled?: boolean
  /** Optional CSS class. */
  className?: string
}

export function PresetForm({ initialValues, onChange, disabled, className }: PresetFormProps) {
  const [values, setValues] = useState<PresetFormValues>(initialValues)
  const [expanded, setExpanded] = useState(false)

  const emit = useCallback(
    (next: PresetFormValues) => {
      setValues(next)
      onChange(next)
    },
    [onChange],
  )

  function handlePhaseChange(index: number, field: keyof PhaseValues, value: string | number) {
    const nextPhases = values.phases.map((p, i) => {
      if (i !== index) return p
      return { ...p, [field]: value }
    })
    emit({ ...values, phases: nextPhases })
  }

  function handlePhaseRemove(index: number) {
    if (values.phases.length <= 1) return
    const nextPhases = values.phases.filter((_, i) => i !== index)
    emit({ ...values, phases: nextPhases })
  }

  function handleAddPhase() {
    const lastPhase = values.phases[values.phases.length - 1]
    const newPhase: PhaseValues = {
      name: `Phase ${values.phases.length + 1}`,
      start_step: lastPhase?.end_step ?? 0,
      end_step: (lastPhase?.end_step ?? 0) + 500,
      lora_rank: lastPhase?.lora_rank ?? 32,
      learning_rate: lastPhase?.learning_rate ?? 0.0001,
      gradient_accumulation: 1,
      differential_guidance: 1.0,
      timestep_bias: 'none',
      save_every: lastPhase?.save_every ?? 100,
    }
    emit({ ...values, phases: [...values.phases, newPhase] })

  }

  function handleDatasetChange(field: keyof DatasetConfigValues, value: number | boolean) {
    emit({ ...values, dataset: { ...values.dataset, [field]: value } })
  }

  function handleSamplingChange(
    field: keyof SamplingConfigValues,
    value: number | SampleSpecValues[],
  ) {
    emit({ ...values, sampling: { ...values.sampling, [field]: value } })
  }


  function handleReset() {
    emit(initialValues)
  }

  // Summary line for collapsed state.
  const totalSteps = values.phases.length > 0
    ? values.phases[values.phases.length - 1].end_step
    : 0
  const phaseCount = values.phases.length

  return (
    <div className={`space-y-3 ${className ?? ''}`}>
      {/* Collapsed header */}
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center justify-between rounded-lg border border-zinc-700
                   bg-zinc-900 px-4 py-3 text-left transition-colors hover:border-zinc-600"
      >
        <div>
          <span className="text-sm font-medium text-zinc-200">Advanced Settings</span>
          <span className="ml-3 text-xs text-zinc-500">
            {phaseCount} phase{phaseCount !== 1 ? 's' : ''}, {totalSteps} total steps,{' '}
            {values.dataset.target_resolution}px
          </span>
        </div>
        {expanded ? (
          <ChevronUp className="h-4 w-4 text-zinc-500" />
        ) : (
          <ChevronDown className="h-4 w-4 text-zinc-500" />
        )}
      </button>

      {expanded && (
        <div className="space-y-4 pl-1">
          {/* Phases */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium text-zinc-300">Training Phases</h3>
              <div className="flex gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleReset}
                  disabled={disabled}
                  className="text-xs"
                >
                  <RotateCcw className="mr-1 h-3 w-3" />
                  Reset
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleAddPhase}
                  disabled={disabled}
                  className="text-xs"
                >
                  + Add Phase
                </Button>
              </div>
            </div>

            {values.phases.map((phase, i) => (
              <PhaseEditor
                key={i}
                phase={phase}
                index={i}
                onChange={handlePhaseChange}
                onRemove={handlePhaseRemove}
                removable={values.phases.length > 1}
                disabled={disabled}
              />
            ))}
          </div>

          {/* Dataset Config */}
          <DatasetConfigEditor
            config={values.dataset}
            onChange={handleDatasetChange}
            disabled={disabled}
          />

          {/* Sampling Config */}
          <SamplingConfigEditor
            config={values.sampling}
            onChange={handleSamplingChange}
            disabled={disabled}
          />
        </div>
      )}
    </div>
  )
}
