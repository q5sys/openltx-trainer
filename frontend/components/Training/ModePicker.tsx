/**
 * ModePicker: training mode selection (Character or Concept).
 *
 * Presents two cards explaining each mode. The selected mode determines
 * which preset is used, which validation rules apply, and which
 * captioning prompt templates are offered.
 */

import { User, Palette } from 'lucide-react'

type TrainingMode = 'character' | 'concept'

interface ModePickerProps {
  /** Currently selected mode. */
  value: TrainingMode
  /** Called when the user selects a mode. */
  onChange: (mode: TrainingMode) => void
  /** Disable selection (e.g., during active training). */
  disabled?: boolean
  /** Optional CSS class. */
  className?: string
}

interface ModeCardProps {
  mode: TrainingMode
  title: string
  description: string
  details: string[]
  icon: React.ReactNode
  selected: boolean
  disabled: boolean
  onClick: () => void
}

function ModeCard({ title, description, details, icon, selected, disabled, onClick }: ModeCardProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`
        flex flex-col items-start gap-2 rounded-lg border p-4 text-left transition-colors
        ${selected
          ? 'border-primary bg-primary/5 ring-1 ring-primary'
          : 'border-border hover:border-primary/50 hover:bg-muted/30'
        }
        ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
      `}
    >
      <div className="flex items-center gap-2">
        <div className={`${selected ? 'text-primary' : 'text-muted-foreground'}`}>
          {icon}
        </div>
        <h3 className="text-sm font-medium">{title}</h3>
      </div>
      <p className="text-xs text-muted-foreground">{description}</p>
      <ul className="text-xs text-muted-foreground/80 space-y-0.5">
        {details.map((detail, i) => (
          <li key={i} className="flex items-start gap-1">
            <span className="text-muted-foreground/40">-</span>
            {detail}
          </li>
        ))}
      </ul>
    </button>
  )
}

export function ModePicker({ value, onChange, disabled = false, className = '' }: ModePickerProps) {
  return (
    <div className={`grid grid-cols-2 gap-3 ${className}`}>
      <ModeCard
        mode="character"
        title="Character"
        description="Train a specific person, character, or consistent subject identity."
        details={[
          '4-phase preset (2500 steps)',
          'Higher LORA ranks for identity capture',
          'Trigger word required (e.g., ohwx)',
          'Best with 15-30 clips of the subject',
        ]}
        icon={<User className="h-5 w-5" />}
        selected={value === 'character'}
        disabled={disabled}
        onClick={() => onChange('character')}
      />
      <ModeCard
        mode="concept"
        title="Concept"
        description="Train a visual style, aesthetic, or environmental concept."
        details={[
          '3-phase preset (2000 steps)',
          'Lower LORA ranks for style transfer',
          'Trigger word optional but recommended',
          'Best with 20-50 diverse example clips',
        ]}
        icon={<Palette className="h-5 w-5" />}
        selected={value === 'concept'}
        disabled={disabled}
        onClick={() => onChange('concept')}
      />
    </div>
  )
}
