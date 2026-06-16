/**
 * PhaseTransitionBanner: displays a notification when the training
 * transitions between phases (e.g., Phase 1 -> Phase 2).
 *
 * Shows the current phase name, the step at which it started, and
 * any changes in training parameters (learning rate, LORA rank).
 */

import { useEffect, useState } from 'react'
import { ArrowRight } from 'lucide-react'

interface PhaseInfo {
  name: string
  displayName: string
  startsAtStep: number
  endsAtStep: number
  loraRank?: number
  learningRate?: number
}

interface PhaseTransitionBannerProps {
  /** Current training step. */
  currentStep: number
  /** List of all phases in order. */
  phases: PhaseInfo[]
  /** Optional CSS class. */
  className?: string
}

function findCurrentPhase(step: number, phases: PhaseInfo[]): PhaseInfo | null {
  for (const phase of phases) {
    if (step >= phase.startsAtStep && step < phase.endsAtStep) {
      return phase
    }
  }
  return null
}

function findPreviousPhase(currentPhase: PhaseInfo, phases: PhaseInfo[]): PhaseInfo | null {
  const idx = phases.indexOf(currentPhase)
  return idx > 0 ? phases[idx - 1] : null
}

export function PhaseTransitionBanner({
  currentStep,
  phases,
  className = '',
}: PhaseTransitionBannerProps) {
  const [showBanner, setShowBanner] = useState(false)
  const [lastSeenPhase, setLastSeenPhase] = useState<string | null>(null)

  const currentPhase = findCurrentPhase(currentStep, phases)
  const previousPhase = currentPhase ? findPreviousPhase(currentPhase, phases) : null

  useEffect(() => {
    if (!currentPhase) return

    if (lastSeenPhase !== null && lastSeenPhase !== currentPhase.name) {
      // Phase changed: show the banner.
      setShowBanner(true)
      const timer = setTimeout(() => setShowBanner(false), 8000)
      return () => clearTimeout(timer)
    }

    setLastSeenPhase(currentPhase.name)
  }, [currentPhase?.name])

  if (!showBanner || !currentPhase) return null

  return (
    <div className={`rounded-lg border border-blue-500/30 bg-blue-500/10 p-3 ${className}`}>
      <div className="flex items-center gap-2 text-sm font-medium text-blue-400">
        {previousPhase && (
          <>
            <span className="text-muted-foreground">{previousPhase.displayName}</span>
            <ArrowRight className="h-4 w-4" />
          </>
        )}
        <span>{currentPhase.displayName}</span>
        <span className="text-xs text-muted-foreground ml-auto">Step {currentStep}</span>
      </div>

      {/* Parameter changes */}
      {previousPhase && (
        <div className="mt-2 flex gap-4 text-xs text-muted-foreground">
          {currentPhase.loraRank !== previousPhase.loraRank && (
            <span>
              Rank: {previousPhase.loraRank} &rarr; {currentPhase.loraRank}
            </span>
          )}
          {currentPhase.learningRate !== previousPhase.learningRate && (
            <span>
              LR: {previousPhase.learningRate?.toExponential(0)} &rarr; {currentPhase.learningRate?.toExponential(0)}
            </span>
          )}
        </div>
      )}
    </div>
  )
}
