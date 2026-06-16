/**
 * Pure SVG loss chart. No external charting library.
 *
 * Renders a line chart of loss values over training steps.
 * Supports optional smoothing via exponential moving average.
 */

import { useMemo } from 'react'
import type { ProgressRecord } from '../../hooks/useTrainingProgress'

interface LossChartProps {
  records: ProgressRecord[]
  height?: number
  smoothingFactor?: number
}

function smoothEma(values: number[], alpha: number): number[] {
  if (values.length === 0) return []
  const result: number[] = [values[0]]
  for (let i = 1; i < values.length; i++) {
    result.push(alpha * values[i] + (1 - alpha) * result[i - 1])
  }
  return result
}

export function LossChart({ records, height = 200, smoothingFactor = 0.3 }: LossChartProps) {
  const chartData = useMemo(() => {
    if (records.length < 2) return null

    const steps = records.map(r => r.step)
    const rawLoss = records.map(r => r.loss)
    const smoothedLoss = smoothEma(rawLoss, smoothingFactor)

    const minStep = steps[0]
    const maxStep = steps[steps.length - 1]
    const stepRange = maxStep - minStep || 1

    // Use smoothed values for Y range to avoid outlier spikes dominating
    const allValues = [...smoothedLoss, ...rawLoss]
    const minLoss = Math.min(...allValues)
    const maxLoss = Math.max(...allValues)
    const lossRange = maxLoss - minLoss || 1

    // Add 10% padding to Y range
    const yMin = minLoss - lossRange * 0.1
    const yMax = maxLoss + lossRange * 0.1
    const yRange = yMax - yMin

    return { steps, rawLoss, smoothedLoss, minStep, stepRange, yMin, yRange }
  }, [records, smoothingFactor])

  if (!chartData) {
    return (
      <div
        className="flex items-center justify-center bg-zinc-900 rounded-lg border border-zinc-800"
        style={{ height }}
      >
        <span className="text-xs text-zinc-600">
          {records.length === 0 ? 'No progress data yet' : 'Waiting for more data points...'}
        </span>
      </div>
    )
  }

  const { steps, rawLoss, smoothedLoss, minStep, stepRange, yMin, yRange } = chartData
  const padding = { top: 20, right: 20, bottom: 30, left: 50 }
  const width = 600
  const plotWidth = width - padding.left - padding.right
  const plotHeight = height - padding.top - padding.bottom

  function toX(step: number): number {
    return padding.left + ((step - minStep) / stepRange) * plotWidth
  }

  function toY(loss: number): number {
    return padding.top + (1 - (loss - yMin) / yRange) * plotHeight
  }

  // Build SVG path strings
  const rawPath = steps.map((s, i) => `${i === 0 ? 'M' : 'L'}${toX(s)},${toY(rawLoss[i])}`).join(' ')
  const smoothPath = steps
    .map((s, i) => `${i === 0 ? 'M' : 'L'}${toX(s)},${toY(smoothedLoss[i])}`)
    .join(' ')

  // Y-axis tick marks (5 ticks)
  const yTicks = Array.from({ length: 5 }, (_, i) => {
    const value = yMin + (yRange * i) / 4
    return { y: toY(value), label: value.toFixed(3) }
  })

  // X-axis tick marks (5 ticks)
  const xTicks = Array.from({ length: 5 }, (_, i) => {
    const step = minStep + (stepRange * i) / 4
    return { x: toX(step), label: Math.round(step).toString() }
  })

  return (
    <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium text-zinc-400">Loss</span>
        <div className="flex items-center gap-3 text-xs text-zinc-500">
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-0.5 bg-zinc-600 rounded" />
            Raw
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-0.5 bg-blue-400 rounded" />
            Smoothed
          </span>
        </div>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full" preserveAspectRatio="xMidYMid meet">
        {/* Grid lines */}
        {yTicks.map((tick, i) => (
          <line
            key={`yg-${i}`}
            x1={padding.left}
            y1={tick.y}
            x2={width - padding.right}
            y2={tick.y}
            stroke="#27272a"
            strokeWidth={1}
          />
        ))}

        {/* Y-axis labels */}
        {yTicks.map((tick, i) => (
          <text
            key={`yl-${i}`}
            x={padding.left - 6}
            y={tick.y + 3}
            textAnchor="end"
            className="fill-zinc-500"
            fontSize={10}
          >
            {tick.label}
          </text>
        ))}

        {/* X-axis labels */}
        {xTicks.map((tick, i) => (
          <text
            key={`xl-${i}`}
            x={tick.x}
            y={height - 6}
            textAnchor="middle"
            className="fill-zinc-500"
            fontSize={10}
          >
            {tick.label}
          </text>
        ))}

        {/* Raw loss line (dimmer) */}
        <path d={rawPath} fill="none" stroke="#52525b" strokeWidth={1} opacity={0.6} />

        {/* Smoothed loss line (bright) */}
        <path d={smoothPath} fill="none" stroke="#60a5fa" strokeWidth={2} />
      </svg>
    </div>
  )
}
