/**
 * Measured VRAM benchmark table for the Training tab.
 *
 * Renders every cell of the master sweep (``GET /api/training/vram-sweep``)
 * so the operator can browse all (profile, quant, blocks_resident)
 * combinations and click any row to apply it to the VRAM-mode knobs.
 * This is the "let me choose myself" complement to the auto-tune
 * recommendation.
 *
 * The table is collapsible (loads its data lazily on first expand),
 * filterable by profile, and sortable by every numeric column.
 */

import { useMemo, useState } from 'react'
import { ChevronDown, ChevronRight, ArrowUpDown } from 'lucide-react'
import {
  useVramSweep,
  type SweepProfile,
  type SweepQuant,
  type VramSweepCell,
} from '../../hooks/useVramSweep'
import type { LowVramMode } from '../../hooks/useVramAutoTune'

// Sweep quant labels ("bf16") map to the low_vram_mode knob ("off").
function quantToLowVramMode(quant: SweepQuant): LowVramMode {
  return quant === 'bf16' ? 'off' : quant
}

type SortKey = 'quant' | 'blocks_resident_on_gpu' | 'peak_vram_gb' | 'runtime_s'
type SortDir = 'asc' | 'desc'

interface VramBenchmarkTableProps {
  /** Apply a chosen cell to the parent's VRAM-mode knobs. */
  onApply: (mode: LowVramMode, blocksResident: number) => void
  /** Currently selected mode/blocks, used to highlight the active row. */
  selectedMode: LowVramMode
  selectedBlocks: number
  disabled?: boolean
}

export function VramBenchmarkTable({
  onApply,
  selectedMode,
  selectedBlocks,
  disabled = false,
}: VramBenchmarkTableProps) {
  const { result, loading, error, fetchSweep } = useVramSweep()
  const [expanded, setExpanded] = useState(false)
  const [profileFilter, setProfileFilter] = useState<SweepProfile>('video')
  const [sortKey, setSortKey] = useState<SortKey>('peak_vram_gb')
  const [sortDir, setSortDir] = useState<SortDir>('asc')

  const toggleExpanded = () => {
    const next = !expanded
    setExpanded(next)
    if (next && !result && !loading) {
      void fetchSweep()
    }
  }

  const setSort = (key: SortKey) => {
    if (key === sortKey) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir('asc')
    }
  }

  const rows = useMemo(() => {
    if (!result) return []
    const quantRank: Record<SweepQuant, number> = { nf4: 0, fp8: 1, bf16: 2 }
    const filtered = result.cells.filter((c) => c.profile === profileFilter)
    const sorted = [...filtered].sort((a, b) => {
      let cmp: number
      if (sortKey === 'quant') {
        cmp = quantRank[a.quant] - quantRank[b.quant]
      } else {
        cmp = a[sortKey] - b[sortKey]
      }
      return sortDir === 'asc' ? cmp : -cmp
    })
    return sorted
  }, [result, profileFilter, sortKey, sortDir])

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/40">
      <button
        type="button"
        onClick={toggleExpanded}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm text-zinc-300 hover:text-zinc-100"
      >
        {expanded ? (
          <ChevronDown className="h-4 w-4" />
        ) : (
          <ChevronRight className="h-4 w-4" />
        )}
        <span className="font-medium">Browse measured benchmarks</span>
        <span className="text-xs text-zinc-500">
          (all tested quant / block-swap combinations)
        </span>
      </button>

      {expanded && (
        <div className="border-t border-zinc-800 px-3 py-3 space-y-3">
          {loading && <p className="text-xs text-zinc-500">Loading benchmark data...</p>}
          {error && <p className="text-xs text-red-400">{error}</p>}

          {result && (
            <>
              <div className="flex items-center justify-between gap-3">
                <div className="inline-flex rounded-md border border-zinc-700 overflow-hidden">
                  {(['video', 'image'] as SweepProfile[]).map((p) => (
                    <button
                      key={p}
                      type="button"
                      onClick={() => setProfileFilter(p)}
                      className={`px-3 py-1 text-xs capitalize ${
                        profileFilter === p
                          ? 'bg-blue-600 text-white'
                          : 'bg-zinc-900 text-zinc-400 hover:text-zinc-200'
                      }`}
                    >
                      {p}
                    </button>
                  ))}
                </div>
                <p className="text-xs text-zinc-600">
                  Click a row to apply. {result.total_blocks} total blocks.
                </p>
              </div>

              <div className="max-h-72 overflow-y-auto rounded border border-zinc-800">
                <table className="w-full text-xs">
                  <thead className="sticky top-0 bg-zinc-900 text-zinc-400">
                    <tr>
                      <HeaderCell label="Quant" col="quant" sortKey={sortKey} sortDir={sortDir} onSort={setSort} />
                      <HeaderCell
                        label="Blocks resident"
                        col="blocks_resident_on_gpu"
                        sortKey={sortKey}
                        sortDir={sortDir}
                        onSort={setSort}
                      />
                      <HeaderCell
                        label="Peak VRAM (GB)"
                        col="peak_vram_gb"
                        sortKey={sortKey}
                        sortDir={sortDir}
                        onSort={setSort}
                      />
                      <HeaderCell
                        label="50-step time (s)"
                        col="runtime_s"
                        sortKey={sortKey}
                        sortDir={sortDir}
                        onSort={setSort}
                      />
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((cell) => (
                      <BenchmarkRow
                        key={`${cell.profile}-${cell.quant}-${cell.blocks_resident_on_gpu}`}
                        cell={cell}
                        active={
                          quantToLowVramMode(cell.quant) === selectedMode &&
                          cell.blocks_resident_on_gpu === selectedBlocks
                        }
                        disabled={disabled}
                        onClick={() =>
                          onApply(quantToLowVramMode(cell.quant), cell.blocks_resident_on_gpu)
                        }
                      />
                    ))}
                  </tbody>
                </table>
              </div>

              <p className="text-[11px] text-zinc-600">
                Source: {result.source}. Lower time is faster; fewer resident blocks lower
                peak VRAM but run slower (more block swapping).
              </p>
            </>
          )}
        </div>
      )}
    </div>
  )
}

function HeaderCell({
  label,
  col,
  sortKey,
  sortDir,
  onSort,
}: {
  label: string
  col: SortKey
  sortKey: SortKey
  sortDir: SortDir
  onSort: (key: SortKey) => void
}) {
  const active = sortKey === col
  return (
    <th className="px-3 py-2 text-left font-medium">
      <button
        type="button"
        onClick={() => onSort(col)}
        className={`inline-flex items-center gap-1 ${active ? 'text-zinc-100' : 'hover:text-zinc-200'}`}
      >
        {label}
        <ArrowUpDown className="h-3 w-3 opacity-60" />
        {active && <span className="text-[10px]">{sortDir === 'asc' ? '↑' : '↓'}</span>}
      </button>
    </th>
  )
}

function BenchmarkRow({
  cell,
  active,
  disabled,
  onClick,
}: {
  cell: VramSweepCell
  active: boolean
  disabled: boolean
  onClick: () => void
}) {
  return (
    <tr
      onClick={disabled ? undefined : onClick}
      className={`border-t border-zinc-800 ${
        active ? 'bg-blue-900/30 text-blue-200' : 'text-zinc-300'
      } ${disabled ? 'opacity-50' : 'cursor-pointer hover:bg-zinc-800/60'}`}
    >
      <td className="px-3 py-1.5 uppercase">{cell.quant}</td>
      <td className="px-3 py-1.5">{cell.blocks_resident_on_gpu}</td>
      <td className="px-3 py-1.5">{cell.peak_vram_gb.toFixed(2)}</td>
      <td className="px-3 py-1.5">{cell.runtime_s}</td>
    </tr>
  )
}
