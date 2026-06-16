import type { VerificationJobStatus } from '../../hooks/useVerification'

interface ComparisonGridProps {
  jobs: VerificationJobStatus[]
}

export function ComparisonGrid({ jobs }: ComparisonGridProps) {
  if (jobs.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-zinc-600 text-sm">
        No comparison results yet. Generate multiple samples to compare.
      </div>
    )
  }

  const gridCols = jobs.length <= 2 ? 'grid-cols-2' : 'grid-cols-2'

  return (
    <div className={`grid ${gridCols} gap-2`}>
      {jobs.map(job => (
        <div key={job.generation_id} className="bg-zinc-900 rounded border border-zinc-800 overflow-hidden">
          <div className="aspect-video bg-zinc-950 flex items-center justify-center">
            {job.status === 'completed' && job.output_path ? (
              <video
                src={`file://${job.output_path}`}
                controls
                loop
                muted
                className="w-full h-full object-contain"
              />
            ) : job.status === 'errored' ? (
              <span className="text-red-400 text-xs">Error: {job.error_message}</span>
            ) : (
              <div className="text-zinc-600 text-xs">
                {job.status} ({Math.round(job.progress * 100)}%)
              </div>
            )}
          </div>
          <div className="p-2">
            <p className="text-xs text-zinc-400 truncate">{job.prompt}</p>
            <p className="text-xs text-zinc-600">Seed: {job.seed}</p>
          </div>
        </div>
      ))}
    </div>
  )
}
