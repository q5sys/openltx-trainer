import { ArrowLeft, Database, Cpu, Activity, CheckCircle } from 'lucide-react'
import { useProjects } from '../contexts/ProjectContext'
import { useView } from '../contexts/ViewContext'
import { Button } from '../components/ui/button'
import { DatasetTab } from '../components/Dataset/DatasetTab'
import { TrainingTab } from '../components/Training/TrainingTab'
import { MonitorTab } from '../components/Monitor/MonitorTab'
import { VerifyTab } from '../components/Verify/VerifyTab'
import type { ProjectTab } from '../types/project'

const TABS: { id: ProjectTab; label: string; icon: React.ReactNode }[] = [
  { id: 'dataset', label: 'Dataset', icon: <Database className="h-4 w-4" /> },
  { id: 'training', label: 'Training', icon: <Cpu className="h-4 w-4" /> },
  { id: 'monitor', label: 'Monitor', icon: <Activity className="h-4 w-4" /> },
  { id: 'verify', label: 'Verify', icon: <CheckCircle className="h-4 w-4" /> },
]

function TabPlaceholder({ tab }: { tab: ProjectTab }) {
  const labels: Record<ProjectTab, string> = {
    dataset: 'Dataset management: import sources, cut clips, browse and caption your training data.',
    training: 'Training configuration: select mode, configure hyperparameters, and start training.',
    monitor: 'Training monitor: view live progress, loss curves, and sample outputs.',
    verify: 'Verification: test your trained LORA with prompt-based generation.',
  }

  return (
    <div className="flex-1 flex items-center justify-center">
      <div className="text-center max-w-md">
        <h3 className="text-lg font-medium text-zinc-300 mb-2 capitalize">{tab}</h3>
        <p className="text-sm text-zinc-500">{labels[tab]}</p>
        <p className="text-xs text-zinc-600 mt-4">This tab will be implemented in a future step.</p>
      </div>
    </div>
  )
}

export function Project() {
  const { activeProject, currentTab, setCurrentTab } = useProjects()
  const { goHome } = useView()

  if (!activeProject) {
    return null
  }

  return (
    <div className="flex flex-col h-full bg-zinc-950">
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-3 border-b border-zinc-800">
        <Button variant="ghost" size="sm" onClick={goHome} className="h-8 w-8 p-0">
          <ArrowLeft className="h-4 w-4" />
        </Button>
        <div className="flex-1 min-w-0">
          <h2 className="text-sm font-medium text-zinc-100 truncate">{activeProject.name}</h2>
          <p className="text-xs text-zinc-500 capitalize">{activeProject.mode} mode</p>
        </div>
      </div>

      {/* Tab bar */}
      <div className="flex border-b border-zinc-800">
        {TABS.map(tab => (
          <button
            key={tab.id}
            onClick={() => setCurrentTab(tab.id)}
            className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium transition-colors border-b-2 ${
              currentTab === tab.id
                ? 'border-blue-500 text-blue-400'
                : 'border-transparent text-zinc-500 hover:text-zinc-300'
            }`}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content - flex-1 min-h-0 ensures the tab can shrink and scroll */}
      <div className="flex-1 min-h-0 flex flex-col">
        {currentTab === 'dataset' ? (
          <DatasetTab />
        ) : currentTab === 'training' ? (
          <TrainingTab />
        ) : currentTab === 'monitor' ? (
          <MonitorTab />
        ) : currentTab === 'verify' ? (
          <VerifyTab />
        ) : (
          <TabPlaceholder tab={currentTab} />
        )}
      </div>
    </div>
  )
}
