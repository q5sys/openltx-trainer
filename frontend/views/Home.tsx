import { useState } from 'react'
import { Plus, Folder, MoreVertical, Trash2, Pencil, User, Image, Film, Lightbulb } from 'lucide-react'
import { useProjects } from '../contexts/ProjectContext'
import { useView } from '../contexts/ViewContext'
import { LtxLogo } from '../components/LtxLogo'
import { Button } from '../components/ui/button'
import type { TrainingProject, TrainingMode, TrainingProfile } from '../types/project'

/**
 * The selectable training types in the New Training Project dialog. Each
 * maps to a (mode, profile) pair on the created project. "concept" is
 * deferred for now and rendered as a disabled card.
 */
type NewProjectChoice = 'character_image' | 'character_video' | 'concept'

function formatDate(iso: string): string {
  const date = new Date(iso)
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function ModeLabel({ mode }: { mode: TrainingMode }) {
  if (mode === 'character') {
    return (
      <span className="inline-flex items-center gap-1 text-xs text-blue-400">
        <User className="h-3 w-3" /> Character
      </span>
    )
  }
  return (
    <span className="inline-flex items-center gap-1 text-xs text-amber-400">
      <Lightbulb className="h-3 w-3" /> Concept
    </span>
  )
}

function ProjectCard({ project, onOpen, onDelete, onRename }: {
  project: TrainingProject
  onOpen: () => void
  onDelete: () => void
  onRename: () => void
}) {
  const [showMenu, setShowMenu] = useState(false)

  return (
    <div
      className="group relative bg-zinc-900 rounded-lg overflow-hidden border border-zinc-800 hover:border-zinc-700 transition-colors cursor-pointer"
      onClick={onOpen}
    >
      <div className="aspect-video bg-zinc-800 flex items-center justify-center">
        <Folder className="h-12 w-12 text-zinc-600" />
      </div>
      <div className="p-3">
        <h3 className="text-sm font-medium text-zinc-200 truncate">{project.name}</h3>
        <div className="flex items-center justify-between mt-1">
          <ModeLabel mode={project.mode} />
          {project.trigger && (
            <span className="text-xs text-zinc-500 font-mono truncate max-w-[80px]">{project.trigger}</span>
          )}
        </div>
        <p className="text-xs text-zinc-500 mt-1">{formatDate(project.updatedAt)}</p>
      </div>
      <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
        <button
          onClick={(e) => { e.stopPropagation(); setShowMenu(!showMenu) }}
          className="h-7 w-7 flex items-center justify-center rounded bg-zinc-800/80 text-zinc-400 hover:text-white"
        >
          <MoreVertical className="h-4 w-4" />
        </button>
        {showMenu && (
          <div className="absolute right-0 top-8 w-36 bg-zinc-800 border border-zinc-700 rounded-md shadow-lg z-10">
            <button
              onClick={(e) => { e.stopPropagation(); setShowMenu(false); onRename() }}
              className="flex items-center gap-2 w-full px-3 py-2 text-sm text-zinc-300 hover:bg-zinc-700"
            >
              <Pencil className="h-3.5 w-3.5" /> Rename
            </button>
            <button
              onClick={(e) => { e.stopPropagation(); setShowMenu(false); onDelete() }}
              className="flex items-center gap-2 w-full px-3 py-2 text-sm text-red-400 hover:bg-zinc-700"
            >
              <Trash2 className="h-3.5 w-3.5" /> Delete
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

// --- New Training Project Dialog ---

function NewProjectDialog({ onClose, onCreate }: {
  onClose: () => void
  onCreate: (name: string, mode: TrainingMode, profile: TrainingProfile) => void
}) {
  const [name, setName] = useState('')
  const [choice, setChoice] = useState<NewProjectChoice>('character_image')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const trimmed = name.trim()
    if (!trimmed) return
    // Concept is deferred; its card is disabled and never selectable.
    if (choice === 'character_image') {
      onCreate(trimmed, 'character', 'image')
    } else if (choice === 'character_video') {
      onCreate(trimmed, 'character', 'video')
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div
        className="bg-zinc-900 border border-zinc-700 rounded-lg shadow-xl w-full max-w-lg p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="text-lg font-semibold text-zinc-100 mb-4">New Training Project</h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm text-zinc-400 mb-1">Project Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="My LORA Project"
              autoFocus
              className="w-full bg-zinc-800 border border-zinc-700 rounded-md px-3 py-2 text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-zinc-500"
            />
          </div>

          <div>
            <label className="block text-sm text-zinc-400 mb-2">Training Type</label>
            <div className="grid grid-cols-3 gap-3">
              <button
                type="button"
                onClick={() => setChoice('character_image')}
                className={`flex flex-col items-center gap-2 p-4 rounded-lg border text-sm transition-colors ${
                  choice === 'character_image'
                    ? 'border-blue-500 bg-blue-500/10 text-blue-400'
                    : 'border-zinc-700 bg-zinc-800 text-zinc-400 hover:border-zinc-600'
                }`}
              >
                <Image className="h-6 w-6" />
                <span className="font-medium">Character from Images</span>
                <span className="text-xs text-zinc-500 text-center">Train a person or face from a still-image dataset</span>
              </button>
              <button
                type="button"
                onClick={() => setChoice('character_video')}
                className={`flex flex-col items-center gap-2 p-4 rounded-lg border text-sm transition-colors ${
                  choice === 'character_video'
                    ? 'border-blue-500 bg-blue-500/10 text-blue-400'
                    : 'border-zinc-700 bg-zinc-800 text-zinc-400 hover:border-zinc-600'
                }`}
              >
                <Film className="h-6 w-6" />
                <span className="font-medium">Character from Video</span>
                <span className="text-xs text-zinc-500 text-center">Train a person or face from video clips</span>
              </button>
              <button
                type="button"
                disabled
                aria-disabled="true"
                title="Concept training is coming soon."
                className="flex flex-col items-center gap-2 p-4 rounded-lg border text-sm border-zinc-800 bg-zinc-900 text-zinc-600 cursor-not-allowed opacity-60"
              >
                <Lightbulb className="h-6 w-6" />
                <span className="font-medium">Concept</span>
                <span className="text-xs text-zinc-600 text-center">Style or object training (coming soon)</span>
              </button>
            </div>
          </div>

          <div className="flex justify-end gap-2 pt-2">
            <Button type="button" variant="outline" size="sm" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" size="sm" disabled={!name.trim()}>
              Create Project
            </Button>
          </div>
        </form>
      </div>
    </div>
  )
}

// --- Home View ---

export function Home() {
  const { projects, getProject, createProject, deleteProject, renameProject } = useProjects()
  const { openProject } = useView()
  const [showNewDialog, setShowNewDialog] = useState(false)

  const handleCreate = (name: string, mode: TrainingMode, profile: TrainingProfile) => {
    const id = createProject(name, mode, profile)
    setShowNewDialog(false)
    openProject(id)
  }

  const handleDelete = (id: string) => {
    if (confirm('Delete this project? This cannot be undone.')) {
      deleteProject(id)
    }
  }

  const handleRename = (id: string) => {
    const project = getProject(id)
    if (!project) return
    const newName = prompt('Rename project:', project.name)
    if (newName && newName.trim()) {
      renameProject(id, newName.trim())
    }
  }

  return (
    <div className="h-full bg-zinc-950 overflow-auto">
      <div className="max-w-5xl mx-auto px-6 py-8">
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-3">
            <LtxLogo className="h-8 w-8" />
            <h1 className="text-xl font-semibold text-zinc-100">OpenLTX Trainer</h1>
          </div>
          <Button onClick={() => setShowNewDialog(true)} size="sm">
            <Plus className="h-4 w-4 mr-2" />
            New Training Project
          </Button>
        </div>

        {projects.length === 0 ? (
          <div className="text-center py-20">
            <Folder className="h-16 w-16 text-zinc-700 mx-auto mb-4" />
            <h2 className="text-lg font-medium text-zinc-400 mb-2">No training projects yet</h2>
            <p className="text-sm text-zinc-500 mb-6">Create a new project to start training a LORA.</p>
            <Button onClick={() => setShowNewDialog(true)}>
              <Plus className="h-4 w-4 mr-2" />
              Create Project
            </Button>
          </div>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {projects.map(project => (
              <ProjectCard
                key={project.id}
                project={project}
                onOpen={() => openProject(project.id)}
                onDelete={() => handleDelete(project.id)}
                onRename={() => handleRename(project.id)}
              />
            ))}
          </div>
        )}
      </div>

      {showNewDialog && (
        <NewProjectDialog
          onClose={() => setShowNewDialog(false)}
          onCreate={handleCreate}
        />
      )}
    </div>
  )
}
