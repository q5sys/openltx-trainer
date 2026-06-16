import React, { createContext, useCallback, useContext, useState } from 'react'
import {
  type TrainingProject,
  type TrainingMode,
  type TrainingProfile,
  type ProjectTab,
  createTrainingProject,
  normalizeProject,
} from '../types/project'

interface ProjectContextType {
  projects: TrainingProject[]
  activeProject: TrainingProject | null
  currentTab: ProjectTab
  getProject: (id: string) => TrainingProject | undefined
  createProject: (name: string, mode: TrainingMode, profile: TrainingProfile) => string
  deleteProject: (id: string) => void
  renameProject: (id: string, name: string) => void
  updateProject: (id: string, updates: Partial<Omit<TrainingProject, 'id' | 'createdAt'>>) => void
  activateProject: (id: string) => void
  clearActiveProject: () => void
  setCurrentTab: (tab: ProjectTab) => void
}

const ProjectContext = createContext<ProjectContextType | null>(null)

const STORAGE_KEY = 'openltx-training-projects'

function loadProjects(): TrainingProject[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    // Backfill the profile field on projects created before the
    // image/video split so they keep the historical video behaviour.
    return (JSON.parse(raw) as TrainingProject[]).map(normalizeProject)
  } catch {
    return []
  }
}

function saveProjects(projects: TrainingProject[]): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(projects))
}

export function ProjectProvider({ children }: { children: React.ReactNode }) {
  const [projects, setProjects] = useState<TrainingProject[]>(loadProjects)
  const [activeProjectId, setActiveProjectId] = useState<string | null>(null)
  const [currentTab, setCurrentTab] = useState<ProjectTab>('dataset')

  const activeProject = projects.find(p => p.id === activeProjectId) ?? null

  const getProject = useCallback((id: string) => {
    return projects.find(p => p.id === id)
  }, [projects])

  const createNewProject = useCallback((name: string, mode: TrainingMode, profile: TrainingProfile) => {
    const project = createTrainingProject(name, mode, profile)
    const updated = [project, ...projects]
    setProjects(updated)
    saveProjects(updated)
    return project.id
  }, [projects])

  const deleteProject = useCallback((id: string) => {
    const updated = projects.filter(p => p.id !== id)
    setProjects(updated)
    saveProjects(updated)
    if (activeProjectId === id) setActiveProjectId(null)
  }, [projects, activeProjectId])

  const renameProject = useCallback((id: string, name: string) => {
    const updated = projects.map(p =>
      p.id === id ? { ...p, name, updatedAt: new Date().toISOString() } : p
    )
    setProjects(updated)
    saveProjects(updated)
  }, [projects])

  const updateProject = useCallback((id: string, updates: Partial<Omit<TrainingProject, 'id' | 'createdAt'>>) => {
    const updated = projects.map(p =>
      p.id === id ? { ...p, ...updates, updatedAt: new Date().toISOString() } : p
    )
    setProjects(updated)
    saveProjects(updated)
  }, [projects])

  const activateProject = useCallback((id: string) => {
    setActiveProjectId(id)
  }, [])

  const clearActiveProject = useCallback(() => {
    setActiveProjectId(null)
  }, [])

  return (
    <ProjectContext.Provider value={{
      projects,
      activeProject,
      currentTab,
      getProject,
      createProject: createNewProject,
      deleteProject,
      renameProject,
      updateProject,
      activateProject,
      clearActiveProject,
      setCurrentTab,
    }}>
      {children}
    </ProjectContext.Provider>
  )
}

export function useProjects() {
  const context = useContext(ProjectContext)
  if (!context) {
    throw new Error('useProjects must be used within a ProjectProvider')
  }
  return context
}
