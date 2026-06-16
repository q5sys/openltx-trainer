/**
 * DatasetTab: container for the dataset sub-views (Sources, Cutter, Browser, Caption).
 * Includes the TriggerWordPanel and ValidationPanel in the header area.
 */

import { useState, useEffect, useCallback } from 'react'
import { useProjects } from '../../contexts/ProjectContext'
import { useDataset } from '../../hooks/useDataset'
import { SourceList } from './Sources/SourceList'
import { SceneProposalList } from './Cutter/SceneProposalList'
import { ClipGrid } from './Browser/ClipGrid'
import { CaptionPanel } from './Caption/CaptionPanel'
import { TriggerWordPanel } from './TriggerWord/TriggerWordPanel'
import { ValidationPanel } from './Validation/ValidationPanel'
import type { SourceMediaInfo } from '../../types/dataset'

type DatasetSubView = 'sources' | 'cutter' | 'browser' | 'caption'

export function DatasetTab() {
  const { activeProject, updateProject } = useProjects()
  const dataset = useDataset()
  const [subView, setSubView] = useState<DatasetSubView>('sources')
  const [activeSource, setActiveSource] = useState<SourceMediaInfo | null>(null)

  const datasetDir = activeProject?.datasetPath || ''

  // Auto-scan directory for media files whenever sources list is empty.
  // This populates the sources panel and auto-imports new images.
  useEffect(() => {
    if (datasetDir && dataset.sources.length === 0 && !dataset.isLoading) {
      void dataset.scanAndImport(datasetDir)
    }
  }, [datasetDir])

  // Load clips whenever the user switches to any sub-view. Sources view also
  // needs the clip list now so it can flag sources that have already been
  // imported as a clip.
  useEffect(() => {
    if (datasetDir) {
      void dataset.listClips(datasetDir)
    }
  }, [subView, datasetDir])

  // Image-profile projects train from still images and never cut video,
  // so the Cutter sub-tab is hidden for them (see
  // memory-bank/feature_character_from_images.md). Video and concept
  // projects keep the full tab set.
  const isImageProfile = activeProject?.profile === 'image'
  const datasetSubViews: DatasetSubView[] = isImageProfile
    ? ['sources', 'browser', 'caption']
    : ['sources', 'cutter', 'browser', 'caption']

  const handleSourceSelected = (source: SourceMediaInfo) => {
    // The Cutter does not exist for image projects; selecting a source
    // there is a no-op because images are imported directly.
    if (isImageProfile) return
    setActiveSource(source)
    setSubView('cutter')
  }

  const handleClipsCreated = () => {
    setSubView('browser')
    if (datasetDir) {
      void dataset.listClips(datasetDir)
    }
  }

  // Called after "Import as Clip" finishes on the Sources tab. We deliberately
  // do NOT switch to the browser view here: the user is likely importing
  // several sources back-to-back and should stay on the Sources tab. We only
  // refresh the clip list so the per-row "Imported" flag updates immediately.
  const handleSourceClipImported = () => {
    if (datasetDir) {
      void dataset.listClips(datasetDir)
    }
  }

  const handleTriggerChange = (trigger: string) => {
    if (activeProject) {
      updateProject(activeProject.id, { trigger })
    }
  }

  const handleRefresh = useCallback(() => {
    if (datasetDir) {
      void dataset.listClips(datasetDir)
      if (activeProject?.trigger) {
        void dataset.validateDataset(datasetDir, activeProject.trigger)
      }
    }
  }, [datasetDir, activeProject?.trigger])

  const handleValidate = useCallback(() => {
    if (datasetDir) {
      void dataset.validateDataset(datasetDir, activeProject?.trigger ?? null)
    }
  }, [datasetDir, activeProject?.trigger])

  const handleClearDataset = useCallback(async () => {
    if (!datasetDir || !activeProject) return
    const confirmed = window.confirm(
      'This will delete all imported clips from the dataset. The original source files will not be affected. Continue?'
    )
    if (confirmed) {
      await dataset.deleteAllClips(datasetDir)
    }
  }, [datasetDir, activeProject])

  const handleRemoveDataset = useCallback(() => {
    if (!activeProject) return
    const confirmed = window.confirm(
      `WARNING: This will unlink the dataset directory from this project.\n\nThe files in "${datasetDir}" will NOT be deleted. To delete the folder and all its contents, you must do so manually from your file manager.\n\nContinue?`
    )
    if (confirmed) {
      updateProject(activeProject.id, { datasetPath: '' })
    }
  }, [activeProject, datasetDir])

  if (!activeProject) return null

  // If no dataset path is set yet, prompt to initialize one.
  if (!datasetDir) {
    const handleChooseDir = async () => {
      try {
        const result = await window.electronAPI.showOpenDirectoryDialog({ title: 'Choose Dataset Directory' })
        if (result) {
          updateProject(activeProject.id, { datasetPath: result })
        }
      } catch {
        // User cancelled.
      }
    }

    const handleDragOver = (e: React.DragEvent) => {
      e.preventDefault()
      e.stopPropagation()
      e.dataTransfer.dropEffect = 'copy'
    }

    const handleDrop = (e: React.DragEvent) => {
      e.preventDefault()
      e.stopPropagation()

      console.log('[DatasetTab] Drop event fired')
      console.log('[DatasetTab] files count:', e.dataTransfer.files.length)
      console.log('[DatasetTab] types:', Array.from(e.dataTransfer.types))

      // Approach 1: Electron exposes .path on File objects for local drag-drop.
      const files = Array.from(e.dataTransfer.files)
      for (const file of files) {
        const filePath = (file as unknown as { path?: string }).path
        console.log('[DatasetTab] file:', file.name, 'type:', file.type, 'size:', file.size, 'path:', filePath)
        if (filePath) {
          updateProject(activeProject.id, { datasetPath: filePath })
          return
        }
      }

      // Approach 2: On Linux, file managers may send text/uri-list instead of files.
      const uriList = e.dataTransfer.getData('text/uri-list')
      console.log('[DatasetTab] uri-list:', uriList)
      if (uriList) {
        const firstUri = uriList.split('\n').find(line => line.startsWith('file://'))
        if (firstUri) {
          const dirPath = decodeURIComponent(firstUri.replace('file://', '').trim())
          console.log('[DatasetTab] Resolved path from URI:', dirPath)
          if (dirPath) {
            updateProject(activeProject.id, { datasetPath: dirPath })
            return
          }
        }
      }

      // Approach 3: Plain text (some file managers send paths as text/plain).
      const plainText = e.dataTransfer.getData('text/plain')
      console.log('[DatasetTab] text/plain:', plainText)
      if (plainText && plainText.startsWith('/')) {
        updateProject(activeProject.id, { datasetPath: plainText.trim() })
        return
      }

      console.warn('[DatasetTab] Could not resolve folder path from drop event.')
    }

    return (
      <div
        className="flex-1 flex items-center justify-center"
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        <div className="text-center max-w-md">
          <h3 className="text-lg font-medium text-zinc-300 mb-2">No Dataset Directory</h3>
          <p className="text-sm text-zinc-500 mb-4">
            Set a dataset directory for this project to begin importing sources.
          </p>
          <div
            className="border-2 border-dashed border-zinc-700 rounded-lg p-8 mb-4 hover:border-blue-500 transition-colors cursor-pointer"
            onClick={handleChooseDir}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
          >
            <p className="text-sm text-zinc-400 mb-2">Drag and drop a folder here</p>
            <p className="text-xs text-zinc-600">or click to browse</p>
          </div>
          <button
            onClick={handleChooseDir}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded transition-colors"
          >
            Choose Dataset Directory
          </button>
        </div>
      </div>
    )
  }

  // Compute trigger stats from validation result or clips.
  const triggerPresentCount = dataset.validation?.stats.trigger_present ?? 0
  const captionedCount = dataset.validation?.stats.captioned ?? dataset.clips.filter(c => c.caption.trim()).length

  return (
    <div className="flex flex-col h-full">
      {/* Validation panel */}
      <ValidationPanel
        validation={dataset.validation}
        isLoading={dataset.isLoading}
        onValidate={handleValidate}
      />

      {/* Sub-view tabs */}
      <div className="flex border-b border-zinc-800 px-4">
        {datasetSubViews.map(view => (
          <button
            key={view}
            onClick={() => setSubView(view)}
            className={`px-3 py-2 text-xs font-medium capitalize transition-colors border-b-2 ${
              subView === view
                ? 'border-blue-500 text-blue-400'
                : 'border-transparent text-zinc-500 hover:text-zinc-300'
            }`}
          >
            {view}
          </button>
        ))}
        <div className="flex-1" />
        {dataset.clips.length > 0 && (
          <span className="text-xs text-zinc-500 self-center mr-3">
            {dataset.clips.length} clip{dataset.clips.length !== 1 ? 's' : ''}
          </span>
        )}
        <button
          onClick={handleClearDataset}
          disabled={dataset.isLoading || dataset.clips.length === 0}
          className="px-2 py-1 text-xs text-red-400 hover:text-red-300 hover:bg-red-900/20 rounded transition-colors disabled:opacity-30 self-center mr-1"
          title="Delete all imported clips"
        >
          Clear Clips
        </button>
        <button
          onClick={handleRemoveDataset}
          disabled={dataset.isLoading}
          className="px-2 py-1 text-xs text-red-400 hover:text-red-300 hover:bg-red-900/20 rounded transition-colors disabled:opacity-30 self-center"
          title="Unlink dataset directory from project"
        >
          Remove Dataset
        </button>
      </div>

      {/* Import progress overlay */}
      {dataset.importProgress && (
        <div className="px-4 py-3 bg-zinc-800/90 border-b border-zinc-700 flex items-center gap-4">
          <div className="flex-1">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs text-zinc-300 font-medium">
                Importing files... {dataset.importProgress.current} / {dataset.importProgress.total}
              </span>
              <span className="text-xs text-zinc-500">
                {Math.round((dataset.importProgress.current / dataset.importProgress.total) * 100)}%
              </span>
            </div>
            <div className="w-full bg-zinc-700 rounded-full h-1.5">
              <div
                className="bg-blue-500 h-1.5 rounded-full transition-all duration-200"
                style={{ width: `${(dataset.importProgress.current / dataset.importProgress.total) * 100}%` }}
              />
            </div>
            <p className="text-xs text-zinc-500 mt-1 truncate">
              {dataset.importProgress.currentFile}
            </p>
          </div>
          <button
            onClick={dataset.cancelImport}
            className="px-3 py-1.5 text-xs text-red-400 hover:text-red-300 bg-red-900/20 hover:bg-red-900/40 border border-red-800/50 rounded transition-colors shrink-0"
          >
            Cancel
          </button>
        </div>
      )}

      {/* Error banner */}
      {dataset.error && (
        <div className="px-4 py-2 bg-red-900/30 border-b border-red-800 text-red-400 text-xs flex items-center justify-between">
          <span>{dataset.error}</span>
          <button onClick={dataset.clearError} className="text-red-300 hover:text-red-100 ml-2">
            Dismiss
          </button>
        </div>
      )}

      {/* Sub-view content */}
      <div className="flex-1 overflow-auto min-h-0">
        {subView === 'sources' && (
          <SourceList
            sources={dataset.sources}
            clips={dataset.clips}
            isLoading={dataset.isLoading}
            datasetDir={datasetDir}
            onProbe={dataset.probeSource}
            onSourceSelected={handleSourceSelected}
            onImportImage={dataset.importImage}
            onCreateBatch={dataset.createClipsBatch}
            onClipsImported={handleSourceClipImported}
          />
        )}

        {subView === 'cutter' && activeSource && (
          <SceneProposalList
            source={activeSource}
            scenes={dataset.scenes}
            isLoading={dataset.isLoading}
            datasetDir={datasetDir}
            onDetectScenes={dataset.detectScenes}
            onCreateClip={dataset.createClip}
            onCreateBatch={dataset.createClipsBatch}
            onDone={handleClipsCreated}
            onClearScenes={dataset.clearScenes}
          />
        )}

        {subView === 'browser' && (
          <ClipGrid
            clips={dataset.clips}
            datasetDir={datasetDir}
            isLoading={dataset.isLoading}
            onDeleteClip={dataset.deleteClip}
            onUpdateCaption={dataset.updateCaption}
          />
        )}

        {subView === 'cutter' && !activeSource && (
          <div className="flex-1 flex items-center justify-center h-full">
            <div className="text-center">
              <p className="text-sm text-zinc-500">
                Select a video source from the Sources tab to cut clips.
              </p>
              <p className="text-xs text-zinc-600 mt-2">
                The Cutter is for video files only. Images are imported directly into the dataset.
              </p>
            </div>
          </div>
        )}

        {subView === 'caption' && (
          <>
            <TriggerWordPanel
              trigger={activeProject.trigger}
              datasetDir={datasetDir}
              clipCount={dataset.clips.length}
              triggerPresentCount={triggerPresentCount}
              captionedCount={captionedCount}
              onTriggerChange={handleTriggerChange}
              onValidateTrigger={dataset.validateTrigger}
              onPrependTrigger={dataset.prependTrigger}
              onRefresh={handleRefresh}
            />
            <p className="px-4 text-xs text-zinc-600 -mt-2 mb-2">
              Apply the trigger word after captions have been generated. It will be prepended to each caption.
            </p>
            <CaptionPanel
              clips={dataset.clips}
              datasetDir={datasetDir}
              onCaptionsUpdated={() => void dataset.listClips(datasetDir)}
            />
          </>
        )}
      </div>
    </div>
  )
}
