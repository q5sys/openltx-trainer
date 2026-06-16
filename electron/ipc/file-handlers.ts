import { dialog } from 'electron'
import path from 'path'
import fs from 'fs'
import { getAllowedRoots } from '../config'
import { logger } from '../logger'
import { getMainWindow } from '../window'
import { validatePath, approvePath } from '../path-validation'
import { getProjectAssetsPath, setProjectAssetsPath } from '../app-state'
import { handle } from './typed-handle'

const MIME_TYPES: Record<string, string> = {
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.gif': 'image/gif',
  '.mp3': 'audio/mpeg',
  '.wav': 'audio/wav',
  '.ogg': 'audio/ogg',
  '.aac': 'audio/aac',
  '.flac': 'audio/flac',
  '.m4a': 'audio/mp4',
  '.mp4': 'video/mp4',
  '.webm': 'video/webm',
  '.mkv': 'video/x-matroska',
  '.mov': 'video/quicktime',
}

function readLocalFileAsBase64(filePath: string): { data: string; mimeType: string } {
  const data = fs.readFileSync(filePath)
  const base64 = data.toString('base64')
  const ext = path.extname(filePath).toLowerCase()
  const mimeType = MIME_TYPES[ext] || 'application/octet-stream'
  return { data: base64, mimeType }
}

function searchDirectoryForFilesImpl(dir: string, filenames: string[]): Record<string, string> {
  const results: Record<string, string> = {}
  const remaining = new Set(filenames.map(f => f.toLowerCase()))

  const walk = (currentDir: string, depth: number) => {
    if (remaining.size === 0 || depth > 10) return
    try {
      const entries = fs.readdirSync(currentDir, { withFileTypes: true })
      for (const entry of entries) {
        if (remaining.size === 0) break
        const fullPath = path.join(currentDir, entry.name)
        if (entry.isFile()) {
          const lower = entry.name.toLowerCase()
          if (remaining.has(lower)) {
            results[lower] = fullPath
            remaining.delete(lower)
          }
        } else if (entry.isDirectory() && !entry.name.startsWith('.')) {
          walk(fullPath, depth + 1)
        }
      }
    } catch {
      // Skip directories we can't read (permissions, etc.)
    }
  }

  walk(dir, 0)
  return results
}

export function registerFileHandlers(): void {
  handle('openLtxApiKeyPage', async () => {
    const { shell } = await import('electron')
    await shell.openExternal('https://console.ltx.video/api-keys/')
    return true
  })

  handle('openLtxBillingPage', async () => {
    const { shell } = await import('electron')
    await shell.openExternal('https://console.ltx.video/billings/#buy')
    return true
  })

  handle('openFalApiKeyPage', async () => {
    const { shell } = await import('electron')
    await shell.openExternal('https://fal.ai/dashboard/keys')
    return true
  })

  handle('openHuggingFaceRepo', async ({ repoId }) => {
    const { shell } = await import('electron')
    await shell.openExternal(`https://huggingface.co/${repoId}`)
    return true
  })

  const HF_AUTHORIZE_URL = 'https://huggingface.co/oauth/authorize'

  handle('openHuggingFaceAuth', async (params) => {
    const { shell } = await import('electron')
    const url = new URL(HF_AUTHORIZE_URL)
    url.searchParams.set('client_id', params.clientId)
    url.searchParams.set('redirect_uri', params.redirectUri)
    url.searchParams.set('response_type', 'code')
    url.searchParams.set('scope', params.scope)
    url.searchParams.set('state', params.state)
    url.searchParams.set('code_challenge', params.codeChallenge)
    url.searchParams.set('code_challenge_method', params.codeChallengeMethod)
    await shell.openExternal(url.toString())
    return true
  })

  handle('openParentFolderOfFile', async ({ filePath }) => {
    const { shell } = await import('electron')
    const normalizedPath = validatePath(filePath, getAllowedRoots())
    const parentDir = path.dirname(normalizedPath)
    if (!fs.existsSync(parentDir) || !fs.statSync(parentDir).isDirectory()) {
      throw new Error(`Parent directory not found: ${parentDir}`)
    }
    shell.openPath(parentDir)
  })

  handle('showItemInFolder', async ({ filePath }) => {
    const { shell } = await import('electron')
    shell.showItemInFolder(filePath)
  })

  handle('readLocalFile', async ({ filePath }) => {
    try {
      const normalizedPath = validatePath(filePath, getAllowedRoots())

      if (!fs.existsSync(normalizedPath)) {
        throw new Error(`File not found: ${normalizedPath}`)
      }

      return readLocalFileAsBase64(normalizedPath)
    } catch (error) {
      logger.error( `Error reading local file: ${error}`)
      throw error
    }
  })

  handle('showSaveDialog', async ({ title, defaultPath, filters }) => {
    const mainWindow = getMainWindow()
    if (!mainWindow) return null
    const result = await dialog.showSaveDialog(mainWindow, {
      title: title || 'Save File',
      defaultPath,
      filters: filters || [],
    })
    if (result.canceled || !result.filePath) return null
    approvePath(result.filePath)
    return result.filePath
  })

  handle('saveFile', async ({ filePath, data, encoding }) => {
    try {
      validatePath(filePath, getAllowedRoots())
      if (encoding === 'base64') {
        fs.writeFileSync(filePath, Buffer.from(data, 'base64'))
      } else {
        fs.writeFileSync(filePath, data, 'utf-8')
      }
      return { success: true, path: filePath }
    } catch (error) {
      logger.error( `Error saving file: ${error}`)
      return { success: false, error: String(error) }
    }
  })

  handle('saveBinaryFile', async ({ filePath, data }) => {
    try {
      validatePath(filePath, getAllowedRoots())
      fs.writeFileSync(filePath, Buffer.from(data))
      return { success: true, path: filePath }
    } catch (error) {
      logger.error( `Error saving binary file: ${error}`)
      return { success: false, error: String(error) }
    }
  })

  handle('showOpenDirectoryDialog', async ({ title }) => {
    const mainWindow = getMainWindow()
    if (!mainWindow) return null
    const result = await dialog.showOpenDialog(mainWindow, {
      title: title || 'Select Folder',
      properties: ['openDirectory', 'createDirectory'],
    })
    if (result.canceled || result.filePaths.length === 0) return null
    approvePath(result.filePaths[0])
    return result.filePaths[0]
  })

  handle('searchDirectoryForFiles', ({ directory, filenames }) => {
    return searchDirectoryForFilesImpl(directory, filenames)
  })

  handle('getProjectAssetsPath', () => {
    return getProjectAssetsPath()
  })

  handle('openProjectAssetsPathChangeDialog', async () => {
    try {
      const mainWindow = getMainWindow()
      if (!mainWindow) return { success: false, error: 'No window' }
      const result = await dialog.showOpenDialog(mainWindow, {
        title: 'Select Project Assets Path',
        properties: ['openDirectory', 'createDirectory'],
      })
      if (result.canceled || result.filePaths.length === 0) return { success: false, error: 'cancelled' }
      const selectedPath = path.resolve(result.filePaths[0])
      setProjectAssetsPath(selectedPath)
      approvePath(selectedPath)
      return { success: true, path: selectedPath }
    } catch (error) {
      return { success: false, error: String(error) }
    }
  })

  handle('checkFilesExist', ({ filePaths }) => {
    const results: Record<string, boolean> = {}
    for (const p of filePaths) {
      try {
        results[p] = fs.existsSync(p)
      } catch {
        results[p] = false
      }
    }
    return results
  })

  handle('showOpenFileDialog', async ({ title, filters, properties }) => {
    const mainWindow = getMainWindow()
    if (!mainWindow) return null
    const props: any[] = ['openFile']
    if (properties?.includes('multiSelections')) props.push('multiSelections')
    const result = await dialog.showOpenDialog(mainWindow, {
      title: title || 'Select File',
      filters: filters || [],
      properties: props,
    })
    if (result.canceled || result.filePaths.length === 0) return null
    for (const fp of result.filePaths) {
      approvePath(fp)
    }
    return result.filePaths
  })

  // Dataset source import (video/image files with multiSelections)
  handle('addDatasetSource', async ({ title, filters }) => {
    const mainWindow = getMainWindow()
    if (!mainWindow) return null
    const result = await dialog.showOpenDialog(mainWindow, {
      title: title || 'Add Dataset Sources',
      filters: filters || [
        { name: 'Video Files', extensions: ['mp4', 'mov', 'avi', 'mkv', 'webm'] },
        { name: 'Image Files', extensions: ['jpg', 'jpeg', 'png', 'webp', 'bmp'] },
        { name: 'All Files', extensions: ['*'] },
      ],
      properties: ['openFile', 'multiSelections'],
    })
    if (result.canceled || result.filePaths.length === 0) return null
    for (const fp of result.filePaths) {
      approvePath(fp)
    }
    return result.filePaths
  })

  // LORA export: save dialog for LORA checkpoint
  handle('showSaveLoraDialog', async ({ defaultName }) => {
    const mainWindow = getMainWindow()
    if (!mainWindow) return null
    const result = await dialog.showSaveDialog(mainWindow, {
      title: 'Export LORA',
      defaultPath: defaultName || 'lora.safetensors',
      filters: [
        { name: 'SafeTensors', extensions: ['safetensors'] },
      ],
    })
    if (result.canceled || !result.filePath) return null
    approvePath(result.filePath)
    return result.filePath
  })

  // Reveal a checkpoint file in the system file manager
  handle('revealCheckpoint', async ({ filePath }) => {
    const { shell } = await import('electron')
    shell.showItemInFolder(filePath)
  })

}
