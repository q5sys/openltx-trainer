import './app-paths'

// Prevent EPIPE crashes when stdout/stderr pipes are broken (e.g., terminal
// closed while Electron is still running). Without this, any console.log call
// from third-party code (updater, etc.) would throw an uncaught exception.
for (const stream of [process.stdout, process.stderr]) {
  stream?.on('error', (err: NodeJS.ErrnoException) => {
    if (err.code === 'EPIPE') return
    throw err
  })
}

import { app } from 'electron'
import { setupCSP } from './csp'
import { registerAppHandlers } from './ipc/app-handlers'
import { registerFileHandlers } from './ipc/file-handlers'
import { registerLogHandlers } from './ipc/log-handlers'
import { logger } from './logger'
import { initSessionLog } from './logging-management'
import { stopPythonBackend } from './python-backend'
import { initAutoUpdater } from './updater'
import { createWindow, getMainWindow } from './window'
import { sendAnalyticsEvent } from './analytics'

function logAppVersion(): void {
  if (!app.isPackaged) {
    logger.info('[LTX Desktop] Running in development mode')
  } else {
    logger.info(`[LTX Desktop] Version ${app.getVersion()}`)
  }
}

const gotLock = app.requestSingleInstanceLock()

if (!gotLock) {
  app.quit()
} else {
  initSessionLog()
  logAppVersion()

  registerAppHandlers()
  registerFileHandlers()
  registerLogHandlers()

  app.on('second-instance', () => {
    const mainWindow = getMainWindow()
    if (mainWindow) {
      if (mainWindow.isMinimized()) {
        mainWindow.restore()
      }
      if (!mainWindow.isVisible()) {
        mainWindow.show()
      }
      mainWindow.focus()
      return
    }
    if (app.isReady()) {
      createWindow()
    }
  })

  app.whenReady().then(async () => {
    setupCSP()
    createWindow()
    initAutoUpdater()
    // Python setup + backend start are now driven by the renderer via IPC

    // Fire analytics event (no-op if user hasn't opted in)
    void sendAnalyticsEvent('ltxdesktop_app_launched')
  })

  app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
      stopPythonBackend()
      app.quit()
    }
  })

  app.on('activate', () => {
    if (getMainWindow() === null) {
      createWindow()
    }
  })

  app.on('before-quit', () => {
    stopPythonBackend()
  })
}
