import { z } from 'zod'

const fileFilter = z.object({ name: z.string(), extensions: z.array(z.string()) })

function ipcResult<T extends z.ZodRawShape>(valueShape: T) {
  return z.discriminatedUnion('success', [
    z.object({ success: z.literal(true), ...valueShape }),
    z.object({ success: z.literal(false), error: z.string() }),
  ])
}

export type IpcResult<T extends z.ZodRawShape> = z.infer<ReturnType<typeof ipcResult<T>>>

const logsResponse = z.object({
  logPath: z.string(),
  lines: z.array(z.string()),
  error: z.string().optional(),
})

const backendHealthStatus = z.object({
  status: z.enum(['alive', 'restarting', 'dead']),
  exitCode: z.number().nullable().optional(),
})

export type BackendHealthStatus = z.infer<typeof backendHealthStatus>

export const electronAPISchemas = {
  // App info
  getBackend: {
    input: z.object({}),
    output: z.object({ url: z.string(), token: z.string(), adminToken: z.string() }),
  },
  getModelsPath: {
    input: z.object({}),
    output: z.string(),
  },
  readLocalFile: {
    input: z.object({ filePath: z.string() }),
    output: z.object({ data: z.string(), mimeType: z.string() }),
  },
  checkGpu: {
    input: z.object({}),
    output: z.object({ available: z.boolean(), name: z.string().optional(), vram: z.number().optional() }),
  },
  getAppInfo: {
    input: z.object({}),
    output: z.object({ version: z.string(), isPackaged: z.boolean(), modelsPath: z.string(), userDataPath: z.string() }),
  },

  // First-run setup
  checkFirstRun: {
    input: z.object({}),
    output: z.object({ needsSetup: z.boolean(), needsLicense: z.boolean() }),
  },
  acceptLicense: {
    input: z.object({}),
    output: z.boolean(),
  },
  completeSetup: {
    input: z.object({}),
    output: z.boolean(),
  },
  fetchLicenseText: {
    input: z.object({}),
    output: z.string(),
  },
  getNoticesText: {
    input: z.object({}),
    output: z.string(),
  },

  // Open external pages / folders
  openLtxApiKeyPage: {
    input: z.object({}),
    output: z.boolean(),
  },
  openLtxBillingPage: {
    input: z.object({}),
    output: z.boolean(),
  },
  openFalApiKeyPage: {
    input: z.object({}),
    output: z.boolean(),
  },
  openHuggingFaceRepo: {
    input: z.object({ repoId: z.string() }),
    output: z.boolean(),
  },
  openHuggingFaceAuth: {
    input: z.object({
      clientId: z.string(),
      redirectUri: z.string(),
      scope: z.string(),
      state: z.string(),
      codeChallenge: z.string(),
      codeChallengeMethod: z.string(),
    }),
    output: z.boolean(),
  },
  openParentFolderOfFile: {
    input: z.object({ filePath: z.string() }),
    output: z.void(),
  },
  showItemInFolder: {
    input: z.object({ filePath: z.string() }),
    output: z.void(),
  },

  // Logs
  getLogs: {
    input: z.object({}),
    output: logsResponse,
  },
  getLogPath: {
    input: z.object({}),
    output: z.object({ logPath: z.string(), logDir: z.string() }),
  },
  openLogFolder: {
    input: z.object({}),
    output: z.boolean(),
  },

  // Paths
  getResourcePath: {
    input: z.object({}),
    output: z.string().nullable(),
  },
  getDownloadsPath: {
    input: z.object({}),
    output: z.string(),
  },

  getProjectAssetsPath: {
    input: z.object({}),
    output: z.string(),
  },
  openProjectAssetsPathChangeDialog: {
    input: z.object({}),
    output: ipcResult({ path: z.string() }),
  },

  // File dialogs & save
  showSaveDialog: {
    input: z.object({
      title: z.string().optional(),
      defaultPath: z.string().optional(),
      filters: z.array(fileFilter).optional(),
    }),
    output: z.string().nullable(),
  },
  saveFile: {
    input: z.object({ filePath: z.string(), data: z.string(), encoding: z.string().optional() }),
    output: ipcResult({ path: z.string() }),
  },
  saveBinaryFile: {
    input: z.object({ filePath: z.string(), data: z.instanceof(ArrayBuffer) }),
    output: ipcResult({ path: z.string() }),
  },
  showOpenDirectoryDialog: {
    input: z.object({ title: z.string().optional() }),
    output: z.string().nullable(),
  },
  searchDirectoryForFiles: {
    input: z.object({ directory: z.string(), filenames: z.array(z.string()) }),
    output: z.record(z.string(), z.string()),
  },
  checkFilesExist: {
    input: z.object({ filePaths: z.array(z.string()) }),
    output: z.record(z.string(), z.boolean()),
  },
  showOpenFileDialog: {
    input: z.object({
      title: z.string().optional(),
      filters: z.array(fileFilter).optional(),
      properties: z.array(z.string()).optional(),
    }),
    output: z.array(z.string()).nullable(),
  },

  // Dataset source import
  addDatasetSource: {
    input: z.object({
      title: z.string().optional(),
      filters: z.array(fileFilter).optional(),
    }),
    output: z.array(z.string()).nullable(),
  },

  // LORA export helpers
  showSaveLoraDialog: {
    input: z.object({
      defaultName: z.string().optional(),
    }),
    output: z.string().nullable(),
  },
  revealCheckpoint: {
    input: z.object({ filePath: z.string() }),
    output: z.void(),
  },

  // Python setup
  checkPythonReady: {
    input: z.object({}),
    output: z.object({ ready: z.boolean() }),
  },
  startPythonSetup: {
    input: z.object({}),
    output: z.void(),
  },
  startPythonBackend: {
    input: z.object({}),
    output: z.void(),
  },
  getBackendHealthStatus: {
    input: z.object({}),
    output: backendHealthStatus.nullable(),
  },

  // Logging
  writeLog: {
    input: z.object({ level: z.string(), message: z.string() }),
    output: z.void(),
  },

  // Models
  openModelsDirChangeDialog: {
    input: z.object({}),
    output: ipcResult({ path: z.string() }),
  },

  // Analytics
  getAnalyticsState: {
    input: z.object({}),
    output: z.object({ analyticsEnabled: z.boolean(), installationId: z.string() }),
  },
  setAnalyticsEnabled: {
    input: z.object({ enabled: z.boolean() }),
    output: z.void(),
  },
  sendAnalyticsEvent: {
    input: z.object({ eventName: z.string(), extraDetails: z.record(z.string(), z.unknown()).nullable().optional() }),
    output: z.void(),
  },
} as const

type Schemas = typeof electronAPISchemas

type InvokeAPI = {
  [K in keyof Schemas]: z.infer<Schemas[K]['input']> extends Record<string, never>
    ? () => Promise<z.infer<Schemas[K]['output']>>
    : (input: z.infer<Schemas[K]['input']>) => Promise<z.infer<Schemas[K]['output']>>
}

export type ElectronAPI = InvokeAPI & {
  onPythonSetupProgress: (cb: (data: unknown) => void) => void
  removePythonSetupProgress: () => void
  onBackendHealthStatus: (cb: (data: BackendHealthStatus) => void) => (() => void)
  getPathForFile: (file: File) => string
  platform: string
  hfGatingEnabled: boolean
}
