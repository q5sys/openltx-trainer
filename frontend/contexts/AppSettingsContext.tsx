import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { resetBackendCredentials } from '../lib/backend'
import { ApiClient } from '../lib/api-client'

export interface ModelDirs {
  baseModels: string
  captioner: string
  trainedLoras: string
}

export interface TrainingDefaults {
  saveOptimizerState: boolean
  keepLastNCheckpoints: number
  sampleOnSave: boolean
  autoAdvancePhases: boolean
  transformerQuantization: string
  textEncoderQuantization: string
}

export interface CaptioningDefaults {
  backend: string
  modelFamily: string
  modelSize: string
  abliterated: boolean
  quantization: string
  captionerIdleTimeoutSeconds: number
}

export interface CaptioningApiKeys {
  gemini: string
  openai: string
  anthropic: string
  openaiCompatible: {
    baseUrl: string
    apiKey: string
  }
}

export interface VerificationDefaults {
  defaultCfg: number
  defaultFrames: number
  defaultSize: number[]
}

export interface AppSettings {
  defaultGpuIndex: number
  modelDirs: ModelDirs
  trainingDefaults: TrainingDefaults
  captioningDefaults: CaptioningDefaults
  captioningApiKeys: CaptioningApiKeys
  verificationDefaults: VerificationDefaults
}

export const DEFAULT_APP_SETTINGS: AppSettings = {
  defaultGpuIndex: 0,
  modelDirs: {
    baseModels: 'auto',
    captioner: 'auto',
    trainedLoras: 'auto',
  },
  trainingDefaults: {
    saveOptimizerState: true,
    keepLastNCheckpoints: 0,
    sampleOnSave: true,
    autoAdvancePhases: true,
    transformerQuantization: 'float8',
    textEncoderQuantization: 'float8',
  },
  captioningDefaults: {
    backend: 'qwen_vl_local',
    modelFamily: 'qwen3-vl',
    modelSize: '4B',
    abliterated: false,
    quantization: 'fp16',
    captionerIdleTimeoutSeconds: 300,
  },
  captioningApiKeys: {
    gemini: '',
    openai: '',
    anthropic: '',
    openaiCompatible: {
      baseUrl: '',
      apiKey: '',
    },
  },
  verificationDefaults: {
    defaultCfg: 10.0,
    defaultFrames: 49,
    defaultSize: [512, 512],
  },
}

type BackendProcessStatus = 'alive' | 'restarting' | 'dead'

interface AppSettingsContextValue {
  settings: AppSettings
  isLoaded: boolean
  runtimePolicyLoaded: boolean
  updateSettings: (patch: Partial<AppSettings> | ((prev: AppSettings) => AppSettings)) => void
  refreshSettings: () => Promise<void>
}

const AppSettingsContext = createContext<AppSettingsContextValue | null>(null)

function toBackendProcessStatus(value: unknown): BackendProcessStatus | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const record = value as { status?: unknown }
  if (record.status === 'alive' || record.status === 'restarting' || record.status === 'dead') {
    return record.status
  }
  return null
}

function normalizeAppSettings(data: Partial<AppSettings>): AppSettings {
  return {
    defaultGpuIndex: data.defaultGpuIndex ?? DEFAULT_APP_SETTINGS.defaultGpuIndex,
    modelDirs: { ...DEFAULT_APP_SETTINGS.modelDirs, ...data.modelDirs },
    trainingDefaults: { ...DEFAULT_APP_SETTINGS.trainingDefaults, ...data.trainingDefaults },
    captioningDefaults: { ...DEFAULT_APP_SETTINGS.captioningDefaults, ...data.captioningDefaults },
    captioningApiKeys: {
      ...DEFAULT_APP_SETTINGS.captioningApiKeys,
      ...data.captioningApiKeys,
      openaiCompatible: {
        ...DEFAULT_APP_SETTINGS.captioningApiKeys.openaiCompatible,
        ...data.captioningApiKeys?.openaiCompatible,
      },
    },
    verificationDefaults: { ...DEFAULT_APP_SETTINGS.verificationDefaults, ...data.verificationDefaults },
  }
}


export function AppSettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_APP_SETTINGS)
  const [isLoaded, setIsLoaded] = useState(false)
  const [runtimePolicyLoaded, setRuntimePolicyLoaded] = useState(false)
  const [backendProcessStatus, setBackendProcessStatus] = useState<BackendProcessStatus | null>(null)

  useEffect(() => {
    if (backendProcessStatus !== 'alive') return

    let cancelled = false
    setRuntimePolicyLoaded(false)

    const fetchRuntimePolicy = async () => {
      const result = await ApiClient.getRuntimePolicy()
      if (!result.ok) {
        if (!cancelled) {
          setRuntimePolicyLoaded(true)
        }
        return
      }

      if (!cancelled) {
        setRuntimePolicyLoaded(true)
      }
    }

    void fetchRuntimePolicy()

    return () => {
      cancelled = true
    }
  }, [backendProcessStatus])

  useEffect(() => {
    let cancelled = false

    const applyStatus = (value: unknown) => {
      const nextStatus = toBackendProcessStatus(value)
      if (!nextStatus || cancelled) {
        return
      }
      if (nextStatus === 'alive') {
        resetBackendCredentials()
      }
      setBackendProcessStatus(nextStatus)
    }

    const unsubscribe = window.electronAPI.onBackendHealthStatus((data) => {
      applyStatus(data)
    })

    void window.electronAPI.getBackendHealthStatus()
      .then((snapshot) => {
        applyStatus(snapshot)
      })
      .catch(() => {
        // Snapshot is optional at startup; subscription continues to listen for pushes.
      })

    return () => {
      cancelled = true
      unsubscribe()
    }
  }, [])

  const refreshSettings = useCallback(async () => {
    const result = await ApiClient.getSettings()
    if (!result.ok) {
      throw new Error(result.error.message)
    }
    setSettings(normalizeAppSettings(result.data as unknown as Partial<AppSettings>))
    setIsLoaded(true)
  }, [])

  useEffect(() => {
    if (isLoaded || backendProcessStatus !== 'alive') return

    let cancelled = false
    let retryTimer: ReturnType<typeof setTimeout> | null = null

    const fetchSettings = async () => {
      try {
        await refreshSettings()
        if (cancelled) return
      } catch {
        if (!cancelled) {
          retryTimer = setTimeout(fetchSettings, 1000)
        }
      }
    }

    fetchSettings()

    return () => {
      cancelled = true
      if (retryTimer) clearTimeout(retryTimer)
    }
  }, [backendProcessStatus, isLoaded, refreshSettings])

  useEffect(() => {
    if (!isLoaded || backendProcessStatus !== 'alive') return
    const syncTimer = setTimeout(async () => {
      const result = await ApiClient.updateSettings(settings as never)
      if (!result.ok) {
        // Best-effort settings sync.
      }
    }, 150)
    return () => clearTimeout(syncTimer)
  }, [backendProcessStatus, isLoaded, settings])

  const updateSettings = useCallback((patch: Partial<AppSettings> | ((prev: AppSettings) => AppSettings)) => {
    if (typeof patch === 'function') {
      setSettings((prev) => patch(prev))
      return
    }
    setSettings((prev) => normalizeAppSettings({ ...prev, ...patch }))
  }, [])

  const contextValue = useMemo<AppSettingsContextValue>(
    () => ({
      settings,
      isLoaded,
      runtimePolicyLoaded,
      updateSettings,
      refreshSettings,
    }),
    [isLoaded, refreshSettings, runtimePolicyLoaded, settings, updateSettings],
  )

  return <AppSettingsContext.Provider value={contextValue}>{children}</AppSettingsContext.Provider>
}

export function useAppSettings() {
  const context = useContext(AppSettingsContext)
  if (!context) {
    throw new Error('useAppSettings must be used within AppSettingsProvider')
  }
  return context
}
