import { Check, Info, Settings, X } from 'lucide-react'
import React, { useEffect, useState } from 'react'
import { Button } from './ui/button'
import { useAppSettings, type AppSettings } from '../contexts/AppSettingsContext'
import { useHfAuth } from '../hooks/use-hf-auth'
import { backendFetch } from '../lib/backend'

interface SettingsModalProps {
  isOpen: boolean
  onClose: () => void
  initialTab?: TabId
}

type TabId = 'general' | 'training' | 'captioning' | 'verification' | 'about'

export function SettingsModal({ isOpen, onClose, initialTab }: SettingsModalProps) {
  const { settings, updateSettings } = useAppSettings()
  const [activeTab, setActiveTab] = useState<TabId>('general')
  const { hfAuthStatus, hfAuthPolling, startHuggingFaceLogin, handleHuggingFaceLogout } = useHfAuth(isOpen)
  const [appVersion, setAppVersion] = useState('')
  const [gpuDevices, setGpuDevices] = useState<{ index: number; name: string }[]>([])

  // API key input state
  const [geminiKeyInput, setGeminiKeyInput] = useState('')
  const [openaiKeyInput, setOpenaiKeyInput] = useState('')
  const [anthropicKeyInput, setAnthropicKeyInput] = useState('')
  const [openaiCompatUrlInput, setOpenaiCompatUrlInput] = useState('')
  const [openaiCompatKeyInput, setOpenaiCompatKeyInput] = useState('')

  useEffect(() => {
    if (isOpen && initialTab) {
      setActiveTab(initialTab)
    }
  }, [isOpen, initialTab])

  useEffect(() => {
    if (activeTab !== 'about' || appVersion) return
    window.electronAPI.getAppInfo().then(info => setAppVersion(info.version)).catch(() => {})
  }, [activeTab, appVersion])

  useEffect(() => {
    if (!isOpen) return
    backendFetch('/api/gpu-list')
      .then(r => r.json())
      .then((data: { devices: { index: number; name: string }[] }) => setGpuDevices(data.devices))
      .catch(() => {})
  }, [isOpen])

  if (!isOpen) return null

  const tabs = [
    { id: 'general' as TabId, label: 'General', icon: Settings },
    { id: 'training' as TabId, label: 'Training', icon: Settings },
    { id: 'captioning' as TabId, label: 'Captioning', icon: Settings },
    { id: 'verification' as TabId, label: 'Verification', icon: Settings },
    { id: 'about' as TabId, label: 'About', icon: Info },
  ]

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />

      <div className="relative bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl w-full max-w-2xl mx-4">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800">
          <div className="flex items-center gap-2">
            <Settings className="h-5 w-5 text-zinc-400" />
            <h2 className="text-lg font-semibold text-white">Settings</h2>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            className="h-8 w-8 text-zinc-400 hover:text-white hover:bg-zinc-800"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-zinc-800 overflow-x-auto">
          {tabs.map((tab) => {
            const Icon = tab.icon
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors whitespace-nowrap ${
                  activeTab === tab.id
                    ? 'text-white border-b-2 border-blue-500 -mb-px'
                    : 'text-zinc-400 hover:text-white'
                }`}
              >
                <Icon className="h-4 w-4" />
                {tab.label}
              </button>
            )
          })}
        </div>

        {/* Content */}
        <div className="px-6 py-5 space-y-6 h-[60vh] overflow-y-auto">

          {/* ===== GENERAL TAB ===== */}
          {activeTab === 'general' && (
            <>
              {/* GPU Selection */}
              <SettingsSection title="Default GPU" description="Select which GPU to use for training and inference.">
                <select
                  value={settings.defaultGpuIndex}
                  onChange={(e) => updateSettings({ defaultGpuIndex: parseInt(e.target.value) || 0 })}
                  className="px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {gpuDevices.length > 0 ? (
                    gpuDevices.map((gpu) => (
                      <option key={gpu.index} value={gpu.index}>
                        Device {gpu.index}: {gpu.name}
                      </option>
                    ))
                  ) : (
                    <option value={settings.defaultGpuIndex}>
                      Device {settings.defaultGpuIndex}
                    </option>
                  )}
                </select>
              </SettingsSection>

              {/* Model Directories */}
              <SettingsSection title="Model Directories" description="Where model files are stored. Use 'auto' for default locations.">
                <div className="space-y-3">
                  <DirField label="Base Models" value={settings.modelDirs.baseModels}
                    onChange={(v) => updateSettings({ modelDirs: { ...settings.modelDirs, baseModels: v } })} />
                  <DirField label="Captioner" value={settings.modelDirs.captioner}
                    onChange={(v) => updateSettings({ modelDirs: { ...settings.modelDirs, captioner: v } })} />
                  <DirField label="Trained LORAs" value={settings.modelDirs.trainedLoras}
                    onChange={(v) => updateSettings({ modelDirs: { ...settings.modelDirs, trainedLoras: v } })} />
                </div>
              </SettingsSection>

            </>
          )}

          {/* ===== TRAINING TAB ===== */}
          {activeTab === 'training' && (
            <>
              <ToggleSection
                title="Save Optimizer State"
                description="Save optimizer state with checkpoints for resumable training. Uses more disk space."
                enabled={settings.trainingDefaults.saveOptimizerState}
                onToggle={() => updateSettings({
                  trainingDefaults: { ...settings.trainingDefaults, saveOptimizerState: !settings.trainingDefaults.saveOptimizerState }
                })}
                color="blue"
              />

              <ToggleSection
                title="Sample on Save"
                description="Generate a sample video each time a checkpoint is saved."
                enabled={settings.trainingDefaults.sampleOnSave}
                onToggle={() => updateSettings({
                  trainingDefaults: { ...settings.trainingDefaults, sampleOnSave: !settings.trainingDefaults.sampleOnSave }
                })}
                color="blue"
              />

              <ToggleSection
                title="Auto-Advance Phases"
                description="Automatically move to the next training phase when the current one completes."
                enabled={settings.trainingDefaults.autoAdvancePhases}
                onToggle={() => updateSettings({
                  trainingDefaults: { ...settings.trainingDefaults, autoAdvancePhases: !settings.trainingDefaults.autoAdvancePhases }
                })}
                color="blue"
              />

              <SettingsSection title="Transformer Quantization" description="Quantization precision for the transformer during training.">
                <select
                  value={settings.trainingDefaults.transformerQuantization}
                  onChange={(e) => updateSettings({
                    trainingDefaults: { ...settings.trainingDefaults, transformerQuantization: e.target.value }
                  })}
                  className="px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="float8">Float 8 (default)</option>
                  <option value="4bit">4 bit</option>
                  <option value="2bit">2 bit</option>
                </select>
              </SettingsSection>

              <SettingsSection title="Text Encoder Quantization" description="Quantization precision for the text encoder during training.">
                <select
                  value={settings.trainingDefaults.textEncoderQuantization}
                  onChange={(e) => updateSettings({
                    trainingDefaults: { ...settings.trainingDefaults, textEncoderQuantization: e.target.value }
                  })}
                  className="px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="float8">Float 8 (default)</option>
                  <option value="4bit">4 bit</option>
                  <option value="2bit">2 bit</option>
                </select>
              </SettingsSection>

              <SettingsSection title="Keep Last N Checkpoints" description="Number of recent checkpoints to keep. 0 means keep all.">
                <input
                  type="number"
                  min="0"
                  max="100"
                  value={settings.trainingDefaults.keepLastNCheckpoints}
                  onChange={(e) => updateSettings({
                    trainingDefaults: { ...settings.trainingDefaults, keepLastNCheckpoints: Math.max(0, Math.min(100, parseInt(e.target.value) || 0)) }
                  })}
                  className="w-20 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white text-center focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </SettingsSection>
            </>
          )}

          {/* ===== CAPTIONING TAB ===== */}
          {activeTab === 'captioning' && (
            <>
              {/* --- Local Model Settings --- */}
              <SettingsSection title="Captioning Backend" description="Default captioning engine for new projects.">
                <select
                  value={settings.captioningDefaults.backend}
                  onChange={(e) => updateSettings({
                    captioningDefaults: { ...settings.captioningDefaults, backend: e.target.value }
                  })}
                  className="px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="qwen_vl_local">Qwen VL (Local)</option>
                  <option value="gemini_api">Gemini API</option>
                  <option value="openai_api">OpenAI API</option>
                  <option value="anthropic_api">Anthropic API</option>
                  <option value="openai_compatible">OpenAI Compatible</option>
                </select>
              </SettingsSection>

              <div className="space-y-3 pt-4 border-t border-zinc-800">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <label className="text-sm font-medium text-white">Model Family</label>
                    <p className="text-xs text-zinc-500 leading-relaxed mt-1">Model family for local captioning.</p>
                    <div className="flex items-center gap-3 mt-2">
                      <label className="inline-flex items-center gap-2 text-xs text-zinc-400 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={settings.captioningDefaults.abliterated}
                          onChange={() => updateSettings({
                            captioningDefaults: { ...settings.captioningDefaults, abliterated: !settings.captioningDefaults.abliterated }
                          })}
                          className="rounded border-zinc-600 bg-zinc-800 text-blue-500 focus:ring-blue-500 focus:ring-offset-0"
                        />
                        Abliterated (uncensored)
                      </label>
                    </div>
                  </div>
                  <input
                    type="text"
                    value={settings.captioningDefaults.modelFamily}
                    onChange={(e) => updateSettings({
                      captioningDefaults: { ...settings.captioningDefaults, modelFamily: e.target.value }
                    })}
                    className="w-40 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              </div>

              <SettingsSection title="Model Size" description="Model size variant for local captioning.">
                <select
                  value={settings.captioningDefaults.modelSize}
                  onChange={(e) => updateSettings({
                    captioningDefaults: { ...settings.captioningDefaults, modelSize: e.target.value }
                  })}
                  className="px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="2B">2B</option>
                  <option value="4B">4B</option>
                  <option value="8B">8B</option>
                  <option value="32B">32B</option>
                </select>
              </SettingsSection>

              <SettingsSection title="Quantization" description="Quantization level for local captioning model.">
                <select
                  value={settings.captioningDefaults.quantization}
                  onChange={(e) => updateSettings({
                    captioningDefaults: { ...settings.captioningDefaults, quantization: e.target.value }
                  })}
                  className="px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="fp16">FP16 (full precision)</option>
                  <option value="8bit">8-bit (moderate VRAM savings)</option>
                  <option value="4bit">4-bit (reduced VRAM)</option>
                </select>
              </SettingsSection>

              <SettingsSection title="Idle Timeout" description="Seconds before unloading the captioning model from memory.">
                <input
                  type="number"
                  min="0"
                  max="3600"
                  value={settings.captioningDefaults.captionerIdleTimeoutSeconds}
                  onChange={(e) => updateSettings({
                    captioningDefaults: { ...settings.captioningDefaults, captionerIdleTimeoutSeconds: Math.max(0, Math.min(3600, parseInt(e.target.value) || 300)) }
                  })}
                  className="w-24 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white text-center focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </SettingsSection>

              {/* --- Divider: API Configuration --- */}
              <div className="pt-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="h-px flex-1 bg-zinc-700" />
                  <span className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">API Configuration</span>
                  <div className="h-px flex-1 bg-zinc-700" />
                </div>

                <div className="space-y-4">
                  {/* Gemini fields */}
                  {settings.captioningDefaults.backend === 'gemini_api' && (
                    <ApiKeySection
                      title="Gemini API Key"
                      description=""
                      masked={settings.captioningApiKeys.gemini}
                      inputValue={geminiKeyInput}
                      onInputChange={setGeminiKeyInput}
                      onSave={() => {
                        updateSettings({ captioningApiKeys: { ...settings.captioningApiKeys, gemini: geminiKeyInput.trim() } })
                        setGeminiKeyInput('')
                      }}
                      onClear={() => updateSettings({ captioningApiKeys: { ...settings.captioningApiKeys, gemini: '' } })}
                      linkUrl="https://aistudio.google.com/app/apikey"
                      linkText="Get Gemini API key"
                    />
                  )}

                  {/* OpenAI fields */}
                  {settings.captioningDefaults.backend === 'openai_api' && (
                    <ApiKeySection
                      title="OpenAI API Key"
                      description=""
                      masked={settings.captioningApiKeys.openai}
                      inputValue={openaiKeyInput}
                      onInputChange={setOpenaiKeyInput}
                      onSave={() => {
                        updateSettings({ captioningApiKeys: { ...settings.captioningApiKeys, openai: openaiKeyInput.trim() } })
                        setOpenaiKeyInput('')
                      }}
                      onClear={() => updateSettings({ captioningApiKeys: { ...settings.captioningApiKeys, openai: '' } })}
                    />
                  )}

                  {/* Anthropic fields */}
                  {settings.captioningDefaults.backend === 'anthropic_api' && (
                    <ApiKeySection
                      title="Anthropic API Key"
                      description=""
                      masked={settings.captioningApiKeys.anthropic}
                      inputValue={anthropicKeyInput}
                      onInputChange={setAnthropicKeyInput}
                      onSave={() => {
                        updateSettings({ captioningApiKeys: { ...settings.captioningApiKeys, anthropic: anthropicKeyInput.trim() } })
                        setAnthropicKeyInput('')
                      }}
                      onClear={() => updateSettings({ captioningApiKeys: { ...settings.captioningApiKeys, anthropic: '' } })}
                    />
                  )}

                  {/* OpenAI Compatible fields */}
                  {settings.captioningDefaults.backend === 'openai_compatible' && (
                    <div className="space-y-3">
                      <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                        <input
                          type="text"
                          value={openaiCompatUrlInput || settings.captioningApiKeys.openaiCompatible.baseUrl}
                          onChange={(e) => setOpenaiCompatUrlInput(e.target.value)}
                          placeholder="Base URL (e.g. https://api.example.com/v1)"
                          onKeyDown={(e) => e.stopPropagation()}
                          className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                        <div className="flex gap-2">
                          <input
                            type="password"
                            value={openaiCompatKeyInput}
                            onChange={(e) => setOpenaiCompatKeyInput(e.target.value)}
                            placeholder={settings.captioningApiKeys.openaiCompatible.apiKey ? 'Enter new key to replace...' : 'API key...'}
                            onKeyDown={(e) => e.stopPropagation()}
                            className="flex-1 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
                          />
                          <button
                            onClick={() => {
                              updateSettings({
                                captioningApiKeys: {
                                  ...settings.captioningApiKeys,
                                  openaiCompatible: {
                                    baseUrl: openaiCompatUrlInput.trim() || settings.captioningApiKeys.openaiCompatible.baseUrl,
                                    apiKey: openaiCompatKeyInput.trim() || settings.captioningApiKeys.openaiCompatible.apiKey,
                                  }
                                }
                              })
                              setOpenaiCompatKeyInput('')
                            }}
                            disabled={!openaiCompatUrlInput.trim() && !openaiCompatKeyInput.trim()}
                            className="px-3 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-500 disabled:bg-zinc-700 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors"
                          >
                            Save
                          </button>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Local backend: no API key needed */}
                  {settings.captioningDefaults.backend === 'qwen_vl_local' && (
                    <p className="text-xs text-zinc-500 italic">No API key required for local captioning.</p>
                  )}

                  {/* HuggingFace Account */}
                  {window.electronAPI.hfGatingEnabled && (
                    <div className="space-y-4 pt-4 border-t border-zinc-800">
                      <h3 className="text-sm font-semibold text-white">HuggingFace</h3>
                      <p className="text-xs text-zinc-500">Sign in to HuggingFace to download model files.</p>
                      <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                        <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
                          hfAuthStatus === 'authenticated'
                            ? 'bg-green-500/10 text-green-400'
                            : 'bg-amber-500/10 text-amber-400'
                        }`}>
                          {hfAuthStatus === 'authenticated' ? (
                            <><Check className="h-3 w-3" /> Signed in</>
                          ) : (
                            <>Not signed in</>
                          )}
                        </div>
                        {hfAuthStatus === 'authenticated' ? (
                          <button
                            onClick={handleHuggingFaceLogout}
                            className="px-3 py-2 bg-zinc-700 text-white text-sm rounded-lg hover:bg-zinc-600 transition-colors"
                          >
                            Sign out
                          </button>
                        ) : (
                          <button
                            onClick={startHuggingFaceLogin}
                            disabled={hfAuthPolling}
                            className="px-3 py-2 bg-indigo-600 text-white text-sm rounded-lg hover:bg-indigo-500 disabled:bg-zinc-700 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors"
                          >
                            {hfAuthPolling ? 'Waiting for sign in...' : 'Sign in with HuggingFace'}
                          </button>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </>
          )}

          {/* ===== VERIFICATION TAB ===== */}
          {activeTab === 'verification' && (
            <>
              <SettingsSection title="Default CFG Scale" description="Classifier-free guidance scale for verification generations.">
                <input
                  type="number"
                  min="1"
                  max="30"
                  step="0.5"
                  value={settings.verificationDefaults.defaultCfg}
                  onChange={(e) => updateSettings({
                    verificationDefaults: { ...settings.verificationDefaults, defaultCfg: Math.max(1, Math.min(30, parseFloat(e.target.value) || 10)) }
                  })}
                  className="w-20 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white text-center focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </SettingsSection>

              <SettingsSection title="Default Frames" description="Number of frames for verification video generation.">
                <input
                  type="number"
                  min="1"
                  max="257"
                  value={settings.verificationDefaults.defaultFrames}
                  onChange={(e) => updateSettings({
                    verificationDefaults: { ...settings.verificationDefaults, defaultFrames: Math.max(1, Math.min(257, parseInt(e.target.value) || 49)) }
                  })}
                  className="w-20 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white text-center focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </SettingsSection>

              <SettingsSection title="Default Size" description="Width x Height for verification video generation.">
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    min="64"
                    max="1024"
                    step="64"
                    value={settings.verificationDefaults.defaultSize[0] ?? 512}
                    onChange={(e) => {
                      const w = Math.max(64, Math.min(1024, parseInt(e.target.value) || 512))
                      updateSettings({
                        verificationDefaults: { ...settings.verificationDefaults, defaultSize: [w, settings.verificationDefaults.defaultSize[1] ?? 512] }
                      })
                    }}
                    className="w-20 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white text-center focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  <span className="text-zinc-500 text-sm">x</span>
                  <input
                    type="number"
                    min="64"
                    max="1024"
                    step="64"
                    value={settings.verificationDefaults.defaultSize[1] ?? 512}
                    onChange={(e) => {
                      const h = Math.max(64, Math.min(1024, parseInt(e.target.value) || 512))
                      updateSettings({
                        verificationDefaults: { ...settings.verificationDefaults, defaultSize: [settings.verificationDefaults.defaultSize[0] ?? 512, h] }
                      })
                    }}
                    className="w-20 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white text-center focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              </SettingsSection>
            </>
          )}

          {/* ===== ABOUT TAB ===== */}
          {activeTab === 'about' && (
            <div className="space-y-6">
              <div className="text-center space-y-2">
                <h3 className="text-lg font-bold text-white">OpenLTX Trainer</h3>
                <p className="text-sm text-zinc-400">Version {appVersion || '...'}</p>
                <p className="text-xs text-zinc-500">LORA Training for LTX-Video 2.3</p>
              </div>

              <div className="bg-zinc-800/50 rounded-lg p-4 space-y-2">
                <div className="flex items-center gap-2">
                  <Info className="h-4 w-4 text-blue-400" />
                  <span className="text-sm font-medium text-white">License</span>
                </div>
                <p className="text-xs text-zinc-400">
                  Licensed under the Apache License, Version 2.0
                </p>
              </div>

              <p className="text-center text-xs text-zinc-600">
                Based on LTX Desktop by Lightricks
              </p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-zinc-800 flex justify-end">
          <Button
            onClick={onClose}
            className="bg-zinc-700 hover:bg-zinc-600 text-white"
          >
            Done
          </Button>
        </div>
      </div>
    </div>
  )
}

/* ===== Reusable sub-components ===== */

function SettingsSection({ title, description, children }: {
  title: string
  description: string
  children: React.ReactNode
}) {
  return (
    <div className="space-y-3 pt-4 border-t border-zinc-800 first:pt-0 first:border-t-0">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1">
          <label className="text-sm font-medium text-white">{title}</label>
          <p className="text-xs text-zinc-500 leading-relaxed mt-1">{description}</p>
        </div>
        {children}
      </div>
    </div>
  )
}

function ToggleSection({ title, description, enabled, onToggle, color }: {
  title: string
  description: string
  enabled: boolean
  onToggle: () => void
  color: 'blue' | 'orange' | 'violet' | 'emerald'
}) {
  const colorMap = {
    blue: { bg: 'bg-blue-500', indicator: 'bg-blue-400', indicatorBg: 'bg-blue-500/10', indicatorText: 'text-blue-400' },
    orange: { bg: 'bg-orange-500', indicator: 'bg-orange-400', indicatorBg: 'bg-orange-500/10', indicatorText: 'text-orange-400' },
    violet: { bg: 'bg-violet-500', indicator: 'bg-violet-400', indicatorBg: 'bg-violet-500/10', indicatorText: 'text-violet-400' },
    emerald: { bg: 'bg-emerald-500', indicator: 'bg-emerald-400', indicatorBg: 'bg-emerald-500/10', indicatorText: 'text-emerald-400' },
  }
  const colors = colorMap[color]

  return (
    <div className="space-y-3 pt-4 border-t border-zinc-800 first:pt-0 first:border-t-0">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1">
          <label className="text-sm font-medium text-white">{title}</label>
          <p className="text-xs text-zinc-500 leading-relaxed mt-1">{description}</p>
        </div>
        <button
          onClick={onToggle}
          className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
            enabled ? colors.bg : 'bg-zinc-700'
          }`}
        >
          <span
            className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
              enabled ? 'translate-x-5' : 'translate-x-0'
            }`}
          />
        </button>
      </div>
    </div>
  )
}

function DirField({ label, value, onChange }: {
  label: string
  value: string
  onChange: (value: string) => void
}) {
  const handleBrowse = async () => {
    const selected = await window.electronAPI.showOpenDirectoryDialog({ title: `Select ${label} Directory` })
    if (selected) {
      onChange(selected)
    }
  }

  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-zinc-400 w-28 flex-shrink-0">{label}</span>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="auto"
        onKeyDown={(e) => e.stopPropagation()}
        className="flex-1 px-3 py-2 rounded-lg bg-zinc-800 border border-zinc-700 text-zinc-300 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
      />
      <button
        onClick={handleBrowse}
        className="px-3 py-2 bg-zinc-700 text-zinc-300 text-xs rounded-lg hover:bg-zinc-600 transition-colors flex-shrink-0"
      >
        Browse
      </button>
    </div>
  )
}

function ApiKeySection({ title, description, masked, inputValue, onInputChange, onSave, onClear, linkUrl, linkText }: {
  title: string
  description: string
  masked: string
  inputValue: string
  onInputChange: (value: string) => void
  onSave: () => void
  onClear: () => void
  linkUrl?: string
  linkText?: string
}) {
  const hasKey = masked.length > 0

  return (
    <div className="space-y-3 pt-4 border-t border-zinc-800 first:pt-0 first:border-t-0">
      <h3 className="text-sm font-semibold text-white">{title}</h3>
      <p className="text-xs text-zinc-500">{description}</p>
      <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
        <div className="flex gap-2">
          <input
            type="password"
            value={inputValue}
            onChange={(e) => onInputChange(e.target.value)}
            placeholder={hasKey ? 'Enter new key to replace...' : 'Enter API key...'}
            onKeyDown={(e) => e.stopPropagation()}
            className="flex-1 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={onSave}
            disabled={!inputValue.trim()}
            className="px-3 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-500 disabled:bg-zinc-700 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors"
          >
            Save
          </button>
        </div>
        <div className="flex items-center justify-between">
          <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
            hasKey ? 'bg-green-500/10 text-green-400' : 'bg-zinc-800 text-zinc-500'
          }`}>
            {hasKey ? (
              <><Check className="h-3 w-3" /> Key configured ({masked})</>
            ) : (
              <>Not configured</>
            )}
          </div>
          {hasKey && (
            <button
              onClick={onClear}
              className="text-xs text-red-400 hover:text-red-300 transition-colors"
            >
              Clear
            </button>
          )}
        </div>
        {linkUrl && linkText && (
          <a
            href={linkUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-blue-400 hover:text-blue-300 transition-colors underline underline-offset-2"
          >
            {linkText} &rarr;
          </a>
        )}
      </div>
    </div>
  )
}

export type { AppSettings, TabId as SettingsTabId }
