# macOS Platform Stubs

This document catalogs all macOS-specific code paths that are currently stubbed out. The application is developed Linux-first; macOS support will be restored in a future release.

## Electron Layer

### `electron/python-setup.ts`
- macOS Python detection uses `which python3`
- MAC-STUB: Homebrew Python paths may differ from system Python
- MAC-STUB: Apple Silicon (arm64) vs Intel (x64) detection for correct pip packages

### `electron/gpu.ts`
- GPU detection uses NVIDIA-specific APIs (pynvml)
- MAC-STUB: macOS has no NVIDIA GPU support; MPS (Metal Performance Shaders) would need separate detection
- MAC-STUB: `checkGpu` will always return `available: false` on macOS

### `electron/main.ts`
- macOS auto-updater uses DMG target
- MAC-STUB: `electron-builder.yml` references macOS config but is untested with new product name

### `resources/`
- `resources/entitlements.mac.plist` exists for code signing
- MAC-STUB: Not verified against current app capabilities

## Backend Layer

### GPU/CUDA Dependencies
- All training and verification pipelines assume CUDA GPU availability
- MAC-STUB: PyTorch MPS backend could theoretically work for inference but is untested
- MAC-STUB: Training on macOS is not supported (no CUDA, MPS training is experimental)

### `backend/services/caption_pipeline/captioner_worker.py`
- Loads models with `device_map="auto"` which defaults to CUDA
- MAC-STUB: Would need explicit `device="mps"` for Apple Silicon

### `backend/services/gpu_info/`
- Uses pynvml for GPU memory and utilization queries
- MAC-STUB: No equivalent for MPS; would need `subprocess` calls to `system_profiler`

## Frontend Layer

No macOS-specific stubs in the frontend layer.

## Resolution Plan

macOS support is lower priority than Linux and Windows due to GPU limitations:
1. Captioning (inference only) could work on Apple Silicon via MPS
2. Training requires CUDA and will not be supported on macOS
3. Verification generation requires CUDA and will not be supported on macOS
4. The app could run in "remote-only" mode on macOS (API captioning, no local training)
