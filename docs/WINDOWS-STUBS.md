# Windows Platform Stubs

This document catalogs all Windows-specific code paths that are currently stubbed out. The application is developed Linux-first; Windows support will be restored in a future release.

## Electron Layer

### `electron/python-setup.ts`
- Windows Python path detection uses `where python` instead of `which python3`
- Windows-specific `.venv\Scripts\python.exe` path
- WIN-STUB: `prepare-python.ps1` script exists but is not tested against new backend dependencies

### `electron/app-paths.ts`
- Windows uses `%APPDATA%` for settings storage
- WIN-STUB: Path separators handled via `path.join()` (should work cross-platform)

### `electron/main.ts`
- Windows auto-updater uses NSIS installer target
- WIN-STUB: `electron-builder.yml` still references Windows build config but is untested

### `scripts/`
- `scripts/prepare-python.ps1` and `scripts/setup-dev.ps1` exist for Windows
- WIN-STUB: Not updated for new training dependencies (peft, bitsandbytes, etc.)
- `scripts/create-installer.ps1` exists but targets the old product name

## Backend Layer

### `backend/services/caption_pipeline/captioner_worker.py`
- Subprocess spawning uses Unix-style process management
- WIN-STUB: May need `creationflags=subprocess.CREATE_NO_WINDOW` on Windows

### `backend/training_worker/ltx_train_worker.py`
- Process spawning and signal handling assumes Unix signals
- WIN-STUB: `SIGTERM` handling may need Windows-specific `CTRL_BREAK_EVENT`

### `backend/services/dataset_pipeline/dataset_pipeline_impl.py`
- ffmpeg invocation assumes `ffmpeg` is on PATH
- WIN-STUB: May need to bundle ffmpeg or use `ffmpeg.exe` path detection

## Frontend Layer

No Windows-specific stubs in the frontend layer. All platform branching is handled in the Electron main process.

## Resolution Plan

Each WIN-STUB should be resolved before the first Windows release:
1. Test Python setup scripts with new dependencies
2. Test subprocess spawning (caption worker, training worker) on Windows
3. Test ffmpeg availability and path detection
4. Update electron-builder.yml with correct product name and Windows targets
5. Run full test suite on Windows
