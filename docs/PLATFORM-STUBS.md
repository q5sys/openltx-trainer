# Platform Stub Audit

**Date**: 2026-05-28

## Overview

OpenLTX Trainer is a Linux-first application. Windows and macOS code paths inherited from the upstream LTX Desktop project are preserved but not actively tested or maintained.

## Current Status

As of Step 15 of the refactor, a grep audit of the codebase found:

- **WIN-STUB markers**: 0 found
- **MAC-STUB markers**: 0 found

No platform-specific code paths have been explicitly stubbed out. The inherited platform code from LTX Desktop remains intact in the Electron layer (e.g., `electron/main.ts`, `electron/python-setup.ts`, `electron/window.ts`, build scripts).

## Platform-Specific Code Locations

The following files contain platform-conditional logic inherited from LTX Desktop:

| File | What it does |
|---|---|
| `electron/main.ts` | macOS dock behavior, Windows taskbar |
| `electron/python-setup.ts` | Platform-specific Python discovery and venv paths |
| `electron/python-backend.ts` | Platform-specific process spawning |
| `electron/app-paths.ts` | Platform-specific app data directories |
| `scripts/prepare-python.sh` | Linux/macOS Python setup |
| `scripts/prepare-python.ps1` | Windows Python setup |
| `scripts/local-build.sh` | Linux/macOS build |
| `scripts/local-build.ps1` | Windows build |
| `electron-builder.yml` | Multi-platform installer config |

## Expectations

- **Linux**: Fully supported. Primary development and testing platform.
- **Windows**: Build scripts exist. Electron shell should work. Training worker and CUDA paths are untested. Community contributions welcome.
- **macOS**: Build scripts exist. MPS (Apple Silicon GPU) is not supported for LTX-Video training. The app may run for dataset preparation and captioning tasks only.

## Adding a Stub

If you encounter a platform-specific code path that does not work on Windows or macOS, mark it with a comment:

```typescript
// WIN-STUB: <description of what needs to change for Windows>
```

```typescript
// MAC-STUB: <description of what needs to change for macOS>
```

Then add an entry to this document describing the stub location and what functionality it affects.
