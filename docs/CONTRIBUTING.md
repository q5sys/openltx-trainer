# Contributing to OpenLTX Trainer

Thanks for taking the time to contribute!

## Getting started (development)

Prereqs:

- Node.js 20+
- `uv` (Python package manager)
- Python 3.13+
- Git
- Linux with NVIDIA GPU (for full functionality)

Setup:

```bash
pnpm install
pnpm setup:dev
```

Run:

```bash
pnpm dev
```

Debug:

```bash
pnpm dev:debug
```

Typecheck:

```bash
pnpm typecheck
```

## Project Architecture

Three-layer architecture:

- **Frontend** (`frontend/`): React 18 + TypeScript + Tailwind CSS
- **Electron** (`electron/`): Main process managing app lifecycle, IPC, Python backend
- **Backend** (`backend/`): Python FastAPI server (port 8000)

Backend request flow: `_routes/* (thin) -> AppHandler -> handlers/* (logic) -> services/* (side effects) + state/* (mutations)`

See `memory-bank/refactor-plans/` for detailed design documents.

## What we accept right now

- Bug fixes and small improvements
- Documentation updates
- Small, targeted UI fixes
- Backend service implementations (with matching fake for tests)
- New training presets (place in `backend/training_worker/presets/`)
- Training engine module implementations (CUDA training loop, sampler, checkpoint logic)
- Captioning pipeline improvements (new model support, prompt template tuning)

**Frontend policy:** the frontend is under active refactor. The 4-tab project view (Dataset, Training, Monitor, Verify) is stable. Avoid large UI/state rewrites. Open an issue first so we can align on the target direction.

## Training Modes

The app supports two training modes with distinct behavior:

- **Character**: trigger word required in captions, 20-200 clip range, person-focused captioning prompts
- **Concept**: trigger word optional, 10-500 clip range, style/aesthetic-focused captioning prompts

When adding validation rules or captioning templates, ensure they diverge correctly per mode. See `backend/handlers/dataset_validation_handler.py` for the mode-specific constants.

## Proposing larger work

Before starting a larger change, please open an issue with:

- The problem you are trying to solve
- The proposed approach (1-2 paragraphs is fine)
- Scope (areas/files likely to change)
- Any UX or compatibility impact

Wait for maintainer alignment before investing in a major refactor.

## Code Standards

### Backend (Python)

- Pyright strict mode is enforced (`backend/pyrightconfig.json`)
- No `unittest.mock` in tests. Use `ServiceBundle` fakes only.
- Fakes live in `tests/fakes/`
- Integration-first testing with Starlette `TestClient`
- See `backend/architecture.md` for detailed patterns

### Frontend (TypeScript)

- Strict mode with `noUnusedLocals`, `noUnusedParameters`
- State management via React contexts only (no Redux/Zustand)
- Backend calls must use `backendFetch` from `frontend/lib/backend.ts`
- Styling with Tailwind + `class-variance-authority` + `clsx` + `tailwind-merge`

## Checks

At minimum, run:

- Type checking:

```bash
pnpm typecheck
```

- Backend tests:

```bash
pnpm backend:test
```

- Frontend build:

```bash
pnpm build:frontend
```
