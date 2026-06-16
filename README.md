# OpenLTX Trainer

A desktop application for training LORA models for LTX-Video 2.3, built on a fork of [LTX Desktop](https://github.com/Lightricks/LTX-Desktop). Linux-first, with NVIDIA CUDA GPUs (8GB+ VRAM). This is a community project, not affiliated with Lightricks.

> **Status: Alpha.** Core pipeline is implemented end-to-end. Real GPU training loop is a placeholder pending integration with a training backend (ai-toolkit or musubi-tuner). The fake training mode works for full UI testing.

## What This App Does

OpenLTX Trainer provides a complete local pipeline for fine-tuning LTX-Video 2.3 models using LORA:

1. **Dataset Preparation** - Import source videos, scene-detect and cut clips, organize into a training dataset
2. **Captioning** - Auto-caption clips using local Qwen3-VL (4B model, ~9GB download) or remote VLM APIs, with manual editing and mode-specific prompt templates
3. **Training** - Configure and run LORA training with character and concept presets, multi-phase workflows, advanced parameter editing, and cost/time estimates
4. **Monitoring** - Real-time loss charts, phase transition banners, sample preview strips, and log tailing
5. **Verification** - Generate test videos with trained LORAs to evaluate quality before export

## Requirements

- Linux (primary target)
- NVIDIA GPU with 8GB+ VRAM (16GB+ recommended)
- CUDA toolkit
- Node.js 20+, pnpm
- Python 3.13+ (managed via `uv`)

Windows and macOS support is not actively maintained. See [docs/PLATFORM-STUBS.md](docs/PLATFORM-STUBS.md) for details.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/q5sys/LTX-Desktop-LORA-Trainer.git
cd LTX-Desktop-LORA-Trainer

# Install dependencies and set up dev environment
pnpm install
pnpm setup:dev

# Start the development server
pnpm dev
```

## Development Commands

| Command | Purpose |
|---|---|
| `pnpm dev` | Start dev server (Vite + Electron + Python backend) |
| `pnpm typecheck` | Run TypeScript and Python type checks |
| `pnpm backend:test` | Run Python backend tests |
| `pnpm build` | Full platform build |

## Architecture

Three-layer architecture inherited from LTX Desktop:

- **Frontend**: React 18 + TypeScript + Tailwind CSS
- **Electron**: Main process managing app lifecycle, IPC, Python backend
- **Backend**: Python FastAPI server handling ML model orchestration

See `memory-bank/refactor-plans/` for detailed design documents.

## Troubleshooting

### Backend fails to start
Ensure Python 3.13+ is available and `uv` is installed. Run `pnpm setup:dev` to initialize the Python environment. Check that port 8000 is not already in use.

### "Module not found" errors in backend
Run `cd backend && uv sync` to install Python dependencies.

### TypeScript compilation errors
Run `pnpm typecheck:ts` to see detailed errors. Ensure you have run `pnpm install` after pulling.

### GPU not detected
Verify NVIDIA drivers and CUDA toolkit are installed: `nvidia-smi` should show your GPU. The app requires CUDA-capable NVIDIA GPUs.

### Captioning model download is slow
Qwen3-VL models are downloaded from HuggingFace on first use. The 4B model is approximately 8GB. Ensure you have a stable internet connection and sufficient disk space in your HuggingFace cache directory.

### Training runs out of memory
Reduce batch size in the training preset, or use a smaller resolution. 24GB VRAM is recommended for comfortable training. 8GB VRAM may work with aggressive memory optimization settings.

## License

Apache-2.0. See [LICENSE.txt](LICENSE.txt).

This project is a fork of [LTX Desktop](https://github.com/Lightricks/LTX-Desktop) by Lightricks. See [NOTICES.md](NOTICES.md) for attribution.
