"""Sample generation during training.

At configured sample steps, the worker pauses training briefly,
runs a forward generation pass with the project's sample prompts,
and writes MP4 files to the samples/ directory.
"""

from __future__ import annotations

from pathlib import Path


def samples_dir(job_dir: Path) -> Path:
    """Return the samples directory for a job, creating it if needed.

    Resolved through ``artifacts.artifacts_root`` so previews land in the
    user's dataset folder when the supervisor points the job there, and
    under the job directory otherwise.
    """
    from training_worker.engine.artifacts import artifacts_root

    d = artifacts_root(job_dir) / "samples"
    d.mkdir(parents=True, exist_ok=True)
    return d


def sample_path(job_dir: Path, step: int) -> Path:
    """Return the path for a sample video at a given step."""
    return samples_dir(job_dir) / f"step_{step:06d}.mp4"


def list_samples(job_dir: Path) -> list[dict[str, str | int]]:
    """List all sample files with their step numbers.

    Sample files are written by ``sample_generation.generate_samples``
    as ``step_<NNNNNN>_prompt_<NN>.mp4`` (one per sample spec in a
    cycle). Older runs may have produced the legacy ``step_<NNNNNN>.mp4``
    name. Both layouts are accepted here: we parse the run of digits that
    immediately follows the ``step_`` prefix and ignore the rest of the
    stem, so the per-prompt suffix no longer causes every file to be
    discarded on an ``int()`` failure.

    Reads from the resolved artifacts root so the Monitor finds previews
    whether they were written under the job directory or in the user's
    dataset folder.
    """
    from training_worker.engine.artifacts import artifacts_root

    sdir = artifacts_root(job_dir) / "samples"
    if not sdir.exists():
        return []

    results: list[dict[str, str | int]] = []
    for f in sorted(sdir.glob("step_*.mp4")):
        step = _parse_step(f.stem)
        if step is None:
            continue
        results.append({"step": step, "path": str(f)})
    return results


def _parse_step(stem: str) -> int | None:
    """Extract the step number from a sample filename stem.

    Accepts both ``step_000100`` and ``step_000100_prompt_00``. Returns
    ``None`` when the stem does not start with ``step_`` followed by at
    least one digit.
    """
    if not stem.startswith("step_"):
        return None
    rest = stem[len("step_") :]
    digits = ""
    for ch in rest:
        if ch.isdigit():
            digits += ch
        else:
            break
    if not digits:
        return None
    return int(digits)

