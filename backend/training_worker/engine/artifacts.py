"""Resolve where a job's checkpoints and samples are written.

By default a job's artifacts live under its job directory
(``job_dir/checkpoints`` and ``job_dir/samples``). That keeps unit
tests and direct CLI runs self-contained.

The desktop app, however, wants the trained LoRA checkpoints and the
preview samples to land in the user's dataset folder, sitting alongside
the ``clips/`` and ``images/`` the user curated, so they are easy to
find after training. The supervisor records that destination by writing
a small pointer file (``artifacts_root.json``) into the job directory at
start time. Both the writers (the training loop) and the readers (the
supervisor's list endpoints) call ``artifacts_root`` here, so they always
agree on a single location without threading a new path argument through
every function.

Resolution rules:
    * If ``job_dir/artifacts_root.json`` exists and names a usable path,
      return that path (the dataset output folder).
    * Otherwise return ``job_dir`` itself (the historical behaviour).

We deliberately use a pointer FILE rather than a symlink because this
app ships on Windows, where creating symlinks needs elevated privileges.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Pointer file name written into the job directory by the supervisor.
ARTIFACTS_POINTER_FILE = "artifacts_root.json"


def artifacts_root(job_dir: Path) -> Path:
    """Return the base directory for this job's checkpoints and samples.

    Reads the optional ``artifacts_root.json`` pointer in ``job_dir``.
    Falls back to ``job_dir`` when the pointer is missing or unreadable,
    so callers never have to special-case the no-pointer path.
    """
    pointer = job_dir / ARTIFACTS_POINTER_FILE
    if pointer.exists():
        try:
            data = json.loads(pointer.read_text(encoding="utf-8"))
            root = data.get("artifacts_root")
            if isinstance(root, str) and root.strip():
                return Path(root)
        except (json.JSONDecodeError, OSError, AttributeError):
            # A corrupt pointer must never crash training or listing; we
            # just fall back to the job directory.
            logger.warning(
                "Unreadable artifacts pointer at %s; using job_dir for artifacts.",
                pointer,
            )
    return job_dir


def write_artifacts_root(job_dir: Path, artifacts_root_path: Path) -> None:
    """Record the external artifacts root for ``job_dir``.

    Called by the supervisor at job start. After this, every
    ``artifacts_root(job_dir)`` call (writer or reader) resolves to
    ``artifacts_root_path``.
    """
    pointer = job_dir / ARTIFACTS_POINTER_FILE
    pointer.write_text(
        json.dumps({"artifacts_root": str(artifacts_root_path)}),
        encoding="utf-8",
    )
