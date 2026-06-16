"""Control file reader for worker subprocess.

The supervisor writes commands to control.json in the job directory.
The worker reads this file at every step boundary to check for
pause, resume, or cancel commands.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal


ControlCommand = Literal["run", "pause", "cancel"]


def read_control(job_dir: Path) -> ControlCommand:
    """Read the current control command from control.json.

    Returns "run" if the file does not exist or cannot be parsed.
    """
    control_path = job_dir / "control.json"
    if not control_path.exists():
        return "run"
    try:
        data = json.loads(control_path.read_text())
        command = data.get("command", "run")
        if command in ("run", "pause", "cancel"):
            return command  # type: ignore[return-value]
        return "run"
    except (json.JSONDecodeError, OSError):
        return "run"


def write_control(job_dir: Path, command: ControlCommand) -> None:
    """Write a control command to control.json."""
    control_path = job_dir / "control.json"
    control_path.write_text(json.dumps({"command": command}))
