"""Training worker subprocess package.

The training worker runs as a separate Python process, not inside the
FastAPI server. The supervisor service spawns it and communicates via
filesystem-based IPC (control.json, progress.jsonl, job.json).
"""
