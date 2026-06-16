"""Dataset validation handler.

Pure-Python validation logic for dataset readiness checks.
No GPU required. Used by both the /api/dataset/validate endpoint
and the training pre-flight check.

Validation rules and captioning prompt templates diverge by training
mode (character vs concept).
"""

from __future__ import annotations

import re
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from handlers.base import StateHandlerBase
from services.dataset_pipeline.dataset_pipeline import ClipRecord, DatasetPipeline
from state.app_state_types import AppState

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig


# ---------------------------------------------------------------------------
# Training mode type
# ---------------------------------------------------------------------------

TrainingMode = Literal["character", "concept"]


# ---------------------------------------------------------------------------
# Trigger word rules
# ---------------------------------------------------------------------------

_TRIGGER_PATTERN = re.compile(r"^[A-Za-z0-9_]{2,32}$")

# Words that conflict with normal caption vocabulary.
TRIGGER_STOPLIST: set[str] = {
    "the", "a", "an", "video", "clip", "scene", "person",
    "woman", "man", "character", "object", "style", "art",
    "image", "photo", "picture", "film", "movie",
}

# Photography terms that are bad for video LORA captions.
PHOTOGRAPHY_TERMS: set[str] = {
    "photograph", "photographer", "photographic",
    "still", "picture", "snapshot", "portrait photo",
}


# ---------------------------------------------------------------------------
# Mode-specific validation thresholds
# ---------------------------------------------------------------------------

# Minimum clips recommended per mode.
MODE_MIN_CLIPS: dict[TrainingMode, int] = {
    "character": 20,
    "concept": 10,
}

# Maximum clips before warning per mode.
MODE_MAX_CLIPS: dict[TrainingMode, int] = {
    "character": 200,
    "concept": 500,
}

# Whether trigger word is required in captions for this mode.
MODE_TRIGGER_REQUIRED: dict[TrainingMode, bool] = {
    "character": True,
    "concept": False,
}


# ---------------------------------------------------------------------------
# Mode-specific captioning prompt templates
# ---------------------------------------------------------------------------

CAPTIONING_TEMPLATES: dict[TrainingMode, list[str]] = {
    "character": [
        "Describe the person in this video in detail. Focus on their physical appearance, "
        "clothing, posture, and actions. Begin the description with the word '{trigger}'.",
        "Write a detailed caption for this video clip of a specific person called '{trigger}'. "
        "Describe what they look like, what they are wearing, and what they are doing.",
    ],
    "concept": [
        "Describe the visual style, mood, and aesthetic of this video. Focus on colors, "
        "lighting, composition, and artistic technique.",
        "Write a detailed caption describing the visual concept shown in this video. "
        "Focus on the overall look and feel rather than specific objects or people.",
    ],
}


def get_captioning_templates(mode: TrainingMode, trigger: str | None = None) -> list[str]:
    """Return captioning prompt templates for the given mode.

    If a trigger word is provided, it replaces the {trigger} placeholder.
    """
    templates = CAPTIONING_TEMPLATES[mode]
    if trigger:
        return [t.replace("{trigger}", trigger) for t in templates]
    return list(templates)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ValidationIssue(BaseModel):
    """A single validation error or warning."""
    code: str
    msg: str
    clip_id: str | None = None


class DatasetStats(BaseModel):
    """Aggregate dataset statistics."""
    clip_count: int
    image_count: int
    captioned: int
    trigger_present: int
    with_audio: int
    without_audio: int
    total_duration_s: float


class TriggerValidationResult(BaseModel):
    """Result of validating a trigger word string."""
    valid: bool
    error: str | None = None
    warning: str | None = None


class DatasetValidationResult(BaseModel):
    """Full dataset validation result."""
    valid: bool
    errors: list[ValidationIssue]
    warnings: list[ValidationIssue]
    stats: DatasetStats


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class DatasetValidationHandler(StateHandlerBase):
    """Validates dataset readiness for training."""

    def __init__(
        self,
        state: AppState,
        lock: RLock,
        config: RuntimeConfig,
        dataset_pipeline: DatasetPipeline,
    ) -> None:
        super().__init__(state, lock, config)
        self._pipeline = dataset_pipeline

    def validate_trigger_word(self, trigger: str) -> TriggerValidationResult:
        """Validate a trigger word string against naming rules."""
        if not trigger:
            return TriggerValidationResult(valid=False, error="Trigger word cannot be empty.")

        if not _TRIGGER_PATTERN.match(trigger):
            return TriggerValidationResult(
                valid=False,
                error="Trigger must be 2-32 characters: ASCII letters, digits, or underscore.",
            )

        lower = trigger.lower()
        if lower in TRIGGER_STOPLIST:
            return TriggerValidationResult(
                valid=True,
                warning=f"'{trigger}' is a common word and may conflict with normal vocabulary.",
            )

        return TriggerValidationResult(valid=True)

    def validate_dataset(
        self,
        dataset_dir: str,
        trigger: str | None,
        mode: TrainingMode = "character",
    ) -> DatasetValidationResult:
        """Run all validation checks on a dataset directory.

        Validation rules diverge per mode:
        - character: trigger is required in every caption, min 20 clips, max 200
        - concept: trigger is optional, min 10 clips, max 500
        """
        clips = self._pipeline.list_clips(Path(dataset_dir))

        errors: list[ValidationIssue] = []
        warnings: list[ValidationIssue] = []

        min_clips = MODE_MIN_CLIPS[mode]
        max_clips = MODE_MAX_CLIPS[mode]
        trigger_required = MODE_TRIGGER_REQUIRED[mode]

        # Compute stats.
        clip_count = len(clips)
        image_count = sum(1 for c in clips if c.duration_s == 0.0)
        captioned = sum(1 for c in clips if c.caption.strip())
        trigger_present = 0
        with_audio = sum(1 for c in clips if c.has_audio)
        without_audio = clip_count - with_audio
        total_duration = sum(c.duration_s for c in clips)

        if trigger:
            trigger_present = sum(
                1 for c in clips if trigger in c.caption
            )

        stats = DatasetStats(
            clip_count=clip_count,
            image_count=image_count,
            captioned=captioned,
            trigger_present=trigger_present,
            with_audio=with_audio,
            without_audio=without_audio,
            total_duration_s=round(total_duration, 2),
        )

        # --- Errors (block training) ---

        if clip_count == 0:
            errors.append(ValidationIssue(
                code="NO_CLIPS",
                msg="Dataset has zero clips. Import source media first.",
            ))
            return DatasetValidationResult(
                valid=False, errors=errors, warnings=warnings, stats=stats,
            )

        # Missing captions.
        for clip in clips:
            if not clip.caption.strip():
                errors.append(ValidationIssue(
                    code="MISSING_CAPTION",
                    msg=f"Clip '{clip.filename}' has no caption.",
                    clip_id=clip.clip_id,
                ))

        # Missing trigger: error for character mode, warning for concept mode.
        if trigger:
            for clip in clips:
                if clip.caption.strip() and trigger not in clip.caption:
                    if trigger_required:
                        errors.append(ValidationIssue(
                            code="MISSING_TRIGGER",
                            msg=f"Clip '{clip.filename}' caption does not contain trigger '{trigger}'.",
                            clip_id=clip.clip_id,
                        ))
                    else:
                        warnings.append(ValidationIssue(
                            code="MISSING_TRIGGER",
                            msg=f"Clip '{clip.filename}' caption does not contain trigger '{trigger}' (optional for concept mode).",
                            clip_id=clip.clip_id,
                        ))

        # Inconsistent resolution.
        resolutions = {(c.width, c.height) for c in clips}
        if len(resolutions) > 1:
            res_str = ", ".join(f"{w}x{h}" for w, h in sorted(resolutions))
            errors.append(ValidationIssue(
                code="INCONSISTENT_RESOLUTION",
                msg=f"Clips have mixed resolutions: {res_str}. Normalize to a single resolution.",
            ))

        # --- Warnings (allow training but flag) ---

        if clip_count < min_clips:
            warnings.append(ValidationIssue(
                code="TINY_DATASET",
                msg=f"Only {clip_count} clips. Recommended minimum is {min_clips} for {mode} training.",
            ))

        if clip_count > max_clips:
            warnings.append(ValidationIssue(
                code="HUGE_DATASET",
                msg=f"{clip_count} clips is very large for {mode} training. Training will be slow.",
            ))

        # Per-clip warnings.
        for clip in clips:
            caption = clip.caption.strip()
            if not caption:
                continue

            if len(caption) > 500:
                warnings.append(ValidationIssue(
                    code="CAPTION_TOO_LONG",
                    msg=f"Clip '{clip.filename}' caption is {len(caption)} chars (max recommended: 500).",
                    clip_id=clip.clip_id,
                ))

            if len(caption) < 20:
                warnings.append(ValidationIssue(
                    code="CAPTION_TOO_SHORT",
                    msg=f"Clip '{clip.filename}' caption is only {len(caption)} chars.",
                    clip_id=clip.clip_id,
                ))

            if trigger and caption:
                # Trigger at end only.
                if caption.endswith(trigger) and not caption.startswith(trigger):
                    warnings.append(ValidationIssue(
                        code="TRIGGER_AT_END",
                        msg=f"Clip '{clip.filename}': trigger appears only at end of caption.",
                        clip_id=clip.clip_id,
                    ))

                # Trigger repeated.
                count = caption.count(trigger)
                if count > 1:
                    warnings.append(ValidationIssue(
                        code="TRIGGER_REPEATED",
                        msg=f"Clip '{clip.filename}': trigger appears {count} times.",
                        clip_id=clip.clip_id,
                    ))

            # Photography terms.
            caption_lower = caption.lower()
            for term in PHOTOGRAPHY_TERMS:
                if term in caption_lower:
                    warnings.append(ValidationIssue(
                        code="PHOTOGRAPHY_TERMS",
                        msg=f"Clip '{clip.filename}' caption contains '{term}' (bad for video LORA).",
                        clip_id=clip.clip_id,
                    ))
                    break

            # Video clip without audio.
            if clip.duration_s > 0.0 and not clip.has_audio:
                warnings.append(ValidationIssue(
                    code="VIDEO_CLIP_HAS_NO_AUDIO",
                    msg=f"Clip '{clip.filename}' is a video but has no audio track.",
                    clip_id=clip.clip_id,
                ))

        is_valid = len(errors) == 0

        return DatasetValidationResult(
            valid=is_valid,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )

    def audit_trigger_in_captions(
        self,
        dataset_dir: str,
        trigger: str,
    ) -> list[ClipRecord]:
        """Return clips whose captions do not contain the trigger word."""
        clips = self._pipeline.list_clips(Path(dataset_dir))
        return [c for c in clips if c.caption.strip() and trigger not in c.caption]

    def prepend_trigger_to_captions(
        self,
        dataset_dir: str,
        trigger: str,
        clip_ids: list[str] | None = None,
    ) -> int:
        """Prepend trigger word to captions that lack it.

        If clip_ids is None, applies to all clips missing the trigger.
        Returns the number of captions modified.
        """
        clips = self._pipeline.list_clips(Path(dataset_dir))
        ds_path = Path(dataset_dir)
        modified = 0

        for clip in clips:
            if clip_ids is not None and clip.clip_id not in clip_ids:
                continue

            caption = clip.caption.strip()
            if not caption:
                continue

            if trigger in caption:
                continue

            # Prepend the trigger as its own sentence ("trigger. caption")
            # so the token is isolated from the description that follows.
            # If the trigger already ends with sentence punctuation, do not
            # add a second period.
            if trigger.endswith((".", "!", "?")):
                new_caption = f"{trigger} {caption}"
            else:
                new_caption = f"{trigger}. {caption}"
            # Write to the caption file.
            for subdir in ("clips", "images"):
                caption_path = ds_path / subdir / f"{clip.clip_id}.txt"

                if caption_path.exists():
                    caption_path.write_text(new_caption)
                    modified += 1
                    break

        return modified
