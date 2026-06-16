"""Centralized logging policy for request and background exception paths."""

from __future__ import annotations

import logging
import re

from fastapi import Request

from _routes._errors import HTTPError

logger = logging.getLogger(__name__)

# Matches common API key patterns: Bearer tokens, sk-* keys, hex strings 32+ chars,
# HuggingFace tokens, and filesystem paths that may contain tokens.
_API_KEY_PATTERN = re.compile(
    r"""("""
    r"""(?:Bearer\s+)[A-Za-z0-9\-_\.]{8,}"""  # Bearer tokens
    r"""|sk-[A-Za-z0-9]{8,}"""  # OpenAI-style keys
    r"""|AIza[A-Za-z0-9\-_]{30,}"""  # Google API keys
    r"""|(?:key-)[A-Za-z0-9]{8,}"""  # Generic key- prefixed
    r"""|hf_[A-Za-z0-9]{8,}"""  # HuggingFace tokens
    r"""|(?:token=)[A-Za-z0-9\-_\.]{8,}"""  # URL query token params
    r""")""",
    re.VERBOSE,
)


def redact_api_keys(text: str) -> str:
    """Replace API key patterns with a redacted placeholder.

    Keeps the first 4 characters of each match for debugging, replaces the rest
    with '***REDACTED***'.
    """

    def _redact(match: re.Match[str]) -> str:
        value = match.group(0)
        if len(value) <= 8:
            return "***REDACTED***"
        return value[:4] + "***REDACTED***"

    return _API_KEY_PATTERN.sub(_redact, text)


def log_http_error(request: Request, exc: HTTPError) -> None:
    """Log typed HTTP errors with policy-based traceback behavior."""
    if 500 <= exc.status_code <= 599:
        logger.error(
            "HTTP error on %s %s: [%s] %s",
            request.method,
            request.url.path,
            exc.status_code,
            exc.detail,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return

    logger.warning(
        "HTTP error on %s %s: [%s] %s",
        request.method,
        request.url.path,
        exc.status_code,
        exc.detail,
    )


def log_unhandled_exception(request: Request, exc: Exception) -> None:
    """Log unhandled request exceptions with full traceback."""
    logger.error(
        "Unhandled error on %s %s",
        request.method,
        request.url.path,
        exc_info=(type(exc), exc, exc.__traceback__),
    )


def log_background_exception(task_name: str, exc: Exception) -> None:
    """Log unhandled background task exceptions with full traceback."""
    logger.error(
        "Unhandled background error in task '%s'",
        task_name,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
