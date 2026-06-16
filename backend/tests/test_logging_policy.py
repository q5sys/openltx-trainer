"""Logging policy tests for traceback ownership and duplication prevention."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Event

from services.task_runner.threading_runner import ThreadingRunner


def _policy_records(caplog, *, contains: str) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == "logging_policy" and contains in record.getMessage()]


def test_http_500_logs_single_traceback(caplog, test_state, monkeypatch) -> None:
    caplog.set_level(logging.WARNING)

    def _boom() -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(test_state.health, "get_health", _boom)

    from app_factory import create_app
    from starlette.testclient import TestClient

    app = create_app(handler=test_state)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/health")

    assert response.status_code == 500
    records = _policy_records(caplog, contains="Unhandled error on GET /health")
    assert len(records) == 1
    assert records[0].exc_info is not None


def test_http_400_logs_without_traceback(caplog, client) -> None:
    caplog.set_level(logging.WARNING)

    response = client.get("/api/models/download/progress", params={"sessionId": "nonexistent"})

    assert response.status_code == 404
    records = _policy_records(caplog, contains="HTTP error on GET /api/models/download/progress: [404]")
    assert len(records) == 1
    assert records[0].exc_info is None


def test_unhandled_exception_logs_single_traceback(caplog, test_state, monkeypatch) -> None:
    caplog.set_level(logging.ERROR)

    def _boom():
        raise RuntimeError("unhandled-boom")

    monkeypatch.setattr(test_state.runtime_policy, "get_runtime_policy", _boom)

    from app_factory import create_app
    from starlette.testclient import TestClient

    app = create_app(handler=test_state)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/runtime-policy")

    assert response.status_code == 500
    records = _policy_records(caplog, contains="Unhandled error on GET /api/runtime-policy")
    assert len(records) == 1
    assert records[0].exc_info is not None


def test_background_task_error_does_not_log_at_boundary(caplog) -> None:
    caplog.set_level(logging.ERROR)
    runner = ThreadingRunner()
    done = Event()

    errors: list[Exception] = []

    def _fail():
        raise RuntimeError("bg-boom")

    def _on_error(exc: Exception) -> None:
        errors.append(exc)
        done.set()

    runner.run_background(_fail, task_name="test-task", on_error=_on_error)
    done.wait(timeout=2)

    assert len(errors) == 1
    policy_records = _policy_records(caplog, contains="bg-boom")
    assert len(policy_records) == 0
