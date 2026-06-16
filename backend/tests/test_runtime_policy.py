"""Tests for /api/runtime-policy endpoint."""

from __future__ import annotations


def test_runtime_policy_always_local(client):
    response = client.get("/api/runtime-policy")
    assert response.status_code == 200
    assert response.json() == {"force_api_generations": False}
