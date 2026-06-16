"""Unit tests for the Gemma3 ``use_cache`` recursive override.

The reason this matters
-----------------------
Stage E was OOMing on a 96 GiB GPU during caption precache because
Gemma3's ``use_cache=True`` plus ``cache_implementation="hybrid"``
default causes a fresh ``HybridCache`` sized for the model's full
``max_position_embeddings`` (131072 tokens for Gemma3-12B) to be
allocated on every forward pass. That alone reserves roughly 48 GiB
of KV cache, on top of the model's own 24 GiB of BF16 weights.

``training_worker.engine.text_encoding._set_use_cache_false_recursive``
walks a module tree and flips every nested HF ``config.use_cache``
attribute to ``False`` before the encode pass runs. These tests verify
the walker visits nested modules, ignores non-HF objects, and tolerates
modules whose ``children`` method raises.

These tests use plain Python objects with the same duck-typed surface
the walker expects (``config.use_cache`` attribute, ``children()``
generator). No HuggingFace, no mocks, no GPU.
"""

from __future__ import annotations

from typing import Any

from training_worker.engine.text_encoding import _set_use_cache_false_recursive


class FakeConfig:
    """Stand-in for ``transformers.PretrainedConfig`` with just ``use_cache``."""

    def __init__(self, use_cache: bool) -> None:
        self.use_cache = use_cache


class FakeModule:
    """Stand-in for ``torch.nn.Module`` that exposes ``config`` + ``children()``.

    Mirrors the attribute surface ``_set_use_cache_false_recursive``
    relies on: a possibly-None ``config`` attribute, and a ``children``
    callable that returns an iterable of child modules.
    """

    def __init__(
        self,
        config: FakeConfig | None = None,
        children: list[Any] | None = None,
    ) -> None:
        self.config: FakeConfig | None = config
        self._children: list[Any] = children or []

    def children(self) -> list[Any]:
        return list(self._children)


class FakeModuleWithBrokenChildren:
    """A module whose ``children()`` raises. The walker must not crash."""

    def __init__(self, config: FakeConfig | None) -> None:
        self.config = config

    def children(self) -> list[Any]:
        raise RuntimeError("children() blew up")


def test_flips_root_module_use_cache_to_false() -> None:
    """The walker flips ``use_cache=True`` on the root module."""
    root = FakeModule(config=FakeConfig(use_cache=True))
    _set_use_cache_false_recursive(root)
    assert root.config is not None
    assert root.config.use_cache is False


def test_flips_nested_modules_use_cache_to_false() -> None:
    """Children, grandchildren, and great-grandchildren all get flipped."""
    leaf = FakeModule(config=FakeConfig(use_cache=True))
    mid = FakeModule(config=FakeConfig(use_cache=True), children=[leaf])
    root = FakeModule(config=FakeConfig(use_cache=True), children=[mid])
    _set_use_cache_false_recursive(root)
    assert root.config is not None and root.config.use_cache is False
    assert mid.config is not None and mid.config.use_cache is False
    assert leaf.config is not None and leaf.config.use_cache is False


def test_modules_without_config_are_skipped() -> None:
    """Modules with ``config=None`` are touched but not flipped."""
    nested_leaf = FakeModule(config=FakeConfig(use_cache=True))
    no_config_mid = FakeModule(config=None, children=[nested_leaf])
    root = FakeModule(config=None, children=[no_config_mid])
    _set_use_cache_false_recursive(root)
    assert nested_leaf.config is not None
    assert nested_leaf.config.use_cache is False


def test_module_with_config_missing_use_cache_attr_is_ignored() -> None:
    """A ``config`` object without ``use_cache`` does not raise."""

    class ConfigWithoutUseCache:
        pass

    weird = FakeModule()
    weird.config = ConfigWithoutUseCache()  # type: ignore[assignment]
    _set_use_cache_false_recursive(weird)


def test_broken_children_does_not_crash() -> None:
    """If ``children()`` raises, the walker skips that branch."""
    config = FakeConfig(use_cache=True)
    root = FakeModuleWithBrokenChildren(config=config)
    _set_use_cache_false_recursive(root)
    assert root.config is not None
    assert root.config.use_cache is False


def test_none_module_is_a_noop() -> None:
    """Passing ``None`` to the walker must not raise."""
    _set_use_cache_false_recursive(None)


def test_cycle_in_module_graph_terminates() -> None:
    """If two modules refer to each other, the walker still terminates."""
    leaf = FakeModule(config=FakeConfig(use_cache=True))
    root = FakeModule(config=FakeConfig(use_cache=True), children=[leaf])
    # introduce a back-edge: leaf points at root
    leaf._children = [root]
    _set_use_cache_false_recursive(root)
    assert root.config is not None and root.config.use_cache is False
    assert leaf.config is not None and leaf.config.use_cache is False
