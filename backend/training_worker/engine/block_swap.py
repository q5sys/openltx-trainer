"""Sliding-window CPU offload of LTX-2 transformer blocks ("block swap").

Stage F technique #2 (per ``memory-bank/feature_real_training.md``).

LTX-Video 2.3 stacks N transformer blocks (48 in the default 22B
configuration). Keeping all N resident on the GPU costs ~44 GB at
BF16. On smaller cards we trade time for VRAM by keeping only a
sliding window of ``blocks_resident_on_gpu`` blocks on the GPU and
pre-staging the next one off CPU pinned memory while the current
block runs forward.

Design contract:

- The transformer's ``transformer_blocks`` is an ``nn.ModuleList``;
  we register a forward pre-hook and a forward post-hook on every
  block so the swap logic is invisible to the rest of the training
  loop. ``BlockSwapper.register()`` returns a context-manager-style
  ``BlockSwapHandle``; calling ``handle.release()`` (or letting it go
  out of scope via ``__del__``) restores every block to the original
  device so a subsequent eval / sample-generation pass starts from a
  clean state.
- We use pinned host memory + non-blocking copies so the swap-in
  overlaps with the currently-running block's matmul. Without
  pinning the swap-in stalls the CUDA stream and the throughput hit
  becomes 3-4x instead of the ~1.5x the plan budgets for.
- The window size ``K = blocks_resident_on_gpu`` is the number of
  blocks we keep on the GPU at any one time. ``K = 0`` means "no
  swap, leave everything resident"; ``K >= len(blocks)`` is the same
  thing. ``K = 1`` is the minimum and only works on PCIe 5.0 cards
  where the swap latency stays under the block's compute time.

We do NOT swap the LoRA-target submodules separately from the block
that contains them. peft hooks the Linear ``forward``; once the
block is on GPU the LoRA forwards just work. When the block moves
back to CPU peft's hook moves too.

Gradient checkpointing (Stage F technique #3) is enabled separately
via the model_loading bundle; it composes additively with block
swap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


@dataclass
class BlockSwapHandle:
    """Live handle returned by ``BlockSwapper.register``.

    Keeps a reference to the hooks so they can be removed at
    teardown, and to the original device so the blocks can be moved
    back into place when the user requests an eval / sample run.
    """

    transformer: Any
    blocks: Any
    target_device: "torch.device"
    cpu_device: "torch.device"
    window_size: int
    hook_handles: list[Any] = field(default_factory=list[Any])
    pinned_state_dicts: dict[int, dict[str, Any]] = field(default_factory=dict[int, dict[str, Any]])
    released: bool = False
    # Index of the single tail block currently swapped onto the GPU, or
    # None when no tail block is resident. The pre-hook evicts THIS block
    # before loading the next one (single-slot tail swap). Tracking the
    # previously-active block instead of computing an eviction index from
    # the traversal position is what makes the swapper TRAVERSAL-AGNOSTIC:
    # it bounds residency to head+1 during BOTH the forward walk (0..N-1)
    # and the gradient-checkpoint recompute during backward (N-1..0).
    active_swapped_index: int | None = None



    def release(self) -> None:
        """Remove hooks and restore the resident-window / CPU-tail split.

        Called at the end of training, or before a sample-generation
        pass that wants a clean forward without per-block swap
        overhead. Idempotent: subsequent calls are no-ops.

        Important: we do NOT move every block onto the GPU here. On a
        memory-constrained card the whole point of block swap is that
        the full block stack never fits in VRAM at once. Forcing all
        N blocks onto the GPU at teardown re-materialises the entire
        transformer and OOMs (this was the secondary failure observed
        in the Stage F 16 GB smoke run). Instead we restore the same
        placement the swapper installed at ``register`` time: the
        first ``window_size`` blocks on the GPU, the remaining blocks
        on CPU. The forward hooks are removed, so any later pass must
        either re-register block swap or run on a card that can hold
        the resident window plus whatever it streams in.

        An inert handle (window_size <= 0 or >= num_blocks) leaves the
        blocks exactly where they are; there is nothing to restore.
        """
        if self.released:
            return
        for h in self.hook_handles:
            try:
                h.remove()
            except Exception:  # noqa: BLE001 - best-effort teardown
                logger.exception("Failed to remove block-swap hook; continuing.")
        self.hook_handles.clear()

        import torch

        num_blocks = len(self.blocks)
        swap_was_active = 0 < self.window_size < num_blocks
        if swap_was_active:
            with torch.no_grad():
                for index, block in enumerate(self.blocks):
                    # Teardown must be best-effort. ``release()`` runs in
                    # ``phase_manager``'s ``finally`` block, so it fires
                    # even when training already failed with a CUDA OOM.
                    # In that case the allocator is full and moving a
                    # block onto the GPU raises a SECOND OutOfMemoryError;
                    # if we let it propagate it overwrites the real
                    # training error in ``job.json`` (the operator then
                    # sees a misleading 32 MiB teardown OOM instead of the
                    # true forward-pass OOM). Swallow per-block move
                    # failures so the original exception survives.
                    try:
                        if index < self.window_size:
                            _ensure_on_device(block, self.target_device)
                        else:
                            _ensure_on_cpu(block, self.cpu_device)
                    except Exception:  # noqa: BLE001 - best-effort teardown
                        logger.exception(
                            "Block-swap release: failed to restore placement "
                            "for block %d; continuing teardown.",
                            index,
                        )
        self.pinned_state_dicts.clear()
        self.released = True


class BlockSwapper:
    """Configurable per-transformer block swapper.

    Construct with the desired window size (``blocks_resident_on_gpu``)
    then call ``register(transformer)`` to install hooks. The hooks
    capture closures over ``self`` so the runtime state (which blocks
    are currently resident) lives on this object.
    """

    def __init__(
        self,
        window_size: int,
        target_device: "torch.device | None" = None,
    ) -> None:
        """Initialize a swapper with the given window size.

        Args:
            window_size: Number of transformer blocks to keep on the
                GPU at any one time. Must be >= 1 to call
                ``register``. ``window_size == 0`` is a no-op sentinel
                used by the configuration layer to mean "block swap
                disabled".
            target_device: GPU device that the resident-window blocks
                should live on. When ``None`` (the historical default),
                the swapper infers the device from the first block's
                parameters. Stage F's CPU-first load path leaves every
                block on CPU before block-swap registers, so the caller
                must pass an explicit ``target_device`` in that case.
        """
        if window_size < 0:
            raise ValueError(f"window_size must be >= 0, got {window_size}")
        self.window_size = window_size
        self.target_device = target_device

    def register(self, transformer: Any) -> BlockSwapHandle:
        """Install pre / post forward hooks on every transformer block.

        Reads ``transformer.transformer_blocks`` (the LTX-2 naming),
        moves all but the first ``window_size`` blocks to CPU pinned
        memory, and installs hooks that swap blocks in / out as the
        forward pass walks down the stack.

        Returns a ``BlockSwapHandle`` whose ``release()`` method must
        be called when the training loop exits, otherwise the hooks
        will fire on subsequent forward passes (e.g., sample
        generation) and corrupt the device placements.
        """
        import torch

        blocks = getattr(transformer, "transformer_blocks", None)
        if blocks is None:
            raise RuntimeError(
                "BlockSwapper.register: transformer has no 'transformer_blocks' "
                "attribute. Is this an LTX-2 LTXModel?"
            )
        num_blocks = len(blocks)
        # Resolve the target device. The constructor argument wins; if
        # absent, fall back to the legacy "look at block[0]" probe. The
        # probe is wrong on the Stage F CPU-first load path because
        # every block lives on CPU at this point; callers on that path
        # must therefore pass ``target_device`` explicitly.
        target_device = self.target_device or _device_of(blocks[0])

        if self.window_size <= 0 or self.window_size >= num_blocks:
            logger.info(
                "Block swap is a no-op: window_size=%d, num_blocks=%d. Returning inert handle.",
                self.window_size,
                num_blocks,
            )
            return BlockSwapHandle(
                transformer=transformer,
                blocks=blocks,
                target_device=target_device,
                cpu_device=torch.device("cpu"),
                window_size=self.window_size,
            )

        cpu_device = torch.device("cpu")

        # Move every block except the first ``window_size`` to CPU
        # pinned memory. Pinning is the difference between ~1.5x
        # slowdown and ~4x slowdown on PCIe 4.0 x16.
        with torch.no_grad():
            for index, block in enumerate(blocks):
                if index < self.window_size:
                    block.to(target_device)
                else:
                    _move_to_cpu_pinned(block)

        handle = BlockSwapHandle(
            transformer=transformer,
            blocks=blocks,
            target_device=target_device,
            cpu_device=cpu_device,
            window_size=self.window_size,
        )

        # Install ONLY a forward pre-hook per block. Eviction is driven
        # from the pre-hook of the NEXT swapped block (single-slot tail
        # swap), never from a post-hook. A post-hook would also fire
        # during the gradient-checkpoint RECOMPUTE in backward and would
        # evict the block before its own backward had run, stranding the
        # weights the gradient needs. Deferring eviction to the next
        # block's pre-hook guarantees the current block stays resident
        # through its entire forward AND backward.
        for index, block in enumerate(blocks):
            pre = block.register_forward_pre_hook(
                _make_pre_hook(handle=handle, block_index=index, num_blocks=num_blocks)
            )
            handle.hook_handles.append(pre)


        logger.info(
            "Block swap registered: %d blocks total, %d resident on %s.",
            num_blocks,
            self.window_size,
            target_device,
        )
        return handle


def install_block_swap(
    transformer: Any,
    blocks_resident_on_gpu: int,
    target_device: "torch.device | None" = None,
) -> BlockSwapHandle:
    """Convenience wrapper that constructs a ``BlockSwapper`` and registers it.

    The phase manager (see ``phase_manager.py``) calls this so the
    caller does not have to import both ``BlockSwapper`` and
    ``BlockSwapHandle``. Returns the live handle.

    Pass ``target_device`` explicitly when the transformer is still on
    CPU at registration time (the Stage F low-VRAM load order). The
    swapper otherwise infers the device from the first block's
    parameters and would incorrectly leave the resident window on CPU.
    """
    swapper = BlockSwapper(
        window_size=blocks_resident_on_gpu,
        target_device=target_device,
    )
    return swapper.register(transformer)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------



def _device_of(module: Any) -> "torch.device":
    """Return the device of the first parameter of ``module``."""
    import torch

    for parameter in module.parameters():
        return parameter.device
    return torch.device("cpu")


def _is_quantized_param(parameter: Any) -> bool:
    """Return True if ``parameter`` is a quantized tensor subclass.

    Detects bitsandbytes ``Params4bit`` / ``Int8Params`` (which keep a
    SEPARATE ``quant_state`` object holding absmax/code/blocksize) and
    torchao wrapper tensors (whose data class lives under the
    ``torchao`` module namespace). These tensors MUST be moved through
    their own ``.to()`` so the quantization metadata travels with the
    packed bytes.

    A blind ``parameter.data = parameter.data.to(device)`` moves only
    the packed blob and leaves the metadata on the old device; the next
    dequant kernel then reads a cross-device pointer and raises a CUDA
    illegal memory access. That was the Stage F block-swap crash
    (``feature_real_training.md`` Defect B). The class-name check also
    catches a bitsandbytes ``Params4bit`` that has not been packed yet
    (``quant_state is None`` until the first ``.to("cuda")``), so the
    first swap-in still routes through the correct subclass ``.to()``.
    """
    if getattr(parameter, "quant_state", None) is not None:
        return True
    if type(parameter).__name__ in ("Params4bit", "Int8Params"):
        return True
    data = getattr(parameter, "data", None)
    if data is not None and type(data).__module__.startswith("torchao"):
        return True
    return False


def _move_param(
    owner_module: Any,
    attr_name: str,
    parameter: Any,
    target_device: "torch.device",
    *,
    pin: bool = False,
    non_blocking: bool = False,
) -> None:
    """Move one parameter to ``target_device`` preserving quant metadata.

    For a quantized parameter we call the tensor subclass's own
    ``.to(device)`` (which moves the packed data AND its ``quant_state``
    / scale together) and rebind the resulting parameter on its owning
    module via ``setattr``. Rebinding the whole object is required
    because the quantized ``.to()`` returns a NEW tensor-subclass
    instance that cannot be smuggled in through ``.data``. The quantized
    base linear is frozen (``requires_grad=False``) and has no optimizer
    state, so replacing the parameter object is safe.

    For an ordinary float parameter we keep the historical
    ``parameter.data = ...`` assignment, which preserves the Parameter
    object's identity (so the optimizer's reference stays valid) and we
    optionally pin the CPU copy for async host-to-device transfers.
    """
    import torch

    if _is_quantized_param(parameter):
        moved = parameter.to(target_device, non_blocking=non_blocking)
        if not isinstance(moved, torch.nn.Parameter):
            moved = torch.nn.Parameter(moved, requires_grad=parameter.requires_grad)
        setattr(owner_module, attr_name, moved)
        return

    data = parameter.data.to(target_device, non_blocking=non_blocking)
    if pin and target_device.type == "cpu":
        try:
            data = data.pin_memory()
        except RuntimeError:
            # Pinning can fail under memory pressure. Fall back to plain
            # CPU; the swap still works, just slower.
            pass
    parameter.data = data


def _move_module(
    module: Any,
    target_device: "torch.device",
    *,
    skip_trainable: bool,
    pin: bool = False,
    non_blocking: bool = False,
) -> None:
    """Move a block's parameters and buffers to ``target_device``.

    Walks every submodule and moves its direct parameters and buffers
    using a quant-aware per-parameter mover (see ``_move_param``).

    ``skip_trainable`` leaves ``requires_grad=True`` parameters where
    they are. This matters because ``create_lora_adapter`` runs AFTER
    block swap is installed, so the trainable LoRA adapters live INSIDE
    the swapped blocks. They must NEVER be evicted to CPU: the 8-bit
    Adam optimizer holds references to them and builds its moment state
    on the GPU, so bouncing them to CPU would desync the optimizer.
    Both reference trainers (ai-toolkit, musubi-tuner) apply the same
    "skip trainable on offload" rule. Buffers are never quantized, so
    they always take the plain float move path.
    """
    import torch

    with torch.no_grad():
        for submodule in module.modules():
            for attr_name, parameter in list(submodule.named_parameters(recurse=False)):
                if skip_trainable and parameter.requires_grad:
                    continue
                if parameter.device == target_device:
                    continue
                _move_param(
                    submodule,
                    attr_name,
                    parameter,
                    target_device,
                    pin=pin,
                    non_blocking=non_blocking,
                )
            for _buffer_name, buffer in list(submodule.named_buffers(recurse=False)):
                if buffer.device == target_device:
                    continue
                data = buffer.data.to(target_device, non_blocking=non_blocking)
                if pin and target_device.type == "cpu":
                    try:
                        data = data.pin_memory()
                    except RuntimeError:
                        pass
                buffer.data = data


def verify_quant_aware_move(module: Any, target_device: "torch.device") -> None:
    """Round-trip a block CPU<->GPU and assert quant metadata stays colocated.

    Stage F sanity check (HANDOFF Task 1, step 2). Moves ``module`` to
    ``target_device``, back to CPU, then to ``target_device`` again, and
    asserts every quantized parameter's ``quant_state.absmax`` lives on
    the same device as the packed weight. A mismatch is exactly the
    cross-device pointer that produced the original CUDA illegal memory
    access. Raises ``AssertionError`` on mismatch. Intended for manual /
    smoke use; it is not called on the training hot path.
    """
    import torch

    cpu_device = torch.device("cpu")
    _move_module(module, target_device, skip_trainable=False)
    _move_module(module, cpu_device, skip_trainable=True)
    _move_module(module, target_device, skip_trainable=False)
    for _name, parameter in module.named_parameters():
        if not _is_quantized_param(parameter):
            continue
        quant_state = getattr(parameter, "quant_state", None)
        absmax = getattr(quant_state, "absmax", None)
        if absmax is not None:
            assert absmax.device == parameter.device, (
                f"quant_state.absmax on {absmax.device} but weight on "
                f"{parameter.device}; block-swap mover is not quant-aware"
            )


def _move_to_cpu_pinned(module: Any) -> None:
    """Move a block's frozen weights to CPU (pinned where possible).

    Called at ``register`` time on every tail block. Uses the quant-aware
    mover so bitsandbytes NF4 weights keep their ``quant_state``, and
    skips trainable params (LoRA adapters are created later but the guard
    is harmless and keeps the contract uniform).
    """
    import torch

    _move_module(
        module,
        torch.device("cpu"),
        skip_trainable=True,
        pin=True,
    )



def _make_pre_hook(
    handle: BlockSwapHandle,
    block_index: int,
    num_blocks: int,
) -> Any:
    """Build a forward pre-hook implementing single-slot tail swap.

    The first ``window_size`` blocks (the HEAD) stay permanently
    resident on the GPU and their pre-hook does nothing. Every other
    block (the TAIL) is the swap set. When a tail block's forward is
    about to run, the hook:

      1. Evicts the PREVIOUSLY-active tail block back to CPU (if any
         and if it is a different block). Eviction happens HERE, in the
         next block's pre-hook, rather than in the current block's
         post-hook, because a post-hook also fires during the
         gradient-checkpoint recompute in backward and would evict the
         block before its own backward had a chance to run, stranding
         the weights autograd needs. Deferring to the next block keeps
         the current block resident through its full forward AND its
         backward.
      2. Loads the requested tail block onto the GPU.

    This is TRAVERSAL-AGNOSTIC: it bounds the resident tail to ONE block
    whether the blocks are visited in forward order (0..N-1) or in the
    reverse order autograd uses while recomputing checkpointed blocks
    during backward (N-1..0). The old design used a fixed eviction
    index (``block_index - window_size + 1``) that only made sense for
    forward traversal and was a no-op during backward, which let every
    recomputed block accumulate on the GPU and caused the Stage F OOM.

    The prefetch of the next block was also removed: in forward order it
    helped overlap, but in backward order it pulls in a block whose
    backward is already finished, which is exactly the accumulation bug.
    Correctness first; prefetch overlap can return later as an explicit
    reverse-aware optimisation.
    """

    def pre_hook(_module: Any, _inputs: Any) -> None:
        if handle.released:
            return
        # Head blocks are permanently resident; nothing to swap.
        if block_index < handle.window_size:
            return
        # Evict the previously-active tail block before loading this one.
        previous = handle.active_swapped_index
        if previous is not None and previous != block_index:
            _evict_probe_before(previous)
            _ensure_on_cpu(handle.blocks[previous], handle.cpu_device)
            _evict_probe_after(previous)
        # Load the requested tail block and record it as active.
        _ensure_on_device(handle.blocks[block_index], handle.target_device)
        handle.active_swapped_index = block_index

    return pre_hook


# ---------------------------------------------------------------------------
# Eviction diagnostic (NF4 floor investigation, env-gated, removable)
# ---------------------------------------------------------------------------
#
# The attributed memory snapshot (see
# ``memory-bank/feature_video_training_and_vram_investigation.md``)
# showed that at ``blocks_resident=1`` all 48 NF4 blocks stay live on the
# GPU (3.76 GB through the quantized branch of ``_move_param``), even
# though the snapshot is taken at the step boundary AFTER backward and
# the optimizer step, when the autograd graph is already freed. The
# single-slot swapper is supposed to leave only ~2 blocks resident, so a
# retained reference is keeping every evicted block's GPU copy alive.
#
# This probe instruments the eviction itself: it logs
# ``torch.cuda.memory_allocated()`` immediately before and after each
# ``_ensure_on_cpu`` call. If allocated DROPS by ~one block per evict,
# eviction frees correctly and the retainer is elsewhere (autograd-save);
# if allocated does NOT drop, the evict-to-CPU is failing to release the
# GPU storage and the fix belongs in the quantized branch of
# ``_move_param``. It is completely inert unless ``OPENLTX_MEM_DEBUG=1``
# and never runs on a normal training pass.


def _evict_probe_enabled() -> bool:
    import os

    return os.environ.get("OPENLTX_MEM_DEBUG", "") in ("1", "true", "True")


def _evict_probe_before(block_index: int) -> None:
    if not _evict_probe_enabled():
        return
    import torch

    if not torch.cuda.is_available():
        return
    _EVICT_PROBE_STATE["before"] = torch.cuda.memory_allocated()


def _evict_probe_after(block_index: int) -> None:
    if not _evict_probe_enabled():
        return
    import torch

    if not torch.cuda.is_available():
        return
    before = _EVICT_PROBE_STATE.get("before", 0)
    after = torch.cuda.memory_allocated()
    delta_mb = (before - after) / (1024**2)
    count = _EVICT_PROBE_STATE.get("count", 0) + 1
    _EVICT_PROBE_STATE["count"] = count
    # The forward pass frees ~one block (419 MB) per evict cleanly. The
    # bug shows up where an evict frees little or nothing, i.e. the GPU
    # weight is still referenced (suspected: gradient-checkpoint recompute
    # during backward). So log the first 12 evictions (forward baseline)
    # AND every evict that frees less than 100 MB (the suspected leaks),
    # capped so the log stays bounded. ``grad_enabled`` is logged because
    # the backward recompute runs with grad enabled inside the checkpoint.
    leaky = delta_mb < 100.0
    leaky_count = _EVICT_PROBE_STATE.get("leaky_count", 0)
    if leaky:
        leaky_count += 1
        _EVICT_PROBE_STATE["leaky_count"] = leaky_count
    if count <= 12 or (leaky and leaky_count <= 60):
        logger.info(
            "OPENLTX_MEM_DEBUG evict block %d (#%d%s): allocated %.3f -> %.3f GB "
            "(freed %.1f MB) grad=%s",
            block_index,
            count,
            " LEAK" if leaky else "",
            before / (1024**3),
            after / (1024**3),
            delta_mb,
            torch.is_grad_enabled(),
        )


_EVICT_PROBE_STATE: dict[str, int] = {}





def _ensure_on_device(
    module: Any,
    target_device: "torch.device",
    non_blocking: bool = False,
) -> None:
    """Move ``module`` to ``target_device`` if it is not already there.

    Quant-aware: bitsandbytes ``Params4bit`` / torchao wrapper tensors
    are moved through their own ``.to()`` so their ``quant_state`` /
    scale travels with the packed weight (see ``_move_param``). The old
    ``parameter.data = parameter.data.to(...)`` here was Defect B: it
    moved the 4-bit blob but stranded the ``quant_state`` on CPU, and
    the next ``dequantize_4bit`` faulted with a CUDA illegal memory
    access. We do not skip trainable params on a move TO the GPU; the
    resident LoRA adapters belong on the device.
    """
    _move_module(
        module,
        target_device,
        skip_trainable=False,
        non_blocking=non_blocking,
    )


def _ensure_on_cpu(module: Any, cpu_device: "torch.device") -> None:
    """Move ``module``'s frozen weights back to CPU pinned memory.

    Quant-aware and trainable-skipping. Evicting a block to CPU must
    leave the trainable LoRA adapters on the GPU (the optimizer holds
    references to them and tracks their moments on the device), so we
    pass ``skip_trainable=True``. Quantized base weights move through
    their subclass ``.to()`` so ``quant_state`` stays colocated.
    """
    _move_module(
        module,
        cpu_device,
        skip_trainable=True,
        pin=True,
    )

