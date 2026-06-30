"""FOCUS split-forward attention-metadata builder (plan §8, step 4).

The reduced forward runs in three regimes inside a single denoising step, each
needing its own FlashInfer attention metadata (rebuilt via
``init_forward_metadata`` between phases — the ``reinit_attn_backend`` pattern in
model_runner.forward_split_prefill):

  Phase P  (prefix)  : L0 full + L1 QKV+RoPE+fill — q=B, kv=context+B  (the
                       ORIGINAL dLLM-extend metadata; no rebuild needed).
  Phase A1 (L1 attn) : q=|S|, kv=context+B   (retained queries attend the full
                       block KV written in Phase P).
  Phase S  (L2..L)   : q=|S|, kv=context+|S| (retained KV written to the block's
                       contiguous prefix; evicted slots are never read).

Both reduced phases collapse to the full forward when |S|=B (the α→∞ anchor),
which is the correctness gate for the split.

Why this works with NO custom kernel (R1, resolved): SGLang's FlashInfer dLLM
extend builds token-granular ``kv_indices`` from ``req_to_token`` — per request
it reads the contiguous slice ``req_to_token[req, 0 : paged_kernel_lens]`` where
``paged_kernel_lens = seq_lens`` and the query count = ``seq_lens - prefix_lens``
(qo_indptr). So a phase is selected purely by its ``seq_lens`` / ``prefix_lens``,
provided the retained block KV occupies a contiguous prefix of the block region
(guaranteed by writing retained L≥2 KV to the block's first |S| slots).

This module is intentionally pure tensor math (no model / CUDA dependency) so it
can be unit-tested in isolation; ``make_focus_phase_batch`` is a thin applier
that stamps the computed fields onto a shallow-copied ForwardBatch.
"""

import copy
from typing import Optional, Tuple

import torch

PHASE_A1 = "A1"  # L1 attention: q=|S|, kv=context+B
PHASE_S = "S"  # layers 2..L: q=|S|, kv=context+|S|


def compute_focus_phase_lens(
    phase: str,
    seq_lens: torch.Tensor,
    block_size: int,
    new_lens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-phase (seq_lens, extend_prefix_lens, extend_seq_lens) for the suffix.

    The original dLLM-extend metadata (per request b) is::

        seq_lens[b]          = context_b + B          (full KV in req_to_token)
        extend_prefix_lens[b]= context_b              (so query count = B)
        extend_seq_lens[b]   = B

    where ``context_b = seq_lens[b] - block_size``. The reduced phases keep the
    SAME ``req_to_token`` slice base (0) and shrink either the query count alone
    (A1) or both the query count and the KV length (S):

      Phase A1: q=|S|, kv unchanged (context+B)::
          seq_lens[b]           = context_b + B           (unchanged)
          extend_seq_lens[b]    = |S_b|
          extend_prefix_lens[b] = seq_lens[b] - |S_b|     (= context+B-|S|)

      Phase S: q=|S|, kv=context+|S|::
          seq_lens[b]           = context_b + |S_b|
          extend_seq_lens[b]    = |S_b|
          extend_prefix_lens[b] = context_b              (= original seq_lens-B)

    Non-causality (dLLM block attention is ENCODER_ONLY ⇒ causal=False) makes the
    physical position of the |S| queries within the sequence irrelevant to
    masking, so handing FlashInfer a shrunk query count + the right KV length is
    exactly the paper's reduced attention — no per-token mask needed.

    Args:
        phase: ``"A1"`` or ``"S"``.
        seq_lens: [bs] int original dLLM-extend seq_lens (= context+B per req).
        block_size: B.
        new_lens: [bs] int retained count |S_b| (from ``build_retained_index``).

    Returns:
        (phase_seq_lens, extend_prefix_lens, extend_seq_lens), all [bs] int64.
    """
    seq_lens = seq_lens.to(torch.int64)
    new_lens = new_lens.to(torch.int64).to(seq_lens.device)
    context_lens = seq_lens - block_size
    if phase == PHASE_A1:
        phase_seq_lens = seq_lens.clone()
        extend_seq_lens = new_lens.clone()
        extend_prefix_lens = phase_seq_lens - new_lens
    elif phase == PHASE_S:
        phase_seq_lens = context_lens + new_lens
        extend_seq_lens = new_lens.clone()
        extend_prefix_lens = context_lens.clone()
    else:
        raise ValueError(f"unknown FOCUS phase {phase!r} (expected 'A1' or 'S')")
    return phase_seq_lens, extend_prefix_lens, extend_seq_lens


def build_phase_s_out_cache_loc(
    out_cache_loc: torch.Tensor,
    block_size: int,
    new_lens: torch.Tensor,
) -> torch.Tensor:
    """Physical KV slots for the retained tokens at L≥2 (block-prefix compaction).

    The original ``out_cache_loc`` is block-contiguous: request b owns the slice
    ``out_cache_loc[b*B : (b+1)*B]`` (the B block positions). The reduced suffix
    writes each retained token's L≥2 KV to the FIRST |S_b| slots of its block, so
    the kept KV occupies a contiguous prefix of the block region in
    ``req_to_token`` and a single ``req_to_token[req, 0:context+|S|]`` slice reads
    exactly context+retained. We therefore gather the first |S_b| slots per block.

    RoPE/positions are per-token and already applied to k before caching, so the
    physical slot order within the block is irrelevant under non-causal block
    attention (R3).

    Args:
        out_cache_loc: [bs*B] original block KV slot ids, block-contiguous.
        block_size: B.
        new_lens: [bs] retained count |S_b|.

    Returns:
        [sum(|S_b|)] long: retained KV slots, request-major.
    """
    bs = new_lens.numel()
    if bs == 0:
        return out_cache_loc.new_empty(0)
    # Vectorized (§A2): gather the first |S_b| slots of each block via a
    # ``arange(B) < |S_b|`` prefix mask over the [bs, B] block grid — no D2H sync.
    grid = out_cache_loc[: bs * block_size].view(bs, block_size)
    cols = torch.arange(block_size, device=new_lens.device)
    prefix_mask = cols.unsqueeze(0) < new_lens.to(torch.int64).unsqueeze(1)
    return grid[prefix_mask]


def make_focus_phase_batch(
    base_fb,
    phase: str,
    block_size: int,
    new_lens: torch.Tensor,
    compact_input_ids: torch.Tensor,
    compact_positions: torch.Tensor,
    compact_out_cache_loc: Optional[torch.Tensor] = None,
    new_lens_cpu: Optional[torch.Tensor] = None,
):
    """Shallow-copy ``base_fb`` and stamp the per-phase reduced-forward fields.

    The returned ForwardBatch is fed to ``model_runner`` with
    ``reinit_attn_backend=True`` so FlashInfer rebuilds its metadata from the new
    ``seq_lens`` / ``extend_prefix_lens``. Only the fields the suffix touches are
    overwritten; everything else (req_pool_indices, kv pool, etc.) is shared.

    For Phase A1 no KV is written (the L1 attention runs read-only against the
    full-block KV from Phase P), so ``out_cache_loc`` is left as the base value
    and the caller must invoke attention with ``save_kv_cache=False``. For Phase
    S, pass ``compact_out_cache_loc`` from ``build_phase_s_out_cache_loc``.

    §A3 (de-sync): the FlashInfer prefill plan reads the host lists
    ``seq_lens_cpu`` / ``extend_prefix_lens_cpu`` / ``extend_seq_lens_cpu``. All
    three are deterministic host arithmetic on ``base_fb.seq_lens_cpu`` (already
    on host), ``block_size`` and ``|S_b|``. The caller passes ``new_lens_cpu``
    (one ``new_lens.cpu()`` per step shared across both phases) so this builder
    does ZERO additional D2H — the device fields come from device arithmetic, the
    host fields from host arithmetic. (When ``new_lens_cpu`` is omitted we fall
    back to a local ``.cpu()`` so the unit tests and any ad-hoc caller still work.)
    """
    # Device-side per-phase lens (no sync) for the tensors the GPU path reads.
    phase_seq_lens, extend_prefix_lens, extend_seq_lens = compute_focus_phase_lens(
        phase, base_fb.seq_lens, block_size, new_lens
    )
    # Host-side per-phase lens (no sync; seq_lens_cpu is already host) for the
    # FlashInfer plan's *_cpu lists.
    if new_lens_cpu is None:
        new_lens_cpu = new_lens.detach().cpu()
    seq_lens_cpu = base_fb.seq_lens_cpu
    (
        phase_seq_lens_cpu,
        extend_prefix_lens_cpu,
        extend_seq_lens_cpu,
    ) = compute_focus_phase_lens(phase, seq_lens_cpu, block_size, new_lens_cpu)

    fb = copy.copy(base_fb)
    fb.input_ids = compact_input_ids
    fb.positions = compact_positions
    fb.seq_lens = phase_seq_lens
    fb.seq_lens_cpu = phase_seq_lens_cpu
    fb.seq_lens_sum = int(phase_seq_lens_cpu.sum())
    fb.extend_seq_lens = extend_seq_lens.to(torch.int32)
    fb.extend_prefix_lens = extend_prefix_lens.to(torch.int32)
    fb.extend_seq_lens_cpu = extend_seq_lens_cpu.tolist()
    fb.extend_prefix_lens_cpu = extend_prefix_lens_cpu.tolist()
    fb.extend_num_tokens = int(extend_seq_lens_cpu.sum())
    if compact_out_cache_loc is not None:
        fb.out_cache_loc = compact_out_cache_loc
    # The importance side-channel only fires in Phase P; the suffix must not
    # recompute it (and its uniform-CSR layout no longer matches |S|).
    fb.focus_view = None
    return fb
