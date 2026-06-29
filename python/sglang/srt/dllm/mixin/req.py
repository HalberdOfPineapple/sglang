from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.dllm.config import DllmConfig

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class DllmReqPhase(str, enum.Enum):
    STAGING_PREFILL = "staging_prefill"
    STAGING_DECODE = "staging_decode"
    INCOMING_PREFILL = "incoming_prefill"
    INCOMING_DECODE = "incoming_decode"


@dataclass
class FocusState:
    """Tracks cumulative decoding statistics and block progress per request."""

    block_length: int
    token_sum: int = 0           # Cumulative decoded tokens across all steps
    total_steps: int = 0          # Total denoising steps taken
    rightmost_processed: int = -1 # Furthest decoded position in current block

    @property
    def avg_decoded_tokens(self) -> float:
        """Cumulative mean N̄_decoded (Eq. 19) for dynamic budgeting."""
        return self.token_sum / max(self.total_steps, 1)

    def reset_for_new_block(self):
        """Reset per-block state (keep cumulative statistics)."""
        self.rightmost_processed = -1


@dataclass
class DelayedCacheState:
    """Manages Intra-Block KV Cache with Neighbor-Aware Stability (DC+)."""

    block_length: int
    uncached_positions: torch.Tensor  # BoolTensor[block_length], True = needs computation
    needs_warmup: bool = True         # First step processes all positions

    def __post_init__(self):
        if self.uncached_positions is None:
            self.uncached_positions = torch.ones(self.block_length, dtype=torch.bool)

    def get_processing_indices(self) -> torch.Tensor:
        """Returns indices of positions to compute this step."""
        return torch.where(self.uncached_positions)[0]

    def update_from_mask(self, dllm_mask: torch.Tensor, mask_id: int):
        """Apply Neighbor-Aware Stability: mark position i cached when both i and i+1 decoded."""
        unmasked = (dllm_mask != mask_id)
        # ready_mask[i] = unmasked[i] AND unmasked[i+1]
        ready_mask = torch.zeros_like(unmasked)
        if len(unmasked) > 1:
            ready_mask[:-1] = unmasked[:-1] & unmasked[1:]
        # Last position ready when decoded (no right neighbor)
        if len(unmasked) > 0:
            ready_mask[-1] = unmasked[-1]
        self.uncached_positions = self.uncached_positions & (~ready_mask)
        self.needs_warmup = False

    def reset_for_new_block(self):
        """Reset for new block (all positions uncached)."""
        self.uncached_positions.fill_(True)
        self.needs_warmup = True


class ReqDllmMixin:
    def init_diffusion_llm(self: Req, dllm_config: DllmConfig):
        self.dllm_phase: Optional[DllmReqPhase] = None
        self.dllm_block_offset = 0
        self.dllm_config = dllm_config

        # Initialize FOCUS state if using Focus algorithm
        if self.dllm_config is not None and self.dllm_config.algorithm == "Focus":
            self.focus_state = FocusState(block_length=dllm_config.block_size)
            self.delayed_cache_state = DelayedCacheState(
                block_length=dllm_config.block_size,
                uncached_positions=None  # Will be initialized in __post_init__
            )
        else:
            self.focus_state = None
            self.delayed_cache_state = None

        if self.dllm_config is not None:
            if len(self.origin_input_ids) < self.dllm_config.block_size:
                self.dllm_phase = DllmReqPhase.INCOMING_DECODE
            else:
                self.dllm_phase = DllmReqPhase.INCOMING_PREFILL

    def is_dllm(self: Req) -> bool:
        return self.dllm_config is not None

    def is_dllm_prefill(self: Req) -> bool:
        return self.dllm_phase in [
            DllmReqPhase.STAGING_PREFILL,
            DllmReqPhase.INCOMING_PREFILL,
        ]

    def determine_dllm_phase(self: Req):
        prefix_length = len(self.prefix_indices)
        min_required_length = prefix_length + self.dllm_config.block_size

        if len(self.fill_ids) < min_required_length:
            # still incoming stage
            return

        input_block = self.fill_ids[prefix_length:min_required_length]
        is_prefill_phase = self.dllm_config.mask_id not in input_block

        if is_prefill_phase:
            self.dllm_phase = DllmReqPhase.STAGING_PREFILL
        else:
            self.dllm_phase = DllmReqPhase.STAGING_DECODE

    def _init_fill_ids_for_dllm(self: Req):
        self.dllm_block_offset = (
            0
            if not self.fill_ids
            else self.dllm_block_offset + self.dllm_config.block_size
        )
        self.fill_ids = (
            self.origin_input_ids
            + self.output_ids
            + [self.dllm_config.mask_id] * self.dllm_config.block_size
        )

        # Reset FOCUS state for new block
        if self.focus_state is not None:
            self.focus_state.reset_for_new_block()
        if self.delayed_cache_state is not None:
            self.delayed_cache_state.reset_for_new_block()

    def _update_block_offset_for_dllm(self):
        prefix_len = len(self.prefix_indices)
        assert (
            prefix_len % self.dllm_config.block_size == 0
        ), f"Unexpected prefix len: {prefix_len}"
        if prefix_len > self.dllm_block_offset:
            self.dllm_block_offset = prefix_len
