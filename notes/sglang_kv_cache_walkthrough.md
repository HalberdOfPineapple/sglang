# SGLang KV Cache: Architecture & Code Walkthrough

A detailed walkthrough of how SGLang manages the KV cache: the cooperating modules, each
component's responsibilities, and the concrete code paths for allocation, the write path,
prefix matching, caching, and eviction. File/line references point at the current tree under
`python/sglang/srt/mem_cache/` unless noted. dLLM-specific notes are called out where the
denoising decode pattern diverges from autoregressive (AR) decoding.

## 1. High-level overview

SGLang deliberately splits "KV cache" into **three layers of indirection** so that prefix
sharing, paging, and physical storage are decoupled (`memory_pool.py:20-25`):

```
 logical request                     indices                       physical bytes
 ┌────────────────────┐   maps    ┌───────────────────────┐  owns  ┌────────────────────┐
 │  ReqToTokenPool    │ ───────►  │ TokenToKVPoolAllocator │ ────► │      KVCache       │
 │ req_pool_idx, pos  │  token →  │  free-list of slots /  │ slot  │ k_buffer/v_buffer  │
 │   → slot index     │  slot     │  pages over the pool   │ →ptr  │ per layer (GPU)    │
 └────────────────────┘           └───────────────────────┘        └────────────────────┘
          ▲                                   ▲                              ▲
          │ writes per-position slots         │ alloc()/free() slots         │ set_kv_buffer()
          │                                   │                              │ during attention
  ┌───────┴────────────────────────────────── ┴──────────────────────────── ┴───────┐
  │                          BasePrefixCache (tree_cache)                             │
  │   RadixCache / ChunkCache / SWA / Mamba / HiRadix — decides what to keep, reuse,  │
  │   evict; holds refs into the allocator; drives match_prefix / cache_* / evict     │
  └──────────────────────────────────────────────────────────────────────────────────┘
```

The split exists because of three distinct concerns:

- **`ReqToTokenPool`** — a `[max_reqs, max_context_len]` int32 table answering *"for request
  R, where is the KV slot for token position i?"*. It is the per-request **token→slot map**.
- **`BaseTokenToKVPoolAllocator`** — a free-list over slot/page indices answering *"give me N
  fresh slots"* / *"these slots are free again"*. It owns no tensors of KV data, only the
  bookkeeping of which slots are in use.
- **`KVCache`** (`MHATokenToKVPool`, `MLATokenToKVPool`, …) — the actual GPU tensors holding
  K/V bytes, indexed by slot. The attention kernel reads/writes these.

Sitting above all three is the **prefix cache** (`tree_cache`, a `BasePrefixCache`). It is the
*policy* layer: it decides which token ranges are worth keeping for reuse across requests
(radix tree), takes a reference-counted hold on those slots in the allocator so they are not
recycled, and runs eviction when the allocator runs low. When prefix caching is disabled, a
degenerate `ChunkCache` stands in so the scheduler code path is uniform.

### Module map

| File | Role |
|---|---|
| `memory_pool.py` | `ReqToTokenPool`, `KVCache` ABC, `MHATokenToKVPool`, `MLATokenToKVPool`, FP4/NoOp variants, `MambaPool`, `HybridReqToTokenPool` |
| `allocator.py` | `BaseTokenToKVPoolAllocator`, `TokenToKVPoolAllocator` (page_size=1), `PagedTokenToKVPoolAllocator` (+Triton alloc kernels) |
| `base_prefix_cache.py` | `BasePrefixCache` ABC + the param/result dataclasses (`MatchResult`, `EvictParams`, …) |
| `radix_cache.py` | `RadixKey`, `TreeNode`, `RadixCache` (the default prefix cache) |
| `chunk_cache.py` | `ChunkCache` / `SWAChunkCache` — no-prefix-sharing fallback |
| `common.py` | Scheduler-facing helpers: `alloc_for_extend`, `alloc_for_decode`, `write_cache_indices`, `evict_from_tree_cache`, `release_kv_cache` |
| `swa_*`, `mamba_*`, `hiradix_*`, `unified_radix_cache.py` | Specialized caches (sliding-window, Mamba hybrid, host-offload/HiCache, unified multi-component) |
| `cache_init_params.py` | `CacheInitParams` — the single config struct passed to every cache |
| `model_executor/model_runner_kv_cache_mixin.py` | Construction & sizing of the pools at startup |

### Who constructs what

At startup the model runner's `init_memory_pool` (`model_runner_kv_cache_mixin.py:896`)
builds, in order: `req_to_token_pool` → `token_to_kv_pool` (the `KVCache`) →
`token_to_kv_pool_allocator` (wrapping the `KVCache`). The selection is driven by model family
and flags (`model_runner_kv_cache_mixin.py:316, 541, 643, 731-747`):

- MLA models (DeepSeek) → `MLATokenToKVPool` (single fused latent buffer).
- Everything else → `MHATokenToKVPool` (separate K and V buffers).
- `page_size == 1` → `TokenToKVPoolAllocator`; `page_size > 1` → `PagedTokenToKVPoolAllocator`.
- Hybrid SWA → `SWATokenToKVPoolAllocator`; Mamba hybrid → `HybridReqToTokenPool` + `MambaPool`.

The scheduler then picks the **prefix cache** (`scheduler.py:900-977`): `ChunkCache` when
radix is disabled, `RadixCache` by default, or `SWARadixCache` / `MambaRadixCache` /
`HiRadixCache` / `UnifiedRadixCache` for the specialized regimes. All of them receive the same
`req_to_token_pool` and `token_to_kv_pool_allocator` handles, so the three storage layers are
shared while the policy layer varies.

## 2. Per-component analysis

### 2.1 `ReqToTokenPool` — the token→slot map (`memory_pool.py:128-193`)

```python
self.req_to_token = torch.zeros((size + 1, max_context_len), dtype=torch.int32, device=device)
self.free_slots = list(range(1, self._alloc_size))
```

- A dense 2-D int32 table on GPU. Row `req_pool_idx` holds, at column `i`, the **KV slot
  index** for the i-th token of that request. `write(indices, values)` is a plain
  `self.req_to_token[indices] = values` (`:154`).
- **Row 0 is a padding row.** CUDA-graph padded batches default `req_pool_indices` to 0, so
  dummy reads/writes for padding tokens land in row 0 harmlessly (`:142-145`).
- `alloc(reqs)` hands out free rows but **reuses an existing `req_pool_idx`** for requests that
  already have one — chunked prefill continuing across chunks, or a dLLM block re-using its
  slot. The assertion at `:170-172` enforces that a reusing request is either chunked or has
  already-committed KV.
- This pool's capacity bounds **max concurrent requests** (`--max-running-requests`), not total
  tokens.

dLLM note: under iterative denoising a request keeps the *same* `req_pool_idx` across many
denoising steps of a block, and its row records slots for all positions in the active block —
including still-masked positions whose KV will be overwritten as tokens get committed. The
reuse path in `alloc` (`:163`) is what makes that stable across steps.

### 2.2 `KVCache` and its subclasses — physical storage (`memory_pool.py:693-1108`)

The `KVCache` ABC (`:693`) fixes the contract every attention backend relies on:
`get_key_buffer`, `get_value_buffer`, `get_kv_buffer`, `set_kv_buffer`. `store_dtype` may
differ from logical `dtype` (FP8 stored as uint8 because `index_put` is unimplemented for
float8 — `:710-714`).

**MHA pool** (`MHATokenToKVPool`, `:789`) — separate per-layer buffers:

```python
self.k_buffer = [torch.zeros((size + page_size, head_num, head_dim), store_dtype, device) ...]
self.v_buffer = [torch.zeros((size + page_size, head_num, v_head_dim), store_dtype, device) ...]
```

- Indexed by **slot**: `k_buffer[layer][slot]` is one token's K vector. `size + page_size`
  rows because slot 0..page_size-1 are the padding slots.
- The write path `set_kv_buffer` (`:1047`) casts to store dtype, optionally applies k/v scales,
  then calls `_set_kv_buffer_impl` (`:91`), which prefers a fused `store_cache` JIT kernel and
  falls back to `k_cache[indices] = k; v_cache[indices] = v`. During CUDA-graph capture it
  splits K and V onto an alt stream to overlap the two scatter writes (`:116-122`).
- `get_key_buffer` (`:1025`) optionally blocks on a `layer_transfer_counter` — this is the hook
  for **layer-wise KV loading** in HiCache: the attention kernel for layer L waits until the
  host→device transfer of layer L's KV finished.
- `data_ptrs` / `data_strides` (`:923-940`) are flat tensors of raw pointers used by the
  whole-cache `move_kv_cache` Triton kernel (used by spec-decode page reshuffles).

**MLA pool** (`MLATokenToKVPool`, `:1618`) — one fused latent buffer per layer:

```python
self.kv_buffer = [torch.zeros((size + page_size, 1, kv_lora_rank + qk_rope_head_dim), ...) ...]
```

- DeepSeek-style MLA stores a single compressed latent (`kv_lora_rank`) plus a RoPE part
  instead of full K and V, so memory per token is far smaller. `get_value_buffer` just slices
  the first `kv_lora_rank` dims of the same buffer (`:1718-1726`). `set_mla_kv_buffer` (`:1750`)
  writes the nope/rope parts via Triton, with FP8-quant variants for NSA.

**Variants**: `MHATokenToKVPoolFP4` (`:1246`, packs to 4-bit + scale buffers),
`NoOpMHATokenToKVPool` (`:1136`, embedding/prefill-only — allocates KB-sized placeholders and
raises if anyone actually writes), `MLATokenToKVPoolFP4`, and NPU variants.

### 2.3 Allocators — the slot/page free-list (`allocator.py`)

`BaseTokenToKVPoolAllocator` (`:35`) holds two index tensors, `free_pages` and `release_pages`,
plus a `free_group` batching mechanism. Key design points:

- **`need_sort`** (`:51`): in P/D-disaggregation, freed slots accumulate in `release_pages` and
  are only merged+sorted into `free_pages` lazily when an allocation would otherwise fail
  (`merge_and_sort_free`, `:86-92`). This keeps freed indices contiguous-ish for transfer
  engines without paying a sort on every free.
- **`free_group_begin/end`** (`:77-84`): defers a batch of frees and concatenates them into one
  `free()` call — used so a whole batch's frees become one tensor op.
- **`backup_state` / `restore_state`** (`:71-75`): snapshot the free-lists. CUDA-graph capture
  and speculative decoding use this to roll back tentative allocations.

**`TokenToKVPoolAllocator`** (page_size=1, `:121`) is the simple case: `free_pages` is just
`arange(1, size+1)`; `alloc(n)` slices the first n (`:148-157`); `free` concatenates back. Slot
0 is reserved as the dummy/padding slot (`:135-138`).

**`PagedTokenToKVPoolAllocator`** (`:362`) allocates **page-aligned, contiguous** slot runs so
that paged attention kernels see whole pages. The interesting part is that it has three
purpose-built entry points instead of one generic `alloc`:

- `alloc(n)` (`:386`) — n must be page-multiple; returns `pages[:,None]*page_size + arange`.
- `alloc_extend(...)` (`:409`) — prefill/extend. A request's prefix already occupies a partial
  last page; the new tokens must (1) fill that partial page, (2) take some full fresh pages,
  (3) start a new partial page. The Triton `alloc_extend_kernel` (`:240`) computes all three
  parts per request in parallel (Part 1/2/3 in the kernel body, `:274-323`). This is the hot
  path that turns "prefix_lens, seq_lens, last_loc" into a flat `out_indices` of new slots.
- `alloc_decode(...)` (`:459`) — one token per request; `alloc_decode_kernel` (`:326`) decides
  per request whether the new token reuses the current page's next slot or opens a fresh page.

`free` (`:498`) reduces slot indices back to **unique page indices** (`free_index //
page_size`) before returning them, since a page is the unit of reuse.

Why Triton kernels here? Allocation must be a single GPU op over the whole batch (no
per-request Python loop on the hot path), and the page arithmetic — partial-page fill,
cross-page boundaries, page-aligned output — is exactly what `alloc_extend_kernel` expresses
without a host round-trip.

### 2.4 `BasePrefixCache` — the policy interface (`base_prefix_cache.py:196`)

Every cache implements: `match_prefix`, `cache_unfinished_req`, `cache_finished_req`, `evict`,
`inc_lock_ref`, `dec_lock_ref`, plus size accessors. The unified dataclasses (`MatchResult`,
`EvictParams`, `IncLockRefResult`, …) keep a single signature across the simple radix cache and
the complex hybrid/host-offload caches. `MatchResult` (`:145`) is the central return type:
`device_indices` (the matched KV slots on GPU), `last_device_node` (where to anchor the lock),
and HiCache-only fields (`last_host_node`, `best_match_node`, `host_hit_length`).

### 2.5 `RadixCache` — prefix sharing via a radix tree (`radix_cache.py:269`)

This is the default and the most important policy implementation.

**`RadixKey`** (`:66`) wraps the token-id sequence plus an `extra_key` namespace tag. The
`extra_key` keeps otherwise-identical prefixes **disjoint** when they must not share state —
different LoRA adapters, cache salts, or RAG contexts (`match_prefix` docstring, `:360-396`).
`is_bigram` mode supports EAGLE speculative decoding, where the cache key is over token *pairs*.
`page_aligned` truncates the key to a page multiple before any tree op (`:121-125`).

**`TreeNode`** (`:206`): `children` (dict keyed by `child_key`), `key` (a `RadixKey` segment),
`value` (an int64 tensor of the **KV slot indices** for that segment), `lock_ref`, and access
metadata. `value is None` ⇒ node **evicted** (`:233-235`). Host-offload fields (`host_value`,
`hash_value`, `host_ref_counter`) support HiCache.

Core invariants the tree maintains:

- The concatenation of `value` tensors along a root→node path **is** the list of KV slots for
  that token prefix — exactly what gets written into `req_to_token` and handed to attention.
- `evictable_size_` + `protected_size_` track how many tokens are reclaimable vs. locked.
- `lock_ref > 0` ⇒ node (and all ancestors) protected from eviction. A running request holds a
  lock on its `last_node`.

**`match_prefix`** (`:360`) → `_match_prefix_helper` (`:645`): walks children by `child_key`,
matching as far as possible. If a match ends *inside* a stored segment, `_split_node` (`:671`)
splits that node so the boundary is exposed (the shared prefix becomes a parent, the divergent
tail a child). Returns the concatenated slot tensor + terminal node.

**`insert`** (`:420`) → `_insert_helper` (`:701`): same walk, but creates/splits nodes so the
full key is present, cloning the relevant slice of the slot `value` tensor into new nodes.
Returns `total_prefix_length` = how much of the inserted key already existed.

**Eviction** (`evict`, `:560`): builds a heap over `evictable_leaves` keyed by the configured
strategy's priority (LRU/LFU/FIFO/…, `evict_policy.py`), pops leaves, frees their `value` slots
back to the allocator (`token_to_kv_pool_allocator.free(x.value)`), deletes the node, and
re-pushes a parent that just became a childless leaf. Only **unlocked leaves** are evictable —
`_update_leaf_status` (`:783`) maintains the `evictable_leaves` set: a node is an evictable leaf
iff it's not evicted, not locked, and has no live children.

**Lock ref-counting** (`inc_lock_ref` `:589` / `dec_lock_ref` `:604`): walk node→root; the
first lock on a node moves its `len(key)` tokens from `evictable_size_` to `protected_size_`
(and the last unlock moves them back). This is how a running request pins its prefix so a
concurrent eviction can't pull slots out from under an in-flight forward pass.

### 2.6 `ChunkCache` — the no-sharing fallback (`chunk_cache.py:35`)

When radix caching is off, `ChunkCache` satisfies the same interface but shares nothing:
`match_prefix` always returns empty (`:67`), `insert` is a no-op, and `cache_finished_req`
(`:79`) simply frees the request's whole committed KV range back to the allocator.
`cache_unfinished_req` (`:87`) just stashes the current slots in `req.prefix_indices` so the
*same* request can continue in the next chunk. `disable` is a property hard-wired to `True`
(`:60-62`), which is how the scheduler detects "no prefix matching." `SWAChunkCache` (`:113`)
adds sliding-window eviction of out-of-window tokens.

## 3. Key workflows

### 3.1 Prefill / extend allocation — `alloc_for_extend` (`common.py:429`)

This runs when a batch of new (or chunk-continuing) requests enters prefill. The orchestration
lives in `common.py`, not in the pools themselves:

```
alloc_for_extend(batch):
  1. batch.maybe_evict_swa()                         # drop out-of-window SWA tokens
  2. prefix_tensors = [r.prefix_indices ...]         # slots already matched per req
  3. req_pool_indices = alloc_req_slots(...)         # rows in ReqToTokenPool
  4. if page_size == 1: out_cache_loc = alloc_token_slots(tree_cache, extend_num_tokens)
     else:              out_cache_loc = alloc_paged_token_slots_extend(... last_loc ...)
  5. write_cache_indices(out_cache_loc, ...)         # fill req_to_token rows
  return out_cache_loc, req_pool_indices_device, req_pool_indices
```

Step-by-step:

- **`alloc_req_slots`** (`common.py:398`) gets `ReqToTokenPool` rows, first evicting Mamba state
  if needed.
- **`alloc_token_slots`** (`common.py:302`) is the slot allocation wrapper. Critically it calls
  **`evict_from_tree_cache` first** (`:308`) — if the allocator's free list is smaller than the
  request, it asks the prefix cache to evict that many tokens *before* trying `allocator.alloc`.
  Only then does it `alloc`; `None` means true OOM and raises with a dump of the tree (`:316`).
- **`alloc_paged_token_slots_extend`** (`common.py:356`) over-estimates the eviction target by
  `+len(seqs)*page_size` (each request may open a partial page) and routes to the Triton
  `alloc_extend`. `last_loc` is the last occupied slot of each request's prefix, needed to
  continue filling its partial page.
- **`write_cache_indices`** (`common.py:104`) writes both the **prefix** slots (matched from the
  tree) and the **new** slots into the request's `req_to_token` row, via the Triton
  `write_req_to_token_pool_triton` kernel (`:53`) when the backend supports it, else a Python
  loop (`:135`). After this, `req_to_token[req_pool_idx, 0:seq_len]` is the full slot list for
  the request, ready for attention.

```
ReqToTokenPool row for req R after write_cache_indices:
  col:   0        prefix_len        seq_len
         │  prefix slots │ newly-alloc slots │ (from tree match) (from alloc_extend)
         └───────────────┴───────────────────┘
                         out_cache_loc fills this part
```

### 3.2 Decode allocation — `alloc_for_decode` (`common.py:524`)

Each decode step appends `token_per_req` tokens (1 for AR, >1 for speculative/dLLM block):

- `page_size == 1` → `alloc_token_slots(bs * token_per_req)`.
- `page_size > 1` → `alloc_paged_token_slots_decode` (`:495`) → Triton `alloc_decode`, with
  `last_loc = req_to_token[req_pool_indices, seq_lens-1]` (the current tail slot).
- Then a single `req_to_token_pool.write((req_pool_indices, locs), out_cache_loc)` records the
  new tail slot per request (`:559`).

dLLM note: for diffusion decoding `token_per_req` corresponds to the number of positions
committed/advanced per denoising iteration. The allocation arithmetic is identical — what
differs is upstream: which positions are "new" each step and how many forward passes touch the
same allocated slots before they are committed. KV reuse across denoising steps is valid only
where the model treats already-committed tokens as fixed context; masked positions whose
predictions change between steps must have their KV re-written, not reused.

### 3.3 The write path — getting K/V bytes into the pool

Allocation only reserves *slots*; the actual K/V tensors are written **inside the attention
layer** during the forward pass. The plumbing:

```
ForwardBatch.out_cache_loc  ──►  RadixAttention.forward / unified_attention_with_output
                                 (radix_attention.py:150)
                                          │
                                          ▼
                       attn_backend.forward(..., save_kv_cache=True)
                                          │  (flashinfer_backend.py:804)
                                          ▼
        forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v, k_scale, v_scale)
                                          │  (memory_pool.py:1047)
                                          ▼
                 k_buffer[layer][cache_loc] = k ;  v_buffer[layer][cache_loc] = v
```

`out_cache_loc` is exactly the `out_cache_loc` produced by the allocator in §3.1/§3.2 — the
flat list of slots for the tokens computed this step. So the attention backend scatters this
step's freshly-computed K/V into precisely the slots the scheduler reserved, and the *same*
slots are recorded in `req_to_token` for future reads. The read side is
`get_kv_buffer(layer_id)` handed to the paged attention kernel together with the page table
derived from `req_to_token` (`flashinfer_backend.py:815`).

### 3.4 Prefix matching at admission

Before a request is scheduled, `Req.init_next_round_input` (`schedule_batch.py:1024-1035`) calls
`tree_cache.match_prefix(...)`. The returned `device_indices` become `req.prefix_indices` (the
slots reused for free) and `req.last_node` is locked. The number of matched tokens is subtracted
from the work prefill has to do — this is the whole point of prefix caching: a shared system
prompt is computed once and every later request reuses those slots.

### 3.5 Caching an unfinished request — `cache_unfinished_req` (`radix_cache.py:487`)

Called when a request yields mid-generation (chunked prefill boundary, or after each
extend/decode in some modes). It:

1. Reads the request's current slots from `req_to_token` (`:493`).
2. `insert`s the `(token_ids → slots)` mapping into the tree (`:503`).
3. **Frees the duplicate slots** in `[cache_protected_len : new_prefix_len]` — the part that
   already existed in the tree, so two requests don't double-own the same logical prefix's
   physical slots (`:513`).
4. Re-runs `match_prefix` to get the canonical (possibly newly-split) slot tensor and **rewrites
   the request's `req_to_token` row** to point at the tree-owned slots (`:518-530`).
5. Moves the lock from the old `last_node` to the new one (`:538-539`).

`cache_protected_len` (`:532-536`) tracks the partial trailing page that lives in
`req.prefix_indices` but is *not* yet in the tree (page-alignment), so it gets freed correctly
later rather than leaking.

### 3.6 Caching a finished request — `cache_finished_req` (`radix_cache.py:440`)

On completion: take the committed token range, build the page-aligned `RadixKey`, `insert` it,
free the slots that were duplicates of what's already in the tree (`:472`), free the
**unaligned tail** that can't be page-stored (`:481`), and `dec_lock_ref` the request's node so
its prefix becomes evictable again (`:484`). If insertion is disabled (deterministic mode), it
simply frees the whole range (`:475-478`). `release_kv_cache` in `common.py:566` is the
scheduler-side wrapper that also returns the `ReqToTokenPool` row and any Mamba state.

### 3.7 Eviction under pressure

Eviction is **pull-based**, triggered lazily by allocation, not by a background thread. The
chain on a memory-tight `alloc`:

```
alloc_token_slots / alloc_paged_*  →  evict_from_tree_cache(tree_cache, num_tokens)
                                      (common.py:330)
   if allocator.available_size() < num_tokens:
       tree_cache.evict(EvictParams(num_tokens=num_tokens))   # radix_cache.py:560
           heap over evictable_leaves by strategy priority
           pop leaf → allocator.free(leaf.value) → delete node → maybe re-push parent
   then allocator.alloc(num_tokens)   # now succeeds, or raise OOM
```

For the SWA hybrid allocator, `evict_from_tree_cache` computes separate full-window and
sliding-window deficits and passes both in `EvictParams` (`common.py:339-349`). Locked nodes are
never in `evictable_leaves`, so an in-flight request's prefix is safe.

## 4. End-to-end picture

Putting the layers and workflows together for one request's lifetime:

```
 admit:   match_prefix ───────────────► req.prefix_indices (reused slots), lock last_node
 prefill: alloc_req_slots ────────────► ReqToTokenPool row
          alloc_for_extend ──evict?──► allocator.alloc_extend ──► out_cache_loc (new slots)
          write_cache_indices ────────► req_to_token[R, :seq_len] = prefix ++ new slots
          forward: set_kv_buffer ─────► k_buffer/v_buffer[layer][out_cache_loc] = K,V
          cache_unfinished_req ───────► insert into radix tree, dedup-free, re-lock
 decode:  (loop) alloc_for_decode ────► one slot/req, write tail, set_kv_buffer
 finish:  cache_finished_req ─────────► insert committed prefix, free tail+dups, dec_lock_ref
          release_kv_cache ───────────► free ReqToTokenPool row (+ Mamba state)
 later:   evict ──────────────────────► free unlocked leaf slots back to allocator
```

The three storage layers never make policy decisions; the prefix cache never touches GPU bytes.
That separation is what lets SGLang support page_size variation, MLA vs MHA, sliding-window,
Mamba-hybrid, and host-offload by swapping one box in the diagram while the scheduler-facing
`common.py` helpers and the `BasePrefixCache` interface stay fixed.

## 5. Open questions / where to look next

- **Host offload (HiCache)**: `hiradix_cache.py` + `memory_pool_host.py` add an L2 CPU tier and
  the `layer_transfer_counter` synchronization hinted at in `get_key_buffer`. Worth a separate
  walkthrough.
- **Unified multi-component cache**: `unified_radix_cache.py` + `unified_cache_components/`
  generalize the tree to validate per-component (FULL/SWA/Mamba) — this is the future-facing
  path the `MatchResult.best_match_node` field exists for.
- **dLLM specifics**: confirm in the dLLM decode path (`python/sglang/srt/dllm/`) exactly which
  positions are passed in `out_cache_loc` per denoising step and whether masked-position KV is
  re-written each step or reused — this determines KV-cache reuse validity under denoising and
  is the key correctness question for any KV optimization targeting diffusion decoding.
```
