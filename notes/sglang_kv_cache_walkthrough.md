# SGLang KV Cache: Architecture & Code Walkthrough

This note traces SGLang's KV cache subsystem end to end: where every component is **constructed** (init site), where it is **called** (usage site, with the full call chain from the scheduler entry point), and the concrete code for allocation, the write path, prefix matching, caching, and eviction. Every claim is anchored to `file:line` in the current tree. dLLM-specific divergences from autoregressive (AR) decoding are flagged inline.

## 1. The three storage layers plus a policy layer

SGLang splits "KV cache" into three layers of indirection so that prefix sharing, paging, and physical byte storage are decoupled, with a policy layer (`tree_cache`) on top. The docstring at `memory_pool.py:20` states the same split.

```
 logical request                     indices                       physical bytes
 ┌────────────────────┐   maps    ┌────────────────────────┐ owns  ┌────────────────────┐
 │  ReqToTokenPool    │ ───────►  │ TokenToKVPoolAllocator  │ ───► │      KVCache       │
 │ (req_pool_idx,pos) │  token →  │  free-list of slots /   │ slot │ k_buffer/v_buffer  │
 │   → slot index     │  slot     │  pages over the pool    │ →row │ per layer (GPU)    │
 └────────────────────┘           └────────────────────────┘       └────────────────────┘
          ▲ write rows                     ▲ alloc()/free()                 ▲ set_kv_buffer()
          │                                │                                │ during attention
  ┌───────┴────────────────────────────────┴────────────────────────────── ┴───────┐
  │                         BasePrefixCache  (self.tree_cache)                       │
  │  RadixCache / ChunkCache / SWA / Mamba / HiRadix: decides what to keep, reuse,   │
  │  evict; holds refs into the allocator; drives match_prefix / cache_* / evict.    │
  └─────────────────────────────────────────────────────────────────────────────────┘
```

The three lower boxes are pure mechanism and never make policy decisions. The top box is pure policy and never touches GPU bytes. That separation is what lets one box be swapped (MLA vs MHA, page_size=1 vs paged, sliding-window, Mamba-hybrid, host-offload) while the scheduler-facing helpers in `common.py` and the `BasePrefixCache` interface stay fixed.

| Component      | Class / file                                                 | What it is                                                   |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Token→slot map | `ReqToTokenPool` (`memory_pool.py:128`)                      | `[max_reqs+1, max_ctx]` int32 table: row `req_pool_idx`, col `i` → KV slot of token `i`. Bounds max concurrent requests. |
| Slot free-list | `TokenToKVPoolAllocator` (`allocator.py:121`), `PagedTokenToKVPoolAllocator` (`allocator.py:362`) | Owns no KV bytes; hands out / reclaims slot or page indices. |
| Physical bytes | `MHATokenToKVPool` (`memory_pool.py:789`), `MLATokenToKVPool` (`memory_pool.py:1618`) | Per-layer GPU tensors of K/V, indexed by slot. The attention kernel reads/writes these. |
| Policy         | `RadixCache` (`radix_cache.py:269`), `ChunkCache` (`chunk_cache.py:35`) | Prefix sharing, ref-counted locking, eviction.               |

## 2. Initialization: where each component is born

The construction order is fixed and one-directional — each layer is handed the layer below it. 

The whole chain runs once per worker at startup.

```
Scheduler.init_model_worker() -> Scheduler.init_tp_model_worker() 
  -> TpModelWorker._init_model_runner() 
  	-> ModelRunner(...) initialization
  		-> ModelRunner.initialize(pre_model_load_memory)
```

```
ModelRunner.initialize(pre_model_load_memory)              model_runner.py:600
  ├─ load_model()                                          model_runner.py:643
  ├─ init_memory_pool(pre_model_load_memory)               model_runner.py:737
  │     └─ ModelRunner.init_memory_pool(...)               model_runner_kv_cache_mixin.py:896
  │           └─ _apply_memory_pool_config(cfg) builds, in order:
  │                 1. self.req_to_token_pool   = ReqToTokenPool(...)            mixin:316
  │                 2. self.token_to_kv_pool    = MHATokenToKVPool(...)          mixin:643
  │                                              | MLATokenToKVPool(...)         mixin:541
  │                 3. self.token_to_kv_pool_allocator =
  │                        TokenToKVPoolAllocator(... kvcache=self.token_to_kv_pool) mixin:732   (page_size==1)
  │                      | PagedTokenToKVPoolAllocator(... kvcache=...)          mixin:740        (page_size>1)
  └─ init_attention_backend()                              model_runner.py:754
```

Step 3 wires the layers together: the allocator is constructed with `kvcache=self.token_to_kv_pool` (`mixin:736`), so `allocator.get_kvcache()` reaches the physical pool. The selection logic: MLA models (DeepSeek) get `MLATokenToKVPool` (`mixin:541`), everything else gets `MHATokenToKVPool` (`mixin:643`); `page_size==1` gets the simple allocator (`mixin:732`), `page_size>1` gets the paged one (`mixin:740`); hybrid SWA and Mamba take the `SWATokenToKVPoolAllocator` / `HybridReqToTokenPool` branches (`mixin:680, 294`). Sizes (`max_total_num_tokens`, etc.) come from `_resolve_memory_pool_config`, which profiles free GPU memory after the model weights are loaded.

The pools then travel up to the scheduler. `TpModelWorker` holds the `ModelRunner`; `get_memory_pool()` (`tp_worker.py:90`) returns the req pool and the allocator. The scheduler grabs them in its constructor:

```
Scheduler.__init__:
  self.req_to_token_pool, self.token_to_kv_pool_allocator = self.tp_worker.get_memory_pool()   scheduler.py:838
  params = CacheInitParams(req_to_token_pool=..., token_to_kv_pool_allocator=..., page_size=..., eviction_policy=...)   scheduler.py:~860
  self.tree_cache = RadixCache(params)        scheduler.py:971    (default)
                  | ChunkCache(params)         scheduler.py:900    (radix disabled + chunked prefill)
                  | SWARadixCache / MambaRadixCache / HiRadixCache / UnifiedRadixCache   scheduler.py:911-963
```

So after startup: the **`KVCache` (token_to_kv_pool) lives in the `ModelRunner`**; the **allocator and `ReqToTokenPool` are shared** by both the `ModelRunner` and the `Scheduler`; the **`tree_cache` lives in the `Scheduler`** and holds references to the same allocator and req pool through `CacheInitParams`. `CacheInitParams` (`cache_init_params.py:14`) is the single config struct every cache variant receives, which is why they all share one signature.

One subtlety for the write path later: the physical `KVCache` is *not* handed to the scheduler at all. Instead `ForwardBatch.init_new` pulls it straight from the model runner — `token_to_kv_pool=model_runner.token_to_kv_pool` (`forward_batch_info.py:484`) — so the attention layer can write bytes without going through the scheduler.

## 3. The scheduler loop: the single entry point for all usage

Every KV cache operation is reached from one place: the scheduler's event loop (`scheduler.py:1551`).

```
event_loop_normal():                                       scheduler.py:1551
  while True:
    recv_reqs = self.recv_requests()                       # new requests into waiting_queue
    self.process_input_requests(recv_reqs)
    batch = self.get_next_batch_to_run()                   scheduler.py:2501   ← decides prefill vs decode
    result = self.run_batch(batch)                         scheduler.py:3007   ← forward pass (write path)
    self.process_batch_result(batch, result)               ← caching + freeing finished reqs
```

`get_next_batch_to_run` (`scheduler.py:2501`) is the fork: it first tries to build a **prefill** batch via `get_new_batch_prefill` (`scheduler.py:2580/2627`); if there is nothing to prefill, it advances the running batch one **decode** step via `update_running_batch` (`scheduler.py:2604/2909`). The next three subsections follow each path down to the pools.

## 4. Prefill / extend: from waiting queue to allocated slots

This is "where a batch enters prefill." Trace:

```
get_next_batch_to_run()                                    scheduler.py:2580
  └─ get_new_batch_prefill()                               scheduler.py:2627
       └─ _get_new_batch_prefill_raw():                    scheduler.py:~2680
            for req in waiting_queue (via PrefillAdder):
              req.init_next_round_input(self.tree_cache)    scheduler.py:2766   ← PREFIX MATCH happens here
              adder.add_one_req(req, ...)                   scheduler.py:2767   ← admission/budget check
            new_batch = ScheduleBatch.init_new(can_run_list, req_to_token_pool,
                                  token_to_kv_pool_allocator, tree_cache, ...)   scheduler.py:2826
            new_batch.prepare_for_extend()                  scheduler.py:2843   ← ALLOCATION happens here
```

### 4.1 Prefix match at admission (`init_next_round_input` → `match_prefix`)

For each candidate request, `req.init_next_round_input(self.tree_cache)` (`scheduler.py:2766`) calls `tree_cache.match_prefix(...)` (`schedule_batch.py:1027`). The returned `MatchResult.device_indices` becomes `req.prefix_indices` (slots reused for free, already on GPU), and `req.last_node` is the tree node to lock. This is the entire payoff of prefix caching: a shared system prompt is computed once, and every later request reuses those exact slots instead of recomputing them. In `RadixCache.match_prefix` (`radix_cache.py:360`), the work is done by `_match_prefix_helper` (`radix_cache.py:645`), which walks children by `child_key`, and `_split_node` (`radix_cache.py:671`) if a match ends inside a stored segment so the boundary becomes a real node.

### 4.2 Allocation (`prepare_for_extend` → `alloc_for_extend`)

`prepare_for_extend` (`schedule_batch.py:1688`) sets `forward_mode = EXTEND` (or `DLLM_EXTEND` for diffusion, `schedule_batch.py:1693`), computes `prefix_lens`/`extend_lens`/`seq_lens` from `req.fill_ids` and `req.prefix_indices` (`schedule_batch.py:1697-1702`), and then calls `alloc_for_extend(self)` (`schedule_batch.py:1749`). That helper (`common.py:429`) orchestrates the three lower layers:

```
alloc_for_extend(batch):                                   common.py:429
  1. batch.maybe_evict_swa()                               # drop out-of-window SWA tokens
  2. req_pool_indices = alloc_req_slots(req_to_token_pool, reqs, tree_cache)   common.py:398  → ReqToTokenPool.alloc()  memory_pool.py:160
  3. if page_size == 1:
        out_cache_loc = alloc_token_slots(tree_cache, extend_num_tokens)        common.py:302
     else:
        out_cache_loc = alloc_paged_token_slots_extend(tree_cache, prefix_lens, seq_lens, last_loc, ...)   common.py:356
  4. write_cache_indices(out_cache_loc, req_pool_indices, prefix_lens, seq_lens, ...)   common.py:104
```

Key points, layer by layer:

- **`alloc_req_slots`** (`common.py:398`) gets `ReqToTokenPool` rows. `ReqToTokenPool.alloc` (`memory_pool.py:160`) reuses an existing `req_pool_idx` for chunked-prefill or dLLM requests that already have one (`memory_pool.py:163`), and otherwise hands out free rows.
- **`alloc_token_slots`** (`common.py:302`) is the slot-allocation wrapper, and it does eviction *first*: `evict_from_tree_cache(tree_cache, num_tokens)` (`common.py:308`) before `allocator.alloc(num_tokens)` (`common.py:314`). If `alloc` still returns `None` it is genuine OOM and it raises after dumping the tree (`common.py:316-325`). This is the only place the allocator and the prefix cache meet on the alloc side.
- **`alloc_paged_token_slots_extend`** (`common.py:356`) over-estimates the eviction target by `+len(seqs)*page_size` (each request may open a partial page) and routes into the Triton `alloc_extend` kernel (`allocator.py:409`, kernel at `allocator.py:240`), which fills each request's partial last page, then full fresh pages, then a new partial page (Parts 1/2/3 in the kernel, `allocator.py:274-323`). `last_loc` (the last occupied slot of the prefix) tells it where the partial page continues.
- **`write_cache_indices`** (`common.py:104`) writes both the **prefix** slots and the **new** slots into the request's `req_to_token` row, via the Triton `write_req_to_token_pool_triton` kernel (`common.py:53`) or a Python fallback (`common.py:135`). After this, `req_to_token[req_pool_idx, 0:seq_len]` is the full slot list, ready for attention to read.

```
ReqToTokenPool row for req R after write_cache_indices:
  col:   0          prefix_len            seq_len
         │ prefix slots │ newly-alloc slots │
         │ (tree match) │ (alloc_extend)    │
         └──────────────┴───────────────────┘
                        out_cache_loc fills this part
```

dLLM note: in `DLLM_EXTEND` the same machinery allocates slots for a whole denoising block, including positions that are still masked. The `req_pool_idx` reuse path (`memory_pool.py:163`) is what keeps a block's slots stable across the many denoising forward passes that touch them before tokens are committed.

## 5. Decode: one step of the running batch

When there is no prefill work, `get_next_batch_to_run` advances the running batch:

```
get_next_batch_to_run()                                    scheduler.py:2604
  └─ update_running_batch(self.running_batch)              scheduler.py:2909
       └─ batch.prepare_for_decode()                       scheduler.py:2995
            └─ alloc_for_decode(self, token_per_req=1)      schedule_batch.py:2335 → common.py:524
```

`prepare_for_decode` (`schedule_batch.py:2280`) sets `forward_mode = DECODE`, moves the last sampled tokens into `input_ids` (`schedule_batch.py:2328`), and calls `alloc_for_decode(self, token_per_req=1)` (`schedule_batch.py:2335`). `alloc_for_decode` (`common.py:524`) appends `token_per_req` slots per request: page_size 1 → `alloc_token_slots(bs*token_per_req)` (`common.py:538`); paged → `alloc_paged_token_slots_decode` (`common.py:495`) → Triton `alloc_decode` (`allocator.py:459`) using `last_loc = req_to_token[req_pool_indices, seq_lens-1]` (`common.py:541`). Then a single `req_to_token_pool.write((req_pool_indices, locs), out_cache_loc)` records the new tail slot per request (`common.py:559`).

dLLM note: for diffusion decoding `token_per_req` is the number of positions advanced per denoising iteration. The slot arithmetic is identical; what differs upstream is which positions are "new" each step and how many forward passes reuse the same slots before commit. KV reuse across denoising steps is valid only where the model treats already-committed tokens as fixed context — masked positions whose predictions change between steps must have their KV re-written, not reused.

## 6. The write path: getting K/V bytes into the pool during the forward

Allocation only reserves slot indices (`out_cache_loc`). The actual K/V tensors are scattered into those slots inside the attention layer during `run_batch` (`scheduler.py:3007`):

```
run_batch(batch)                                           scheduler.py:3007
  → model_runner.forward(forward_batch)                    # forward_batch carries out_cache_loc + token_to_kv_pool
      → model layers → RadixAttention.forward / unified_attention_with_output   radix_attention.py:150
          → attn_backend.forward(..., save_kv_cache=True)
              → forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v, k_scale, v_scale)   flashinfer_backend.py:805
                  → k_buffer[layer][cache_loc] = k ;  v_buffer[layer][cache_loc] = v                     memory_pool.py:1047
```

`cache_loc` is exactly the `out_cache_loc` produced by the allocator in §4/§5 (`flashinfer_backend.py:792`). So the attention backend scatters this step's freshly computed K/V into precisely the slots the scheduler reserved, and those same slots are recorded in `req_to_token` for future reads. The read side is `get_kv_buffer(layer_id)` (`memory_pool.py:1044`) handed to the paged-attention kernel together with the page table derived from `req_to_token` (`flashinfer_backend.py:815`). `set_kv_buffer` (`memory_pool.py:1047`) casts to `store_dtype`, applies optional k/v scales, then calls `_set_kv_buffer_impl` (`memory_pool.py:91`), which prefers a fused `store_cache` JIT kernel and otherwise does the plain scatter, splitting K and V onto an alt stream during CUDA-graph capture (`memory_pool.py:116`). MLA writes the fused latent through `set_mla_kv_buffer` (`memory_pool.py:1750`) instead.

This is the only path that touches GPU KV bytes, and it deliberately bypasses the scheduler: `ForwardBatch` got `token_to_kv_pool` directly from the model runner at `forward_batch_info.py:484`.

## 7. Caching results and freeing: after the forward

`process_batch_result` (called at `scheduler.py:1567`) routes into the output-processor mixin, which decides per request what to keep and what to free:

```
process_batch_result → scheduler_output_processor_mixin.py:
   finished req:                 release_kv_cache(req, self.tree_cache)            line 103/256/390/657
   unfinished (kept for reuse):  maybe_cache_unfinished_req(req, self.tree_cache)  line 259/393
   chunked prefill boundary:     maybe_cache_unfinished_req(req, tree_cache, chunked=True)   scheduler.py:2463
```

### 7.1 `cache_unfinished_req` — promote in-flight slots into the shared tree (`radix_cache.py:487`)

Called when a request yields mid-generation (a chunked-prefill boundary, or between extend/decode in some modes). It reads the request's current slots from `req_to_token` (`radix_cache.py:493`), `insert`s the `(token_ids → slots)` mapping into the tree (`radix_cache.py:503`), frees the **duplicate** slots in `[cache_protected_len : new_prefix_len]` that already existed in the tree so two requests never double-own the same physical prefix (`radix_cache.py:513`), re-runs `match_prefix` to get the canonical (possibly newly split) slot tensor, **rewrites the request's `req_to_token` row** to point at the tree-owned slots (`radix_cache.py:518-530`), and moves the lock from the old `last_node` to the new one (`radix_cache.py:538`). `cache_protected_len` (`radix_cache.py:532`) tracks the partial trailing page that lives in `req.prefix_indices` but is not yet page-aligned into the tree, so it is freed later rather than leaked.

### 7.2 `cache_finished_req` — final insert and unlock (`radix_cache.py:440`)

On completion (reached via `release_kv_cache`, `common.py:566`): build the page-aligned `RadixKey` over committed tokens, `insert` it (`radix_cache.py:468`), free slots that duplicate what's already in the tree (`radix_cache.py:472`), free the unaligned tail that cannot be page-stored (`radix_cache.py:481`), and `dec_lock_ref` the request's node so its prefix becomes evictable again (`radix_cache.py:484`). `release_kv_cache` (`common.py:566`) is the scheduler-side wrapper that also returns the `ReqToTokenPool` row (`common.py:619`) and any Mamba state. When prefix caching is off, `ChunkCache.cache_finished_req` (`chunk_cache.py:79`) just frees the whole committed range.

### 7.3 Lock ref-counting — why an in-flight prefix is safe

`inc_lock_ref` (`radix_cache.py:589`) / `dec_lock_ref` (`radix_cache.py:604`) walk node→root; the first lock on a node moves its `len(key)` tokens from `evictable_size_` to `protected_size_`, and the last unlock moves them back. A running request holds a lock on its `last_node` for its whole lifetime, so a concurrent eviction can never pull slots out from under an in-flight forward pass. Only unlocked leaves are ever evictable.

## 8. Eviction: pull-based, triggered by allocation

There is no background eviction thread. Eviction happens lazily, the moment an allocation would otherwise fail, through the `evict_from_tree_cache` call inside every alloc wrapper (`common.py:308, 369, 506`):

```
alloc_token_slots / alloc_paged_*                          common.py:302 / 356 / 495
  └─ evict_from_tree_cache(tree_cache, num_tokens)          common.py:330
        if allocator.available_size() < num_tokens:
            tree_cache.evict(EvictParams(num_tokens=num_tokens))     radix_cache.py:560
               heap over evictable_leaves keyed by strategy priority (LRU/LFU/FIFO/…)   evict_policy.py
               pop leaf → allocator.free(leaf.value) → _delete_leaf(leaf) → maybe re-push parent
  └─ allocator.alloc(num_tokens)                            # now succeeds, or raise OOM
```

`RadixCache.evict` (`radix_cache.py:560`) heapifies `evictable_leaves`, pops the lowest-priority leaf, frees its `value` slots back to the allocator (`radix_cache.py:576`), deletes the node, and re-pushes a parent that just became a childless unlocked leaf (`radix_cache.py:580`). `_update_leaf_status` (`radix_cache.py:783`) keeps the `evictable_leaves` set correct: a node is an evictable leaf iff it is not evicted, not locked, and has no live children. For the SWA hybrid allocator, `evict_from_tree_cache` computes separate full-window and sliding-window deficits and passes both in `EvictParams` (`common.py:339-349`).

## 9. One request's full lifetime (everything together)

```
admit   get_new_batch_prefill → init_next_round_input → match_prefix     reuse prefix slots, lock last_node
        (scheduler.py:2766 → radix_cache.py:360)
prefill prepare_for_extend → alloc_for_extend                            ReqToTokenPool row + new slots
        (schedule_batch.py:1688 → common.py:429)
          ├─ alloc_req_slots          (common.py:398)                    ReqToTokenPool.alloc
          ├─ alloc_token/paged_slots  (common.py:302/356)                evict-then-allocator.alloc
          └─ write_cache_indices      (common.py:104)                    req_to_token[R,:seq]=prefix++new
forward run_batch → set_kv_buffer                                        K/V bytes into k/v_buffer[layer][slot]
        (scheduler.py:3007 → memory_pool.py:1047)
cache   cache_unfinished_req                                            insert into radix tree, dedup-free, re-lock
        (scheduler_output_processor_mixin.py:259 → radix_cache.py:487)
decode  (loop) update_running_batch → prepare_for_decode → alloc_for_decode   one slot/req, write tail, set_kv_buffer
        (scheduler.py:2909 → schedule_batch.py:2280 → common.py:524)
finish  release_kv_cache → cache_finished_req                          insert committed prefix, free tail+dups, dec_lock_ref, free row
        (scheduler_output_processor_mixin.py:103 → common.py:566 → radix_cache.py:440)
evict   (under pressure, on next alloc) evict_from_tree_cache → evict   free unlocked-leaf slots back to allocator
        (common.py:330 → radix_cache.py:560)
```

## 10. Where to look next

- **Host offload (HiCache)**: `hiradix_cache.py` + `memory_pool_host.py` add an L2 CPU tier; the `layer_transfer_counter` hook in `get_key_buffer` (`memory_pool.py:1029`) is where layer-wise host→device loading synchronizes with attention. Constructed at `scheduler.py:944`.
- **Unified multi-component cache**: `unified_radix_cache.py` + `unified_cache_components/` generalize the tree to validate per-component (FULL/SWA/Mamba); this is what `MatchResult.best_match_node` (`base_prefix_cache.py:156`) exists for. Constructed at `scheduler.py:926`.
- **dLLM specifics**: in `python/sglang/srt/dllm/`, confirm exactly which positions are passed in `out_cache_loc` per denoising step and whether masked-position KV is re-written or reused each step — this determines KV-cache reuse validity under denoising and is the key correctness question for any KV optimization targeting diffusion decoding.