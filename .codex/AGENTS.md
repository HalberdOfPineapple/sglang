# AGENTS.md

## Project Overview

This repository is used for research on distributed optimization for diffusion language models (dLLMs), built on top of SGLang.

The main goal is to understand, modify, and evaluate SGLang-based serving/runtime mechanisms for dLLM workloads. The work may involve inference scheduling, distributed execution, communication/computation overlap, batching behavior, memory management, and runtime-level optimizations specific to diffusion-style language model decoding.

Assume the user is familiar with fundamental ML systems concepts such as distributed training/inference, GPU kernels, NCCL communication, tensor/data/pipeline/expert parallelism, batching, profiling, and memory hierarchy. However, the user is still new to practical dLLM systems, so explanations, notes, and design comments should explicitly connect implementation details to dLLM-specific behavior, especially masked-token denoising, iterative decoding, block-wise generation, confidence-based remasking, and non-autoregressive or semi-autoregressive execution patterns.

## Working Style

When making code changes, prefer small, targeted, and reversible modifications. Avoid large refactors unless explicitly requested. Preserve the original SGLang coding style and runtime assumptions whenever possible.

When analyzing the codebase, first identify the relevant SGLang components and summarize the control flow before proposing changes. Pay attention to where request scheduling, model execution, distributed communication, KV/cache management, sampling, token update logic, and worker orchestration are implemented.

When the task involves dLLM behavior, do not blindly assume standard autoregressive LLM decoding semantics. Carefully distinguish between autoregressive next-token decoding and dLLM iterative denoising. In particular, check how masked tokens are represented, how they participate in attention, how each forward step produces logits or token predictions, how decoded tokens are selected, and how embeddings/cache/state are updated between denoising steps.

## Computing Environment

The code is expected to run on a Kubernetes-managed cluster with containerized environments. Treat container-local system paths as temporary unless explicitly stated otherwise.

### Persistent Storage

Use `/cephfs/$USER` and `/cephfs/shared` for persistent storage.

`/cephfs/$USER` is private to the login account and should be used for personal runtime datasets, source code, experiment outputs, checkpoints, logs, notes, and other important working files.

`/cephfs/shared` is shared within the group and should be used only for data or artifacts intended to be accessible by other group members.

Important code, experiment data, notes, and checkpoints must be stored under CephFS. Do not rely on container-local directories for anything important.

### Temporary or Non-Persistent Paths

The container root filesystem and system paths such as `/`, `/tmp`, `/proc`, `/root`, `/home`, `/etc`, `/var`, and `/usr` are temporary from the perspective of container lifecycle. Files stored there may be lost after container restart, shutdown, rescheduling, or rebuild.

`/root` is the home directory and may be used for installing the runtime environment only. Prefer Miniconda over Anaconda because Anaconda is large and may negatively affect deployment, backup, and host operation. Keep total `/root` usage below 200 GB.

Do not store important code, datasets, checkpoints, or experiment results in `/root`, `/home`, `/tmp`, `/etc`, `/var`, or `/usr`.

### High-IOPS Temporary Storage

`/localssd` is an SSD-backed temporary path. It may be useful for read-only datasets, caches, or temporary files with high IOPS requirements. This path is shared, and administrators may clean it when capacity is full. Only place reproducible or backed-up data there.

### Shared Data Path

`/data` is shared by all users. It may be cleaned according to age or capacity policies, for example by periodically removing old files. Do not place important data there. Use this path sparingly and avoid wasting shared space.

### Recommended Layout

Use a layout similar to the following:

```text
/cephfs/$USER/
  sglang-dllm/
    repo/
    notes/
    scripts/
    configs/
    logs/
    outputs/
    checkpoints/
    datasets/
```

Keep source code, notes, scripts, configs, logs, and experiment outputs under `/cephfs/$USER`. Use `/localssd` only for temporary high-IOPS data that can be regenerated or copied again.

## Notes and Documentation

All research notes, code-reading notes, experiment notes, and implementation summaries should be written as Markdown files under `notes/`.

Do not scatter notes across random directories. Do not put important notes only in chat, terminal output, or temporary files.

Markdown notes should avoid unnecessary blank lines between main contents. Use clear headings, concise paragraphs, and compact bullet lists. Prefer dense but readable technical notes over verbose prose.

When creating notes, use descriptive filenames such as:

```text
notes/dllm_inference_flow.md
notes/sglang_scheduler_reading.md
notes/distributed_execution_findings.md
notes/experiment_YYYYMMDD_short_name.md
```

Each note should include enough context to be useful later, including the relevant files/functions, observed behavior, open questions, and next actions.

## dLLM-Specific Guidance

When reasoning about diffusion language models, explicitly track the following concepts:

* How masked tokens are represented in input IDs, embeddings, attention masks, and model outputs.
* Whether masked positions attend to each other, to decoded tokens, and to previous or future positions.
* What one forward pass returns: logits for all masked positions, logits for selected positions, confidence scores, updated hidden states, or other intermediate states.
* How the system chooses which masked tokens to decode or remask at each step.
* Whether token updates are synchronous across a block or sequential within a block.
* How embeddings or cached states change after a masked token is replaced by a concrete token.
* Whether KV cache reuse is valid under the dLLM decoding pattern.
* How batching and scheduling differ from autoregressive decoding.
* How distributed communication patterns change when multiple tokens are updated per step.

Avoid describing dLLMs as ordinary autoregressive LLMs unless the code path explicitly implements autoregressive fallback behavior.

## SGLang-Specific Guidance

Before modifying SGLang internals, identify the relevant runtime path. Pay special attention to:

* Request lifecycle and scheduler behavior.
* Model worker execution flow.
* Tokenization and detokenization boundaries.
* Sampling and token selection logic.
* Batch construction and batch mutation.
* KV/cache allocation, reuse, and eviction.
* Distributed worker initialization and communication groups.
* CUDA graph usage and constraints.
* Interactions with PyTorch, Triton, NCCL, FlashInfer, and other backend libraries.

When introducing dLLM-specific logic, avoid breaking standard SGLang serving behavior unless the change is intentionally isolated to a dLLM path, flag, backend, or experiment branch.

## Experiment and Profiling Guidance

For experiments, save scripts, configs, logs, and summarized results under `/cephfs/$USER/sglang-dllm/` or another persistent CephFS project directory.

Use clear experiment names and record the following:

* Git commit or branch.
* Model name and configuration.
* Number and type of GPUs.
* Parallelism configuration.
* Batch size and sequence length.
* dLLM decoding configuration, including number of denoising steps, block size, mask ratio, remasking policy, and confidence policy if applicable.
* Throughput, latency, memory usage, communication time, and GPU utilization.
* Any failed runs or unexpected behavior.

When profiling, prefer saving profiler traces and derived summaries under persistent storage. Do not leave important traces only in `/tmp`, `/root`, or container-local paths.

## Safety Rules for File Operations

Before deleting, moving, or overwriting files, check whether the path is persistent or temporary.

Never delete large directories blindly. Avoid commands such as:

```bash
rm -rf /*
rm -rf /cephfs/*
rm -rf /data/*
rm -rf /localssd/*
```

When cleanup is necessary, list target files first and restrict deletion to explicit experiment outputs, caches, or known temporary artifacts.

## Dependency and Environment Rules

Prefer installing Python environments under `/root` only when needed for runtime setup, while keeping the total `/root` usage below 200 GB.

For reproducibility, record important installation commands, package versions, CUDA versions, PyTorch versions, SGLang versions, FlashInfer versions, and relevant environment variables in Markdown notes under `notes/`.

If a dependency build fails, preserve the error log in a note or log file under CephFS before attempting destructive cleanup.

## Expected Agent Behavior

When asked to investigate an issue, first summarize the suspected subsystem and then inspect the relevant files. Provide concrete file paths, function names, and call chains where possible.

When asked to implement a change, explain the intended modification briefly, make the smallest practical code change, and mention how to test it.

When asked to write notes, create Markdown files under `notes/` with compact formatting and no unnecessary blank lines between main sections.

When uncertain about dLLM semantics, mark the uncertainty explicitly and inspect the code or paper assumptions instead of guessing.

When proposing optimizations, distinguish clearly between:

* Algorithmic dLLM changes.
* Runtime scheduling changes.
* Distributed communication changes.
* GPU kernel or backend changes.
* Memory/cache management changes.
* Experimental measurement changes.

The default priority is correctness and interpretability first, then performance optimization.
