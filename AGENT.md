# nano-infer — Agent Guide

## Project Purpose

This is a **learning project**: a minimal LLM inference engine, built from
scratch as a personal version of **nano-vllm / mini-sglang**. The goal is to
understand how modern serving systems (vLLM, SGLang) work by reimplementing
their core ideas in small, readable code — not to be production-grade.

When helping, optimize for:
- **Clarity over cleverness** — this is for learning; prefer explicit,
  well-commented code over opaque optimizations.
- **Incremental steps** — build features one at a time, each verifiable.
- **Explaining the "why"** — when implementing a technique (paged attention,
  continuous batching, etc.), briefly note which paper/system it comes from.

## Scope

Core features to implement (roughly in order):

1. **Model loading & forward pass** — a small HF model (e.g. Qwen/Llama), plain
   PyTorch, no serving yet.
2. **KV cache** — naive first, then **paged KV cache** (vLLM-style blocks).
3. **Scheduler** — continuous batching: prefill / decode phases, admission,
   preemption.
4. **Sampling** — temperature, top-p, top-k, seed handling.
5. **API layer** — minimal OpenAI-compatible or simple HTTP/CLI interface.
6. **Optional extensions** — CUDA graphs, chunked prefill, prefix caching,
   speculative decoding, tensor parallelism, flash/custom kernels.

Keep each feature isolated and understandable; avoid big-bang rewrites.

## Tech Stack

- **Python** (3.10+)
- **PyTorch** — model execution
- **HuggingFace `transformers`** — weights, tokenizer, config only (write the
  forward/serving logic ourselves where practical)
- Reference implementations: `vllm/vllm`, `sgl-project/sglang`,
  `GeeeekExplorer/nano-vllm`

## Code Conventions

- Small, single-responsibility modules. Suggested layout as the project grows:
  ```
  nano_infer/
    engine/        # scheduler, engine loop
    model/         # model definition & forward
    cache/         # KV cache / block manager
    api/           # server / CLI entrypoints
  tests/
  ```
- Type hints on public functions; docstrings for non-obvious logic.
- Comments should explain *invariant and intent* (e.g. "block table maps
  logical token blocks → physical blocks, shared across sequences on fork"),
  not restate the code.
- Avoid heavy abstractions/design patterns until the simple version works.

## Testing & Verification

- Correctness target: **greedy-decoding output must match HuggingFace
  `generate()`** for the same model/prompt. Use this as the golden test.
- Prefer small deterministic tests (fixed seed, tiny model) over flaky
  end-to-end benchmarks.
- Run tests with `pytest`.
- When adding a feature, also add or update a minimal test/demo proving it
  works before moving on.

## Workflow Expectations for the Agent

1. Before writing code, check what already exists (`rg`, `ls`) — this repo
   grows incrementally; don't duplicate or silently restructure modules.
2. Prefer many small commits/steps; state clearly what each change does and
   which vLLM/SGLang concept it mirrors.
3. If a design tradeoff exists (e.g. simplicity vs. throughput), default to
   simplicity and note the alternative in a comment or in the response.
4. Do not add dependencies without asking; keep the dependency surface small
   (torch, transformers, pytest, maybe fastapi later).
5. When unsure about exact semantics of a system being mirrored (e.g. vLLM
   block manager behavior), say so rather than guessing — correctness of the
   *mental model* matters more than matching internals exactly.

## Non-Goals

- Multi-node serving, production hardening, autoscaling.
- Supporting many model architectures — one or two decoder-only models is
  enough.
- Beating benchmarks — performance work is for learning specific techniques,
  not leaderboard numbers.
