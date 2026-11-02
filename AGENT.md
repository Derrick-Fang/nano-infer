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
- Reference implementations:
  - `../nano-vllm` — **model/layers 层**的移植来源（`python/layers/`,
    `python/model/`）
  - `../mini-sglang` — **runtime/引擎层**的结构与命名基准（见下方
    "Runtime Design" 一节）
  - `vllm/vllm`, `sgl-project/sglang` — 概念溯源用

## Code Conventions

- Small, single-responsibility modules. Layout follows the mini-sglang
  aligned design (see "Runtime Design" below); model/layers code stays a
  nano-vllm port:
  ```
  python/          # 包名就叫 python（历史原因）
    core.py        # SamplingParams / Req / Batch（对齐 minisgl/core.py）
    engine/        # Engine（模型执行器）、EngineConfig、Sampler
    scheduler/     # Scheduler、Prefill/DecodeManager、TableManager、CacheManager
    kvcache/       # KV cache pool 与 prefix cache（后期）
    llm.py         # 离线入口 LLM.generate()
    layers/        # nano-vllm 移植：linear/norm/rotary/embedding
    model/         # nano-vllm 移植：qwen3.py、loader.py
  tests/
  ```
- Type hints on public functions; docstrings for non-obvious logic.
- Comments should explain *invariant and intent* (e.g. "block table maps
  logical token blocks → physical blocks, shared across sequences on fork"),
  not restate the code.
- Avoid heavy abstractions/design patterns until the simple version works.

## Runtime Design（对齐 mini-sglang，离线版）

**基准**：`../mini-sglang`（本地仓库）。runtime 层的模块划分、类名、
术语全部向它看齐；**模型/layers 层不做改动**，保持 nano-vllm 移植现状。
只做**离线（offline）**版本：不涉及 online serving、zmq/IPC、多进程、
tokenizer/detokenizer 独立进程——这些在 mini-sglang 里对应
`server/`、`message/`、`tokenizer/`、`distributed/`，**一律不实现**。

### 目录与类名映射

```
python/
├── core.py               # 对齐 minisgl/core.py
│   ├── SamplingParams    # dataclass: temperature/top_k/top_p/ignore_eos/max_tokens, is_greedy
│   ├── Req               # 一个生成请求（对应 vLLM 的 Request / nano-vllm 的 Sequence）
│   └── Batch             # reqs + phase: "prefill" | "decode"
├── engine/
│   ├── config.py         # EngineConfig: model_path, dtype, max_running_req, page_size, ...
│   ├── engine.py         # Engine: 模型执行器（持有 model/sampler/kv_pool/page_table）
│   │                     #   forward_batch(batch, sample_args) -> ForwardOutput
│   └── sample.py         # Sampler + BatchSamplingArgs（先只实现 greedy）
├── scheduler/
│   ├── config.py         # SchedulerConfig(EngineConfig): + max_extend_tokens
│   ├── scheduler.py      # Scheduler: normal_loop 主循环，串联各 Manager
│   ├── prefill.py        # PrefillManager + PrefillAdder（ChunkedReq 后期再加）
│   ├── decode.py         # DecodeManager: running_reqs 集合
│   ├── table.py          # TableManager: table_idx 分配 + token_pool（KV 阶段）
│   ├── cache.py          # CacheManager: page 分配/释放（KV 阶段）
│   └── utils.py          # PendingReq
├── kvcache/              # KV 阶段：base.py(BaseKVCachePool) / naive_cache.py / mha_pool.py
└── llm.py                # LLM(Scheduler): 离线入口 generate(prompts, sampling_params)
```

### 核心命名约定（术语表）

| mini-sglang 术语 | 含义 | 备注 |
|---|---|---|
| `Req` / `uid` | 请求 / 请求 id | 不用 Request/Sequence/req_id |
| `input_ids` | 完整 token 序列（prompt+生成） | 单一增长序列，见下 |
| `cached_len` | 已有 KV 的前缀长度 | 无 KV cache 时恒 0 或等于 device_len |
| `device_len` | 已在引擎里的 token 数 | `remain_len = max_device_len - device_len` |
| `extend_len` | 本次 prefill 前向的 token 数 | `= device_len - cached_len` |
| `complete_one()` | 前向一次后记账 | `cached_len = device_len; device_len += 1` |
| `can_decode` | 是否还能继续生成 | `remain_len > 0` |
| `Batch.phase` | `"prefill"` / `"decode"` | 字符串字面量，不用枚举 |
| `pending_list` / `running_reqs` | 等待队列 / 运行集合 | 分别归 Prefill/DecodeManager 管 |
| `schedule_next_batch()` | 各 Manager 的调度入口 | prefill 优先于 decode |
| `page` / `page_table` / `page_size` | KV cache 分页 | 用 page 不用 block（sglang 术语） |
| `table_idx` / `token_pool` | 请求在 page_table 的行号 / token 暂存池 | KV 阶段引入 |
| `prefill_budget` (`max_extend_tokens`) | 单次 prefill 的 token 预算 | chunked prefill 用 |
| `inflight_tokens` | decode 在途请求的预留量 | 准入估算用 |
| `ForwardOutput` | `forward_batch` 的返回 | next_tokens（gpu/cpu 之分去掉，单机单份） |

### 生命周期模型（与 nano-vllm 的关键差异）

- **Req 不携带 status 枚举**。请求处于什么阶段由"它在哪个 Manager 的
  哪个队列里"表达：`pending_list`（等待 prefill）→ `running_reqs`
  （decode 中）→ 从两边移除（结束）。这取代了现有 `Request` 的
  `RequestStatus` 状态机——`python/engine/request.py` 将在 R1 被
  `core.py` 取代并删除。
- 停止判定（eos / max_tokens）在 Scheduler 处理前向结果时做
  （对齐 `_process_last_data`），eos 由 scheduler 从 tokenizer 侧持有。

### 分阶段实施路线（offline）

- **R1 `core.py`（已完成）**：`SamplingParams` / `Req` / `Batch` +
  纯逻辑测试；`python/engine/request.py` 已删除。Req 存 `list[int]`，
  长度记账用 cached_len/device_len/max_device_len 三整数。
- **R2 naive Engine + 离线入口（已完成）**：`Engine.forward_batch`
  无 KV cache、逐 req 全量重算；`Sampler` 只支持 greedy（LLM 入口断言
  `is_greedy`）；`llm.py` 的 `LLM.generate()` 走"全体 prefill → 逐步
  decode"最短路径。**golden test 已通过：混合 batch 的 greedy 输出
  逐 token 对齐 HF `generate()`（`tests/test_greedy_decode.py`）。**
- **R3 Scheduler + 连续批处理（已完成）**：`PrefillManager` /
  `PrefillAdder` / `DecodeManager` / `PendingReq` / `Scheduler.normal_loop`
  （同步：schedule → forward → process，无 overlap）；`LLM` 改为继承
  `Scheduler`。准入只受 `max_running_req` 限制（无内存约束，因为还没
  有 KV cache）；token 预算（`max_extend_tokens`）已保留但 R3 无
  `ChunkedReq`，单条超预算的 prompt 整批放行。测试：名额受限多波次、
  各请求不同 max_tokens、decode 中途异步到达，输出均逐 token 对齐 HF
  （`tests/test_scheduler.py` + `tests/test_continuous_batching.py`）。
- **R4 KV cache**：先 naive（每请求连续增长的 KV），再 paged：
  `TableManager` + `page_table` + `MHAPool` + `CacheManager`，届时引入
  `table_idx`/`token_pool`/`out_loc`，prefill 预算与 `ChunkedReq`。
  RadixCache（prefix caching）作为可选项最后做。
- **R5 采样完整化**：temperature / top_k / top_p（`BatchSamplingArgs`），
  种子处理。
- **R6（可选）** attention backend 抽象与全局 `Context`：只有当
  paged attention 的 metadata 传递开始污染模型签名时才引入
  （对齐 `minisgl/core.py` 的 Context + `attention/`）。

### 对 mini-sglang 的有意偏离（实现时必须在注释中标注）

1. **无消息层**：mini-sglang 离线模式内部仍走 `UserMsg`/`DetokenizeMsg`
   编解码；我们直接让 `LLM` 把 `PendingReq` 喂给 `PrefillManager`，
   省掉整个 `message/` 包。
2. **无 overlap scheduling**：只实现 `normal_loop`，且是完全同步的
   （CPU 上 stream/event 无意义）。
3. **设备无关**：mini-sglang 是 CUDA-only（pin_memory、stream、event、
   pynccl）；本项目当前在 CPU 上跑，相关代码一律不写，不为它们留空壳。
4. **无 CUDA graph / GraphRunner / dummy_req**。
5. **`Req.input_ids` 用 `list[int]`** 而非 host tensor：在引入
   `token_pool`（R4）之前没有设备侧 token 存储，list 更简单；R4 时再评估
   是否换成 tensor。
6. **config 简化**：`EngineConfig`/`SchedulerConfig` 只保留当前用到的字段
   （model_path, dtype, max_running_req, page_size, max_extend_tokens, ...），
   不抄全量字段。

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
