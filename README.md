# woke-baby-llm — Thinking System Language Substrate

A production-grade continuous attractor language model built without attention, transformers, or traditional LLM methods. State follows a physical **trajectory**; meaning is **path-dependent**. The architecture is driven by the **Propagate → Relax → Break → Evolve** cycle that powers the TS-OS.

**Repository:** [github.com/BoggersTheFish/woke-baby-llm](https://github.com/BoggersTheFish/woke-baby-llm)

---

## Architecture overview

```
Corpus / Token stream
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  TorchAttractorLanguageModel  (sandbox.py)              │
│                                                         │
│  embed_window(W ids) → (W, D) embedding matrix          │
│         │                                               │
│         ▼                                               │
│  run_window_dynamics()  ← tension-adaptive Euler loop   │
│    ┌─ step_state_batch  (diffusion + tanh + damping)    │
│    ├─ compute_window_tension  (geometry or entropy)     │
│    └─ early-exit / noise / break on tension threshold   │
│         │                                               │
│  readout_window() → logits → sample_next_token_id()     │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────┐   ┌───────────────────────────┐
│  AttractorStateCache │   │  LLMSubstrateNode          │
│  (state_cache.py)    │   │  (llm_substrate_node.py)   │
│  O(1) per token      │   │  TS-Core graph integration │
│  fast + slow memory  │   │  Propagate → Evolve hook   │
└──────────────────────┘   └───────────────────────────┘
```

**No attention. No transformer blocks. No external model weights.**

The network contains only:
- A learnable diffusion matrix `A` (negative-definite, stable dynamics)
- A `tanh`-bounded nonlinearity with damping
- Fast memory (per-token evolved state) + slow memory (exponential average)
- A linear readout from the symplectic midpoint of fast-start / fast-end states
- A scalar tension `T ≈ |ΔE| + λ(1 − cos(fast, slow)) + μH(logits)` that controls step count, noise, and break events

---

## Project structure

| File / Directory | Wave | Purpose |
|---|---|---|
| `sandbox.py` | Phase 0 | Single-file reference model: training, generation, all dynamics |
| `smoke_test.py` | Phase 0 | 5-assertion integration test (dynamics + TSCore wave cycle) |
| `wave_a_tokenizer.py` | A | tiktoken BPE tokenizer (32k+ vocab) with word-list fallback |
| `dynamics_vectorized.py` | B | Vectorized `MultiHeadDynamics` window step; `torch.compile` wrapper |
| `state_cache.py` | C | Rolling attractor state cache — O(1) per token at inference |
| `data_pipeline.py` | D | Streaming sharded DataLoader (txt / JSONL, multi-worker) |
| `llm_substrate_node.py` | E | Registers model as a native TSCore node; Evolve hook |
| `goat_memory_transitions.py` | F | GOAT-TS ACTIVE → DORMANT → DEEP token state transitions |
| `inference_server.py` | G | FastAPI inference server — OpenAI-compatible `/v1/completions` |
| `Dockerfile` | G | CPU Docker image (swap whl URL for CUDA wheel) |
| `docker-compose.yml` | G | One-command deploy |
| `eval_harness.py` | H | Perplexity + tension metrics + 11-tick WaveCycleRunner |
| `vendor/GOAT-TS` | — | Constraint-graph engine (submodule) |
| `vendor/TS-Core` | — | UniversalLivingGraph + WaveCycleRunner (submodule) |
| `vendor/ts-llm` | — | Tokenizer, hierarchical dynamics, attractor LLM package (submodule) |
| `docs/API_DISCOVERY.md` | — | Verified entrypoints for all three vendored repos |
| `docs/BASELINE.md` | — | Phase 0 baseline recording instructions |

---

## Quick start

### Requirements

- Python 3.10+
- [PyTorch](https://pytorch.org/) (CPU or CUDA)

```bash
git clone --recurse-submodules https://github.com/BoggersTheFish/woke-baby-llm.git
cd woke-baby-llm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Train the baseline model

```bash
python sandbox.py
```

With trajectory contrastive loss (recommended), LR schedule, and metrics CSV:

```bash
python sandbox.py \
  --epoch-metrics-csv metrics.csv \
  --lr 0.001 \
  --lr-decay-every 15 \
  --lr-gamma 0.7 \
  --val-fraction 0.1
```

### Run the smoke test

Verifies dynamics, tension, training step, and TSCore wave cycle all pass:

```bash
python smoke_test.py
```

### Run the evaluation harness

```bash
python eval_harness.py --val-fraction 0.2 --wave-cycles 11 --output eval_results.json
```

### Start the inference server

```bash
pip install fastapi uvicorn
python inference_server.py --host 0.0.0.0 --port 8000
```

Or with Docker:

```bash
docker compose up
```

Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness + tension metrics |
| POST | `/v1/completions` | OpenAI-compatible text completion |
| POST | `/v1/generate` | Direct generate call |
| GET | `/metrics/tension` | Last window tension curve |
| POST | `/ts/propagate` | Trigger one TSCore wave propagation |
| GET | `/ts/tension` | Current TSCore graph tension |

---

## Wave-by-wave implementation log

### Phase 0 — Substrate initialisation

- Synced workspace from `origin/main` (GitHub); tagged `phase-0-baseline`
- Added git submodules: `vendor/GOAT-TS`, `vendor/TS-Core`, `vendor/ts-llm`
- Verified all entrypoints; documented in `docs/API_DISCOVERY.md`
- `smoke_test.py`: 5 assertions all pass on CPU in ~2 s

### Wave A — Tokenizer

`wave_a_tokenizer.py` replaces the hard-coded `_VOCAB_BLOB` with:

- **tiktoken BPE** (default: gpt2 encoding, `vocab_cap=32768`) — real subword tokens
- **Word-list fallback** (offline, deterministic) — uses existing `FULL_VOCAB`
- `recommended_state_dim(vocab_cap)` — scaling table: 512→512, 32k→2048, 64k→4096
- `encode_corpus(sentences, tok)` — lazy encoding for the data pipeline

```python
from wave_a_tokenizer import make_vocab_and_tokenizer
vocab_list, tok = make_vocab_and_tokenizer(vocab_cap=32768)
ids = tok.encode("the quick brown fox")
text = tok.decode(ids)
```

### Wave B — Vectorized dynamics

`dynamics_vectorized.py` provides `VectorizedWindowDynamics` — a drop-in for `model.run_window_dynamics()`:

- Wraps `MultiHeadDynamics` from `vendor/ts-llm` (low-rank diffusion per head + cross-head coupling)
- Reshapes `(B, W, D)` → `(B·W, D)` for a single batched matrix multiply per step — no Python loop
- `torch.compile` wrapper via `get_compiled()` (cached by shape key)
- `run_window_dynamics_vectorized(S, model, vec_dyn)` writes to `model._last_window_tension_curve` for API compatibility
- Parity tests confirm both paths produce finite outputs; cosine similarity reported (different equations → intentional divergence)

### Wave C — State cache

`state_cache.py` provides O(1)-per-token inference:

- `AttractorStateCache` holds `fast_state (D,)` + `slow_memory (D,)` + rolling `phrase_table`
- `step(token_id)` — one dynamics step + slow memory update + phrase table eviction; no full-window re-embedding
- `logits()` — symplectic readout from cached state; no window rebuild
- `warmup(prompt_ids)` — seed cache from prompt before generation
- `generate_with_cache(model, cache, prompt, ...)` — drop-in for `model.generate()`

```python
from state_cache import AttractorStateCache, generate_with_cache
cache = AttractorStateCache(model)
text = generate_with_cache(model, cache, prompt="the cat sat", max_tokens=30)
```

### Wave D — Data pipeline

`data_pipeline.py` replaces the in-memory shuffle with a streaming loader:

- Accepts lists of `.txt` / `.jsonl` files or directories; parses JSONL `{"text": "..."}` automatically
- Sliding-window `(context, target)` pair builder on token id streams
- Shuffle buffer (in-memory random shuffle of `shuffle_buffer` pairs)
- Multi-shard round-robin (`shard_id` / `num_shards`) for data-parallel workers
- `epoch_batches()` yields `(contexts, targets)` directly compatible with `trajectory_contrastive_loss_and_logits`

```python
from data_pipeline import AttractorDataPipeline
pipe = AttractorDataPipeline(sources=["data/corpus.txt"], model=model, batch_size=16)
for contexts, targets in pipe.epoch_batches():
    loss, _ = model.trajectory_contrastive_loss_and_logits(contexts, targets)
```

### Wave E — TS-OS integration shim

`llm_substrate_node.py` closes the language → TS-OS feedback loop:

- Registers `"llm_substrate"` as a native node in `TSCore` with an edge from `"ts_native"`
- `on_batch(model)` — reads `_last_window_tension_curve`, normalises to `[0,1]`, pushes to node activation, calls `ts.propagate_wave()`
- When TSCore tension exceeds `evolve_threshold`, calls `ts.factory_evolve()` (appends a stability node — self-improvement tick)
- Optional HTTP POST to `LLM_HOOK_URL` (BoggersTheAI Evolve endpoint) — fire-and-forget, never blocks training

```python
from llm_substrate_node import LLMSubstrateNode
substrate = LLMSubstrateNode(model)
# After each training batch:
substrate.on_batch(model)
```

### Wave F — GOAT-TS memory transitions

`goat_memory_transitions.py` wires GOAT-TS `memory_manager` into per-token activation tracking:

- One `Node(activation, state: MemoryState)` per vocabulary token
- `tick(contexts)` — boosts usage-proportional activation for seen tokens, then applies exponential decay + ACTIVE / DORMANT / DEEP state transitions
- `activation_bonus(token_id)` — returns a `[0, bonus_scale]` float for ACTIVE tokens (signal injection boost)
- `sweep_config()` — returns tunable knobs (`decay_rate`, `active_threshold`, `dormant_threshold`, `ticks_to_deep`, `bonus_scale`) for automated Wave F hyperparameter sweeps

```
State machine (per token):
  high usage → ACTIVE  (activation ≥ 0.5)
  low  usage → DORMANT (activation < 0.1)
  3 ticks at DORMANT → DEEP  (excluded from bigram bias)
```

### Wave G — Inference server

`inference_server.py` exposes the model via FastAPI:

- OpenAI-compatible `/v1/completions` (drop-in for any client that targets the OpenAI API)
- Uses `AttractorStateCache` for O(1)-per-token latency
- TSCore sidecar: `/ts/propagate` and `/ts/tension` endpoints
- Thread-safe: `threading.Lock()` wraps generate calls
- `Dockerfile` (CPU; swap whl URL for CUDA) + `docker-compose.yml` with healthcheck, volume mounts for checkpoints and corpus

```bash
# Test without FastAPI installed:
python inference_server.py --self-test
```

### Wave H — Evaluation harness

`eval_harness.py` provides the full evaluation loop:

- `compute_perplexity(model, dataset)` — token-level PPL = exp(mean CE)
- `compute_mean_tension(model, dataset)` — mean final window tension across batches
- `compute_traj_contrast(model, dataset)` — mean trajectory contrastive loss
- `run_wave_cycle(model, substrate, dataset, max_ticks=11)` — feeds language batches into TSCore, runs `run_until_stable(max_ticks)`, returns before/after tension delta and evolve count

**Phase 0 eval results (untrained model, 51-line corpus):**

| Metric | Value |
|--------|-------|
| Baseline val PPL | 506 |
| Mean window tension | 0.318 |
| TSCore tension (before WaveCycle) | 0.149 |
| TSCore tension (after 11-tick WaveCycle) | 0.0005 |
| Evolve events triggered | 5 |

TSCore converges cleanly. High PPL is expected for an untrained model — the harness is the measurement instrument.

---

## Core model API

```python
import sandbox as sb

model = sb.TorchAttractorLanguageModel(
    vocab=sb.FULL_VOCAB,      # list[str] — 512 words (baseline) or tiktoken ids
    state_dim=512,            # D — embedding / state dimension
    train_window_size=6,      # W — context window
    max_window_steps=16,      # max tension-adaptive steps per window
)

# Training path
wids = model.window_ids_from_sequence(token_ids)    # list[int] → list[int]
S = model.embed_window(wids)                         # (W, D)
S, logs = model.run_window_dynamics(S)               # (W, D), metrics
logits = model.readout_window(S.reshape(1, -1))      # (V,)

# Batched trajectory contrastive loss
loss, logits = model.trajectory_contrastive_loss_and_logits(contexts, targets)

# Generation
text = model.generate("the quick brown fox", max_tokens=40)

# Prompt comparison (trajectory distance)
sb.compare_prompts(model, "cats eat fish", "fish eat cats")
```

---

## Training objective

Default: **trajectory contrastive loss** with optional auxiliary cross-entropy.

```
L = L_traj + α · L_ce_aux

L_traj = mean(ReLU(0.2 − cos(pred, teacher) + cos(pred, negative)))
  pred   = evolved(context window)
  teacher = evolved(shifted window [x2…xW, next_token])
  negative = shuffled teacher in batch

L_ce_aux = cross-entropy on readout_window(pred)  (keeps readout trainable)
```

Use `--loss-mode ce` for classic next-token CE only.

---

## Tension semantics

Tension `T` is a scalar computed after each dynamics step:

```
T = |ΔE_state| + λ · (1 − cos(fast, slow)) + μ · H(readout_logits)
```

| T < tol | Early exit — attractor is stable |
|---------|----------------------------------|
| T > high | Inject noise — push out of shallow attractor |
| T > break_thresh | Break perturbation — reset trajectory |

Window-level tension uses geometry only by default (`WINDOW_TENSION_USE_ENTROPY = False`):

```
T_window = 1 − mean_cos(token_states, mean_direction)
```

---

## CLI reference

```
python sandbox.py [options]

Training:
  --corpus PATH              Training corpus file (default: data/corpus.txt)
  --val-fraction FLOAT       Fraction held out for validation CE (default: 0)
  --seed INT                 Random seed
  --epoch-copies INT         Repeat corpus N times per epoch before shuffle
  --window-size INT          Context window width W (default: 6)
  --num-dynamics-steps INT   Max window dynamics steps (default: 16)
  --trajectory-batch-size INT  Batch size for contrastive loss (default: 16)
  --loss-mode {trajectory,ce}  Training objective (default: trajectory)
  --token-aux-ce FLOAT       Weight of aux CE loss in trajectory mode (default: 0.2)
  --lr FLOAT                 Learning rate (default: 0.001)
  --lr-decay-every INT       Apply StepLR every N epochs (0 = disabled)
  --lr-gamma FLOAT           StepLR gamma (default: 0.5)

Logging:
  --epoch-metrics-csv PATH   Append per-epoch metrics to CSV
  --log-hard-batch-loss-above FLOAT  Print context when batch loss exceeds threshold
  --baseline-out PATH        Save Phase 0 baseline block to file

Misc:
  --quick-test               Run window sanity checks only (no training)
  --baseline-out PATH        Record Phase 0 baseline snapshot
```

---

## Submodules

Clone with:

```bash
git clone --recurse-submodules https://github.com/BoggersTheFish/woke-baby-llm.git
```

If already cloned:

```bash
git submodule update --init --recursive
```

| Path | Repo | Branch | Role |
|------|------|--------|------|
| `vendor/GOAT-TS` | [GOAT-TS](https://github.com/BoggersTheFish/GOAT-TS) | `main` | Constraint-graph engine, tension semantics, memory transitions |
| `vendor/TS-Core` | [TS-Core](https://github.com/BoggersTheFish/TS-Core) | `master` | UniversalLivingGraph, WaveCycleRunner (Rust + Python fallback) |
| `vendor/ts-llm` | [ts-llm](https://github.com/BoggersTheFish/ts-llm) | `main` | Tokenizer (tiktoken), hierarchical fast/slow dynamics, attractor LLM package |

To update a submodule to latest:

```bash
git -C vendor/<name> pull
git add vendor/<name>
git commit -m "chore: bump <name> submodule"
```

---

## License

[MIT](LICENSE)
