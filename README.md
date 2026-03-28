# BoggersTheLanguageModel — Thinking System Language Substrate

**BoggersTheLanguageModel** is a production-grade continuous attractor language model built without attention, transformers, or traditional LLM methods. State follows a physical **trajectory**; meaning is **path-dependent**. The architecture is driven by the **Propagate → Relax → Break → Evolve** cycle that powers the TS-OS.

**Source repository:** [github.com/BoggersTheFish/idekatp](https://github.com/BoggersTheFish/idekatp) (clone the repo as `idekatp`; the product name is BoggersTheLanguageModel.)

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
│    ┌─ positional coupling + dynamics.step(S, signal)    │
│    │     (SimpleAttractorDynamics or VectorizedWindow)  │
│    ├─ optional GOAT activation_bonus → per-position signal│
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
| `sandbox.py` | Phase 0 | **BoggersTheLanguageModel** core: training, generation, all dynamics |
| `smoke_test.py` | Phase 0 | 5-assertion integration test (dynamics + TSCore wave cycle) |
| `wave_a_tokenizer.py` | A | tiktoken BPE helpers; training uses `sandbox._build_tokenizer()` |
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
git clone --recurse-submodules https://github.com/BoggersTheFish/idekatp.git
cd idekatp
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
python eval_harness.py --val-fraction 0.2 --max-ticks 11 --output eval_results.json
```

(`--wave-cycles` is a deprecated alias for `--max-ticks`.)

### Start the inference server

```bash
pip install fastapi uvicorn
python inference_server.py --host 0.0.0.0 --port 8000
```

Or with Docker:

```bash
docker compose up
# Service name: boggers-language-model (see docker-compose.yml)
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

Training and inference use **`sandbox._build_tokenizer(mode, vocab_cap)`**, which loads `AttractorTokenizer` from `vendor/ts-llm`:

- **`--tokenizer tiktoken`** — gpt2 BPE up to `--vocab-cap` (default 32768)
- **`--tokenizer fallback`** — same BPE, vocab capped at 512 for fast iteration

The model is constructed with **`vocab_size = tok.n_vocab`**; `model.tokenizer` is set for `encode` / `decode`. `sandbox.FULL_VOCAB` remains an empty legacy shim for old imports.

`wave_a_tokenizer.py` still exposes `make_vocab_and_tokenizer()` for scripts that want a standalone helper.

```python
import sandbox as sb
tok = sb._build_tokenizer("tiktoken", 32768)
model = sb.TorchAttractorLanguageModel(tok.n_vocab, train_window_size=6)
model.tokenizer = tok
```

### Wave B — Vectorized dynamics

`dynamics_vectorized.py` provides `VectorizedWindowDynamics`, swappable with **`--dynamics vectorized`** (replaces `model.dynamics` after construction).

Both **`SimpleAttractorDynamics`** and **`VectorizedWindowDynamics`** implement the same **`step(S, signal) → S`** interface used inside `_single_window_step` (positional coupling first, then one dynamics step with the optional GOAT signal tensor).

- Wraps `MultiHeadDynamics` from `vendor/ts-llm` (low-rank diffusion per head + cross-head coupling)
- `run_window_dynamics_vectorized(S, model, vec_dyn)` remains available for alternate call sites
- `torch.compile` wrapper via `get_compiled()` (cached by shape key)
- Parity tests: both paths produce finite outputs; equations differ by design

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
- `on_batch(model)` — reads `_last_window_tension_curve`, normalises to `[0,1]`, pushes to node activation, calls `ts.propagate_wave()` (skips when language tension is below `high_tension_threshold`)
- When TSCore tension exceeds `evolve_threshold`, calls `ts.factory_evolve()` (appends a stability node — self-improvement tick)
- Optional HTTP POST to `LLM_HOOK_URL` (BoggersTheAI Evolve endpoint) — fire-and-forget, never blocks training
- With **`--use-substrate`**, each epoch logs **`evolves`**, **`last_ts_tension`**, and active vs idle batch counts; if TSCore never fired (all batches below threshold), a **single** per-epoch warning is printed. The same substrate fields are appended to **`--epoch-metrics-csv`** as `tscore_evolves` and `tscore_last_tension`.

```python
from llm_substrate_node import LLMSubstrateNode
substrate = LLMSubstrateNode(model)
# After each training batch:
substrate.on_batch(model)
```

### Wave F — GOAT-TS memory transitions

`goat_memory_transitions.py` wires GOAT-TS-style per-token memory into training when you pass **`--use-goat-memory`**:

- One `Node` per vocabulary index (`vocab_size`); labels are string token IDs
- After each batch, `GoatMemoryManager.tick(contexts)` updates activations and ACTIVE / DORMANT / DEEP transitions
- During **window dynamics**, `_single_window_step` builds a `(B, W, D)` signal from **`activation_bonus(token_id)`** at each position (broadcast across `D`), so GOAT affects the actual forward pass—not only `get_signal()` on the legacy single-token path
- `sweep_config()` — tunable knobs for automated sweeps

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

tok = sb._build_tokenizer("fallback", 512)  # or "tiktoken", 32768
model = sb.TorchAttractorLanguageModel(
    vocab_size=tok.n_vocab,
    state_dim=512,
    train_window_size=6,
    max_window_steps=16,
)
model.tokenizer = tok

# Training path
wids = model.window_ids_from_sequence(token_ids)
S = model.embed_window(wids)
S, logs = model.run_window_dynamics(S, context_ids=wids)  # GOAT uses token ids if enabled
logits = model.readout_window(S.reshape(1, -1))

# Batched trajectory contrastive loss
loss, logits = model.trajectory_contrastive_loss_and_logits(contexts, targets)

# Generation
text = model.generate("the quick brown fox", max_tokens=40)

# Prompt comparison (trajectory distance)
sb.compare_prompts(model, "cats eat fish", "fish eat cats")
```

---

## Training objective

Default: **trajectory contrastive loss** with optional auxiliary terms.

```
L = L_traj + w_token · L_token_aux + α · L_readout_aux

L_traj = mean(ReLU(0.2 − cos(pred, teacher) + cos(pred, negative)))
  pred   = evolved(context window)
  teacher = evolved(shifted window [x2…xW, next_token])
  negative = shuffled teacher in batch

L_token_aux = CE on readout_window(flattened pred window) vs target  (--token-aux-ce, default 0.2)
L_readout_aux = CE on readout(final token row of pred) vs target     (--readout-aux-alpha, default 0.15)
```

The single-state **`readout`** head is what inference and `AttractorStateCache` use; **`readout_window`** is the primary training readout. Use `--loss-mode ce` for classic next-token CE only.

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

Data & tokenizer:
  --corpus PATH              Training text (default: data/corpus.txt)
  --dataset-path PATH        Alias for --corpus (takes precedence if set)
  --val-fraction FLOAT       Held-out fraction for val CE / traj eval (default: 0.05; use 0 to disable)
  --tokenizer {tiktoken,fallback}   BPE mode (default: fallback)
  --vocab-cap INT            Max BPE vocab when using tiktoken mode (default: 32768)
  --seq-len INT              Alias for --window-size
  --batch-size INT           Alias for --trajectory-batch-size
  --shuffle-buffer INT       Pipeline shuffle buffer (default: 2048)

Training:
  --window-size INT          Context window W (default: 6)
  --num-dynamics-steps INT   Max tension-adaptive steps per window (default: 16)
  --trajectory-batch-size INT  Batch size for trajectory mode (default: 16, need ≥2)
  --loss-mode {trajectory,ce}
  --token-aux-ce FLOAT       Aux CE on readout_window in trajectory mode (default: 0.2)
  --readout-aux-alpha FLOAT  Aux CE on single-state readout (default: 0.15; 0 = off)
  --lr, --lr-decay-every, --lr-gamma
  --epoch-copies INT         Repeat training lines per epoch
  --seed INT

Device & checkpointing:
  --device auto|cpu|cuda|cuda:N
  --resume-checkpoint PATH
  --save-every N             Save every N optimizer steps (0 = final only)
  --checkpoint-dir PATH      Default: ./checkpoints

Integrations:
  --use-substrate            TSCore LLMSubstrateNode after each batch
  --use-goat-memory          GoatMemoryManager + window-path signal injection
  --dynamics {simple,vectorized}

Logging:
  --epoch-metrics-csv PATH   Per-epoch CSV (see below)
  --log-hard-batch-loss-above FLOAT
  --baseline-out PATH        Phase-0 baseline snapshot text file

Misc:
  --quick-test               Window sanity checks, exit
```

### Epoch metrics CSV columns

When `--epoch-metrics-csv` is set, each row includes: `epoch`, `loss_mode`, `mean_loss`, **`train_ce`** (mean batch CE from `readout_window` logits during the epoch), **`val_ce`** (held-out `mean_cross_entropy_eval`, empty if no val), `train_traj_contrast` (last training batch trajectory loss snapshot), **`val_traj_contrast`** (full val-set mean when val exists), `mean_final_step_tension`, `max_batch_loss`, `lr`, `global_step`, **`tscore_evolves`**, **`tscore_last_tension`** (0 if substrate disabled).

**Validation perplexity:** `PPL_val = exp(val_ce)` when `val_ce` is finite.

### A/B example (GOAT on vs off)

Use the same `--seed`, corpus, and hyperparameters; only add `--use-goat-memory` for the treatment run. Log with `--epoch-metrics-csv` and compare `val_ce` / `mean_loss` curves.

---

## Submodules

Clone with:

```bash
git clone --recurse-submodules https://github.com/BoggersTheFish/idekatp.git
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
