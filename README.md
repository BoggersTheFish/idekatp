# BoggersTheLanguageModel — Thinking System Language Substrate

**BoggersTheLanguageModel** is a production-grade continuous attractor language model built without attention, transformers, or traditional LLM methods. State follows a physical **trajectory**; meaning is **path-dependent**. The architecture is driven by the **Propagate → Relax → Break → Evolve** cycle that powers the TS-OS.

**Primary repository:** [github.com/BoggersTheFish/BoggersTheLLM](https://github.com/BoggersTheFish/BoggersTheLLM). **Alternate mirror:** [github.com/BoggersTheFish/idekatp](https://github.com/BoggersTheFish/idekatp). The product name is **BoggersTheLanguageModel**.

---

## Architecture overview

```
Corpus / Token stream
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  TorchAttractorLanguageModel  (sandbox.py)              │
│                                                         │
│  embed_window / embed_windows_batch → (W, D) or (B, W, D) │
│         │                                               │
│         ▼                                               │
│  run_window_dynamics()  ← outer loop (≤ max_window_steps; optional early exit) │
│    ┌─ positional coupling + dynamics.step(S, signal)    │
│    │     (SimpleAttractorDynamics or VectorizedWindow)  │
│    ├─ optional GOAT activation_bonus → per-position signal│
│    ├─ compute_tension_window (geometry ± entropy)       │
│    │     (alias: compute_window_tension)                │
│    └─ tension-driven breaks (Phase 2: directional escape) │
│         + GOAT transition + high-T renorm                  │
│         │                                               │
│  readout_window() → logits (training / trajectory)      │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────┐   ┌───────────────────────────┐
│  AttractorStateCache │   │  LLMSubstrateNode          │
│  (state_cache.py)    │   │  (llm_substrate_node.py)   │
│  same run_window_    │   │  TS-Core graph integration │
│  dynamics per step   │   │  Propagate → Evolve hook   │
│  readout(fast/slow)  │   │                            │
└──────────────────────┘   └───────────────────────────┘
```

**No attention. No transformer blocks. No external model weights.**

The network contains only:
- A learnable diffusion matrix `A` (negative-definite, stable dynamics) in the **simple** path (`--dynamics simple`), or **low-rank multi-head** diffusion in the **default** **`--dynamics vectorized`** path
- A `tanh`-bounded (simple window step) or **cubic** (vectorized) nonlinearity with damping
- **Window path:** state `S` is `(B, W, D)`; **token path** (`evolve_token`): fast/slow `(D,)` with symplectic blend for `readout`
- **`readout_window`:** linear map `W·D → vocab` (training default); **`readout`:** `D → vocab` (cache / `next_token_logits`)
- **Two tension scalars:** `compute_tension_window(S)` after each window step (geometry + optional entropy); `compute_tension(fast, slow, logits, …)` inside `evolve_token` (energy drift + alignment + entropy)

---

## Project structure

| File / Directory | Wave | Purpose |
|---|---|---|
| `sandbox.py` | Phase 0+ | **BoggersTheLanguageModel** core: training, generation, dynamics |
| `phase05_config.py` | 0.5 | `Phase05Config`: tension weights, batch CSV, adaptive dt, neg-def diffusion |
| `phase1_config.py` | 1 | `Phase1Config`: multi-head drift, window interaction matrix `C`, diversity loss |
| `phase2_config.py` | 2 | `Phase2Config`: directional breaks, residual mixing, `C` reg, head tension weights |
| `smoke_test.py` | Phase 0 | 5-assertion integration test (dynamics + TSCore wave cycle) |
| `tests/test_embed_windows_batch.py` | — | Parity check: `embed_windows_batch` vs stacking `embed_window` per row |
| `wave_a_tokenizer.py` | A | tiktoken BPE helpers; training uses `sandbox._build_tokenizer()` |
| `dynamics_vectorized.py` | B | `VectorizedWindowDynamics`: `step(S, signal)` only; `forward` disabled; `run_window_dynamics_vectorized` → `model.run_window_dynamics` |
| `state_cache.py` | C | Rolling cache: `run_window_dynamics` on `(1,W,D)` aligned with training; `logits()` via `readout` + fast/slow |
| `scripts/ts_workflow_smoke.py` | — | Smoke: `AttractorStateCache` + simple/vectorized `run_window_dynamics` (`.venv/bin/python scripts/ts_workflow_smoke.py`) |
| `data_pipeline.py` | D | Streaming sharded DataLoader (txt / JSONL, multi-worker) |
| `data/generate_corpus.py` | — | Deterministic synthetic `.txt` corpus (tiktoken-sized); CLI + `sandbox` fallback |
| `data/hf_remote_corpus.py` | — | TinyStories / FineWeb-Edu → cached `.txt` for training (`--dataset-source`) |
| `data/__init__.py` | — | Package marker for `data.generate_corpus` imports |
| `llm_substrate_node.py` | E | Registers model as a native TSCore node; Evolve hook |
| `goat_memory_transitions.py` | F | GOAT-TS ACTIVE → DORMANT → DEEP token state transitions |
| `inference_server.py` | G | FastAPI inference server — OpenAI-compatible `/v1/completions` |
| `Dockerfile` | G | CPU Docker image (swap whl URL for CUDA wheel) |
| `docker-compose.yml` | G | One-command deploy |
| `eval_harness.py` | H | Perplexity + tension metrics + 11-tick WaveCycleRunner |
| `vendor/GOAT-TS` | — | Constraint-graph engine (submodule) |
| `vendor/TS-Core` | — | UniversalLivingGraph + WaveCycleRunner (submodule) |
| `vendor/ts-llm` | — | Tokenizer, hierarchical dynamics, attractor LLM package (submodule) |
| `docs/API_DISCOVERY.md` | — | Verified entrypoints for vendored repos + model config surface |
| `docs/BASELINE.md` | — | Phase 0 baseline recording instructions |
| `docs/DEVELOPMENT_ROADMAP.md` | — | Engineering vs validation phases (performance targets, dataset order) |
| `scripts/plot_phase05_metrics.py` | — | Plots `--phase05-batch-metrics-csv` columns (incl. Phase 1–2 extras) |

---

## Quick start

### Requirements

- Python 3.10+
- [PyTorch](https://pytorch.org/) (CPU or CUDA)

```bash
git clone --recurse-submodules https://github.com/BoggersTheFish/BoggersTheLLM.git
cd BoggersTheLLM
python3 -m venv .venv
source .venv/bin/activate   # after this, `python` usually works; without venv use `python3`
pip install -r requirements.txt
```

On **Ubuntu / Linux Mint**, only `python3` may be installed. Either activate `.venv` as above or call `python3 sandbox.py` instead of `python sandbox.py`.

### Train the baseline model

```bash
python3 sandbox.py
```

With trajectory contrastive loss (recommended), LR schedule, metrics CSV, and a fixed epoch count:

```bash
python3 sandbox.py \
  --epoch-metrics-csv metrics.csv \
  --lr 0.001 \
  --lr-decay-every 15 \
  --lr-gamma 0.7 \
  --val-fraction 0.1 \
  --max-epochs 30
```

### First real training run (public corpus + checkpoint + eval JSON)

The default `data/corpus.txt` (or empty path) may trigger a **synthetic** text fallback so integration tests always have enough tokens. For a **real** corpus and metrics that reflect language modeling (not random-like ~506 val perplexity on a toy file), use a Hub dataset or your own large `.txt` / directory.

**Option A — TinyStories via Hugging Face (recommended first real run)**

Requires `datasets` (included in `requirements.txt`). The first run downloads data into `data/cache/hf/`. Optional: set `HF_TOKEN` for higher Hub rate limits.

```bash
pip install -r requirements.txt
mkdir -p checkpoints/real_run

python3 sandbox.py \
  --dataset-source tinystories \
  --tokenizer tiktoken \
  --val-fraction 0.1 \
  --max-epochs 50 \
  --use-goat-memory \
  --use-substrate \
  --lr 0.001 \
  --lr-decay-every 15 \
  --lr-gamma 0.7 \
  --epoch-metrics-csv metrics_real.csv \
  --eval-results-json eval_results.json \
  --checkpoint-dir checkpoints/real_run
```

This writes a final `ckpt_step*.pt` under `--checkpoint-dir` and an **`eval_results.json`** with the same **token-level val split** as training (`val_ce`, `val_ppl`, `val_windows`, checkpoint path). Perplexity should move off the untrained baseline as loss decreases.

**Option B — FineWeb-Edu subset (streaming)**

Uses the `sample-10BT` config and stops after `--hf-max-rows` rows (default 50k):

```bash
python3 sandbox.py \
  --dataset-source fineweb-edu \
  --hf-max-rows 20000 \
  --tokenizer tiktoken \
  --val-fraction 0.1 \
  --max-epochs 50 \
  --eval-results-json eval_results.json \
  --checkpoint-dir checkpoints/fineweb_run
```

**Option C — Your own corpus (TS-OS export or any large text)**

Place UTF-8 text at `data/corpus.txt`, or pass `--corpus /path/to/dir` (merges `.txt` / `.jsonl`). To **disable** automatic synthetic fallback when the file is missing or tiny:

```bash
python3 sandbox.py --corpus data/my_corpus.txt --no-synthetic-fallback --tokenizer tiktoken ...
```

**Materialize HF data only (no training)**

```bash
python3 data/hf_remote_corpus.py tinystories --max-rows 50000 --cache-dir data/cache/hf
# The script prints the path to the generated .txt; pass it to --corpus:
python3 sandbox.py --corpus PATH_PRINTED_ABOVE --tokenizer tiktoken ...
```

**Full Wave H harness** (TSCore before/after tension + same val perplexity) on the same cached file:

```bash
python3 eval_harness.py \
  --dataset-source tinystories \
  --tokenizer tiktoken \
  --model-checkpoint checkpoints/real_run/ckpt_step0001234.pt \
  --output eval_results_wave_h.json
```

### Run the smoke test

Verifies dynamics, tension, training step, and TSCore wave cycle all pass:

```bash
python3 smoke_test.py
```

### Embed batch parity (`embed_windows_batch`)

Training uses a **single batched embedding** for trajectory windows (`embed_windows_batch` on `(B, W)` token ids). To verify it matches **row-wise** `embed_window` (same numerics as the old `torch.stack` loop):

```bash
python3 tests/test_embed_windows_batch.py
```

Prints **`max_abs_diff`**; asserts `allclose` at `1e-6`. Requires PyTorch (same env as training).

### Run the evaluation harness

```bash
python3 eval_harness.py --val-fraction 0.2 --max-ticks 11 --output eval_results.json
```

Use the same Hugging Face corpus as training (ignores `--corpus` when set):

```bash
python3 eval_harness.py --dataset-source tinystories --tokenizer tiktoken \
  --val-fraction 0.2 --max-ticks 11 --output eval_results.json
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

## Usage guide

This section is the operational manual — what to run, how data and validation behave, and how to reproduce runs: how training consumes data, how validation stays honest, how to reproduce runs, and how the main scripts relate to each other. For a compact flag list, see [CLI reference](#cli-reference) below.

### Training entry point (`sandbox.py`)

`python3 sandbox.py` is the full training and baseline run: it builds the tokenizer and `TorchAttractorLanguageModel`, loads (or synthesizes) the corpus, constructs an `AttractorDataPipeline` when possible, runs `num_epochs` of optimization, prints sample generations, and optionally writes checkpoints and CSV metrics.

**Default data mode is streaming (recommended).** The whole corpus file (or merged directory of `.txt` / `.jsonl`) is read as one string, encoded once into a single token sequence, and training uses all sliding windows `(context, target)` along that sequence. Legacy **line-based** mode (`--no-streaming-dataset`) tokenizes each non-empty line separately and skips lines shorter than `window_size + 1`; it also uses `--epoch-copies` to repeat the training line list each epoch. In **stream mode**, `--epoch-copies` is ignored on purpose: repetition is controlled only by `--max-epochs` and by shuffled window sampling, not by duplicating tokens in memory.

**Device and checkpoints.** `--device auto` picks CUDA when available. `--resume-checkpoint` restores weights and optimizer state. `--checkpoint-dir` and `--save-every` control where and how often numbered checkpoints are written.

**Integrations.** `--use-substrate` attaches `LLMSubstrateNode` so language tension can drive TSCore propagation and logging. `--use-goat-memory` enables `GoatMemoryManager` and injects a per-position signal into window dynamics. By default **`--dynamics vectorized`** loads `VectorizedWindowDynamics` from `dynamics_vectorized.py` (falls back to simple if import fails). Use **`--dynamics simple`** for the legacy `SimpleAttractorDynamics` path.

### Corpus, paths, and automatic synthetic text

**Corpus path.** `--dataset-path` wins over `--corpus` if both are set; otherwise the default is `data/corpus.txt`. Paths may be a single file, a directory (all `.txt` / `.jsonl` / `.json` under it, merged), or `.jsonl` with `text` / `content` / `sentence` fields concatenated.

**Hugging Face datasets (`--dataset-source`).** With `--dataset-source tinystories` or `fineweb-edu`, the sandbox ignores `--corpus` / `--dataset-path` for loading and materializes text into `data/cache/hf/` (override with `--hf-cache-dir`). Limits: `--hf-max-rows`, `--hf-max-chars`; `--hf-refresh` rebuilds the cache file. Requires `pip install datasets` (listed in `requirements.txt`).

**Automatic fallback.** If no text files resolve for the path, or if the tokenized sequence is shorter than `20 * window_size` tokens, training prints `Corpus too small — generating synthetic corpus...`, writes a **temporary** UTF-8 file using `data/generate_corpus.py`’s `generate_corpus()` (target length at least 20k tiktoken GPT-2 tokens, scaled up slightly with window size), trains from that file, and deletes the temp file after the run. This keeps local experiments runnable without hand-curating a large corpus. Pass **`--no-synthetic-fallback`** to fail fast instead (recommended when you expect a real corpus file to be present).

**Manual corpus generation.** To persist synthetic text instead of a temp file:

```bash
python3 data/generate_corpus.py --out data/generated.txt --tokens 20000 --seed 42
python3 sandbox.py --corpus data/generated.txt
```

`generate_corpus` grows paragraphs until the **tiktoken** GPT-2 encoding length reaches `--tokens`. Counts may differ slightly from the sandbox tokenizer (`--tokenizer tiktoken` vs `fallback`), but the generated file is always large enough for the small-corpus threshold above.

### Train/validation split (stream mode) and reliable metrics

**Token-level split with a gap.** Validation is a suffix of the token stream. Between the last training token and the first validation token the code skips **`window_size` tokens** so no sliding window’s context crosses into the other split (no train/val leakage). Each side must still have at least `window_size + 1` tokens to form one window.

**Minimum validation windows.** The code targets at least **50** validation windows (`MIN_VAL_WINDOWS`). After the first split, if there are fewer than 50 val windows, it **recomputes the split once** using an effective hold-out fraction of at least `(50 + window_size) / total_tokens`, then logs **`final val_fraction`** (actual `len(val_tokens) / total_tokens`). If 50 windows are still impossible (corpus too small), it prints `Validation set too small (X windows). Metrics will be noisy.` and, at startup, `WARNING: validation unreliable` when `val_windows < 50`. Treat `val_ce` / perplexity on tiny val sets as qualitative only.

**What you see at startup (stream mode).** Lines such as `total_tokens=`, `train_tokens=`, `val_tokens=`, `train_windows=`, `val_windows=`, and `final val_fraction=…` summarize the run. For statistically stable validation on modest corpora, prefer **`--val-fraction 0.2`**–**`0.3`** or more tokens.

### Reproducibility

**Global seed.** `--seed` seeds Python’s `random` module at process start.

**Per-epoch batch order (stream mode).** `AttractorDataPipeline.epoch_batches(epoch_index=epoch)` uses `random.Random(seed + epoch_index)` to shuffle window start indices. Fixing `--seed` fixes the entire sequence of batch orders across epochs; changing the epoch index changes the shuffle, so epochs are not identical copies of the same ordering.

**No stream duplication.** Training does not multiply the train token list by `epoch_copies` in stream mode; multiple passes are real epochs over reshuffled windows.

### Other scripts (when to use them)

| Command | Purpose |
|--------|---------|
| `python3 smoke_test.py` | Fast integration check: model + one dynamics pass + training step + TSCore wave cycle. Run after install or refactors. |
| `python3 tests/test_embed_windows_batch.py` | Confirms batched window embedding matches per-row `embed_window` (prints max abs diff). |
| `python3 eval_harness.py …` | Perplexity, mean tension, trajectory contrast, optional TSCore `WaveCycleRunner` metrics; writes JSON. Use `--dataset-source tinystories` / `fineweb-edu` to match Hub training data. |
| `python3 inference_server.py` | FastAPI server: OpenAI-style `/v1/completions`, cache-backed generation, optional TSCore hooks. Needs `pip install fastapi uvicorn`. |
| `python3 data/generate_corpus.py --out PATH --tokens N` | Offline synthetic corpus for tests or a fixed `data/generated.txt`. |

### Docker and deployment

`docker compose up` builds and runs the inference image from `Dockerfile` / `docker-compose.yml`. Swap the PyTorch wheel in the Dockerfile for CUDA if you need GPU in the container. The compose service name is noted in `docker-compose.yml` (see Quick start).

### How this maps to the rest of the README

- **Architecture and tension:** [Architecture overview](#architecture-overview) and [Tension semantics](#tension-semantics).
- **Loss function:** [Training objective](#training-objective).
- **Per-flag list:** [CLI reference](#cli-reference) and [Epoch metrics CSV columns](#epoch-metrics-csv-columns).
- **Scaling and practical training:** [Scaling and training tips](#scaling-and-training-tips).
- **Lightweight diagnostics:** [Debug mode](#debug-mode).
- **Module-by-module history:** [Wave-by-wave implementation log](#wave-by-wave-implementation-log).

### Scaling and training tips

**Data and validation**

- Prefer **stream mode** (default): one token sequence and sliding windows scale to large corpora. Use **`--dataset-source tinystories`** or **`fineweb-edu`** for a first serious run, or a large UTF-8 **`--corpus`** file / directory.
- For stable **val CE / perplexity**, use enough hold-out tokens: **`--val-fraction 0.1`**–**`0.3`**. If startup warns **`val_windows < 50`**, treat metrics as noisy until you add text or increase the fraction.
- **`--no-synthetic-fallback`** fails fast if the corpus is missing or tiny—useful before long GPU jobs.

**Batch size, window, and steps**

- **Trajectory mode** requires **`--trajectory-batch-size` ≥ 2** (negatives are drawn inside the batch). Larger batches stabilise contrastive training but cost more memory.
- **Window size** (`--window-size`): wider context increases compute per step roughly linearly in `W` (embedding is `W×D`, dynamics run up to **`--num-dynamics-steps`** / **`--max-window-steps`** outer steps). Start with the default or `8`; increase when data and VRAM allow.
- **`--num-dynamics-steps`** / **`--max-window-steps`**: hard cap on outer steps per window. Optional **`--convergence-epsilon`** (with **`--min-attractor-steps`**, default 2) can stop early when state change or tension is stable; default epsilon is **`0`** (full configured outer steps each window). More steps mean deeper relaxation; diminishing returns once tension curves flatten—watch **`mean_final_step_tension`** in epoch CSV.

**Throughput and hardware**

- Use **`--device cuda`** when available. On CUDA, **`torch.set_float32_matmul_precision("high")`** is set, and **`torch.compile`** targets only the inner step (**`dyn._step`** for vectorized, **`dyn._step_rows`** for simple)—not the outer window loop. First epoch can be slower while kernels warm up.
- **`--dynamics vectorized`** (default) can help on GPU at larger `D`; **`--dynamics simple`** is fine for CPU smoke tests and small models.

**Optimisation and stability**

- **`--lr`** 1e-3 with **`StepLR`** (`--lr-decay-every`, `--lr-gamma`) is a reasonable default; lower LR if loss spikes or **`grad_norm`** explodes (see **`--debug`**).
- Keep at least one of **`--token-aux-ce`** or **`--readout-aux-alpha`** on in trajectory mode so the readout heads receive gradients (the script warns if both are zero).
- **Phase 1 window interaction** (`--phase1-enable-window-interaction`) plus **Phase 2** **`--phase2-interaction-reg-weight`** and optional **`--phase2-interaction-decay-tau`** help keep coupling matrix **`C`** from drifting; enable when you see unstable window norms.
- **Checkpoints:** **`--checkpoint-dir`** + **`--save-every`** for long runs; **`--resume-checkpoint`** restores weights and optimizer. Newer code may add parameters—use **`strict=False`** in custom loaders if needed.

**Logging for analysis**

- **`--epoch-metrics-csv`**: one row per epoch (loss, CE, tension, TSCore fields).
- **`--phase05-batch-metrics-csv`** (with **`--phase05-log-metrics`** implied): per-batch diagnostics; plot with **`scripts/plot_phase05_metrics.py`**. When metrics logging is off, the runtime skips heavy tracing arrays and keeps only the tension values needed for control flow.

### Debug mode

Pass **`--debug`** to print a **small number of `[debug]` lines** at meaningful points (no per-step spam):

| When | What you see |
|------|----------------|
| After **resume** (if used) | Starting epoch index and `global_step`. |
| After **model → device** | Parameter count, `state_dim`, train window size, max window steps. |
| After **integrations** | Dynamics class name, `torch.compile` outcome, substrate / GOAT on or off. |
| Before the **training loop** | Epoch range, starting `global_step`, `loss_mode`. |
| **Pipeline** | Streaming on/off, batch size, estimated batches per epoch (or legacy fallback). |
| **Each epoch** | Estimated batches, `report_every` (matches progress snippet cadence), current LR. |
| **First batch only** (trajectory mode) | Loss, **gradient L2 norm**, whether readout logits are all finite. |
| **End of each epoch** | Batch count, approximate window-updates, batches per second. |
| **After training** | Final `global_step`, last epoch id, last mean loss and train CE. |

**`--quick-test --debug`** prints one line before the sanity checks then exits.

Use this when verifying a new machine, a resumed run, or tracking down NaNs; for full traces use **`--phase05-batch-metrics-csv`** instead.

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

`dynamics_vectorized.py` provides **`VectorizedWindowDynamics`**, selected by default via **`--dynamics vectorized`** (replaces `model.dynamics` after construction; use **`--dynamics simple`** for `SimpleAttractorDynamics`).

Both **`SimpleAttractorDynamics`** and **`VectorizedWindowDynamics`** implement the same **`step(S, signal) → S`** interface used inside **`run_window_dynamics`** (positional coupling first, then one dynamics step with the optional GOAT signal tensor). **`run_window_dynamics`** caches per-window static work (positional weight matrix, **`C * mask`** for Phase 1, GOAT bonus vector) outside the time-step loop.

- Wraps **`MultiHeadDynamics`** from `vendor/ts-llm` (low-rank diffusion per head + cross-head coupling); window step uses **cubic** nonlinearity (simple path uses **`tanh`**).
- **`forward` on `VectorizedWindowDynamics` is disabled** (`NotImplementedError`); use **`model.run_window_dynamics`** or **`run_window_dynamics_vectorized`**, which temporarily swaps dynamics and calls **`model.run_window_dynamics`** (optional **`**kwargs`** forwarded to **`run_window_dynamics`**).
- **`torch.compile`:** on CUDA, only **`dyn._step`** (vectorized) or **`dyn._step_rows`** (simple) is compiled—not the full dynamics module.
- **`get_compiled()`** caches compiled `_step` by shape key for smoke / benchmarks.
- Parity tests: both paths produce finite outputs; equations differ by design.

### Wave C — State cache

`state_cache.py` provides **rolling-window** inference aligned with training:

- `AttractorStateCache` holds **`fast_state (D,)`** + **`slow_memory (D,)`** + rolling **`phrase_table`**
- **`step(token_id)`** — builds the last-**W** token ids, applies the same training embedding pipeline (**`Embedding → LayerNorm → row L2`**), runs **`model.run_window_dynamics(S, context_ids=[ids], …)`** with **`S`** **`(1, W, D)`**, then updates fast/slow from the final row and phrase table
- **`logits()`** — **`readout(combined)`** on the symplectic blend of fast/slow (same head as **`next_token_logits`**)
- **`warmup(prompt_ids)`** — seed cache from prompt before generation
- **`generate_with_cache(model, cache, prompt, ...)`** — drop-in for **`model.generate()`**
- Smoke: **`python3 state_cache.py`**; training-aligned check: **`.venv/bin/python scripts/ts_workflow_smoke.py`**

```python
from state_cache import AttractorStateCache, generate_with_cache
cache = AttractorStateCache(model)
text = generate_with_cache(model, cache, prompt="the cat sat", max_tokens=30)
```

### Wave D — Data pipeline

`data_pipeline.py` feeds training with **stream-based tokenization by default** (no dependence on individual lines being long enough):

- **Stream mode (default):** the corpus is read as **full text** (whole `.txt` files; `.jsonl` records concatenated), then `tokenizer.encode(full_text)` produces **one continuous token sequence**. Sliding windows `(context, target)` use `tokens[i : i+W]` → target `tokens[i+W]`. Each epoch shuffles all window start indices, then batches.
- **`train_token_ids=`** — sandbox passes the train split after **token-level** train/val cut with a **gap of `window_size` tokens** between train and val so no sliding window shares context across the split.
- **Shuffle:** `epoch_batches(epoch_index=epoch)` uses `Random(seed + epoch_index)` so each epoch has a **deterministic but different** batch order (reproducible runs).
- **Stream mode** ignores **`--epoch-copies`** (use **`--max-epochs`** instead); duplicating the token stream is not applied.
- **Legacy line mode:** `streaming_dataset=False` keeps per-line encoding (short lines dropped); `shuffle_buffer` refills between batch groups.
- Multi-shard round-robin (`shard_id` / `num_shards`) for data-parallel workers
- Too few tokens (`len < window_size + 1`) raises a clear **“Corpus too small after tokenization”** error.
- **Synthetic fallback:** if the corpus path yields no files or fewer than `20 * window_size` tokens after encoding, `sandbox.py` generates a temporary corpus via **`data/generate_corpus.py`** (see [Usage guide](#usage-guide)).
- **Startup logging:** stream mode prints `total_tokens`, `train_tokens`, `val_tokens`, `train_windows`, `val_windows`, and **`final val_fraction`** after any minimum-val-window adjustment.

```python
from data_pipeline import AttractorDataPipeline
pipe = AttractorDataPipeline(
    sources=["data/corpus.txt"], model=model, batch_size=16, streaming_dataset=True
)
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

Evaluation calls that run trajectory forwards now use a non-mutating path for repulsion memory bookkeeping, so diagnostics do not alter subsequent training behavior.

**Phase 0 eval results (untrained model, 51-line corpus):**

| Metric | Value |
|--------|-------|
| Baseline val PPL | 506 |
| Mean window tension | 0.318 |
| TSCore tension (before WaveCycle) | 0.149 |
| TSCore tension (after 11-tick WaveCycle) | 0.0005 |
| Evolve events triggered | 5 |

TSCore converges cleanly. High PPL is expected for an untrained model — the harness is the measurement instrument.

### Phase 0.5 — Instrumentation and stability

Configuration: **`Phase05Config`** in `phase05_config.py`, passed to **`TorchAttractorLanguageModel(..., phase05=...)`** (CLI: `--phase05-*`).

- **`--phase05-log-metrics`**: collect window-trace and token-evolve diagnostics used for batch CSV and logged scalars. When disabled, the outer loop avoids accumulating tension curves, step diagnostics, and break-tracing arrays.
- **`--phase05-batch-metrics-csv PATH`**: append one row per training batch (implies log metrics). Column list is `PHASE05_BATCH_CSV_HEADER` in `sandbox.py` (tension components, stagnation, trajectory margin, break counts, Phase 1–2 extensions).
- **`--phase05-enforce-negdef-diffusion`**: strictly negative-definite diffusion in the simple dynamics path.
- **`--phase05-adaptive-window-dt`**: EMA-scaled positional timestep from window tension.
- **`--phase05-tension-w w1,w2,w3`**: override weights in `T_total = w1·T_energy + w2·T_align + w3·T_entropy`.
- **`--phase05-multi-negative` / `--phase05-num-negatives` / `--phase05-traj-temperature`**: trajectory contrastive negatives and temperature.

### Phase 1 — Multi-head diffusion and window interaction

Configuration: **`Phase1Config`** in `phase1_config.py` (CLI: `--phase1-*`).

- **`--phase1-num-heads`**, **`--phase1-head-dim-mode {shared,split}`**: parallel drift heads; split mode partitions `D` across heads (`D % H == 0`).
- **`--phase1-enable-window-interaction`**: learnable **`C ∈ ℝ^{W×W}`** applied as `einsum('bid,ij->bjd', S, C)` after each local step (scaled by **`--phase1-interaction-scale`**).
- **`--phase1-head-diversity-weight`**: auxiliary penalty on mean pairwise cosine similarity of head drift directions.
- **`--phase1-enable-per-head-tension`**: when logging, record mean per-head geometry tension (split layout).

### Phase 2 — Directional breaks and stabilised routing

Configuration: **`Phase2Config`** in `phase2_config.py` (CLI: **`--phase2-*`**). No attention or token–token scoring; head-level weighting only.

| Area | Behaviour |
|------|-----------|
| **Breaks** | Default: escape along normalised **`state − prev_state`**, step size **`α = break_base_strength · clamp((T_target − T)/T_target, min, max)`**; tiny delta norm falls back to random unit direction. **`--phase2-disable-directional-break`** restores Gaussian jitter. |
| **Rejection** | **`--phase2-enable-break-rejection`**: revert a break if tension increases and row cosine alignment worsens. |
| **Mixing** | **`state + sigmoid(gate)·W_mix(concat heads)`** when residual mixing is on; **`--phase2-disable-residual-mixing`** uses linear mix only. **`--phase2-mixing-gate-init`** sets initial gate (~0.1 default). |
| **Window `C`** | Optional **`--phase2-interaction-decay-tau`**: multiply **`C`** by **`exp(−|i−j|/τ)`** before the einsum. **`--phase2-interaction-reg-weight`**: add **`‖C−I‖²`** to the trajectory loss (requires window interaction). |
| **Head weights** | **`--phase2-enable-head-tension-weighting`**: combine head drifts with **`softmax(−T_head)`** (requires per-head tension signal in the dynamics path). |
| **Memory hook** | **`--phase2-store-break-memory`**: store last pre/post break window states on the model for future reuse. |

Batch CSV (with `--phase05-batch-metrics-csv`) gains Phase 2 fields when breaks occur: **`phase2_break_direction_norm_mean`**, **`phase2_break_applied_alpha_mean`**, **`phase2_break_delta_tension_mean`**, **`phase2_break_delta_alignment_mean`**, **`phase2_head_weight_entropy`**, **`phase2_interaction_reg_loss`**.

**Checkpoints:** new parameters (for example **`mixing_gate_raw`**, **`phase1_window_C`**) are not in older checkpoints; load with **`strict=False`** or retrain.

**Resume reliability:** optimizer state is restored after model device placement and optimizer tensors are migrated to the active device, preventing Adam CPU/CUDA state mismatch on resumed training.

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
    phase05=sb.Phase05Config(),
    phase1=sb.Phase1Config(),
    phase2=sb.Phase2Config(),
)
model.tokenizer = tok

# Training path
wids = model.window_ids_from_sequence(token_ids)
S = model.embed_window(wids)  # (W, D)
# Batched: context_tensor (B, W) long -> (B, W, D), equivalent to stacking embed_window rows
# S_b = model.embed_windows_batch(context_tensor)
S, logs = model.run_window_dynamics(S, context_ids=wids)  # GOAT uses token ids if enabled
train_logits = model.readout_window(S.reshape(1, -1))  # primary training readout

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

**Per-token path** (`evolve_token`): after each inner step, `compute_tension` returns:

```
T = |ΔE_state| + λ · (1 − cos(fast, slow)) + μ · H(readout_logits)
```

| T < tol | Early exit — attractor is stable |
|---------|----------------------------------|
| T > high | Directional break (Phase 2 default) or Gaussian jitter (`--phase2-disable-directional-break`) |
| T > break_thresh | Same break family on the token path |

**Window path** (`run_window_dynamics`): after each outer iteration, **`compute_tension_window`** (alias **`compute_window_tension`**) uses neighbor energy drift + misalignment + optional readout entropy (see `WINDOW_TENSION_USE_ENTROPY` in `sandbox.py`). The outer loop runs at most **`max_window_steps`** times; if **`convergence_epsilon > 0`** (CLI **`--convergence-epsilon`**, after **`--min-attractor-steps`**), the loop may exit early when state change or tension delta falls below epsilon. Otherwise (default epsilon **`0`**) the loop uses all **`max_window_steps`**. Tension still drives **low-tension escape**, **high-tension breaks**, **GOAT transitions**, and **high-T row renorm** inside each step. Phase 2 breaks use **`state − prev_state`** (**`F.normalize`**) with tension-scaled magnitude; see [Phase 2](#phase-2--directional-breaks-and-stabilised-routing).

---

## CLI reference

```
python3 sandbox.py [options]    # or: source .venv/bin/activate && python sandbox.py

Data & tokenizer:
  --corpus PATH              Training text (default: data/corpus.txt)
  --dataset-path PATH        Alias for --corpus (takes precedence if set)
  --dataset-source {local,tinystories,fineweb-edu}  Hugging Face corpus (requires `datasets`); ignores --corpus
  --hf-cache-dir PATH        HF materialized text cache (default: data/cache/hf)
  --hf-max-rows N            HF rows to read (default: 50000)
  --hf-max-chars N           Optional total character cap (0 = none)
  --hf-refresh               Rebuild HF cache file
  --no-synthetic-fallback    Error if corpus missing/too small instead of temp synthetic text
  --val-fraction FLOAT       Token-level val hold-out in stream mode (default: 0.05). Use ~0.3 if you need many val windows; 0 = off.
  --tokenizer {tiktoken,fallback}   BPE mode (default: fallback)
  --vocab-cap INT            Max BPE vocab when using tiktoken mode (default: 32768)
  --seq-len INT              Alias for --window-size
  --batch-size INT           Alias for --trajectory-batch-size
  --shuffle-buffer INT       Pipeline shuffle buffer (line-based mode only; default: 2048)
  --no-streaming-dataset     Legacy line-based corpus (short lines dropped). Default: stream whole file as tokens.

Training:
  --window-size INT          Context window W (default: 6)
  --num-dynamics-steps INT, --max-window-steps INT
                            Max outer attractor steps per window (default: 16)
  --convergence-epsilon FLOAT  Early exit when ‖ΔS‖ or |ΔT_mean| below this after min steps (0 = use all configured outer steps; try 1e-3 on GPU)
  --min-attractor-steps INT  Minimum outer steps before early exit may trigger (default: 2, ≥2)
  --trajectory-batch-size INT  Batch size for trajectory mode (default: 64, need ≥2)
  --loss-mode {trajectory,ce}
  --token-aux-ce FLOAT       Aux CE on readout_window in trajectory mode (default: 0.2)
  --readout-aux-alpha FLOAT  Aux CE on single-state readout (default: 0.15; 0 = off)
  --grad-clip FLOAT          Optional global grad-norm clip (default: off)
  --lr, --lr-decay-every, --lr-gamma
  --epoch-copies INT         Repeat training lines per epoch
  --max-epochs N, --epochs N Number of training epochs (default: 25)
  --seed INT

Device & checkpointing:
  --device auto|cpu|cuda|cuda:N
  --resume-checkpoint PATH
  --save-every N             Save every N optimizer steps (0 = final only)
  --checkpoint-dir PATH      Default: ./checkpoints

Integrations:
  --use-substrate            TSCore LLMSubstrateNode after each batch
  --use-goat-memory          GoatMemoryManager + window-path signal injection
  --use-lorentz              Lorentzian positional coupling in window dynamics (vectorized path only; default: off)
  --dynamics {simple,vectorized}   Default: vectorized (MultiHeadDynamics); simple = legacy single-matrix drift

Phase 0.5 (instrumentation):
  --phase05-log-metrics      Per-batch diagnostics + window trace for CSV
  --phase05-batch-metrics-csv PATH  Append batch rows (implies log-metrics); see PHASE05_BATCH_CSV_HEADER
  --phase05-enforce-negdef-diffusion
  --phase05-adaptive-window-dt
  --phase05-tension-w w1,w2,w3
  --phase05-multi-negative   Trajectory: multi-shuffle negatives
  --phase05-num-negatives K  (with multi-negative; default 4)
  --phase05-traj-temperature FLOAT

Phase 1 (multi-head + window C):
  --phase1-num-heads H
  --phase1-head-dim-mode {shared,split}
  --phase1-interaction-scale FLOAT
  --phase1-enable-window-interaction
  --phase1-head-diversity-weight FLOAT
  --phase1-enable-per-head-tension

Phase 2 (breaks + stable routing):
  --phase2-disable-directional-break   Legacy Gaussian breaks
  --phase2-break-base-strength, --phase2-break-min-scale, --phase2-break-max-scale
  --phase2-break-t-target FLOAT        Reference T in α scaling
  --phase2-enable-break-rejection
  --phase2-disable-residual-mixing     Linear W_mix only
  --phase2-mixing-gate-init FLOAT
  --phase2-interaction-reg-weight FLOAT  ‖C−I‖² on loss (needs window interaction)
  --phase2-interaction-decay-tau TAU   exp(−|i−j|/τ) mask on C
  --phase2-enable-head-tension-weighting
  --phase2-store-break-memory

Logging:
  --epoch-metrics-csv PATH   Per-epoch CSV (see below)
  --eval-results-json PATH   After training: val CE, val PPL, checkpoint path (same val split as training)
  --log-hard-batch-loss-above FLOAT
  --baseline-out PATH        Phase-0 baseline snapshot text file

Misc:
  --quick-test               Window sanity checks, exit
  --debug                    Concise [debug] lines at setup, each epoch, first-batch grad norm (trajectory)
```

### Epoch metrics CSV columns

When `--epoch-metrics-csv` is set, each row includes: `epoch`, `loss_mode`, `mean_loss`, **`train_ce`** (mean batch CE from `readout_window` logits during the epoch), **`val_ce`** (held-out `mean_cross_entropy_eval`, empty if no val), `train_traj_contrast` (last training batch trajectory loss snapshot), **`val_traj_contrast`** (full val-set mean when val exists), `mean_final_step_tension`, `max_batch_loss`, `lr`, `global_step`, **`tscore_evolves`**, **`tscore_last_tension`** (0 if substrate disabled).

**Per-batch CSV** (`--phase05-batch-metrics-csv`): separate file; one row per optimizer step with window tension curves, trajectory margins, break counters, and (when enabled) Phase 1 / Phase 2 columns — see `PHASE05_BATCH_CSV_HEADER` in `sandbox.py`. Plot with **`python3 scripts/plot_phase05_metrics.py PATH --out DIR`**.

**Validation perplexity:** `PPL_val = exp(val_ce)` when `val_ce` is finite.

### A/B example (GOAT on vs off)

Use the same `--seed`, corpus, and hyperparameters; only add `--use-goat-memory` for the treatment run. Log with `--epoch-metrics-csv` and compare `val_ce` / `mean_loss` curves (or `exp(val_ce)` for val perplexity).

On **tiny corpora** (under a few thousand tokens), val CE and GOAT A/B deltas are **integration checks only**, not evidence of real model quality.

```bash
mkdir -p experiments/goat_ab
# Baseline
python3 sandbox.py \
  --corpus data/corpus.txt \
  --val-fraction 0.3 \
  --seed 42 \
  --device cpu \
  --tokenizer fallback \
  --max-epochs 30 \
  --epoch-metrics-csv experiments/goat_ab/baseline_42.csv
# +GOAT
python3 sandbox.py \
  --corpus data/corpus.txt \
  --val-fraction 0.3 \
  --seed 42 \
  --device cpu \
  --tokenizer fallback \
  --use-goat-memory \
  --max-epochs 30 \
  --epoch-metrics-csv experiments/goat_ab/goat_42.csv
```

---

## Submodules

Clone with:

```bash
git clone --recurse-submodules https://github.com/BoggersTheFish/BoggersTheLLM.git
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

## Phase 0 Baseline (March 29 2026)
First real training run on TinyStories (120 stories).
Mean trajectory loss: **2.3759**
Val trajectory contrast: **0.112059**
Checkpoint: [ckpt_step0000274.pt](https://github.com/BoggersTheFish/BoggersTheLLM/tree/main/checkpoints/first_real_release)

