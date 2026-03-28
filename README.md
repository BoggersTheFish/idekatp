# woke-baby-llm

A **minimal, single-file** PyTorch experiment: a small-vocabulary “language model” built from **continuous dynamics** instead of attention or transformers. State follows a **trajectory**; meaning is **path-dependent** (order and context matter).

Repository: [github.com/BoggersTheFish/woke-baby-llm](https://github.com/BoggersTheFish/woke-baby-llm)

## What it does

- **Embeddings** feed a **context-conditioned signal** (base direction plus a γ-weighted context vector, then renormalized). Recent token signals can be **blended** (light “multi-agent” mix) via a learnable gate.
- **Fast + slow memory**: fast state is stepped by learned dynamics; slow memory uses **decay + learnable slow_lr** (`slow = (1 − slow_decay)·slow + slow_lr·fast`) with a **norm cap** on slow so it cannot dominate.
- **Tension-adaptive partial convergence**: each token runs up to **`max_convergence_steps`** inner dynamics steps. A scalar **tension** \(T \approx |\Delta E| + \lambda(1-\cos(\text{fast},\text{slow})) + \mu H(\text{logits})\) (energy drift, fast/slow misalignment, prediction entropy) drives **early exit** when \(T\) is below **`tension_tol`**, **extra noise** when tension was high, and a **break** perturbation when \(T\) exceeds **`tension_break_thresh`**. This replaces a fixed step count when the attractor is already stable.
- **Symplectic-style readout**: the combined state uses a **midpoint in fast** between the token’s start and end (`0.5·(fast_start + fast_end)`) with static slow for that sub-step, then `w_fast·mid + w_slow·slow`, before normalization and linear readout.
- **Dynamics** (stable): learned diffusion, **tanh**-bounded nonlinearity, **damping**, **β-scaled** input signal, **tension-scaled** Gaussian noise, then **unit normalization** of the fast state after each step.
- **Decoding**: primary path is a **linear readout** from the normalized combined state. A **`next_token_logits_distance`** helper keeps the distance-to-embedding baseline for experiments.
- **Training**: **sliding-window** sequences (default **6** tokens of context → predict next), corpus loaded from **`data/corpus.txt`** by default (one sentence per line; `#` line comments), duplicated and shuffled per epoch, **cross-entropy with label smoothing**, minus entropy bonus, **light bigram logit bias** on embeddings, **anti-repetition** logit shaping on the training context, and **entropy floor** (nats) when logits are too peaked. Optional **train/val split** reports validation CE each epoch (train CE printed for comparison).
- **Generation**: **temperature** sampling with **tension-adaptive** scaling when last tension exceeds the tolerance, **top-k** truncation, **repetition penalties** on recent token ids, optional **debug** attractor / diversity metrics, and **`compare_prompts()`** for trajectory distance between prompts.

There is **no** attention, no transformer blocks, and no external model—only **`sandbox.py`** plus **`requirements.txt`**.

## Limitations and what needs work

This repo is a **research sandbox**, not a production language model.

- **Data vs vocabulary**: The model is trained on **dozens of short sentences** against a **512-word** vocabulary. That is far too little signal for fluent text; the network tends toward **phrase-level templates** even with tension-aware dynamics and decoding.
- **Coherence**: Do not expect topic continuity, factual answers, or long-range syntax. Improvements need **more and more varied training text**, **longer windows**, **more epochs or capacity**, and/or **different objectives**—not only dynamics tweaks.
- **Throughput**: The bundled script runs **on CPU** by default; **pre-training is slow** at full epoch counts. Reduce `NUM_EPOCHS` or corpus size while iterating.

Contributions are welcome if you want to extend the experiment (data pipeline, evaluation, or architecture).

## Requirements

- Python 3.10+ recommended  
- [PyTorch](https://pytorch.org/) and NumPy (see `requirements.txt`)

On many Linux distributions the system Python is **PEP 668** “externally managed”; use a venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python sandbox.py
```

Default training text is **`data/corpus.txt`** (next to `sandbox.py`). Use your own file:

```bash
python sandbox.py --corpus path/to/sentences.txt
```

Useful flags: `--val-fraction 0.05` (validation cross-entropy after each epoch; `0` disables), `--seed 42`, `--epoch-copies 2` (repeat the sentence list per epoch before shuffling), `--baseline-out path` (Phase 0 snapshot; see `docs/BASELINE.md`). **Training objective:** `--loss-mode trajectory` (default) uses **contrastive** alignment between the **evolved** context-window state and the **evolved** shifted teacher window `[x₂…x_W, next_token]`, plus optional `--token-aux-ce` on the readout. Window evolution is **tension-adaptive** (early exit when stable, noise when unstable), not a fixed step count. Use `--trajectory-batch-size` (≥2) for in-batch negatives. Use `--loss-mode ce` for classic next-token CE only. Run `python sandbox.py --help` for details.

**Data scale:** the default corpus is only **dozens of lines**—enough to smoke-test the code, not to train a useful model. For real runs, add **thousands to tens of thousands** of lines (one sentence per line; UTF-8) with words from the vocab. With a tiny hold-out, **val CE is noisy**; the script warns when the validation window count is very small—trust **train CE** and qualitative generations until you have enough lines and a larger val split.

**Diagnostics:** `--epoch-metrics-csv run.csv` appends per-epoch columns (mean loss, CE, traj contrast, **mean final-step tension**, max batch loss, LR) for plotting. `--log-hard-batch-loss-above 0.2` prints the first context in a batch when loss spikes. `--lr-decay-every 10 --lr-gamma 0.5` applies `StepLR` after each epoch block.

### Example run (trajectory mode, default corpus, Mar 2026)

From the repo root with the venv active (CPU; ~12 min for 25 epochs on a typical laptop):

```bash
source .venv/bin/activate
python sandbox.py \
  --epoch-metrics-csv metrics.csv \
  --log-hard-batch-loss-above 0.22 \
  --lr 0.001 \
  --lr-decay-every 15 \
  --lr-gamma 0.7
```

Observed on one run (49 usable train lines after filtering; **val CE is not trustworthy** with only 2 held-out lines):

| Quantity (last epoch) | Order of magnitude |
|------------------------|--------------------|
| `mean_loss` (trajectory objective: contrastive + weighted aux CE) | ~0.17 |
| `train_CE` | ~0.64 |
| `val_CE` | ~8 (noisy; tiny val split) |
| `train_traj_contrast` / `val_traj_contrast` | ~0.05 / ~0.2 (val noisy) |
| `mean_final_T` (mean tension at last adaptive step) | ~0.24 |
| Wall time | ~720 s total pre-training |

**Tension curves:** With geometry-only tension (`WINDOW_TENSION_USE_ENTROPY = False` in `sandbox.py`), final-step tension often stays **above** `WINDOW_TENSION_TOL_GEOMETRY`, so runs may show **`Steps: 16`** (full `MAX_WINDOW_STEPS`) every batch—decaying curves are still useful to see smooth relaxation vs oscillation. For early exit, raise tol slightly or enable entropy in tension (and use the entropy tol/high constants).

**Qualitative checks:** `compare_prompts` on reordered text can show **large L2 / moderate cosine** (context sensitivity). Generation on ~50 lines remains template-like; scale the corpus for language quality.

Full Phase 0 copy-paste block for `docs/BASELINE.md` is maintained in **`docs/BASELINE.md`** (recorded run).

On startup the script prints **corpus coverage**: how many lines are long enough after dropping out-of-vocabulary words. The **512-token vocabulary** is built from (1) a few legacy seed sentences, (2) **all unique words in `data/corpus.txt`**, then (3) filling to 512 from a large word blob. Custom `--corpus` files are not added to the vocab at runtime—use words already in the vocab or edit the default corpus / vocab construction.

This runs pre-training (epochs, sliding windows, progress lines), then sample generations, attractor debug, and prompt comparisons. Training is **CPU-heavy**; adjust epochs or corpus size if needed.

## Performance notes

The script is tuned for **CPU** research runs, not large-batch GPU training.

- **Vocabulary lookup** uses an internal **`word → index` dict** (`O(1)` per token). Avoid `list.index` on the vocab list in hot paths (the bundled code does not).
- **Single-token signals** use a direct **embedding row + LayerNorm** path instead of allocating a new index tensor every time.
- **Tension** uses a compact **dot-product cosine** instead of `cosine_similarity` on stacked vectors.
- **Optimizer** uses `zero_grad(set_to_none=True)` to reduce overhead per step.
- **Threading**: for multi-core CPU matrix work you can set `OMP_NUM_THREADS` / `MKL_NUM_THREADS` in the environment (values depend on your machine; often matches physical cores).

## Main knobs (in `sandbox.py`)

| Idea | Where |
|------|--------|
| Window length / epochs / entropy bonus | `WINDOW_SIZE`, `NUM_EPOCHS`, `ENTROPY_WEIGHT`, `ENTROPY_FLOOR`, `DRIFT_MIN` |
| Window differentiation (anti-collapse) | `WINDOW_NONLINEAR_GAIN` (tanh strength in window path), `POSITION_ASYM_STRENGTH`, `WINDOW_INTERACTION_SCALE_INIT`, `WINDOW_POSITION_GAMMA_INIT` |
| Tension-adaptive window | `MAX_WINDOW_STEPS`; `WINDOW_TENSION_USE_ENTROPY` (if False, tension is geometry-only: energy + λ·mismatch — no readout entropy); separate tol/high for geometry vs entropy (`WINDOW_TENSION_TOL_GEOMETRY`, `WINDOW_TENSION_TOL_ENTROPY`, …); coupling step `WINDOW_INTERACTION_DT_INIT`, `WINDOW_NONLINEAR_GAIN` |
| Data CLI | `--corpus`, `--val-fraction`, `--seed`, `--epoch-copies`, `--baseline-out`, `--window-size`, `--num-dynamics-steps`, `--trajectory-batch-size`, `--quick-test`, `--loss-mode`, `--token-aux-ce`, `--lr`, `--lr-decay-every`, `--lr-gamma`, `--epoch-metrics-csv`, `--log-hard-batch-loss-above` |
| Training regularization | `LABEL_SMOOTHING`, `BIGRAM_TRAIN_WEIGHT`, `TRAIN_LOGIT_NOISE` |
| Generation sampling | `GEN_TOP_K`, `GEN_REPEAT_LOGIT_PENALTY`, `GEN_NO_REPEAT_LAST_EXTRA`, `GEN_TENSION_TEMP_SCALE`, `generation_temperature` (constructor arg) |
| Slow memory | `slow_decay`; learnable `slow_lr` |
| Tension / adaptive steps | `TENSION_LAMBDA`, `TENSION_MU`, `TENSION_TOL`, `MAX_CONVERGENCE_STEPS`, `TENSION_BREAK_THRESH`, `max_convergence_steps` (constructor) |
| Readout / context | `w_fast`, `w_slow`; learnable `gamma`, `agent_blend_weight` |
| Base inner steps per token | `convergence_steps` |
| Dynamics | `beta`, `noise_scale`, `lambda_decay`, `signal_scale` in `SimpleAttractorDynamics` |

## API sketch

- `TorchAttractorLanguageModel` — **training and generation** use the same path: `window_ids_from_sequence` → `embed_window` → `run_window_dynamics` → `readout_window` (`forward_training_window`). Legacy helpers (`get_signal`, `evolve_token`, `next_token_logits`, …) remain for experiments / tension tooling but are not used in the default `generate` loop.
- `encode_prompt` — returns converged window state tensor `(W, D)` after dynamics (for analysis / `compare_prompts`).
- `compare_prompts(model, prompt_a, prompt_b)` — L2 / cosine between flattened window states
- `build_sequence_dataset(tokens, window_size)` — sliding (context, target) pairs
- `generate(..., log_dynamics=True)` — prints per–outer-step metrics (norm change, token variance, cosine to previous step) while sampling

## Submodules (TS-OS integration)

This repo uses git submodules to pin the TS-OS dependency stack. Clone with:

```bash
git clone --recurse-submodules https://github.com/BoggersTheFish/woke-baby-llm.git
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

| Path | Repo | Branch | Role |
|------|------|--------|------|
| `vendor/GOAT-TS` | [GOAT-TS](https://github.com/BoggersTheFish/GOAT-TS) | `main` | Constraint-graph engine, tension semantics |
| `vendor/TS-Core` | [TS-Core](https://github.com/BoggersTheFish/TS-Core) | `master` | UniversalLivingGraph, WaveCycleRunner (Rust + Python) |
| `vendor/ts-llm` | [ts-llm](https://github.com/BoggersTheFish/ts-llm) | `main` | Tokenizer, hierarchical fast/slow dynamics, attractor LLM package |

SHAs are pinned at the time each Wave PR lands. To update a submodule to latest:

```bash
git -C vendor/<name> pull
git add vendor/<name>
git commit -m "chore: bump <name> submodule"
```

## License

[MIT](LICENSE)
