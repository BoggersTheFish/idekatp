# Project status — BoggersTheLLM (April 2026)

This document answers: **where the codebase is today**, **what is solid**, **what is experimental**, and **what to do next**. It is meant to stay roughly in sync with `sandbox.py` and `phase05_config.py`; when in doubt, trust the code.

---

## What this project is

A **continuous attractor language model** without transformer attention: window state `S ∈ (B, W, D)` evolves under **positional coupling**, **learned per-wave energy heads**, optional **tension** and **anchor** terms, **Phase 1/2** window coupling and breaks, then **readout** to vocabulary logits. Training defaults to **trajectory contrastive** loss plus auxiliary CE paths. Decoding should use **`model.generate`** / **`forward_training_window`** so logits match training (**`readout_window_logits`** path, optionally with fusion).

---

## Implemented (production-relevant)

| Area | Status | Notes |
|------|--------|--------|
| **Streaming data + train/val split** | Stable | Token-level split, gap `W`, shuffled windows per epoch |
| **Trajectory-guided supervision** | Optional | Precomputed **`(n_windows, W, D)`** on **`AttractorDataPipeline`**, batch nudge + MSE in **`trajectory_contrastive_loss_and_logits`**; CLI **`--trajectory-guidance-*`** (stream mode) |
| **Batched embedding** | Stable | `embed_windows_batch` parity-tested vs stacked `embed_window` |
| **Window dynamics** | Stable | Outer loop in `run_window_dynamics`; inner step is **energy descent** on sum of per-wave energies + optional λ·tension + anchor distance (not `dynamics.step`) |
| **Multi-wave layout** | Stable | `D = num_waves × wave_dim`; contiguous wave blocks in `D` |
| **Per-wave dynamics (token path)** | Stable | `WaveDynamics` × `num_waves` in `evolve_token` when `model.dynamics` is None |
| **Cross-wave interaction (token path)** | Stable | `wave_interaction` linear on `concat(waves)` + residual strength |
| **Per-wave energy heads (window path)** | Stable | `energy_heads[i]` only sees wave `i`; total energy = sum |
| **Vectorized window dynamics module** | Stable | `VectorizedWindowDynamics` on **`wave_dim`** when `--dynamics vectorized`; `step` used in **`evolve_token`** when `mhd` present |
| **`--dynamics simple`** | Stable | Means **no** `VectorizedWindowDynamics`; token path uses **`wave_dynamics`** (not legacy `SimpleAttractorDynamics` unless attached manually) |
| **Readout** | Stable | `readout_window_logits(S)` → optional `Linear(D,D)` per position (`--readout-fusion`) → `readout_window` |
| **Checkpoints** | Stable | `load_model_from_checkpoint`, legacy `energy_head.*` → broadcast to `energy_heads`; optional missing keys for `wave_interaction`, `readout_fusion`, etc. |
| **Anchor freeze (window path)** | Optional | `--phase05-enable-anchor-freeze`; zeros energy grad on converged wave slices; metrics `frozen_fraction_*` |
| **Inference** | Stable | `generate_sample.py`, `inference_server.py` use checkpoint loader + `generate` |
| **GOAT / substrate** | Optional | Integrated; not required for core LM training |
| **Phase 0.5 / 1 / 2 metrics** | Stable | Batch CSV columns include energy, breaks, optional frozen fraction, per-wave energy string |

---

## Gaps and risks

1. **Documentation lag** — Older notes may still mention `run_window_dynamics` calling `dynamics.step` every outer step; the **window** path is primarily **energy + coupling**. Token path uses `step` / `wave_dynamics`. External tutorials must use the **3-tuple** `epoch_batches` API when copying pipeline examples.
2. **Hyperparameter surface** — `num_waves`, `wave_interaction_strength`, `anchor_freeze_threshold`, `readout_fusion`, and per-wave energy heads multiply knobs; **re-baseline** after architectural toggles.
3. **GPU performance** — Roadmap targets (batch &lt; 1 s, etc.) are **aspirational** until measured on your hardware; profile with `scripts/profile_training_step.py`.
4. **Eval vs training** — `state_cache` / `readout(D)` is **not** training-parity decoding; use **`generate`** for real samples.
5. **Tests** — Smoke and parity tests exist; **no** full regression suite for every Phase 0.5 flag combination.

---

## Recommended next steps (prioritized)

### Near term (engineering)

1. **Profile one training step on target GPU** — confirm bottleneck is window loop vs embedding vs readout.
2. **Single “golden” config** — **CPU:** README Option **A1** + **`docs/TRAINING_RUN_LOG.md`** (TinyStories + `hf-max-chars`, `state_dim=128`, `num_waves=4`). **GPU:** scale up rows/chars and batch size; same vectorized head divisibility rule (`wave_dim % vectorized_num_heads == 0`).
3. **Plot script** — extend `plot_phase05_metrics.py` for `frozen_fraction_*` and `energy_per_wave_means` if you rely on them.

### Near term (science / product)

1. **Ablations** — `num_waves`, `wave_interaction_strength`, anchor freeze on/off, readout fusion on/off on the same data slice.
2. **Stability** — watch `mean_final_T`, `val_ce`, and fixed-prompt generations (see `BASELINE.md`).

### Medium term

1. **Fuse or cache** more of the inner energy step if profiling shows it dominates.
2. **Optional:** apply the same “freeze” idea to **tensor paths** that skip expensive work when entire rows are frozen (currently only gradients are masked).

---

## How to update this file

After a major change to `sandbox.py` (new module lists, new loss terms, or changed defaults), update the **Implemented** table and **Gaps** in the same PR.
