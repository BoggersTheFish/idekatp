# Changelog — BoggersTheLanguageModel

All notable changes to this project are documented here.
Format: [Wave / Phase] — description — date.

The project is documented as **BoggersTheLanguageModel**; canonical source is [github.com/BoggersTheFish/BoggersTheLLM](https://github.com/BoggersTheFish/BoggersTheLLM) (alternate: [idekatp](https://github.com/BoggersTheFish/idekatp)).

---

## Performance — batched trajectory embedding + aux CE (Mar 2026)

- **`sandbox.py`**: **`embed_windows_batch(context_tensor)`** — one **`Embedding` + LayerNorm + row L2** pass for shape **`(B, W)`**; matches stacking **`embed_window`** per row. **`trajectory_contrastive_loss_and_logits`** and **`mean_trajectory_contrastive_eval`** use it for student and teacher (**shifted windows** via **`torch.cat([context[:, 1:], target.unsqueeze(1)], dim=1)`**). **`_aux_ce_loss_batch`** is **fully vectorized** over **`B`** (bigram bias, repeat penalties, entropy floor branch, **`cross_entropy(..., reduction="none")`**).
- **`tests/test_embed_windows_batch.py`**: prints **`max_abs_diff`** between batched vs stacked embeddings; run with **`python3 tests/test_embed_windows_batch.py`**.
- **`.gitignore`**: **`runs/`** for local experiment output.

## Performance — batched validation, cache parity, lighter tracing (Apr 2026)

- **`state_cache.py`**: **`AttractorStateCache.step()`** now uses the same training-time embedding geometry as `embed_window` / `embed_windows_batch`: **`Embedding -> LayerNorm -> row L2`** before `run_window_dynamics`.
- **`sandbox.py`**: **`mean_cross_entropy_eval()`** now evaluates validation in batches with **`embed_windows_batch`**, **`run_window_dynamics`**, and **`readout_window`** instead of one-window-at-a-time `forward_training_window`. Validation shaping is preserved (bigram bias, repeat penalties, label smoothing) but GPU utilization is much better.
- **`sandbox.py`**: **`trajectory_contrastive_loss_and_logits(..., teacher_steps=None)`** optionally runs the detached teacher path with fewer attractor steps for cheaper experiments, restoring `max_window_steps` via `try/finally`.
- **`sandbox.py`**: **`run_window_dynamics`** only materializes tension curves when requested and defers `.item()` conversion for collected step diagnostics until after the outer loop, reducing sync overhead from Phase 0.5 tracing.
- **`goat_memory_transitions.py`**: **`GoatMemoryManager.tick()`** now uses **`torch.bincount`** to count token usage across the batch before applying GOAT activation boosts.
- **`phase05_config.py`** / **README**: clarified that `log_metrics=False` keeps only control-flow tension work and skips heavy tracing.

## Engineering — window dynamics performance (Mar 2026)

- **`sandbox.py`**: **`run_window_dynamics`** caches static per-window tensors (positional coupling weights, Phase‑1 **`C * mask`**, GOAT bonus vector) outside the outer step loop; optional **early convergence** via **`convergence_epsilon`** / **`min_attractor_steps`** (CLI **`--convergence-epsilon`**, **`--min-attractor-steps`**; default epsilon **`0`** preserves full **`max_window_steps`**). **`--dynamics`** default is **`vectorized`**; **`VectorizedWindowDynamics`** is constructed with **`state_dim`**, **`window_size`**, **`max_steps`**. On CUDA: **`torch.set_float32_matmul_precision("high")`**; **`torch.compile`** only **`dyn._step`** (vectorized) or **`dyn._step_rows`** (simple). Phase‑2 directional escape uses **`F.normalize`** on break directions. Last-run diagnostics: **`_last_attractor_steps_used`**, **`_last_final_window_tension_diag`**, **`_last_window_break_count`**, **`_last_convergence_triggered`**.
- **`PHASE05_BATCH_CSV_HEADER`**: appended **`attractor_steps_used`**, **`final_window_tension_diag`**, **`break_count_window`**, **`convergence_triggered`** when batch CSV is enabled.
- **`dynamics_vectorized.py`**: **`run_window_dynamics_vectorized(..., **kwargs)`** forwards extra kwargs to **`model.run_window_dynamics`**.
- **`docs/DEVELOPMENT_ROADMAP.md`**: phased engineering vs validation vs training roadmap.

## Phase 2 — Directional breaks, residual mixing, stable window coupling (Mar 2026)

- **`phase2_config.py`**: `Phase2Config` (directional vs legacy breaks, adaptive α, break rejection, residual head mixing gate, `‖C−I‖²` loss weight, optional distance decay on `C`, head-level `softmax(−T)` drift weighting, break memory buffers).
- **`sandbox.py`**: `TorchAttractorLanguageModel(..., phase2=...)`; `phase2_config_from_args`; CLI `--phase2-*`. Window and token breaks use normalised **`state − prev_state`** with tension-scaled step (fallback random direction if delta norm tiny). Optional revert when tension rises and alignment worsens. `SimpleAttractorDynamics`: **`state + sigmoid(gate)·mixed`** when residual mixing enabled. Interaction step: **`C * exp(−|i−j|/τ)`** when `interaction_decay_tau` set. Trajectory loss adds interaction regulariser when `interaction_reg_weight > 0`.
- **`PHASE05_BATCH_CSV_HEADER`**: `phase2_break_*`, `phase2_head_weight_entropy`, `phase2_interaction_reg_loss` when batch metrics CSV enabled.
- **`scripts/plot_phase05_metrics.py`**: optional plots for Phase 2 columns when present.

## Phase 1 & 0.5 — Config modules (Mar 2026)

- **`phase05_config.py`**, **`phase1_config.py`**: split from `sandbox.py` for `Phase05Config` / `Phase1Config`; wired through `TorchAttractorLanguageModel` and `SimpleAttractorDynamics`.

## Dynamics — unified window loop, cache alignment, vectorized compile (Mar 2026)

- **`sandbox.py`**: Single outer path **`run_window_dynamics`** for training and inference; window tension via **`compute_tension_window`** (alias **`compute_window_tension`**). With **`--dynamics vectorized`**, **`torch.compile`** targets **`dyn._step`** only.
- **`dynamics_vectorized.py`**: **`VectorizedWindowDynamics.forward`** disabled; **`run_window_dynamics_vectorized`** swaps dynamics and delegates to **`model.run_window_dynamics`**.
- **`state_cache.py`**: **`step()`** uses **`run_window_dynamics`** on **`(1, W, D)`** with row-normalized embeddings; **`logits()`** via **`readout`**; slow buffers **`.detach()`** before scalar reads to avoid autograd noise.
- **`scripts/ts_workflow_smoke.py`**: Smoke test for cache + simple/vectorized window dynamics (run with **`.venv/bin/python scripts/ts_workflow_smoke.py`**).

## Data — Hugging Face corpora + training eval JSON (Mar 2026)

- **`data/hf_remote_corpus.py`**: materialize TinyStories (`roneneldan/TinyStories`) or FineWeb-Edu (`HuggingFaceFW/fineweb-edu`, `sample-10BT`) into cached UTF-8 text; CLI `python data/hf_remote_corpus.py tinystories …`.
- **`sandbox.py`**: `--dataset-source {local,tinystories,fineweb-edu}`, `--hf-cache-dir`, `--hf-max-rows`, `--hf-max-chars`, `--hf-refresh`, `--no-synthetic-fallback`, **`--eval-results-json`** (post-training val CE / val PPL, checkpoint path, config).
- **`eval_harness.py`**: same `--dataset-source` / `--hf-*` flags for evaluation on Hub data without a local `--corpus`.
- **`requirements.txt`**: `datasets` package for Hugging Face Hub loading.

## Data — Stream corpus (Mar 2026)

- **Default training path:** entire corpus file(s) → one token stream → sliding windows; line boundaries no longer gate training.
- **Validation:** token-level split (`train_tokens` / `val_tokens`); `build_dataset_from_token_ids` for val windows.
- **`AttractorDataPipeline`:** `streaming_dataset=True` (default), optional `train_token_ids` from sandbox; `epoch_batches` shuffles window indices per epoch.
- **`--no-streaming-dataset`:** restores legacy per-line filtering.
- Logging: `total_tokens`, `train_windows`, `val_windows` instead of “usable lines”.
- Train/val **gap** of `window_size` tokens (no cross-split window leakage). **`epoch_batches(epoch_index)`** shuffles with `seed + epoch_index`. Stream mode **ignores `--epoch-copies`**. Warn when **`val_windows < 50`** (unreliable metrics); tiny-corpus note for GOAT A/B interpretation.

## Phase 0 — Substrate Initialisation (Mar 2026)

- Synced workspace to `origin/main`; tagged `phase-0-baseline`
- Added git submodules: `vendor/GOAT-TS`, `vendor/TS-Core`, `vendor/ts-llm`
- Documented all vendored entrypoints in `docs/API_DISCOVERY.md`
- Added `smoke_test.py`: 5 assertions covering dynamics, tension, training, and TSCore wave cycle

## Wave A — Tokenizer (Mar 2026)

- Added `wave_a_tokenizer.py`: `make_vocab_and_tokenizer()`, `encode_corpus()`, `recommended_state_dim()`
- Added `tiktoken` to `requirements.txt`
- BPE mode (gpt2, vocab_cap=32768) and word-list fallback both verified

## Wave B — Vectorized Dynamics (Mar 2026)

- Added `dynamics_vectorized.py`: `VectorizedWindowDynamics` (MultiHeadDynamics from ts-llm)
- Drop-in `run_window_dynamics_vectorized()` with full API compatibility
- `torch.compile` caching via `get_compiled()`
- Parity tests: both paths produce finite outputs; cosine similarity reported

## Wave C — State Cache (Mar 2026)

- Added `state_cache.py`: `AttractorStateCache`, `generate_with_cache()`
- O(1)-per-token inference using rolling fast + slow state
- Phrase table sliding window; `warmup()` for prompt seeding

## Wave D — Data Pipeline (Mar 2026)

- Added `data_pipeline.py`: `AttractorDataPipeline`
- Streaming txt / JSONL loader with shuffle buffer
- Multi-shard round-robin; yields batches directly compatible with `trajectory_contrastive_loss_and_logits`

## Wave E — TS-OS Integration Shim (Mar 2026)

- Added `llm_substrate_node.py`: `LLMSubstrateNode`
- Registers attractor model as native `TSCore` node
- `on_batch()` pushes language tension into TSCore graph; calls `factory_evolve()` on high tension
- Optional HTTP hook to BoggersTheAI Evolve endpoint via `LLM_HOOK_URL` env var

## Wave F — GOAT-TS Memory Transitions (Mar 2026)

- Added `goat_memory_transitions.py`: `GoatMemoryManager`
- ACTIVE → DORMANT → DEEP transitions per vocabulary token via GOAT-TS `memory_manager`
- `sweep_config()` returns tunable hyperparameter knobs for automated Wave F sweeps
- Robust import path (loads `models.py` and `memory_manager.py` directly to avoid heavy optional deps in GOAT-TS `__init__.py`)

## Wave G — Inference Server (Mar 2026)

- Added `inference_server.py`: FastAPI server, OpenAI-compatible `/v1/completions`
- `AttractorStateCache` backend for O(1) generation latency
- TSCore sidecar: `/ts/propagate`, `/ts/tension`
- Added `Dockerfile` (CPU PyTorch) and `docker-compose.yml` with healthcheck

## Wave H — Evaluation Harness (Mar 2026)

- Added `eval_harness.py`: perplexity, mean tension, trajectory contrastive score
- `run_wave_cycle()`: feeds language batches into TSCore, runs 11-tick `run_until_stable()`
- Phase 0 results (untrained model): TSCore tension 0.149 → 0.0005 after 11 ticks; 5 Evolve events
- Results written to `eval_results.json`
- CLI prefers `--max-ticks`; `--wave-cycles` kept as a deprecated alias

## Integration hardening — training / dynamics / metrics (Mar 2026)

- **GOAT memory:** `activation_bonus` is injected in `_single_window_step` as a per-position `(B, W, D)` signal during window dynamics (not only the legacy `get_signal` path).
- **Dynamics swap:** `SimpleAttractorDynamics` and `VectorizedWindowDynamics` both implement **`step(S, signal)`**; training no longer reaches into `dyn.diffusion` / `dyn.dt` on the model object.
- **Metrics:** `train_ce` in epoch CSV is the mean **training-batch** CE from `readout_window` logits; `val_ce` is held-out eval. **`val_traj_contrast`** is computed on the validation split; **`train_traj_contrast`** is a last-batch trajectory snapshot.
- **TSCore logging:** With `--use-substrate`, each epoch prints evolve delta, `last_ts_tension`, active/idle batch counts; one warning if the substrate never engaged. CSV gains **`tscore_evolves`** and **`tscore_last_tension`**.
- **Docs:** README and this changelog updated to match the above.
