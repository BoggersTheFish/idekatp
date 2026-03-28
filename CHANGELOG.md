# Changelog — BoggersTheLanguageModel

All notable changes to this project are documented here.
Format: [Wave / Phase] — description — date.

The project is documented as **BoggersTheLanguageModel**; canonical source is [github.com/BoggersTheFish/idekatp](https://github.com/BoggersTheFish/idekatp).

---

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
