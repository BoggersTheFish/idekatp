# API Discovery — Vendored Submodules

Verified entrypoints for all three TS-OS vendored repos as of the pinned submodule SHAs.
Last verified: March 2026.

---

## This repository (BoggersTheLanguageModel) — `sandbox.py` integration surface

| Entry | Role |
|-------|------|
| `_build_tokenizer(mode, vocab_cap)` | Loads `AttractorTokenizer` from `vendor/ts-llm` (`tiktoken` or `fallback` cap). |
| `Phase05Config` (`phase05_config.py`) | Instrumentation: log metrics, batch CSV path, adaptive window dt, neg-def diffusion, tension weights, trajectory negatives. |
| `Phase1Config` (`phase1_config.py`) | Multi-head drift (`num_heads`, `head_dim_mode`), window interaction `C`, head diversity weight, per-head tension logging. |
| `Phase2Config` (`phase2_config.py`) | Directional breaks, residual mixing gate, `C` regularisation / distance decay, head tension weighting, break memory. |
| `TorchAttractorLanguageModel(vocab_size, …, phase05=, phase1=, phase2=, convergence_epsilon=, min_attractor_steps=)` | Core model; defaults construct `Phase05Config` / `Phase1Config` / `Phase2Config` if omitted. Optional window early-exit knobs mirror CLI (`convergence_epsilon=0` runs all outer steps). Set `model.tokenizer` for encode/decode. |
| `embed_window(ids)` / `embed_windows_batch(context_tensor)` | Single window **`(W,) → (W, D)`** vs batch **`(B, W) → (B, W, D)`** with the same norm + row-wise unit norm as stacking **`embed_window`** per row. Trajectory training uses the batched path. |
| `run_window_dynamics(..., context_ids=…, convergence_epsilon=, min_attractor_steps=)` | Outer time loop only; caches static coupling / `C` / GOAT tensors once per call. Optional early exit after `min_attractor_steps` when state delta or `|ΔT_mean|` &lt; epsilon (if epsilon &gt; 0). Tension curves are only materialized when requested; collected step diagnostics stay on device until the loop exits. Break/convergence branching avoids `.item()` scalar sync checks in the loop hot path. |
| `model.dynamics.step(S, signal)` | Unified step API on **`SimpleAttractorDynamics`** or **`VectorizedWindowDynamics`** (CLI default **`--dynamics vectorized`**). |
| `phase05_batch_csv_values()` / `PHASE05_BATCH_CSV_HEADER` | Flat row + column names for `--phase05-batch-metrics-csv` (Phase 0.5 + 1 + 2 + optional attractor diagnostics: steps used, final tension, break count, convergence flag). |
| `AttractorDataPipeline` (`data_pipeline.py`) | Streaming train batches when import succeeds; else legacy in-memory shuffle. |
| `mean_cross_entropy_eval(model, dataset, batch_size=…)` | Held-out CE path. Uses batched `embed_windows_batch -> run_window_dynamics -> readout_window` plus the same eval-time shaping as the old single-window path (bigram bias, repeat penalties, label smoothing). |
| `trajectory_contrastive_loss_and_logits(contexts, targets, teacher_steps=None, update_repulsion_memory=True)` | Main training forward. Student path is unchanged; detached teacher path may use fewer outer steps when `teacher_steps` is set. `update_repulsion_memory=False` keeps evaluation/logging forwards from mutating training repulsion history. |

## Inference cache

| Entry | Role |
|-------|------|
| `AttractorStateCache.step(token_id)` (`state_cache.py`) | Rolling inference update. Builds the current window, applies **`Embedding -> LayerNorm -> row L2`** to match training geometry, then calls `run_window_dynamics` on shape `(1, W, D)`. |

---

## GOAT-TS

Python-only. Import root: `vendor/GOAT-TS/src/`.

> **Note:** `vendor/GOAT-TS/src/graph/__init__.py` imports `sentence_transformers`
> (an optional heavy dependency). Load `models.py` and `memory_manager.py` directly
> via `importlib.util.spec_from_file_location` to avoid this — see `goat_memory_transitions.py`
> for the production pattern.

### Data models — `src/graph/models.py`

| Symbol | Type | Description |
|--------|------|-------------|
| `Node` | dataclass (slots) | Graph node: `node_id`, `label`, `activation`, `state: MemoryState`, `embedding`, `position`, `velocity` |
| `Edge` | dataclass (slots) | Directed edge: `src_id`, `dst_id`, `relation`, `weight` |
| `Wave` | dataclass (slots) | Cognitive episode: `wave_id`, `label`, `tension`, `coherence`, `intensity` |
| `Triple` | dataclass (slots) | Subject–relation–object triple with confidence |
| `MemoryState` | StrEnum | `ACTIVE`, `DORMANT`, `DEEP` |
| `NodeType` | StrEnum | `KNOWLEDGE`, `QUESTION`, `HYPOTHESIS`, `SURPRISE`, `EQUATION`, `CLUSTER`, `GOAL` |

### Memory management — `src/memory_manager.py`

```python
from src.memory_manager import memory_tick, apply_decay_and_transitions

nodes, tick_counts = memory_tick(
    nodes,
    low_activation_ticks,
    decay_rate=0.95,
    active_threshold=0.5,    # activation ≥ this → ACTIVE
    dormant_threshold=0.1,   # activation < this → DORMANT
    ticks_to_deep=3,         # consecutive ticks at DORMANT before → DEEP
)
```

Key functions:

| Function | Description |
|----------|-------------|
| `decay_activations(nodes, decay_rate)` | Exponential decay on all activations |
| `transition_states(nodes, ...)` | ACTIVE / DORMANT transitions by threshold |
| `apply_decay_and_transitions(nodes, ...)` | One tick: decay + transition |
| `promote_to_deep_after_ticks(nodes, ticks, ...)` | DORMANT → DEEP after N ticks |
| `memory_tick(nodes, ticks, ...)` | Full tick: decay + ACTIVE/DORMANT + DEEP promotion |

Default constants: `ACTIVE_THRESHOLD=0.5`, `DORMANT_THRESHOLD=0.1`, `TICKS_TO_DEEP=3`, `DEFAULT_DECAY_RATE=0.95`

### Tension — `src/reasoning/tension.py`

```python
from src.reasoning.tension import compute_tension

result = compute_tension(
    positions={"nodeA": np.array([1.0, 0.0]), "nodeB": np.array([0.5, 0.5])},
    expected_distances={("nodeA", "nodeB"): 1.0},
)
# result.score: float (total squared deviation)
# result.high_tension_pairs: list[(src, dst, delta)]  — top 10
```

### Wave propagation — `src/graph/wave_propagation.py`

```python
from src.graph.wave_propagation import run_wave_propagation

nodes, result = run_wave_propagation(
    nodes, edges,
    input_text="the cat sat",   # OR seed_ids=["node1", "node2"]
    max_hops=5,
    decay=0.1,
    threshold=0.1,
    use_torch=True,             # GPU via PyTorch when available
)
# result.activations: dict[node_id, float]
# result.converged: bool
# result.iterations: int
```

Propagation uses interference-weighted adjacency: aligned embeddings (cosine ≥ 0) scale by `ALIGN_FACTOR=1.2`, opposed by `OPPOSE_FACTOR=0.8`.

---

## TS-Core

Rust crate (`vendor/TS-Core/src/rust/`) with PyO3 bindings (`ts_core_kernel`).
Falls back transparently to pure-Python equivalent when the Rust extension is not built.

**No Rust build required** for Waves 0–E.

Import root: `vendor/TS-Core/src/python/`. Must add `vendor/TS-Core` to `sys.path`.

### TSCore — `src/python/core.py`

```python
import sys
sys.path.insert(0, "vendor/TS-Core")
from src.python.core import TSCore

ts = TSCore(
    damping=0.35,
    data_dir=Path("~/.tscore"),
    on_propagate=callback,        # optional: called after each propagate_wave()
    kernel_wave12=False,          # enable 9-phase Wave 12 OS scheduler
)
```

| Method | Description |
|--------|-------------|
| `ts.add_node(node_id, activation, stability)` | Register a node |
| `ts.add_edge(fr, to, weight)` | Add a weighted constraint edge |
| `ts.propagate_wave(*, quiet) → (tension, icarus_line)` | One tick; Rust if built |
| `ts.run_until_stable(max_ticks, eps) → int` | Iterate until `|Δtension| < eps` |
| `ts.measure_tension() → float` | Std-dev of node activations |
| `ts.factory_evolve()` | Self-improvement tick: append a stability node |
| `ts.graph["nodes"][id]["activation"] = v` | Direct graph mutation |

**Tension metric:** standard deviation of node activations — `sqrt(mean((a − ā)²))`. Converges toward 0 as the graph homogenises.

### Language substrate registration pattern

```python
ts.add_node("llm_substrate", activation=0.5, stability=0.5)
ts.add_edge("ts_native", "llm_substrate", weight=1.0)

# After each training batch:
ts.graph["nodes"]["llm_substrate"]["activation"] = float(sandbox_window_tension)
tension, _ = ts.propagate_wave(quiet=True)

if tension > EVOLVE_THRESHOLD:
    ts.factory_evolve()
```

### Wave 12 (9-phase scheduler)

Activated via `ts.kernel_wave12 = True` or `TSCORE_KERNEL_WAVE12=1` env var. Runs a 9-phase strongest-node scheduler each tick:

1. Strongest-node scan
2. Strongest-node lock + spin budget
3. Initial spin (damping × 1.12)
4–6. Three standard propagation passes
7. Icarus wings seal (stabilise low-stability nodes)
8. Self-validation (tension mid-point check)
9. Pages Island persist (final tension, write to JSONL history)

### Rust bindings (Phase 2+)

```bash
cd vendor/TS-Core && maturin develop
```

Exposes `ts_core_kernel.rust_propagate_wave(graph_json, damping)` and `rust_wave12_propagate(graph_json, damping)`. TSCore auto-detects and uses them when present.

---

## ts-llm

Python package: `vendor/ts-llm/attractor_llm/`. Add `vendor/ts-llm` to `sys.path`.

### Tokenizer — `attractor_llm/tokenizer.py`

```python
from attractor_llm.tokenizer import AttractorTokenizer

tok = AttractorTokenizer(
    encoding_name="gpt2",
    vocab_cap=32768,       # max token ids exposed to the model
    use_tiktoken=True,     # False = word-list fallback
)

ids = tok.encode("the quick brown fox")   # list[int]
text = tok.decode(ids)                    # str
n = tok.get_vocab_size()                  # int (= vocab_cap or fallback size)
```

### MultiHeadDynamics — `attractor_llm/torch_core.py`

```python
from attractor_llm.torch_core import MultiHeadDynamics

mhd = MultiHeadDynamics(
    state_dim=512,
    num_heads=4,
    rank=64,           # per-head low-rank factor
    dt=0.09,
    coupling=0.01,     # cross-head alignment pull
)

# One Euler step: (B, D) → (B, D)
s_next = mhd.forward(state, signal)

# Fixed-step convergence: (D,) or (V, D) → same shape
attractor = mhd.converge_fixed(signal, num_steps=16)
```

**Diffusion per head:** `A_h = U_h V_h + diag(d_h)` — low-rank + diagonal. Negative eigenvalues ensure stability.

### HierarchicalAttractorLLM — `attractor_llm/hierarchy.py`

Full model with explicit fast/slow timescale split. Used as the architectural reference for Wave B vectorisation.

---

## Integration map

| Wave | Primary API used |
|------|-----------------|
| 0 smoke | `TSCore.run_until_stable`, `TSCore.measure_tension` |
| A tokenizer | `AttractorTokenizer.encode`, `AttractorTokenizer.decode` |
| B vectorize | `MultiHeadDynamics.forward`, `MultiHeadDynamics.drift` |
| C cache | Internal `fast_state` / `slow_memory` tensors from `sandbox.py` |
| D data | `AttractorDataPipeline` (wave D) feeding `trajectory_contrastive_loss_and_logits` |
| E shim | `TSCore.add_node`, `TSCore.propagate_wave`, `TSCore.factory_evolve` |
| F memory | `memory_tick`, `MemoryState`, `ACTIVE_THRESHOLD` / `DORMANT_THRESHOLD` |
| G serve | `TSCore` sidecar in `inference_server.py` |
| H eval | `TSCore.run_until_stable(max_ticks=11)` |
