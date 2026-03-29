# Phase 0 — Baseline and success criteria (BoggersTheLanguageModel)

Phase 0 locks a **reproducible reference** for **BoggersTheLanguageModel** (`sandbox.py`) before scaling data, windows, or model size. Use it to answer “did the change help?” without guessing.

For runs on a **public or large text corpus** (Hugging Face TinyStories / FineWeb-Edu or your own files), follow README → [First real training run](https://github.com/BoggersTheFish/BoggersTHeLLM#first-real-training-run-public-corpus--checkpoint--eval-json). Use **`--eval-results-json`** with `sandbox.py` to write val CE / val PPL and the checkpoint path alongside training.

## How to record a baseline run

1. From the repo root, with your venv active:

   ```bash
   source .venv/bin/activate
   python sandbox.py --baseline-out docs/BASELINE_LAST_RUN.txt
   ```

   Add any flags you use in production (e.g. trajectory mode, LR schedule, CSV logging):

   ```bash
   python sandbox.py \
     --epoch-metrics-csv metrics.csv \
     --log-hard-batch-loss-above 0.22 \
     --lr 0.001 \
     --lr-decay-every 15 \
     --lr-gamma 0.7 \
     --baseline-out docs/BASELINE_LAST_RUN.txt
   ```

2. The script prints a **Phase 0 baseline** block at the end (metrics + three fixed generations). `--baseline-out` saves that block to a file for git or notes.

3. Copy the printed block into the **“Recorded baseline”** section below when you want a frozen snapshot in git.

**Fixed generation prompts** (defined in `sandbox.py` as `BASELINE_PROMPT_1` … `BASELINE_PROMPT_3`) are always the same so outputs are comparable across runs.

## Metrics to care about

| Metric | Meaning |
|--------|--------|
| **train_ce** (CSV / log) | Mean **per-batch** CE on `readout_window` logits vs targets during training (trajectory mode). **Not** the same as `val_ce`. |
| **val_ce** (CSV / log) | Held-out **`mean_cross_entropy_eval`**. **Ignore absolute value** when the val set is tiny; use trend after scale-up. **Val PPL** ≈ `exp(val_ce)`. |
| **mean_loss** (last epoch) | With `--loss-mode trajectory` (default): full step objective (trajectory contrastive + token aux CE + readout aux weights). |
| **train_traj_contrast** | Trajectory contrastive loss on the **last training batch** of the epoch (diagnostic). |
| **val_traj_contrast** | Mean trajectory contrastive loss over the **full validation** set (when val exists). |
| **mean_final_T** | Mean window tension at the **last** adaptive dynamics step each epoch—track drift via `--epoch-metrics-csv`. |
| **tscore_evolves** / **tscore_last_tension** | With `--use-substrate`: per-epoch evolve delta and last TSCore tension. |

Architecture changes will change absolute numbers—re-record baseline after major `sandbox.py` updates.

## v1 scale-up success check (agreed criteria)

Treat a change as **successful for v1** when **both** hold:

1. **Calibration:** **val_ce** is **lower than this baseline** (same seed/split settings), or at least not worse while **train_ce** improves — i.e. no clear collapse to memorized noise. Only meaningful once the val set has **enough windows**.
2. **Subjective quality:** On the **three fixed prompts**, text shows **less pointless repetition** than the baseline generations, without becoming random gibberish.

Optional: note wall time and epoch count if you change data size or model size.

---

## Recorded baseline

Official snapshot from a **trajectory-mode** run on the default corpus (CPU, seed 42). Command:

```bash
source .venv/bin/activate
python sandbox.py \
  --epoch-metrics-csv metrics.csv \
  --log-hard-batch-loss-above 0.22 \
  --lr 0.001 \
  --lr-decay-every 15 \
  --lr-gamma 0.7
```

| Field | Value |
|-------|--------|
| Date (UTC) | 2026-03-28T15:49:20+00:00 |
| Git commit (training run) | `6858eca` |
| Git commit (docs + trajectory `sandbox.py` on `main`) | `1685e55` |
| Corpus | `data/corpus.txt` — 51 lines loaded, 49 usable (≥7 in-vocab tokens), 2 train / val lines for val split |
| Last epoch | 25/25 |
| Windows per epoch | 180 |
| `mean_loss` (objective) | 0.1737 |
| `train_CE` | 0.6365 |
| `val_CE` | 7.9570 (noisy) |
| `train_traj_contrast` | 0.046814 |
| `val_traj_contrast` | 0.200000 |
| Train wall time (total pre-training) | 718.7 s |
| Notes | LR stepped down with `StepLR` every 15 epochs (`lr-gamma` 0.7). Window dynamics often used all 16 steps (`MAX_WINDOW_STEPS`) because geometry tension stayed above early-exit tolerance. |

### Phase 0 block (copy-paste)

```
--- Phase 0 baseline (copy into docs/BASELINE.md) ---
time_utc: 2026-03-28T15:49:20+00:00
git: 6858eca
corpus: data/corpus.txt
seed: 42  val_fraction: 0.05  epoch_copies: 2
loss_mode: trajectory  token_aux_ce: 0.2
window_size: 6  num_dynamics_steps: 16  num_epochs: 25
last_epoch: 25/25  windows: 180  epoch_sec: 16.0
train_sec_total: 718.7
mean_loss (objective): 0.1737
train_CE: 0.6365  val_CE: 7.9570
train_traj_contrast: 0.046814  val_traj_contrast: 0.200000

--- generation baseline prompt 1 ---
the quick brown fox jumps over the lazy dog and then what happens in the system of mind and reason system yak one dance clear cause we the system patterns effect reason move like tie the lazy in clear reason of effect system ink flow dead lives reason time care the lazy coin clear reason ready pattern come mind lazy

--- generation baseline prompt 2 ---
mind reason cause effect system the flow inside reason cause of the system demands pattern reason the strong of pure reason responds pattern flow the demands of clear reason lid pattern mind patterns flow effect clear reason one us the acts demands the effect system

--- generation baseline prompt 3 ---
effect cause reason mind system the lazy concept pay cost active and dog remains sea cell brown cause action eat mind effect chin reason flow cause into effect one the job of mind patterns share brown us was cause the effect hub action clear reason
--- end baseline ---
```

### Debug attractor (representative)

One prompt with `log_dynamics`-style metrics at end of training:

- **Tension curve:** monotone decay over 16 steps (example final values ~0.24–0.29), not oscillatory.
- **mean_var** (token variance across positions): ~0.00155 — still low; differentiation remains the main qualitative bottleneck at tiny data scale.
- **mean_cos(step):** ~0.998 — smooth step-to-step updates.
- **compare_prompts** (order sensitivity): e.g. L2(window) ~3.0–3.3, cosine ~0.12–0.25 between reordered same-word prompts.

---

## Example row (illustrative only)

Older CE-only runs might show different `mean_loss` semantics. After scaling data, you want **train_CE** and (with a real val split) **val_CE** to move together sensibly, and the three prompt outputs to read **less repetitively** than the snapshot above.
