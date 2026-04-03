# First milestone training (TinyStories)

Goal: coherent multi-token generation from attractor dynamics. Use trajectory loss + auxiliary CE, vectorized dynamics, and logged attractor-step statistics.

## Recommended hyperparameters

| Setting | Value |
|--------|--------|
| Dataset | `--dataset-source tinystories`; use **`--hf-max-chars`** (e.g. `1500000`) for CPU-sized slices (see **`docs/TRAINING_RUN_LOG.md`**) |
| Window | `--window-size 8` |
| State dim | `--state-dim 128` |
| Heads | `--vectorized-num-heads 4` (requires `state_dim % 4 == 0`) |
| Batch | `--batch-size 64` (trajectory mode) |
| Max outer steps | `--num-dynamics-steps 16` (ceiling) |
| Early exit | `--convergence-epsilon 5e-3` or `1e-2` (target mean steps ~4–6) |
| Adaptive inner scale | `--phase05-adaptive-window-dt` |
| Stronger linear contraction | `--vectorized-strong-diffusion` |
| Aux signal | `--token-aux-ce 0.2` and/or `--readout-aux-alpha 0.15` |

## Example command — CPU-friendly slice (verified Apr 2026)

~22 min/epoch on CPU at batch 64; ~3.9 h for 10 epochs. See **`docs/TRAINING_RUN_LOG.md`** for metrics.

```bash
python3 sandbox.py \
  --dataset-source tinystories \
  --hf-max-rows 50000 \
  --hf-max-chars 1500000 \
  --tokenizer tiktoken \
  --vocab-cap 8192 \
  --val-fraction 0.1 \
  --window-size 8 \
  --state-dim 128 \
  --num-waves 4 \
  --vectorized-num-heads 4 \
  --batch-size 64 \
  --max-epochs 10 \
  --lr 0.001 \
  --grad-clip 1.0 \
  --lr-decay-every 5 \
  --lr-gamma 0.8 \
  --epoch-metrics-csv metrics_meaningful.csv \
  --eval-results-json eval_meaningful.json \
  --checkpoint-dir checkpoints/meaningful_run \
  --save-every 500
```

## Example command — larger corpus (GPU / long run)

Materialize TinyStories (adjust `--hf-max-rows` for “full” corpus), train several epochs, save checkpoints and metrics:

```bash
python sandbox.py \
  --dataset-source tinystories \
  --hf-max-rows 500000 \
  --tokenizer tiktoken \
  --window-size 8 \
  --state-dim 128 \
  --batch-size 64 \
  --vectorized-num-heads 4 \
  --vectorized-rank 64 \
  --num-dynamics-steps 16 \
  --convergence-epsilon 0.01 \
  --min-attractor-steps 2 \
  --phase05-adaptive-window-dt \
  --vectorized-strong-diffusion \
  --loss-mode trajectory \
  --token-aux-ce 0.2 \
  --readout-aux-alpha 0.15 \
  --max-epochs 5 \
  --lr 0.001 \
  --save-every 2000 \
  --checkpoint-dir checkpoints/tinystories_milestone \
  --epoch-metrics-csv metrics_epoch.csv \
  --attractor-steps-metrics-csv metrics_attractor_steps.csv
```

## What to watch

- `metrics_attractor_steps.csv`: `mean_steps` trending toward 4–6; `convergence_failures` not dominating.
- `metrics_epoch.csv` (if enabled): trajectory contrastive, train CE, tension.
- Phase 0.5 batch CSV (optional `--phase05-log-metrics --phase05-batch-metrics-csv ...`) for fine-grained tension and step counts.

## After training

`scripts/generate_sample.py` calls **`sandbox.load_model_from_checkpoint`** (vectorized dynamics + `dt` + Lorentz from checkpoint `config`) then **`model.generate`** — same readout path as training.

```bash
python scripts/generate_sample.py \
  --checkpoint checkpoints/tinystories_milestone/ckpt_stepXXXXXXX.pt \
  --tokenizer tiktoken \
  --prompts "Once upon a time" "The scientist discovered"
```

See `docs/FAILURE_ANALYSIS.md` if generation is poor.
