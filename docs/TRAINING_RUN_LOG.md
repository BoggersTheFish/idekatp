# Training run log

Chronological notes for **substantive** `sandbox.py` runs (real corpora, multi-epoch). Integration smokes on `data/corpus.txt` alone are not logged here.

| Field | Meaning |
|--------|--------|
| **Date** | UTC or local as noted |
| **Hardware** | CPU / GPU, rough throughput |
| **Corpus** | HF source + cache keying (`--hf-max-rows`, `--hf-max-chars`) |
| **Metrics** | End-of-run `train_CE`, `val_CE` from epoch summary; `eval_results.json` if used |

---

## 2026-04-02 — TinyStories CPU slice (verified meaningful run)

**Goal:** Real LM signal on consumer CPU without multi-day epochs.

**Corpus:** `--dataset-source tinystories --hf-max-rows 50000 --hf-max-chars 1500000` → cached `data/cache/hf/tinystories_46047f2aaf7b6ad7399c.txt` (~355k BPE tokens after tiktoken; ~1.5M UTF-8 chars materialized).

**Hardware:** CPU, ~3.85–3.9 optimizer steps/s, **~21.5–22 min/epoch**, **~3.9 h** for 10 epochs (4998 batches/epoch, batch 64).

**Model / train:** `window_size=8`, `state_dim=128`, `num_waves=4`, `vectorized_num_heads=4`, `vocab_cap=8192`, `tiktoken`, trajectory + `token_aux_ce=0.2`, `readout_aux_alpha=0.15`, `lr=0.001`, `grad_clip=1.0`, LR decay every 5 epochs ×0.8 from epoch 5.

**Metrics (epoch 10):** `mean_loss` ~3.63, `train_CE` ~3.88, `val_CE` ~4.81 (`val_ppl` ≈ exp(4.81) ≈ 122). `val_traj_contrast` snapshot ~0.096 (full-val mean; not comparable to train trajectory loss scale).

**Qualitative:** Fixed-prompt generations show TinyStories-like story tone, imperfect grammar, and **order-sensitive** `compare_prompts` (nonzero L2, cosine not equal to 1).

**Artifacts:** `checkpoints/meaningful_run/ckpt_step*.pt`, `metrics_meaningful.csv`, `eval_meaningful.json` (local paths; not committed).

**Full command:** see README → [First real training run](../README.md#first-real-training-run-public-corpus--checkpoint--eval-json) → **Option A1 (CPU-sized)**.

---

## How to add an entry

1. Copy the section template above.
2. Keep metrics copy-paste friendly (no need to paste full generation blobs into git).
3. Link checkpoint dir and CSV names if reproducibility matters.
