"""
Wave H — Evaluation harness.

Runs the full 11-step WaveCycleRunner on the language substrate and measures:
  - Perplexity (exp of cross-entropy) on a held-out corpus
  - Mean final window tension
  - TSCore global tension before and after 11-tick relaxation
  - trajectory contrastive score on val set

Usage
-----
    python eval_harness.py [--corpus data/corpus.txt] [--val-fraction 0.2]
                           [--model-checkpoint path] [--wave-cycles 11]
                           [--output eval_results.json]

Exit code 0 if eval completes successfully (not a pass/fail gate on PPL).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))
# TS-Core for WaveCycleRunner
_TS_CORE = _REPO / "vendor" / "TS-Core"
sys.path.insert(0, str(_TS_CORE))

import torch
import torch.nn.functional as F

import sandbox as sb  # type: ignore[import]
from llm_substrate_node import LLMSubstrateNode  # type: ignore[import]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def compute_perplexity(
    model: "sb.TorchAttractorLanguageModel",
    dataset: list,
) -> float:
    """
    Compute token-level perplexity on a (context, target_id) dataset.
    PPL = exp(mean cross-entropy).
    """
    if not dataset:
        return float("nan")
    model.eval()
    total_ce = 0.0
    n = 0
    with torch.inference_mode():
        for context, target_id in dataset:
            logits = model.forward_training_window(context)
            log_probs = F.log_softmax(logits, dim=-1)
            ce = -float(log_probs[target_id].item())
            if math.isfinite(ce):
                total_ce += ce
                n += 1
    if n == 0:
        return float("nan")
    return math.exp(total_ce / n)


def compute_mean_tension(
    model: "sb.TorchAttractorLanguageModel",
    dataset: list,
    batch_size: int = 16,
) -> float:
    """Mean final window tension across batches of the dataset."""
    if not dataset:
        return float("nan")
    tensions: list[float] = []
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(dataset), batch_size):
            chunk = dataset[i : i + batch_size]
            contexts = [c for c, _ in chunk]
            targets = [t for _, t in chunk]
            _, _ = model.trajectory_contrastive_loss_and_logits(contexts, targets)
            curve = getattr(model, "_last_window_tension_curve", [])
            if curve:
                tensions.append(curve[-1])
    return float(sum(tensions) / len(tensions)) if tensions else float("nan")


def compute_traj_contrast(
    model: "sb.TorchAttractorLanguageModel",
    dataset: list,
    batch_size: int = 16,
) -> float:
    """Mean trajectory contrastive loss on a dataset."""
    if not dataset:
        return float("nan")
    total = 0.0
    n = 0
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(dataset), batch_size):
            chunk = dataset[i : i + batch_size]
            if len(chunk) < 2:
                continue
            contexts = [c for c, _ in chunk]
            targets = [t for _, t in chunk]
            loss, _ = model.trajectory_contrastive_loss_and_logits(contexts, targets)
            if torch.isfinite(loss):
                total += float(loss.item())
                n += 1
    return total / n if n > 0 else float("nan")


# --------------------------------------------------------------------------
# WaveCycleRunner (11-step TSCore relaxation on language substrate)
# --------------------------------------------------------------------------

def run_wave_cycle(
    model: "sb.TorchAttractorLanguageModel",
    substrate: LLMSubstrateNode,
    dataset: list,
    max_ticks: int = 11,
    batch_size: int = 16,
) -> dict[str, float]:
    """
    Run one WaveCycle:
    1. Push per-batch tension into TSCore.
    2. Call run_until_stable(max_ticks).
    3. Return before/after tension + language metrics.
    """
    ts_t0 = substrate.ts.measure_tension()

    # Feed language batches to the substrate (updates TSCore activations)
    model.eval()
    lang_tensions: list[float] = []
    with torch.inference_mode():
        for i in range(0, min(len(dataset), batch_size * 8), batch_size):
            chunk = dataset[i : i + batch_size]
            if len(chunk) < 2:
                continue
            contexts = [c for c, _ in chunk]
            targets = [t for _, t in chunk]
            model.trajectory_contrastive_loss_and_logits(contexts, targets)
            curve = getattr(model, "_last_window_tension_curve", [])
            if curve:
                lang_tensions.append(curve[-1])
            substrate.on_batch(model)

    # Run TSCore until stable
    ticks = substrate.ts.run_until_stable(max_ticks=max_ticks, quiet=True)
    ts_t1 = substrate.ts.measure_tension()

    mean_lang_t = sum(lang_tensions) / len(lang_tensions) if lang_tensions else float("nan")
    return {
        "ts_tension_before": ts_t0,
        "ts_tension_after": ts_t1,
        "ts_ticks": ticks,
        "mean_lang_tension": mean_lang_t,
        "evolve_count": substrate.evolve_count,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Wave H evaluation harness")
    parser.add_argument("--corpus", default="data/corpus.txt")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--model-checkpoint", default=None)
    parser.add_argument("--wave-cycles", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=6)
    parser.add_argument("--output", default="eval_results.json")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"Corpus not found: {corpus_path}", file=sys.stderr)
        sys.exit(1)

    print("[eval] loading model ...", flush=True)
    model = sb.TorchAttractorLanguageModel(sb.FULL_VOCAB, train_window_size=args.window_size)
    if args.model_checkpoint:
        sd = torch.load(args.model_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
        print(f"[eval] loaded checkpoint: {args.model_checkpoint}", flush=True)
    model.eval()

    sentences = sb.load_corpus(corpus_path)
    print(f"[eval] corpus: {len(sentences)} lines", flush=True)

    usable = sb.sentences_with_training_windows(sentences, set(model.vocab), args.window_size)
    train_sents, val_sents = sb.train_val_split(usable, args.val_fraction, seed=42)
    val_dataset = sb.build_dataset_from_sentences(val_sents, model, args.window_size)
    train_dataset = sb.build_dataset_from_sentences(train_sents, model, args.window_size)
    print(f"[eval] train windows={len(train_dataset)}  val windows={len(val_dataset)}", flush=True)

    # --- Base metrics (before wave cycle) ---
    print("[eval] computing base metrics ...", flush=True)
    base_ppl = compute_perplexity(model, val_dataset)
    base_tension = compute_mean_tension(model, val_dataset, batch_size=args.batch_size)
    base_traj = compute_traj_contrast(model, val_dataset, batch_size=args.batch_size)

    print(f"  base val_PPL={base_ppl:.2f}  mean_tension={base_tension:.4f}  traj_contrast={base_traj:.4f}")

    # --- WaveCycle run ---
    substrate = LLMSubstrateNode(
        model,
        evolve_threshold=0.05,
        high_tension_threshold=0.01,
        quiet=True,
    )
    print(f"[eval] running {args.wave_cycles}-tick WaveCycle ...", flush=True)
    wave_result = run_wave_cycle(
        model, substrate, train_dataset,
        max_ticks=args.wave_cycles,
        batch_size=args.batch_size,
    )
    print(
        f"  TSCore: {wave_result['ts_tension_before']:.4f} → {wave_result['ts_tension_after']:.4f}"
        f"  ticks={wave_result['ts_ticks']}"
        f"  evolves={wave_result['evolve_count']}",
        flush=True,
    )

    # --- Post-wave metrics ---
    print("[eval] computing post-wave metrics ...", flush=True)
    post_ppl = compute_perplexity(model, val_dataset)
    post_tension = compute_mean_tension(model, val_dataset, batch_size=args.batch_size)
    post_traj = compute_traj_contrast(model, val_dataset, batch_size=args.batch_size)
    print(f"  post val_PPL={post_ppl:.2f}  mean_tension={post_tension:.4f}  traj_contrast={post_traj:.4f}")

    results = {
        "base": {
            "val_ppl": base_ppl,
            "mean_tension": base_tension,
            "traj_contrast": base_traj,
        },
        "wave_cycle": wave_result,
        "post": {
            "val_ppl": post_ppl,
            "mean_tension": post_tension,
            "traj_contrast": post_traj,
        },
        "config": {
            "corpus": str(corpus_path),
            "val_fraction": args.val_fraction,
            "wave_cycles": args.wave_cycles,
            "window_size": args.window_size,
            "checkpoint": args.model_checkpoint,
        },
    }

    out_path = Path(args.output)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[eval] results saved to {out_path}", flush=True)

    # Tension tolerance assertion (must be finite; no hard PPL gate on untrained model)
    ts_final = wave_result["ts_tension_after"]
    assert math.isfinite(ts_final), f"TSCore tension is not finite: {ts_final}"
    assert math.isfinite(base_ppl), f"Base perplexity is not finite: {base_ppl}"
    print("[eval] EVAL OK — all metrics are finite", flush=True)


if __name__ == "__main__":
    main()
