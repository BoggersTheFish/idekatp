"""
Phase 0.5 — instrumentation + stability toggles (no architecture change).

Weights (w1, w2, w3) combine raw tension components:
  T_total = w1 * T_energy + w2 * T_alignment + w3 * T_entropy
Defaults (1.0, TENSION_LAMBDA, TENSION_MU) match pre-Phase05 formulas in sandbox.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Phase05Config:
    log_metrics: bool = False
    """If True, collect per-batch diagnostics and tracing; False keeps only control-flow tension computations."""

    batch_metrics_csv: str | None = None
    """Path to append per-batch CSV rows (requires log_metrics)."""

    enforce_negative_definite_diffusion: bool = False
    """If True, SimpleAttractorDynamics uses D = -(A^T A) - eps I (strictly negative definite)."""

    adaptive_window_dt: bool = False
    """EMA scale on window positional dt from tension (smooth, clamped)."""

    adaptive_dt_spike_thresh: float = 0.22
    adaptive_dt_low_thresh: float = 0.06
    adaptive_dt_smooth: float = 0.12
    adaptive_dt_min_scale: float = 1e-3
    adaptive_dt_max_scale: float = 1.0
    adaptive_dt_spike_factor: float = 0.88
    adaptive_dt_low_factor: float = 1.04

    tension_weights: Tuple[float, float, float] | None = None
    """
    (w_energy, w_align, w_entropy). None = use model buffers tension_lambda / tension_mu
    with energy weight 1.0 (legacy T = E + λ·align + μ·H).
    """

    multi_negative: bool = False
    num_negatives: int = 4
    """Contrastive negatives: random permutations of teacher batch, averaged cosine."""

    trajectory_temperature: float = 1.0
    """Divides (margin) inside ReLU; 1.0 = unchanged."""

    stagnation_delta_thresh: float = 1e-3
    """Fraction of window substeps with mean row-wise ||ΔS|| below this counts as stagnation."""

