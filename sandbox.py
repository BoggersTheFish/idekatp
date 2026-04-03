import argparse
import csv
import datetime
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections import Counter

from tqdm import tqdm
from pathlib import Path

from phase05_config import Phase05Config
from phase1_config import Phase1Config
from phase2_config import Phase2Config

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- repository root and default paths ----------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS_PATH = _REPO_ROOT / "data" / "corpus.txt"

# Minimal built-in sentences used when no corpus file is found.
_FALLBACK_SENTENCES = [
    "the problem appears but the solution exists because the reason is clear therefore the system stays stable",
    "mind understands the cause and the effect creates a stable system",
    "the quick brown fox jumps over the lazy dog and then the pattern flows into the future",
]


def _build_tokenizer(mode: str = "fallback", vocab_cap: int = 32768):
    """
    Build an AttractorTokenizer from vendor/ts-llm.

    mode='tiktoken'  — BPE gpt2 encoding, vocab_cap tokens (default 32768)
    mode='fallback'  — same BPE but capped at 512 for fast prototyping
    """
    _ts_llm = _REPO_ROOT / "vendor" / "ts-llm"
    if str(_ts_llm) not in sys.path:
        sys.path.insert(0, str(_ts_llm))
    try:
        from attractor_llm.tokenizer import AttractorTokenizer  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "vendor/ts-llm missing — run: git submodule update --init --recursive"
        ) from exc
    actual_cap = vocab_cap if mode == "tiktoken" else min(vocab_cap, 512)
    return AttractorTokenizer(
        encoding_name="gpt2",
        vocab_cap=actual_cap,
        use_tiktoken=True,
    )


# Legacy shim so external scripts that import sandbox.FULL_VOCAB keep working.
# This is a placeholder — real vocab comes from _build_tokenizer().
FULL_VOCAB: list[str] = []


# Anti-collapse: trajectory drift in readout; entropy floor for sampling / training logits.
DRIFT_MIN = 0.008
# Min entropy (nats) before extra logit noise; ~log(V) is max. Too low (e.g. 0.02) fires on every confident step.
ENTROPY_FLOOR = 2.0
# Training: lighter exploratory noise when floor triggers (large σ destroys the CE signal).
ENTROPY_FLOOR_NOISE = 0.12
TRAIN_LOGIT_NOISE = 0.005
# Generation: top-k caps tail mass; repeat penalties reduce "effect effect" / same-token loops.
GEN_TOP_K = 28
GEN_REPEAT_LOGIT_PENALTY = 1.35
# Extra penalty on the single most recent token (blocks immediate repeats harder).
GEN_NO_REPEAT_LAST_EXTRA = 5.0
# Training: bigram bias scale (too high pulls logits toward embedding self-similarity loops).
BIGRAM_TRAIN_WEIGHT = 0.025
LABEL_SMOOTHING = 0.06

# Trajectory training (default): match evolved state of context window to evolved state of shifted
# teacher window [x2..xW, next_token]. Auxiliary CE on readout(pred_state) keeps decoding trainable.
TOKEN_AUX_CE_WEIGHT_DEFAULT = 0.2

# --- GOAT-TS-style tension (adaptive dynamics + symplectic readout) ---
# T ≈ |ΔE_state| + λ(1 - cos(fast,slow)) + μ·H(logits); used to adapt inner steps and modulate noise.
TENSION_LAMBDA = 0.35
TENSION_MU = 0.08
TENSION_TOL = 0.85
MAX_CONVERGENCE_STEPS = 12
TENSION_BREAK_THRESH = 2.5
TENSION_NOISE_GAIN = 0.15
GEN_TENSION_TEMP_SCALE = 0.035

# Training: full-window state (W × D), tension-adaptive interaction + step_state (no attention).
NUM_WINDOW_DYNAMICS_STEPS = 8  # legacy default for max_window_steps if not overridden
MAX_WINDOW_STEPS = 16
# Window tension: two regimes — with readout entropy (scale ~0.7–0.9) vs geometry-only (much smaller).
WINDOW_TENSION_USE_ENTROPY = False
WINDOW_TENSION_TOL_GEOMETRY = 0.05
WINDOW_TENSION_HIGH_GEOMETRY = 0.18
WINDOW_TENSION_TOL_ENTROPY = 0.75
WINDOW_TENSION_HIGH_ENTROPY = 0.92
TRAJECTORY_BATCH_SIZE_DEFAULT = 64
# Below this many val windows, val CE / PPL are not statistically reliable.
MIN_VAL_WINDOWS = 50
MIN_VAL_WINDOWS_RELIABLE = MIN_VAL_WINDOWS  # legacy alias
# Larger outer step → more movement per iteration (target mean_cos(step) ~0.98–0.995 vs ~0.997+).
WINDOW_INTERACTION_DT_INIT = 0.09
# Sharper distance decay → less mixing of far positions (reduces over-smoothing vs nonlinearity).
WINDOW_POSITION_GAMMA_INIT = 0.52
# Lower than earlier defaults: coupling was washing token rows together (low mean_var in logs).
WINDOW_INTERACTION_SCALE_INIT = 0.07
# Extra gain on tanh(c) in step_state_batch for the window path only (differentiation vs collapse).
WINDOW_NONLINEAR_GAIN = 4.0
# Scales (1 + strength * tanh(asym) * sign(j−i)); was 0.5, too weak for left/right contrast.
POSITION_ASYM_STRENGTH = 1.25

# Phase 0.5 per-batch CSV columns (see phase05_batch_csv_values).
PHASE05_BATCH_CSV_HEADER = [
    "epoch",
    "batch_idx",
    "global_step",
    "tension_curve_final",
    "student_T_total",
    "student_T_energy",
    "student_T_align",
    "student_T_entropy",
    "outer_mean_T_total",
    "outer_mean_T_energy",
    "outer_mean_T_align",
    "outer_mean_T_entropy",
    "state_norm_mean",
    "state_norm_std",
    "win_delta_l2_mean",
    "stagnation_frac",
    "cos_pos",
    "cos_neg",
    "margin",
    "break_high_count",
    "break_low_jitter_count",
    "break_avg_t_pre",
    "break_avg_t_post",
    "interaction_dt_scale",
    "token_break_high_count",
    "token_low_tension_exit_count",
    "token_break_avg_t_pre",
    "token_break_avg_t_post",
    "loss_traj_core",
    "loss_traj_full",
    "batch_ce",
    "phase1_interaction_rms",
    "phase1_tension_heads_mean",
    "phase1_head_div_loss",
    "phase2_break_direction_norm_mean",
    "phase2_break_applied_alpha_mean",
    "phase2_break_delta_tension_mean",
    "phase2_break_delta_alignment_mean",
    "phase2_head_weight_entropy",
    "phase2_interaction_reg_loss",
    "attractor_steps_used",
    "attractor_steps_mean_lifetime",
    "final_window_tension_diag",
    "break_count_window",
    "convergence_triggered",
    "energy_descent_base_mean",
    "energy_descent_tension_mean",
    "energy_descent_total_mean",
    "energy_per_wave_means",
    "energy_potential_mean",
    "energy_reg_mean_sq",
    "energy_reg_weighted",
    "anchor_energy_mean",
    "anchor_energy_std",
    "frozen_fraction_mean",
    "frozen_fraction_std",
]


def sample_next_token_id(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    recent_token_ids: list,
    repeat_penalty: float,
    no_repeat_last_extra: float,
) -> int:
    """Apply repetition penalty, optional top-k, temperature, multinomial sample."""
    lo = logits.clone()
    for tid in recent_token_ids[-4:]:
        lo[tid] -= repeat_penalty
    if recent_token_ids:
        lo[recent_token_ids[-1]] -= no_repeat_last_extra
    if top_k > 0 and top_k < lo.numel():
        tk_logits, tk_idx = torch.topk(lo, top_k)
        scaled = (tk_logits - tk_logits.max()) / temperature
        probs = F.softmax(scaled, dim=-1)
        j = torch.multinomial(probs, 1).item()
        return int(tk_idx[j].item())
    scaled = (lo - lo.max()) / temperature
    probs = F.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, 1).item())


# ==================== FIXED TORCH MODEL (shape bugs corrected) ====================
class TorchAttractorLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        state_dim=512,
        convergence_steps=4,
        slow_decay=0.05,
        slow_lr=0.05,
        w_fast=1.0,
        w_slow=0.3,
        gamma_init=0.2,
        generation_temperature=1.02,
        max_convergence_steps=MAX_CONVERGENCE_STEPS,
        train_window_size: int = 6,
        max_window_steps: int = MAX_WINDOW_STEPS,
        convergence_epsilon: float = 0.0,
        min_attractor_steps: int = 2,
        num_waves: int = 8,
        wave_interaction_strength: float = 0.05,
        use_readout_fusion: bool = False,
        phase05: Phase05Config | None = None,
        phase1: Phase1Config | None = None,
        phase2: Phase2Config | None = None,
    ):
        super().__init__()
        self.phase05_config = phase05 if phase05 is not None else Phase05Config()
        self.phase1_config = phase1 if phase1 is not None else Phase1Config()
        self.phase2_config = phase2 if phase2 is not None else Phase2Config()
        self.register_buffer("_phase05_interaction_dt_scale", torch.tensor(1.0))
        self.vocab_size = vocab_size
        self.state_dim = state_dim
        self.num_waves = max(1, int(num_waves))
        if self.state_dim % self.num_waves != 0:
            raise ValueError(
                f"state_dim={self.state_dim} must be divisible by num_waves={self.num_waves}"
            )
        self.wave_dim = self.state_dim // self.num_waves
        self.train_window_size = train_window_size
        self.max_window_steps = max_window_steps
        # Early exit for window attractor (0 = disabled; see run_window_dynamics).
        self.convergence_epsilon = float(convergence_epsilon)
        self.min_attractor_steps = max(2, int(min_attractor_steps))
        _wtol = (
            WINDOW_TENSION_TOL_ENTROPY
            if WINDOW_TENSION_USE_ENTROPY
            else WINDOW_TENSION_TOL_GEOMETRY
        )
        _whigh = (
            WINDOW_TENSION_HIGH_ENTROPY
            if WINDOW_TENSION_USE_ENTROPY
            else WINDOW_TENSION_HIGH_GEOMETRY
        )
        self.register_buffer("window_tension_tol", torch.tensor(float(_wtol)))
        self.register_buffer("window_tension_high", torch.tensor(float(_whigh)))
        # Partial updates per token (path-dependent evolution; not full relaxation).
        self.convergence_steps = convergence_steps
        self.max_convergence_steps = max_convergence_steps
        # Slow memory: slow = (1 - slow_decay) * slow + slow_lr * fast (decay prevents unbounded growth).
        self.register_buffer("slow_decay", torch.tensor(float(slow_decay)))
        self.slow_lr = nn.Parameter(torch.tensor(float(slow_lr)))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        # Decode / context mix: symplectic half-step uses w_fast/w_slow on midpoint fast + slow.
        self.register_buffer("w_fast", torch.tensor(float(w_fast)))
        self.register_buffer("w_slow", torch.tensor(float(w_slow)))
        # Extra temperature at generation time (escapes shallow attractors in sampling).
        self.register_buffer("generation_temperature", torch.tensor(float(generation_temperature)))
        # Context-dependent signal injection strength (trajectory sensitivity).
        self.register_buffer("signal_eps", torch.tensor(1e-6))
        self.wave_dynamics = nn.ModuleList(
            [WaveDynamics(self.wave_dim) for _ in range(self.num_waves)]
        )
        _wc = self.num_waves * self.wave_dim
        self.wave_interaction = nn.Linear(_wc, _wc, bias=True)
        nn.init.normal_(self.wave_interaction.weight, std=0.02)
        nn.init.zeros_(self.wave_interaction.bias)
        self.register_buffer(
            "wave_interaction_strength",
            torch.tensor(float(wave_interaction_strength)),
        )
        self.dynamics: nn.Module | None = None
        _w = train_window_size
        _tau = self.phase2_config.interaction_decay_tau
        if _tau is not None and _tau > 0:
            idx = torch.arange(_w, dtype=torch.float32)
            dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
            self.register_buffer(
                "_phase2_C_dist_mask",
                torch.exp(-dist / float(_tau)),
            )
        else:
            self.register_buffer("_phase2_C_dist_mask", torch.ones(_w, _w))
        if self.phase1_config.enable_window_interaction:
            off = 1.0 - torch.eye(_w)
            self.phase1_window_C = nn.Parameter(
                torch.eye(_w) + 0.01 * torch.randn(_w, _w) * off
            )
        else:
            self.register_buffer("phase1_window_C", torch.zeros(_w, _w))
        self.embedder = nn.Embedding(self.vocab_size, state_dim)
        self.norm = nn.LayerNorm(state_dim, elementwise_affine=False)
        self.readout = nn.Linear(self.state_dim, self.vocab_size, bias=False)
        # Training: W·D → V. Prefer :meth:`readout_window_logits` / :meth:`_readout_window_logits` so
        # optional :attr:`readout_fusion` is applied; calling this layer alone skips fusion.
        self.readout_window = nn.Linear(
            train_window_size * state_dim, self.vocab_size, bias=False
        )
        self.use_readout_fusion = bool(use_readout_fusion)
        if self.use_readout_fusion:
            self.readout_fusion = nn.Linear(state_dim, state_dim, bias=True)
            nn.init.xavier_uniform_(self.readout_fusion.weight)
            nn.init.zeros_(self.readout_fusion.bias)
        else:
            self.readout_fusion = None
        # One small MLP per wave: E_i = head_i(wave slice); total head energy = sum_i E_i (grad w.r.t. wave i only from E_i).
        def _one_energy_head() -> nn.Sequential:
            m = nn.Sequential(
                nn.Linear(self.wave_dim, self.wave_dim),
                nn.Tanh(),
                nn.Linear(self.wave_dim, 1),
            )
            for _em in m.modules():
                if isinstance(_em, nn.Linear):
                    nn.init.xavier_uniform_(_em.weight)
                    if _em.bias is not None:
                        nn.init.zeros_(_em.bias)
            return m

        self.energy_heads = nn.ModuleList(
            [_one_energy_head() for _ in range(self.num_waves)]
        )
        # Positional interaction strength (softplus > 0); left/right differ via |i−j|.
        self.position_gamma_raw = nn.Parameter(
            torch.tensor(math.log(math.exp(WINDOW_POSITION_GAMMA_INIT) - 1.0))
        )
        self.interaction_scale_raw = nn.Parameter(
            torch.tensor(math.log(math.exp(WINDOW_INTERACTION_SCALE_INIT) - 1.0))
        )
        self.interaction_dt_raw = nn.Parameter(
            torch.tensor(math.log(math.exp(WINDOW_INTERACTION_DT_INIT) - 1.0))
        )
        # Left vs right neighbor strength (tanh-bounded inside coupling).
        self.position_asym = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("_vocab_ids", torch.arange(self.vocab_size, dtype=torch.long))
        # Unconstrained raw; effective temperature = softplus(raw) > 0 (learnable temp can hit 0 otherwise -> inf logits).
        t0 = 0.12
        self.temperature_raw = nn.Parameter(torch.tensor(math.log(math.exp(t0) - 1.0)))
        # Tension coefficients (buffers; can tune without breaking checkpoints if names stable).
        self.register_buffer("tension_lambda", torch.tensor(float(TENSION_LAMBDA)))
        self.register_buffer("tension_mu", torch.tensor(float(TENSION_MU)))
        self.register_buffer("tension_tol", torch.tensor(float(TENSION_TOL)))
        self.register_buffer("tension_break_thresh", torch.tensor(float(TENSION_BREAK_THRESH)))
        self.tension_noise_gain = nn.Parameter(torch.tensor(float(TENSION_NOISE_GAIN)))
        self.agent_blend_weight = nn.Parameter(torch.tensor(-0.4))
        # Last inner-step tension (float) for generation temperature adaptation.
        self._last_tension_val = 0.0
        # Symplectic readout: fast at start of token vs end (midpoint).
        self._fast_start_snapshot: torch.Tensor | None = None
        # Multi-agent light: recent token signals (normalized embedding directions).
        self._context_ring: list[torch.Tensor] = []
        # Debug: attractor keys and last-step metrics (set by evolve_token when track_attractors=True).
        self.track_attractors = False
        self._attractor_counts: Counter = Counter()
        self._last_state_norm = 0.0
        self._last_state_delta = 0.0
        self._last_combined_norm = 0.0
        self._last_slow_norm = 0.0
        # Trajectory drift pressure in readout (reset at each new sequence via reset_readout_trajectory).
        self._prev_combined = None
        # Filled when collect_dynamics_metrics=True in forward_training_window / run_window_dynamics.
        self._last_dynamics_logs: list[dict] | None = None
        self._last_window_tension_mean: torch.Tensor | None = None
        self._last_adaptive_window_steps: int = 0
        # Mean tension after each outer step (last run_window_dynamics with record_tension_log=True).
        self._last_window_tension_curve: list[float] = []
        # Integration hooks — set by caller after construction.
        self.tokenizer = None           # AttractorTokenizer for encode / decode
        self._goat_mgr = None           # GoatMemoryManager for per-token activation bonuses
        self._last_pred_final_state: torch.Tensor | None = None  # (B, D) after trajectory dynamics
        # Phase 1 repetition fix: rolling mean final-row states for repulsion term (last 8 batches).
        self._repulsion_prev_states: list[torch.Tensor] = []
        self._last_traj_student_tension_mean: torch.Tensor | None = None
        self._goat_transition_context_ids: list[list[int]] | list[int] | None = None
        # Phase 0.5: evolve_token break tallies (cleared each trajectory batch when logging).
        self._phase05_evolve_high_breaks: int = 0
        self._phase05_evolve_low_exits: int = 0
        self._phase05_evolve_t_pre_sum: float = 0.0
        self._phase05_evolve_t_post_sum: float = 0.0
        self._last_traj_cos_pos: float = float("nan")
        self._last_traj_cos_neg: float = float("nan")
        self._last_traj_margin: float = float("nan")
        self._phase05_last_window_trace: dict[str, float] | None = None
        self._phase2_last_break_pre: torch.Tensor | None = None
        self._phase2_last_break_post: torch.Tensor | None = None
        self._phase2_interaction_reg_logged: float = float("nan")
        self._phase2_head_weight_entropy_logged: float = float("nan")
        # Last window run diagnostics (optional logging / CSV extensions).
        self._last_attractor_steps_used: int = 0
        self._last_final_window_tension_diag: float = float("nan")
        self._last_window_break_count: int = 0
        self._last_convergence_triggered: bool = False
        self._attractor_steps_cumulative_sum: float = 0.0
        self._attractor_steps_cumulative_count: int = 0
        # Last inner energy-descent step (updated when logging/metrics enabled).
        self._last_energy_descent_base: float | None = None
        self._last_energy_descent_tension: float | None = None
        self._last_energy_descent_total: float | None = None
        self._last_energy_descent_anchor_mean: float = float("nan")
        self._last_energy_descent_anchor_std: float = float("nan")
        # Mean per-wave scalar energy on last inner descent leaf (B,W averaged); set with record_metrics.
        self._last_energy_descent_per_wave: list[float] | None = None
        # Nearest-anchor stats on student ``S_pred`` (trajectory batch CSV).
        self._phase05_anchor_energy_mean: float = float("nan")
        self._phase05_anchor_energy_std: float = float("nan")
        self._phase05_anchor_contrast_loss: float = float("nan")
        # Per-batch window potential stats (``trajectory_contrastive_loss_and_logits``).
        self._last_batch_energy_mean: float = float("nan")
        self._last_batch_energy_variance: float = float("nan")
        self._last_batch_energy_min: float = float("nan")
        self._last_batch_energy_max: float = float("nan")
        # Adaptive inner dt (shape (B,) during ``run_window_dynamics`` when enabled).
        self._attractor_dt_per_batch: torch.Tensor | None = None
        self._attractor_max_dt_per_batch: torch.Tensor | None = None
        self._last_attractor_dt_curve: list[float] = []

    def _reshape_waves_flat(self, S: torch.Tensor) -> torch.Tensor:
        """``(B, W, D) -> (B * W * num_waves, wave_dim)`` (contiguous wave blocks in D)."""
        B, W, D = S.shape
        if D != self.state_dim:
            raise ValueError(f"expected last dim {self.state_dim}, got {D}")
        return S.reshape(B * W * self.num_waves, self.wave_dim)

    def _window_coupling_flat(self, S: torch.Tensor) -> torch.Tensor:
        """``(B, W, D) -> (B * num_waves, W, wave_dim)`` for per-wave positional coupling."""
        B, W, D = S.shape
        return S.reshape(B, W, self.num_waves, self.wave_dim).permute(0, 2, 1, 3).reshape(
            B * self.num_waves, W, self.wave_dim
        )

    def _from_window_coupling_flat(self, Sfw: torch.Tensor, B: int) -> torch.Tensor:
        """Inverse of :meth:`_window_coupling_flat`."""
        nw, W, wd = self.num_waves, self.train_window_size, self.wave_dim
        return Sfw.reshape(B, nw, W, wd).permute(0, 2, 1, 3).reshape(B, W, self.state_dim)

    def _window_batch_full_state(self, S: torch.Tensor) -> torch.Tensor:
        """
        Normalize to ``(B, W, D)`` with ``W = train_window_size``, ``D = state_dim``.
        Accepts single window ``(W, D)``, batched flat ``(B, W * D)``, or ``(B, W, D)``.
        Wave channels are contiguous in ``D`` (same layout as dynamics / embedding).
        """
        Wm, Dtot = self.train_window_size, self.state_dim
        if S.dim() == 2:
            if S.shape[0] == Wm and S.shape[1] == Dtot:
                return S.unsqueeze(0)
            if S.shape[1] == Wm * Dtot:
                B = S.shape[0]
                return S.reshape(B, Wm, Dtot)
            raise ValueError(
                f"window state: expected ({Wm}, {Dtot}) or (B, {Wm * Dtot}), got {tuple(S.shape)}"
            )
        if S.dim() == 3:
            if S.shape[1] != Wm or S.shape[2] != Dtot:
                raise ValueError(
                    f"window state: expected (B, {Wm}, {Dtot}), got {tuple(S.shape)}"
                )
            return S
        raise ValueError(f"window state expects 2D or 3D tensor, got dim {S.dim()}")

    def _readout_window_logits(self, S: torch.Tensor) -> torch.Tensor:
        """
        Full-window state (all wave channels in ``D``) → vocab logits ``(B, V)``.
        Optional :attr:`readout_fusion` ``Linear(D, D)`` on the last dim per position, then
        existing :attr:`readout_window` on ``flatten(B, W, D)``.
        """
        Sb = self._window_batch_full_state(S)
        if self.readout_fusion is not None:
            Sb = self.readout_fusion(Sb)
        B = Sb.size(0)
        return self.readout_window(
            Sb.reshape(B, self.train_window_size * self.state_dim)
        )

    def readout_window_logits(self, S: torch.Tensor) -> torch.Tensor:
        """Public alias for :meth:`_readout_window_logits` (training / inference parity)."""
        return self._readout_window_logits(S)

    def _per_wave_energy_scalars(self, S: torch.Tensor) -> torch.Tensor:
        """
        Per-wave scalar energy before summing: ``(B, W, num_waves)``.
        ``[..., i] = energy_heads[i](S[..., i, :])`` — gradients from ``E_i`` hit only wave ``i``.
        """
        B, W, D = S.shape
        if D != self.state_dim:
            raise ValueError(f"expected state_dim {self.state_dim}, got {D}")
        Sw = S.reshape(B, W, self.num_waves, self.wave_dim)
        parts: list[torch.Tensor] = []
        for i in range(self.num_waves):
            xi = Sw[:, :, i, :].reshape(B * W, self.wave_dim)
            ei = self.energy_heads[i](xi).view(B, W)
            parts.append(ei)
        return torch.stack(parts, dim=2)

    def _wave_energy_head_per_batch_row(self, S: torch.Tensor) -> torch.Tensor:
        """Per batch row: mean over window of total head energy ``sum_i E_i`` → ``(B,)``."""
        ew = self._per_wave_energy_scalars(S)
        total_bw = ew.sum(dim=2)
        return total_bw.mean(dim=1)

    def _wave_energy_head_global_mean(self, S: torch.Tensor) -> torch.Tensor:
        """Scalar mean over (B, W) of ``sum_i E_i``."""
        ew = self._per_wave_energy_scalars(S)
        return ew.sum(dim=2).mean()

    def _clamp_window_waves_norm(
        self, S: torch.Tensor, floor: float, ceiling: float | None
    ) -> torch.Tensor:
        """L2 clamp on each ``(wave_dim,)`` sub-vector (per batch row, time, wave)."""
        B, W, D = S.shape
        Sfw = self._window_coupling_flat(S)
        out = self._clamp_window_rows_norm(Sfw, floor, ceiling)
        return self._from_window_coupling_flat(out, B)

    def energy(self, window_states: torch.Tensor) -> torch.Tensor:
        """
        Total learned potential per window position: ``(B, W, D) -> (B, W, 1)``.
        ``E = sum_i energy_heads[i](wave_i)`` (per-wave heads; gradient from ``E`` to wave ``i``
        flows only through ``energy_heads[i]``).
        """
        if window_states.dim() != 3:
            raise ValueError(f"energy expects (B, W, D), got shape {tuple(window_states.shape)}")
        b, w, d = window_states.shape
        if d != self.state_dim:
            raise ValueError(f"energy: last dim {d} != state_dim {self.state_dim}")
        ew = self._per_wave_energy_scalars(window_states)
        return ew.sum(dim=2, keepdim=True)

    def _dynamics_euler_dt(self) -> float:
        """Step size used for window inner updates (matches previous ``dynamics.step`` Euler ``dt``)."""
        d = self.dynamics
        raw = getattr(d, "dt", None)
        if raw is not None:
            return float(raw.detach().item() if isinstance(raw, torch.Tensor) else raw)
        mhd = getattr(d, "mhd", None)
        if mhd is not None:
            raw = getattr(mhd, "dt", None)
            if raw is not None:
                return float(raw.detach().item() if isinstance(raw, torch.Tensor) else raw)
        return 0.09

    def _readout_window_logits_for_anchors(self, S: torch.Tensor) -> torch.Tensor:
        """``readout_window`` logits (B, V), temperature-scaled — same shaping as training readout."""
        logits = self._readout_window_logits(S)
        return torch.nan_to_num(
            logits / self.effective_temperature(),
            nan=0.0,
            posinf=0.0,
            neginf=-1e4,
        )

    def _window_anchor_candidate_embeds(self, S: torch.Tensor) -> torch.Tensor:
        """
        Per batch row, top-``k`` token rows from ``readout_window`` logits (embeddings detached).
        Returns ``(B, k, D)`` with ``k = min(anchor_search_topk, V)``.
        """
        if S.dim() != 3:
            raise ValueError(f"anchor candidates expect (B, W, D), got {tuple(S.shape)}")
        B, W, D = S.shape
        if W != self.train_window_size:
            raise ValueError(
                f"anchor candidates: W={W} != train_window_size={self.train_window_size}"
            )
        k = min(
            max(1, int(self.phase05_config.anchor_search_topk)),
            self.vocab_size,
        )
        logits = self._readout_window_logits_for_anchors(S)
        _vals, topk_idx = logits.topk(k=k, dim=-1)
        embed = self.embedder.weight.detach().to(device=S.device, dtype=S.dtype)
        return embed[topk_idx]

    def _window_anchor_nearest_among_candidates(
        self, S: torch.Tensor, cand: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        ``S`` (B, W, D), ``cand`` (B, k, D) detached → min L2 distance (B, W) and nearest vector (B, W, D).
        """
        diff = S.unsqueeze(2) - cand.unsqueeze(1)
        dist = torch.linalg.vector_norm(diff, dim=-1)
        _min_d, argmin_k = dist.min(dim=2)
        B, W = S.shape[0], S.shape[1]
        b_ar = torch.arange(B, device=S.device, dtype=torch.long).unsqueeze(1).expand(B, W)
        nearest_vec = cand[b_ar, argmin_k, :]
        return _min_d, nearest_vec

    def _apply_token_anchor_force(self, S: torch.Tensor) -> torch.Tensor:
        """
        Pull each (B, W) position toward the nearest **readout top-k** token embedding (detached).
        ``S <- S - strength * (S - anchor[nearest])``. No-op when ``anchor_force_strength`` is 0.
        """
        fs = float(self.phase05_config.anchor_force_strength)
        if fs == 0.0:
            return S
        if S.dim() != 3:
            raise ValueError(f"anchor force expects (B, W, D), got {tuple(S.shape)}")
        cand = self._window_anchor_candidate_embeds(S)
        _d, nearest_vec = self._window_anchor_nearest_among_candidates(S, cand)
        return S - fs * (S - nearest_vec)

    def _window_anchor_nearest_distances(self, S: torch.Tensor) -> torch.Tensor:
        """
        Per position L2 distance to nearest **readout top-k** token embedding (anchors detached).
        ``S``: (B, W, D) → ``(B, W)``.
        """
        if S.dim() != 3:
            raise ValueError(f"anchor distances expect (B, W, D), got {tuple(S.shape)}")
        cand = self._window_anchor_candidate_embeds(S)
        min_dist, _ = self._window_anchor_nearest_among_candidates(S, cand)
        return min_dist

    def _window_anchor_nearest_distances_per_wave(self, S: torch.Tensor) -> torch.Tensor:
        """
        Per (batch, position, wave) L2 distance from ``S``'s wave slice to the nearest top-``k``
        anchor's **same** wave slice (anchors detached). ``S``: ``(B, W, D)`` → ``(B, W, num_waves)``.
        """
        if S.dim() != 3:
            raise ValueError(
                f"per-wave anchor distances expect (B, W, D), got {tuple(S.shape)}"
            )
        B, W, D = S.shape
        if D != self.state_dim:
            raise ValueError(f"last dim {D} != state_dim {self.state_dim}")
        cand = self._window_anchor_candidate_embeds(S)
        nw, wd = self.num_waves, self.wave_dim
        S_w = S.reshape(B, W, nw, wd)
        C_w = cand.reshape(B, -1, nw, wd)
        diff = S_w.unsqueeze(2) - C_w.unsqueeze(1)
        dist_k = torch.linalg.vector_norm(diff, dim=-1)
        nearest_dist, _ = dist_k.min(dim=2)
        return nearest_dist

    def _window_anchor_energy_per_batch_row(self, S: torch.Tensor) -> torch.Tensor:
        """Mean nearest-anchor distance over window positions, per batch row: (B,)."""
        return self._window_anchor_nearest_distances(S).mean(dim=1)

    def _window_energy_per_batch_row(self, S: torch.Tensor) -> torch.Tensor:
        """
        Same potential as mean descent objective, reduced per batch row: (B,).
        ``mean(rows)`` matches the scalar loss used for ``autograd.grad``.

        ``E = e_head + λ·T + anchor_λ·anchor_energy_row`` (tension / anchor terms omitted when
        their lambdas are 0).
        """
        lam = float(self.phase05_config.tension_lambda)
        al = float(self.phase05_config.anchor_lambda)
        e_head = self._wave_energy_head_per_batch_row(S)
        out = e_head
        if S.dim() == 3 and S.size(1) >= 2:
            if lam != 0.0:
                T = self.compute_tension_window(S)
                lam_t = torch.as_tensor(lam, device=S.device, dtype=S.dtype)
                out = out + lam_t * T
        if al != 0.0:
            ae = self._window_anchor_energy_per_batch_row(S)
            al_t = torch.as_tensor(al, device=S.device, dtype=S.dtype)
            out = out + al_t * ae
        return out

    @staticmethod
    def _clamp_window_rows_norm(
        S: torch.Tensor, floor: float, ceiling: float | None
    ) -> torch.Tensor:
        """Per-row L2 norm clamp on ``(B, W, D)`` (same idea as vectorized ``_clamp_norm``)."""
        n = torch.linalg.vector_norm(S, dim=-1, keepdim=True)
        out = S
        if floor > 0:
            out = torch.where(n < floor, out * (floor / (n + 1e-15)), out)
            n = torch.linalg.vector_norm(out, dim=-1, keepdim=True)
        if ceiling is not None:
            out = torch.where(n > ceiling, out * (ceiling / n), out)
        return out

    def _window_energy_gradient_step(
        self,
        S_prep: torch.Tensor,
        *,
        record_metrics: bool = False,
        attractor_dt_per_batch: torch.Tensor | None = None,
        anchor_freeze_wave: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        One descent step: ``S <- S - dt * ∇_S E(S)`` with
        ``E = mean(sum_i energy_heads[i](wave_i)) + λ * mean(T) + anchor_λ * mean(anchor_dist)``
        where ``T`` is ``compute_tension_window`` (``λ = phase05_config.tension_lambda``, 0 off) and
        anchor distances use detached ``embedder.weight`` (``anchor_lambda``, 0 off).
        Uses a short inner graph each call to avoid unbounded backprop through time.

        If ``attractor_dt_per_batch`` is a ``(B,)`` tensor, uses per-row ``dt`` and updates it
        from energy before vs after the step (see ``phase05_config.enable_adaptive_attractor_dt``).

        If ``anchor_freeze_wave`` is ``(B, W, num_waves)`` bool (``True`` = frozen), zero gradient
        on the corresponding wave slices so only unstable slices receive ``-dt * grad`` updates.
        """
        B = S_prep.size(0)
        grad_enabled = torch.is_grad_enabled()
        use_adaptive = attractor_dt_per_batch is not None
        lam = float(self.phase05_config.tension_lambda)

        def _scalar_energy(st_leaf: torch.Tensor) -> torch.Tensor:
            return self._window_energy_per_batch_row(st_leaf).mean()

        def _run_step(st_leaf: torch.Tensor, *, create_graph: bool) -> torch.Tensor:
            E_before = self._window_energy_per_batch_row(st_leaf).detach()
            energy = _scalar_energy(st_leaf)
            if record_metrics:
                ew = self._per_wave_energy_scalars(st_leaf)
                base_energy = ew.sum(dim=2).mean()
                with torch.no_grad():
                    pwm = ew.mean(dim=(0, 1))
                    self._last_energy_descent_per_wave = [
                        float(pwm[i].detach().item()) for i in range(self.num_waves)
                    ]
                if st_leaf.size(1) >= 2:
                    tension = self.compute_tension_window(st_leaf).mean()
                else:
                    tension = st_leaf.new_zeros(())
                self._last_energy_descent_base = float(base_energy.detach().item())
                self._last_energy_descent_tension = (
                    float(tension.detach().item())
                    if st_leaf.size(1) >= 2
                    else float("nan")
                )
                self._last_energy_descent_total = float(energy.detach().item())
                al = float(self.phase05_config.anchor_lambda)
                if al != 0.0:
                    with torch.no_grad():
                        nd = self._window_anchor_nearest_distances(st_leaf)
                        ndf = nd.reshape(-1)
                        self._last_energy_descent_anchor_mean = float(ndf.mean().item())
                        self._last_energy_descent_anchor_std = float(
                            ndf.std(unbiased=False).item()
                        )
                else:
                    self._last_energy_descent_anchor_mean = float("nan")
                    self._last_energy_descent_anchor_std = float("nan")
            # Default retain_graph matches create_graph (PyTorch). When create_graph=True
            # (training), we must NOT pass retain_graph=False here — that frees the energy
            # subgraph while grad_st still needs it for the outer loss.backward().
            grad_st = torch.autograd.grad(
                energy,
                st_leaf,
                create_graph=create_graph,
            )[0]
            if anchor_freeze_wave is not None:
                Ws = st_leaf.size(1)
                m = (~anchor_freeze_wave).to(dtype=grad_st.dtype)
                m = m.unsqueeze(-1).expand(B, Ws, self.num_waves, self.wave_dim).reshape(
                    B, Ws, self.state_dim
                )
                grad_st = grad_st * m
            if use_adaptive:
                dt_b = attractor_dt_per_batch.view(B, 1, 1).to(
                    device=st_leaf.device, dtype=st_leaf.dtype
                )
            else:
                dt0 = self._dynamics_euler_dt()
                dt_b = st_leaf.new_full((B, 1, 1), dt0, dtype=st_leaf.dtype)
            S_raw = st_leaf - dt_b * grad_st
            S_mid = torch.nan_to_num(S_raw, nan=0.0, posinf=0.0, neginf=0.0)
            if use_adaptive or grad_enabled:
                S_mid = self._clamp_window_waves_norm(S_mid, 1e-3, 12.0)
            if use_adaptive:
                cfg = self.phase05_config
                max_d = self._attractor_max_dt_per_batch
                if max_d is None:
                    raise RuntimeError(
                        "adaptive attractor dt requires _attractor_max_dt_per_batch"
                    )
                with torch.no_grad():
                    E_after = self._window_energy_per_batch_row(S_mid)
                    shrink = E_after > E_before
                    rel = float(cfg.adaptive_attractor_energy_rel_thresh)
                    decr = E_before - E_after
                    grow = (~shrink) & (decr > rel * (torch.abs(E_before) + 1e-8))
                    shrink_f = float(cfg.adaptive_attractor_dt_shrink)
                    grow_f = float(cfg.adaptive_attractor_dt_grow)
                    min_dt = float(cfg.adaptive_attractor_dt_min)
                    new_dt = torch.where(
                        shrink, attractor_dt_per_batch * shrink_f, attractor_dt_per_batch
                    )
                    new_dt = torch.where(
                        grow,
                        torch.minimum(attractor_dt_per_batch * grow_f, max_d),
                        new_dt,
                    )
                    new_dt.clamp_(min=min_dt)
                    attractor_dt_per_batch.copy_(new_dt)
            return S_mid

        if not grad_enabled:
            st = S_prep.detach().requires_grad_(True)
            with torch.enable_grad():
                S_out = _run_step(st, create_graph=False)
            if use_adaptive:
                return S_out
            return torch.nan_to_num(S_out, nan=0.0, posinf=0.0, neginf=0.0)

        st = S_prep.detach().requires_grad_(True)
        return _run_step(st, create_graph=True)

    def _window_row_cos_mean(self, X: torch.Tensor) -> torch.Tensor:
        """Mean cosine between consecutive window rows (B, W, D)."""
        if X.dim() != 3 or X.size(1) < 2:
            return X.new_zeros(())
        return F.cosine_similarity(X[:, 1:], X[:, :-1], dim=-1).mean()

    def _normalize_window_state_if_enabled(self, S: torch.Tensor) -> torch.Tensor:
        """L2-normalize each ``(wave_dim,)`` sub-vector when config enables it."""
        if not self.phase05_config.enable_state_normalization:
            return S
        B, W, D = S.shape
        Sw = S.reshape(B, W, self.num_waves, self.wave_dim)
        Sw = F.normalize(Sw, dim=-1, eps=1e-12)
        return Sw.reshape(B, W, D)

    def _phase2_break_alpha(self, t_mean: torch.Tensor) -> torch.Tensor:
        """Scalar α = base * clamp((T_tgt - T)/T_tgt, min, max); low T → larger α."""
        p2 = self.phase2_config
        tgt = float(p2.break_t_target)
        tm = t_mean.detach() if torch.is_tensor(t_mean) else torch.as_tensor(t_mean)
        ratio = (tgt - tm) / (tgt + 1e-8)
        sc = torch.clamp(
            ratio,
            float(p2.break_min_scale),
            float(p2.break_max_scale),
        )
        return torch.as_tensor(float(p2.break_base_strength), device=tm.device, dtype=tm.dtype) * sc

    def _phase2_directional_escape(
        self,
        S: torch.Tensor,
        delta_ref: torch.Tensor,
        t_mean: torch.Tensor,
        *,
        row_renorm: bool,
        legacy_scale: float,
    ) -> torch.Tensor:
        """state + α * û(delta); random unit fallback if ||delta|| tiny (vectorised)."""
        if S.dim() == 1:
            return self._phase2_directional_escape(
                S.unsqueeze(0),
                delta_ref.unsqueeze(0),
                t_mean,
                row_renorm=row_renorm,
                legacy_scale=legacy_scale,
            ).squeeze(0)
        p2 = self.phase2_config
        if not p2.enable_directional_break:
            if row_renorm:
                z = S + legacy_scale * torch.randn_like(S)
                return z / (torch.linalg.vector_norm(z, dim=-1, keepdim=True) + 1e-8)
            return S + legacy_scale * torch.randn_like(S)
        dn = torch.linalg.vector_norm(delta_ref, dim=-1, keepdim=True)
        rnd = torch.randn_like(delta_ref)
        dir_from_delta = F.normalize(delta_ref, dim=-1, eps=1e-8)
        dir_from_rnd = F.normalize(rnd, dim=-1, eps=1e-8)
        dir_u = torch.where(dn >= 1e-6, dir_from_delta, dir_from_rnd)
        alpha = self._phase2_break_alpha(t_mean)
        if not torch.is_tensor(alpha):
            alpha = torch.as_tensor(alpha, device=S.device, dtype=S.dtype)
        out = S + alpha * dir_u
        if row_renorm:
            out = out / (torch.linalg.vector_norm(out, dim=-1, keepdim=True) + 1e-8)
        return out

    def phase2_interaction_reg_loss(self) -> torch.Tensor:
        """Pull C toward identity (off-diagonal noise regularised)."""
        p2 = self.phase2_config
        z = self.embedder.weight.new_zeros(())
        if p2.interaction_reg_weight <= 0.0 or not self.phase1_config.enable_window_interaction:
            return z
        W = self.train_window_size
        C = self.phase1_window_C
        I = torch.eye(W, device=C.device, dtype=C.dtype)
        return ((C - I) ** 2).sum()

    def _phase05_tension_w(self) -> tuple[float, float, float]:
        """(w_energy, w_align, w_entropy) for T = w1*E + w2*A + w3*H; None weights → legacy λ, μ."""
        if self.phase05_config.tension_weights is not None:
            return self.phase05_config.tension_weights
        return (1.0, float(self.tension_lambda.item()), float(self.tension_mu.item()))

    def _phase1_global_interaction_step(
        self,
        S: torch.Tensor,
        C_masked: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Learnable cross-position mix: delta[b,j,d] = sum_i S[b,i,d] C[i,j]; add scale * delta.
        No softmax — linear routing only.
        ``C_masked`` optional cached ``C * mask`` for the window (same each attractor step).
        """
        p = self.phase1_config
        if not p.enable_window_interaction:
            self._phase1_last_interaction_rms = None
            return S, None
        if C_masked is None:
            C = self.phase1_window_C.to(device=S.device, dtype=S.dtype)
            mask = self._phase2_C_dist_mask.to(device=S.device, dtype=S.dtype)
            C_eff = C * mask
        else:
            C_eff = C_masked
        delta = torch.einsum("bid,ij->bjd", S, C_eff)
        rms = torch.sqrt(delta.pow(2).mean() + 1e-12)
        self._phase1_last_interaction_rms = rms
        return S + float(p.interaction_scale) * delta, rms

    def _phase1_per_head_tension_means(self, S: torch.Tensor) -> torch.Tensor | None:
        """(B, H) geometry tension per head slice; entropy omitted (no per-head readout)."""
        p = self.phase1_config
        if not p.enable_per_head_tension or p.num_heads < 2:
            return None
        B, W, D = S.shape
        H = p.num_heads
        if D % H != 0:
            return None
        dh = D // H
        parts = S.reshape(B, W, H, dh)
        dpos = parts[:, 1:] - parts[:, :-1]
        T_e = dpos.pow(2).mean(dim=(1, 3))
        cos = F.cosine_similarity(parts[:, :-1], parts[:, 1:], dim=-1)
        T_a = (1.0 - cos).mean(dim=1)
        w1, w2, _w3 = self._phase05_tension_w()
        w1t = torch.as_tensor(w1, device=S.device, dtype=S.dtype)
        w2t = torch.as_tensor(w2, device=S.device, dtype=S.dtype)
        return w1t * T_e + w2t * T_a

    def phase1_head_diversity_loss(self) -> torch.Tensor:
        """Mean pairwise cosine of batch-mean head drift directions; minimise to decorrelate heads."""
        p = self.phase1_config
        dyn = self.dynamics
        z0 = self.embedder.weight.new_zeros(())
        if p.head_diversity_weight <= 0:
            return z0
        if not isinstance(dyn, SimpleAttractorDynamics) or dyn.num_heads < 2:
            return z0
        d = dyn._last_multihead_drifts
        if d is None:
            return z0
        zn = F.normalize(d, dim=-1, eps=1e-8)
        m = F.normalize(zn.mean(dim=0), dim=-1, eps=1e-8)
        sim = torch.mm(m, m.T)
        mask = torch.triu(
            torch.ones_like(sim, dtype=torch.bool), diagonal=1
        )
        n = int(mask.sum().item())
        if n <= 0:
            return z0
        return (sim * mask.to(sim.dtype)).sum() / float(n)

    @property
    def goat_memory(self):
        """Shim so run_window_dynamics can call self.goat_memory.apply_transition(...)."""
        return self

    def apply_transition(self, fast_state: torch.Tensor, fr: str, to: str) -> torch.Tensor:
        """GOAT DORMANT→ACTIVE jitter hook (Phase 1 repetition fix); returns fast_state unchanged if no-op."""
        mgr = self._goat_mgr
        ctx = self._goat_transition_context_ids
        if mgr is None or ctx is None or fr != "DORMANT" or to != "ACTIVE":
            return fast_state
        try:
            from dataclasses import replace

            from goat_memory_transitions import MemoryState
        except Exception:
            return fast_state
        if not ctx:
            rows = []
        elif isinstance(ctx[0], list):
            rows = ctx  # type: ignore[assignment]
        else:
            rows = [ctx]  # type: ignore[list-item]
        for row in rows:
            for tid in row:
                if isinstance(tid, int) and 0 <= tid < len(mgr.nodes):
                    n = mgr.nodes[tid]
                    if n.state == MemoryState.DORMANT:
                        mgr.nodes[tid] = replace(
                            n,
                            state=MemoryState.ACTIVE,
                            activation=max(float(n.activation), float(mgr.active_threshold)),
                        )
        try:
            mgr.invalidate_bonus_cache()
        except Exception:
            pass
        return fast_state

    def reset_readout_trajectory(self):
        """Clear stored combined state for drift pressure (call once per training window / at generate start)."""
        self._prev_combined = None
        self._context_ring = []
        self._fast_start_snapshot = None
        self._last_tension_val = 0.0

    def _state_energy(self, fast: torch.Tensor) -> torch.Tensor:
        return torch.sum(fast * fast)

    def _normalized_token_embedding(self, token_id: int) -> torch.Tensor:
        """Single-token path: LayerNorm row + unit direction (matches batched embed + norm)."""
        row = self.embedder.weight[token_id].unsqueeze(0)
        emb = self.norm(row).squeeze(0)
        n0 = torch.linalg.vector_norm(emb).clamp(min=1e-12)
        return emb / n0

    def compute_tension(
        self,
        fast: torch.Tensor,
        slow: torch.Tensor,
        logits: torch.Tensor,
        prev_energy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scalar tension T and components; logits are vocab logits for entropy term."""
        e = self._state_energy(fast)
        de = torch.abs(e - prev_energy)
        fnf = torch.linalg.vector_norm(fast)
        fns = torch.linalg.vector_norm(slow)
        cos_fs = ((fast * slow).sum() / (fnf * fns + 1e-12)).clamp(-1.0, 1.0)
        div = 1.0 - cos_fs
        probs = F.softmax(logits, dim=-1)
        H = -(probs * (probs.clamp(min=1e-9)).log()).sum(dim=-1)
        w1, w2, w3 = self._phase05_tension_w()
        T = w1 * de + w2 * div + w3 * H
        return T, de, div, H

    def _symplectic_combined(self, fast: torch.Tensor, slow: torch.Tensor) -> torch.Tensor:
        """Half-step (Störmer-style) blend: midpoint in fast, static slow for this sub-step."""
        fast, slow = self._init_dual_state(fast, slow)
        fs = self._fast_start_snapshot
        if fs is None:
            fs = fast
        fast_mid = 0.5 * (fast + fs)
        return self.w_fast * fast_mid + self.w_slow * slow

    def _logits_for_tension(self, fast: torch.Tensor, slow: torch.Tensor) -> torch.Tensor:
        """readout(D) for tension entropy only; next-token training uses readout_window via forward_training_window."""
        combined = self._symplectic_combined(fast, slow)
        state = combined / (torch.linalg.vector_norm(combined) + 1e-8)
        logits = self.readout(state) / self.effective_temperature()
        return torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=-1e4)

    def effective_temperature(self) -> torch.Tensor:
        return F.softplus(self.temperature_raw).clamp(min=1e-6)

    def _context_vector(self, fast_state, slow_state):
        """Unit direction from fast (or weighted combined) for context injection; expects inited dual state."""
        combined = self._symplectic_combined(fast_state, slow_state)
        fast_norm = torch.linalg.vector_norm(fast_state)
        eps = self.signal_eps
        device = fast_state.device
        dtype = fast_state.dtype
        if float(fast_norm.detach()) > 1e-8:
            return fast_state / (fast_norm + eps)
        cn = torch.linalg.vector_norm(combined)
        if float(cn.detach()) > 1e-8:
            return combined / (cn + eps)
        return torch.zeros(self.state_dim, device=device, dtype=dtype)

    def get_signal(self, token_id: int, fast_state=None, slow_state=None) -> torch.Tensor:
        """Context-sensitive input: base embedding + gamma * normalized context; then unit-scale signal."""
        fast_state, slow_state = self._init_dual_state(fast_state, slow_state)
        self._fast_start_snapshot = fast_state.detach().clone()
        base_signal = self._normalized_token_embedding(token_id)
        if len(self._context_ring) >= 2:
            w = torch.sigmoid(self.agent_blend_weight)
            ring_mean = torch.stack(self._context_ring).mean(0)
            base_signal = (1.0 - w) * base_signal + w * ring_mean
        self._context_ring.append(base_signal.detach().clone())
        if len(self._context_ring) > 4:
            self._context_ring.pop(0)
        context_vector = self._context_vector(fast_state, slow_state)
        signal = base_signal + self.gamma * context_vector
        sn = torch.linalg.vector_norm(signal)
        signal = signal / (sn + self.signal_eps)
        # Phase 8: GOAT memory activation bonus — shifts signal direction toward active tokens.
        if self._goat_mgr is not None:
            bonus = float(self._goat_mgr.activation_bonus(token_id))
            if bonus > 0.0:
                signal = signal + bonus
                sn2 = torch.linalg.vector_norm(signal)
                signal = signal / (sn2 + self.signal_eps)
        return signal

    def all_signals(self, fast_state, slow_state):
        """All vocab signals in one batched pass (avoids 512× Python loop and duplicate graphs)."""
        fast_state, slow_state = self._init_dual_state(fast_state, slow_state)
        ids = self._vocab_ids.to(device=fast_state.device)
        emb = self.norm(self.embedder(ids))
        n0 = torch.linalg.vector_norm(emb, dim=-1, keepdim=True).clamp(min=1e-12)
        base_signals = emb / n0
        ctx = self._context_vector(fast_state, slow_state)
        signals = base_signals + self.gamma * ctx
        sn = torch.linalg.vector_norm(signals, dim=-1, keepdim=True).clamp(min=1e-12)
        return signals / (sn + self.signal_eps)

    def _init_dual_state(self, fast_state, slow_state):
        if fast_state is None:
            fast_state = torch.zeros(self.state_dim, device=self.embedder.weight.device, dtype=self.embedder.weight.dtype)
        if slow_state is None:
            slow_state = torch.zeros(self.state_dim, device=self.embedder.weight.device, dtype=self.embedder.weight.dtype)
        return fast_state, slow_state

    def _wave_dynamics_evolve(
        self,
        fw: torch.Tensor,
        sigw: torch.Tensor,
        noise_scale_mul: float | torch.Tensor,
    ) -> torch.Tensor:
        """Apply :class:`WaveDynamics` per channel; ``fw`` / ``sigw`` are ``(num_waves, wave_dim)``."""
        outs: list[torch.Tensor] = []
        for i in range(self.num_waves):
            x = fw[i : i + 1].unsqueeze(0)
            s = sigw[i : i + 1].unsqueeze(0)
            y = self.wave_dynamics[i](x, s, noise_scale_mul=noise_scale_mul)
            outs.append(y.squeeze(0))
        return torch.cat(outs, dim=0)

    def _apply_wave_cross_interaction(self, fw: torch.Tensor) -> torch.Tensor:
        """
        Weak mixing across wave channels after local dynamics.

        ``interaction = linear(concat(waves))`` reshaped back to ``(num_waves, wave_dim)``,
        then ``fw + wave_interaction_strength * interaction`` (gradients through Linear).
        """
        if fw.dim() != 2 or fw.shape[0] != self.num_waves or fw.shape[1] != self.wave_dim:
            raise ValueError(
                f"_apply_wave_cross_interaction expected ({self.num_waves}, {self.wave_dim}), "
                f"got {tuple(fw.shape)}"
            )
        flat = fw.reshape(-1)
        inter = self.wave_interaction(flat).view(self.num_waves, self.wave_dim)
        alpha = self.wave_interaction_strength.to(device=fw.device, dtype=fw.dtype)
        return fw + alpha * inter

    def evolve_token(self, fast_state, slow_state, signal, num_steps=None):
        """Tension-adaptive inner steps on fast_state, then slow memory; symplectic readout uses token start/end fast."""
        fast_state, slow_state = self._init_dual_state(fast_state, slow_state)
        if self._fast_start_snapshot is None:
            self._fast_start_snapshot = fast_state.detach().clone()
        base = int(num_steps) if num_steps is not None else self.convergence_steps
        max_steps = self.max_convergence_steps
        prev_energy = self._state_energy(fast_state)
        brk = float(self.tension_break_thresh)
        tol = float(self.tension_tol)
        i = 0
        while i < max_steps:
            prev_fast = fast_state.detach()
            fast_before_dyn = fast_state
            t_prev = self._last_tension_val
            noise_mul = (1.0 + F.softplus(self.tension_noise_gain) * min(t_prev, 3.0)).detach()
            fw = fast_state.view(self.num_waves, self.wave_dim)
            sigw = signal.view(self.num_waves, self.wave_dim)
            dyn = self.dynamics
            if dyn is not None and getattr(dyn, "mhd", None) is not None:
                fw_b = fw.unsqueeze(1)
                sig_b = sigw.unsqueeze(1)
                fw_out = dyn.step(fw_b, sig_b).squeeze(1)
            elif dyn is not None:
                fw_out = dyn.step(fw, sigw)
            else:
                fw_out = self._wave_dynamics_evolve(fw, sigw, noise_mul)
            fw_out = self._apply_wave_cross_interaction(fw_out)
            fast_state = fw_out.reshape(self.state_dim)
            logits_t = self._logits_for_tension(fast_state, slow_state)
            T, _de, _div, _H = self.compute_tension(
                fast_state, slow_state, logits_t, prev_energy
            )
            prev_energy = self._state_energy(fast_state)
            t_item = T.detach().item()
            self._last_tension_val = t_item
            if t_item > brk:
                if self.phase05_config.log_metrics:
                    self._phase05_evolve_high_breaks += 1
                    T_pre = T.detach().item()
                delta_im = fast_state - fast_before_dyn
                t_for_alpha = torch.as_tensor(t_item, device=fast_state.device, dtype=fast_state.dtype)
                cos_pre = F.cosine_similarity(
                    fast_state.unsqueeze(0), fast_before_dyn.unsqueeze(0), dim=-1
                ).squeeze()
                fast_pre_break = fast_state
                prop = self._phase2_directional_escape(
                    fast_state,
                    delta_im,
                    t_for_alpha,
                    row_renorm=True,
                    legacy_scale=0.02,
                )
                logits_post = self._logits_for_tension(prop, slow_state)
                T_post, _, _, _ = self.compute_tension(
                    prop, slow_state, logits_post, prev_energy
                )
                cos_post = F.cosine_similarity(
                    prop.unsqueeze(0), fast_before_dyn.unsqueeze(0), dim=-1
                ).squeeze()
                rejected = False
                if self.phase2_config.enable_break_rejection:
                    if float(T_post.detach()) > float(T.detach()) and float(
                        cos_post.detach()
                    ) < float(cos_pre.detach()):
                        prop = fast_pre_break
                        rejected = True
                fast_state = prop
                if self.phase2_config.store_break_memory:
                    self._phase2_last_break_pre = fast_pre_break.detach().clone()
                    self._phase2_last_break_post = fast_state.detach().clone()
                if self.phase05_config.log_metrics:
                    self._phase05_evolve_t_pre_sum += T_pre
                    self._phase05_evolve_t_post_sum += (
                        float(T.detach().item())
                        if rejected
                        else float(T_post.detach().item())
                    )
            self._last_state_norm = float(torch.linalg.vector_norm(fast_state.detach()))
            self._last_state_delta = float(
                torch.linalg.vector_norm((fast_state - prev_fast).detach())
            )
            if self.track_attractors:
                print(
                    f"  [dyn] ||fast||={self._last_state_norm:.4f}  "
                    f"||Δfast||={self._last_state_delta:.4f}  T={t_item:.4f}"
                )
            i += 1
            if i >= base and t_item < tol:
                if self.phase05_config.log_metrics:
                    self._phase05_evolve_low_exits += 1
                break
        slow_state = (1.0 - self.slow_decay) * slow_state + self.slow_lr * fast_state
        sn_slow = torch.linalg.vector_norm(slow_state)
        if float(sn_slow.detach()) > 0.5:
            slow_state = slow_state * (0.5 / (sn_slow + 1e-12))
        combined = self._symplectic_combined(fast_state, slow_state)
        self._last_slow_norm = float(torch.linalg.vector_norm(slow_state.detach()))
        self._last_combined_norm = float(torch.linalg.vector_norm(combined.detach()))
        if self.track_attractors:
            aid = torch.round(combined, decimals=2)
            key = aid.detach().cpu().numpy().tobytes()
            self._attractor_counts[key] += 1
            print(
                f"  [token] ||fast||={self._last_state_norm:.4f}  ||slow||={self._last_slow_norm:.4f}  "
                f"||combined||={self._last_combined_norm:.4f}  attractor_id[:4]={aid[:4].tolist()}"
            )
        return fast_state, slow_state

    def step_token(self, fast_state, slow_state, signal):
        """Single dynamics update per token (num_steps=1); use for maximal path dependence."""
        return self.evolve_token(fast_state, slow_state, signal, num_steps=1)

    def combined_state(self, fast_state, slow_state):
        fast_state, slow_state = self._init_dual_state(fast_state, slow_state)
        return self._symplectic_combined(fast_state, slow_state)

    def next_token_logits(self, fast_state, slow_state):
        """Legacy D→V readout on fast/slow blend; prefer forward_training_window for training-aligned logits."""
        combined = self.combined_state(fast_state, slow_state)
        if self._prev_combined is not None:
            prev = self._prev_combined.to(device=combined.device, dtype=combined.dtype)
            drift = torch.linalg.vector_norm(combined - prev)
            if float(drift.detach()) < DRIFT_MIN:
                combined = combined + torch.randn_like(combined) * 0.05
        if not self.training:
            combined = combined + torch.randn_like(combined) * 0.01
        state = combined / (torch.linalg.vector_norm(combined) + 1e-8)
        logits = self.readout(state)
        logits = logits / self.effective_temperature()
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=-1e4)
        self._prev_combined = combined.detach().clone()
        return logits

    def next_token_logits_distance(self, fast_state, slow_state):
        """Distance-to-embedding decoding (baseline / comparison experiments)."""
        state = self.combined_state(fast_state, slow_state)
        all_signals = self.all_signals(fast_state, slow_state)
        dists = torch.linalg.vector_norm(all_signals - state.unsqueeze(0), dim=-1)
        logits = -dists / self.effective_temperature()
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=-1e4)
        return logits

    def compute_tension_window_components(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Window tension decomposition (B,).

        T_energy: mean squared delta along positions (legacy energy term).
        T_alignment: mean (1 - cosine) between consecutive rows (positional misalignment).
        T_entropy: readout entropy when WINDOW_TENSION_USE_ENTROPY else 0.
        T_total: w1*T_energy + w2*T_alignment + w3*T_entropy (Phase 0.5 weights).
        """
        assert state.dim() == 3 and state.size(1) >= 2
        w1, w2, w3 = self._phase05_tension_w()
        w1_t = torch.as_tensor(w1, device=state.device, dtype=state.dtype)
        w2_t = torch.as_tensor(w2, device=state.device, dtype=state.dtype)
        w3_t = torch.as_tensor(w3, device=state.device, dtype=state.dtype)
        delta = state[:, 1:] - state[:, :-1]
        T_energy = delta.pow(2).mean(dim=(1, 2))
        cos = F.cosine_similarity(state[:, 1:], state[:, :-1], dim=-1)
        T_alignment = (1.0 - cos).mean(dim=1)
        if WINDOW_TENSION_USE_ENTROPY:
            logits = self._readout_window_logits(state)
            probs = F.softmax(logits, dim=-1)
            T_entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
        else:
            T_entropy = torch.zeros(state.size(0), device=state.device, dtype=state.dtype)
        T_total = w1_t * T_energy + w2_t * T_alignment + w3_t * T_entropy
        return T_total, T_energy, T_alignment, T_entropy

    def compute_tension_window(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: (B, W, D) normalized states

        Returns:
            scalar tension per batch (B,). Also enters window energy descent as
            ``λ * mean(T)`` when ``phase05_config.tension_lambda`` is non-zero
            (see :meth:`_window_energy_gradient_step`).
        """
        T, _, _, _ = self.compute_tension_window_components(state)
        return T

    def compute_window_tension(self, state: torch.Tensor) -> torch.Tensor:
        """Legacy alias for :meth:`compute_tension_window`."""
        return self.compute_tension_window(state)

    def _single_window_step(
        self,
        S: torch.Tensor,
        context_ids: list[list[int]] | None = None,
        context_tensor: torch.Tensor | None = None,
        *,
        pos_weights: torch.Tensor | None = None,
        pos_wsum: torch.Tensor | None = None,
        C_masked: torch.Tensor | None = None,
        goat_bonus_vec: torch.Tensor | None = None,
        log_energy_metrics: bool = False,
        attractor_dt_per_batch: torch.Tensor | None = None,
        anchor_freeze_wave: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One coupling + relaxation step; S is (B, W, D).

        Positional coupling and optional GOAT signal are applied additively to the
        state, then (if ``phase05_config.anchor_force_strength > 0``) each position
        is pulled toward its nearest detached token embedding, then the window descends
        the learned potential via ``S <- S - dt * ∇ E(S)`` with ``E`` from
        :meth:`_window_energy_gradient_step`. Phase-1 global interaction runs after that step.

        If GOAT memory is active and context_tensor is provided (B, W) long), an
        activation-bonus signal is gathered on GPU (no per-step Python over B×W).

        Optional caches (fixed for one ``run_window_dynamics``): positional weights,
        Phase-1 ``C * mask``, GOAT ``(V,)`` bonus vector.

        When ``phase05_config.enable_state_normalization`` is True, applies
        ``F.normalize(S, dim=-1)`` before returning (differentiable).
        """
        B, W, D = S.shape
        assert W == self.train_window_size
        assert D == self.state_dim
        pos_g = F.softplus(self.position_gamma_raw) + 1e-6
        isc = F.softplus(self.interaction_scale_raw)
        idt = F.softplus(self.interaction_dt_raw)
        if self.phase05_config.adaptive_window_dt:
            idt = idt * self._phase05_interaction_dt_scale.to(
                device=idt.device, dtype=idt.dtype
            )
        Sfw = self._window_coupling_flat(S)
        if pos_weights is not None and pos_wsum is not None:
            delta = positional_coupling_delta_from_weights(Sfw, pos_weights, pos_wsum)
        else:
            delta = positional_coupling_delta(Sfw, pos_g, self.position_asym)
        Sfw = Sfw + idt * isc * delta
        S = self._from_window_coupling_flat(Sfw, B)

        # Build GOAT activation-bonus signal (Bug 1 fix).
        # bonuses[b, t] is a scalar; we broadcast it across the D dimension.
        if self._goat_mgr is not None and context_tensor is not None:
            V = len(self._goat_mgr.nodes)
            valid = (context_tensor >= 0) & (context_tensor < V)
            safe = context_tensor.clamp(min=0, max=max(V - 1, 0))
            if goat_bonus_vec is not None:
                bv = goat_bonus_vec
            else:
                bv = self._goat_mgr.bonus_tensor(S.device, S.dtype)
            bonuses_2d = bv[safe] * valid.to(dtype=S.dtype)
            signal = bonuses_2d.unsqueeze(-1).expand(B, W, D)
        else:
            signal = None

        # Inject coupling positions + GOAT signal, optional anchor pull, then energy descent.
        if signal is not None:
            S_prep = S + signal
        else:
            S_prep = S
        S_prep = self._apply_token_anchor_force(S_prep)
        S = self._window_energy_gradient_step(
            S_prep,
            record_metrics=log_energy_metrics,
            attractor_dt_per_batch=attractor_dt_per_batch,
            anchor_freeze_wave=anchor_freeze_wave,
        )
        if not torch.is_grad_enabled() and attractor_dt_per_batch is None:
            S = self._clamp_window_waves_norm(S, 1e-3, 12.0)

        S, _ = self._phase1_global_interaction_step(S, C_masked=C_masked)
        return self._normalize_window_state_if_enabled(S)

    def run_window_dynamics(
        self,
        S: torch.Tensor,
        collect_metrics: bool = False,
        record_tension_log: bool = True,
        context_ids: list[list[int]] | None = None,
        target_states: torch.Tensor | None = None,
        *,
        convergence_epsilon: float | None = None,
        min_attractor_steps: int | None = None,
        return_intermediate_states: bool = False,
    ) -> tuple[torch.Tensor, list[dict] | None, list[torch.Tensor] | None]:
        """
        Tension-adaptive evolution: (W, D) or (B, W, D). Each outer step applies coupling,
        optional GOAT signal, optional token-anchor pull (:meth:`_apply_token_anchor_force`),
        then energy-head gradient descent with a detached state leaf so backprop does not
        chain across steps (``energy_heads`` parameters still receive gradients when training).
        If record_tension_log, fills _last_window_tension_curve with mean(T) after each step
        (use False on teacher-only passes so the student curve is preserved).

        context_ids: list of per-batch token ID windows (B × W) used to compute
        GOAT activation bonuses inside each dynamics step.

        target_states: optional ``(B, W, D)`` same layout as batched ``S`` (after any batch
        unsqueeze). When set and ``phase05_config.trajectory_guidance_nudge_scale > 0``, each
        outer step applies ``S <- S + β * (T.detach() - S)`` before coupling and energy descent.

        Optional early exit (after ``min_attractor_steps``): if ``convergence_epsilon > 0``,
        stop when the batch-mean window potential (``_window_energy_per_batch_row``) changes
        by less than epsilon versus the previous outer step. ``0`` disables early exit.
        ``max_window_steps`` remains a hard cap. Uses model defaults when kwargs are None.
        Updates ``_attractor_steps_cumulative_*`` for a running mean step count across calls.

        If ``return_intermediate_states``, also returns each post-step window tensor
        ``(B, W, D)`` (after tension/jitter for that outer step), excluding the initial embed.

        With ``phase05_config.enable_adaptive_attractor_dt``, each batch row carries its own
        inner step size ``dt`` (initialized from dynamics ``dt``, capped at that value when
        growing). See ``_last_attractor_dt_curve`` and window trace dt fields when logging.

        With ``phase05_config.enable_anchor_freeze``, after each outer step per-wave distances to
        the nearest readout top-``k`` anchor slice gate energy gradients on the next step
        (coupling and anchor pull still update the full state).
        """
        single = S.dim() == 2
        if single:
            S = S.unsqueeze(0)
            # Wrap single window in a batch list so _single_window_step can index it.
            if context_ids is not None and len(context_ids) > 0 and not isinstance(context_ids[0], list):
                context_ids = [context_ids]  # type: ignore[list-item]
            if target_states is not None and target_states.dim() == 2:
                target_states = target_states.unsqueeze(0)
        B, W, D = S.shape
        if target_states is not None:
            if target_states.shape != S.shape:
                raise ValueError(
                    f"target_states shape {tuple(target_states.shape)} must match state "
                    f"{tuple(S.shape)}"
                )
        assert W == self.train_window_size
        self._last_energy_descent_base = None
        self._last_energy_descent_tension = None
        self._last_energy_descent_total = None
        self._last_energy_descent_per_wave = None
        step_logs: list[dict] | None = [] if collect_metrics else None
        step_log_tensors: list[dict[str, torch.Tensor]] | None = [] if collect_metrics else None
        tension_curve_tensors: list[torch.Tensor] | None = [] if record_tension_log else None
        thigh = self.window_tension_high.to(device=S.device, dtype=S.dtype)
        zero_long = torch.zeros((), device=S.device, dtype=torch.long)
        consecutive_low_t_steps_t = zero_long.clone()
        context_tensor: torch.Tensor | None = None
        if context_ids is not None:
            context_tensor = torch.as_tensor(
                context_ids, dtype=torch.long, device=S.device
            )
            if context_tensor.dim() == 1:
                context_tensor = context_tensor.unsqueeze(0)
        trace = self.phase05_config.log_metrics
        if not trace:
            self._phase05_last_window_trace = None
        if trace:
            z = torch.zeros((), device=S.device, dtype=S.dtype)
            _acc_T = z.clone()
            _acc_Te = z.clone()
            _acc_Ta = z.clone()
            _acc_Th = z.clone()
            _acc_rnm = z.clone()
            _acc_rns = z.clone()
            _acc_dl2 = z.clone()
            _acc_stag = z.clone()
            _n_trace = 0
            _br_high = 0
            _br_low = 0
            _t_pre_h = z.clone()
            _t_post_h = z.clone()
            _n_h_ev = 0
            _acc_p1_ir = z.clone()
            _n_p1_ir = 0
            _acc_p1_th = z.clone()
            _n_p1_th = 0
            _acc_p2_dn = z.clone()
            _acc_p2_al = z.clone()
            _acc_p2_dt = z.clone()
            _acc_p2_dc = z.clone()
            _n_p2_br = 0
        S_prev: torch.Tensor | None = None
        stag_eps = float(self.phase05_config.stagnation_delta_thresh)
        # Per-window static tensors (parameters fixed for this forward).
        pos_g_pre = F.softplus(self.position_gamma_raw) + 1e-6
        pos_weights, pos_wsum = positional_coupling_weights_static(
            W, pos_g_pre, self.position_asym, S.device, S.dtype
        )
        C_masked: torch.Tensor | None = None
        if self.phase1_config.enable_window_interaction:
            Cm = self.phase1_window_C.to(device=S.device, dtype=S.dtype)
            maskm = self._phase2_C_dist_mask.to(device=S.device, dtype=S.dtype)
            C_masked = Cm * maskm
        goat_bonus_vec: torch.Tensor | None = None
        if self._goat_mgr is not None and context_tensor is not None:
            goat_bonus_vec = self._goat_mgr.bonus_tensor(S.device, S.dtype)
        conv_eps = float(self.convergence_epsilon if convergence_epsilon is None else convergence_epsilon)
        min_steps = int(self.min_attractor_steps if min_attractor_steps is None else min_attractor_steps)
        min_steps = max(2, min_steps)
        convergence_triggered = False
        break_count_window = 0
        energy_trace = bool(trace or collect_metrics)
        _acc_e_base = 0.0
        _acc_e_tension = 0.0
        _acc_e_total = 0.0
        _n_energy = 0
        _acc_e_anchor_mean = 0.0
        _acc_e_anchor_std = 0.0
        _n_e_anchor = 0
        _acc_e_per_wave = [0.0] * self.num_waves
        prev_energy: float | None = None
        steps_used = 0
        t_mean = torch.tensor(float("nan"), device=S.device, dtype=S.dtype)
        intermediate_states: list[torch.Tensor] | None = (
            [] if return_intermediate_states else None
        )
        self._last_attractor_dt_curve = []
        self._attractor_dt_per_batch = None
        self._attractor_max_dt_per_batch = None
        _acc_adt_mean = 0.0
        _acc_adt_min = float("inf")
        _acc_adt_max = float("-inf")
        _n_adt = 0
        if self.phase05_config.enable_adaptive_attractor_dt:
            _bd = float(self._dynamics_euler_dt())
            self._attractor_max_dt_per_batch = S.new_full((B,), _bd)
            self._attractor_dt_per_batch = self._attractor_max_dt_per_batch.clone()
        _freeze_enabled = bool(self.phase05_config.enable_anchor_freeze)
        _anchor_freeze_wave: torch.Tensor | None = None
        _freeze_age: torch.Tensor | None = None
        if _freeze_enabled:
            _freeze_age = torch.zeros(
                B, self.train_window_size, self.num_waves,
                device=S.device,
                dtype=torch.long,
            )
        _acc_ff_mean = 0.0
        _acc_ff_std = 0.0
        _n_ff = 0
        _guide_beta = float(self.phase05_config.trajectory_guidance_nudge_scale)
        for step in range(self.max_window_steps):
            steps_used = step + 1
            S_prev_outer = S
            if target_states is not None and _guide_beta > 0.0:
                ts_n = target_states.to(device=S.device, dtype=S.dtype)
                S = S + _guide_beta * (ts_n.detach() - S)
            S0 = S.detach().clone() if collect_metrics else None
            S = self._single_window_step(
                S,
                context_ids=context_ids,
                context_tensor=context_tensor,
                pos_weights=pos_weights,
                pos_wsum=pos_wsum,
                C_masked=C_masked,
                goat_bonus_vec=goat_bonus_vec,
                log_energy_metrics=energy_trace,
                attractor_dt_per_batch=self._attractor_dt_per_batch,
                anchor_freeze_wave=_anchor_freeze_wave,
            )
            if self._attractor_dt_per_batch is not None:
                _dm = float(self._attractor_dt_per_batch.mean().detach().item())
                self._last_attractor_dt_curve.append(_dm)
                if trace:
                    _acc_adt_mean += _dm
                    _acc_adt_min = min(
                        _acc_adt_min,
                        float(self._attractor_dt_per_batch.min().detach().item()),
                    )
                    _acc_adt_max = max(
                        _acc_adt_max,
                        float(self._attractor_dt_per_batch.max().detach().item()),
                    )
                    _n_adt += 1
            if _freeze_enabled and _freeze_age is not None:
                with torch.no_grad():
                    _dist_pw = self._window_anchor_nearest_distances_per_wave(S)
                    _thr = torch.as_tensor(
                        float(self.phase05_config.anchor_freeze_threshold),
                        device=S.device,
                        dtype=_dist_pw.dtype,
                    )
                    _stable = _dist_pw < _thr
                    _freeze_age = torch.where(
                        _stable,
                        _freeze_age + 1,
                        torch.zeros_like(_freeze_age),
                    )
                    _max_age = int(self.phase05_config.anchor_freeze_max_age)
                    if _max_age > 0:
                        _anchor_freeze_wave = _stable & (_freeze_age <= _max_age)
                    else:
                        _anchor_freeze_wave = _stable
                _row_ff = _anchor_freeze_wave.float().mean(dim=(1, 2))
                _ff_m = float(_row_ff.mean().item())
                _ff_s = (
                    float(_row_ff.std(unbiased=False).item())
                    if B > 1
                    else 0.0
                )
                _acc_ff_mean += _ff_m
                _acc_ff_std += _ff_s
                _n_ff += 1
            if energy_trace and self._last_energy_descent_total is not None:
                _acc_e_base += float(self._last_energy_descent_base or 0.0)
                _te = self._last_energy_descent_tension
                if _te is not None and math.isfinite(_te):
                    _acc_e_tension += _te
                _acc_e_total += float(self._last_energy_descent_total)
                _n_energy += 1
                _pw = self._last_energy_descent_per_wave
                if _pw is not None and len(_pw) == self.num_waves:
                    for _iw in range(self.num_waves):
                        _acc_e_per_wave[_iw] += float(_pw[_iw])
                if float(self.phase05_config.anchor_lambda) != 0.0:
                    _am = getattr(
                        self, "_last_energy_descent_anchor_mean", float("nan")
                    )
                    _asd = getattr(
                        self, "_last_energy_descent_anchor_std", float("nan")
                    )
                    if math.isfinite(_am) and math.isfinite(_asd):
                        _acc_e_anchor_mean += _am
                        _acc_e_anchor_std += _asd
                        _n_e_anchor += 1
            T, Te, Ta, Th = self.compute_tension_window_components(S)
            t_mean = T.mean()
            self._last_window_tension_mean = t_mean.detach()
            self._last_adaptive_window_steps = step + 1
            if self.phase05_config.adaptive_window_dt:
                tm = t_mean.detach()
                sc = self._phase05_interaction_dt_scale
                cfg = self.phase05_config
                spike = (tm > cfg.adaptive_dt_spike_thresh).to(dtype=torch.float32)
                low = (tm < cfg.adaptive_dt_low_thresh).to(dtype=torch.float32)
                fac = (
                    1.0
                    - spike
                    - low
                    + spike * cfg.adaptive_dt_spike_factor
                    + low * cfg.adaptive_dt_low_factor
                )
                tgt = (sc * fac).clamp(
                    cfg.adaptive_dt_min_scale, cfg.adaptive_dt_max_scale
                )
                sm = cfg.adaptive_dt_smooth
                self._phase05_interaction_dt_scale.copy_((1.0 - sm) * sc + sm * tgt)
            if tension_curve_tensors is not None:
                tension_curve_tensors.append(t_mean.detach())
            if trace:
                with torch.no_grad():
                    _acc_T = _acc_T + t_mean
                    _acc_Te = _acc_Te + Te.mean()
                    _acc_Ta = _acc_Ta + Ta.mean()
                    _acc_Th = _acc_Th + Th.mean()
                    rn = torch.linalg.vector_norm(S, dim=-1)
                    _acc_rnm = _acc_rnm + rn.mean()
                    _acc_rns = _acc_rns + rn.std(unbiased=False)
                    dpos = S[:, 1:] - S[:, :-1]
                    _acc_dl2 = _acc_dl2 + torch.linalg.vector_norm(dpos, dim=-1).mean()
                    if S_prev is not None:
                        dstep = torch.linalg.vector_norm(S - S_prev, dim=-1)
                        _acc_stag = _acc_stag + (dstep < stag_eps).float().mean()
                    _n_trace += 1
                    lrms = getattr(self, "_phase1_last_interaction_rms", None)
                    if lrms is not None:
                        _acc_p1_ir = _acc_p1_ir + lrms
                        _n_p1_ir += 1
                    thm = self._phase1_per_head_tension_means(S)
                    if thm is not None:
                        _acc_p1_th = _acc_p1_th + thm.mean()
                        _n_p1_th += 1
                S_prev = S.detach()
            if (
                collect_metrics
                and S0 is not None
                and step_logs is not None
                and step_log_tensors is not None
            ):
                with torch.no_grad():
                    diff = S - S0
                    _eb = self._last_energy_descent_base
                    _et = self._last_energy_descent_tension
                    _eo = self._last_energy_descent_total
                    _elog: dict[str, torch.Tensor] = {
                        "norm_delta": torch.linalg.vector_norm(diff).detach(),
                        "token_var_mean": S.var(
                            dim=(0, 1), unbiased=False
                        ).mean().detach(),
                        "cosine_to_prev": F.cosine_similarity(
                            S.flatten(), S0.flatten(), dim=0
                        ).detach(),
                        "mean_row_norm": torch.linalg.vector_norm(
                            S, dim=-1
                        ).mean().detach(),
                        "energy_descent_base": torch.as_tensor(
                            float(_eb) if _eb is not None else float("nan"),
                            device=S.device,
                            dtype=S.dtype,
                        ),
                        "energy_descent_tension": torch.as_tensor(
                            float(_et) if _et is not None else float("nan"),
                            device=S.device,
                            dtype=S.dtype,
                        ),
                        "energy_descent_total": torch.as_tensor(
                            float(_eo) if _eo is not None else float("nan"),
                            device=S.device,
                            dtype=S.dtype,
                        ),
                    }
                    _lpw = self._last_energy_descent_per_wave
                    if _lpw is not None:
                        for _wi, _v in enumerate(_lpw):
                            _elog[f"energy_wave_{_wi}"] = torch.as_tensor(
                                float(_v),
                                device=S.device,
                                dtype=S.dtype,
                            )
                    step_log_tensors.append(_elog)
            is_low = t_mean < 0.08
            consecutive_low_t_steps_t = torch.where(
                is_low,
                consecutive_low_t_steps_t + 1,
                torch.zeros((), device=S.device, dtype=torch.long),
            )
            need_jitter = is_low & (consecutive_low_t_steps_t >= 4)
            # Low-tension break: directional escape along (S − S_prev_outer) or legacy Gaussian jitter.
            if bool(torch.any(need_jitter)):
                break_count_window += 1
                S_low_bef = S
                delta_lo = S - S_prev_outer
                T_pre_l = self.compute_tension_window(S_low_bef).mean().detach()
                cos_pre_l = self._window_row_cos_mean(S_low_bef).detach()
                S_try_l = self._phase2_directional_escape(
                    S_low_bef,
                    delta_lo,
                    t_mean,
                    row_renorm=False,
                    legacy_scale=0.015,
                )
                T_post_l = self.compute_tension_window(S_try_l).mean().detach()
                cos_post_l = self._window_row_cos_mean(S_try_l).detach()
                if self.phase2_config.enable_break_rejection:
                    reject_l = (T_post_l > T_pre_l) & (cos_post_l < cos_pre_l)
                    if bool(torch.any(reject_l)):
                        S_try_l = S_low_bef
                S = self._normalize_window_state_if_enabled(S_try_l)
                if trace:
                    _br_low += 1
                    with torch.no_grad():
                        _acc_p2_dn = _acc_p2_dn + torch.linalg.vector_norm(
                            delta_lo, dim=-1
                        ).mean()
                        _acc_p2_al = _acc_p2_al + self._phase2_break_alpha(t_mean)
                        _acc_p2_dt = _acc_p2_dt + (T_post_l - T_pre_l)
                        _acc_p2_dc = _acc_p2_dc + (cos_post_l - cos_pre_l)
                        _n_p2_br += 1
                if self.phase2_config.store_break_memory:
                    self._phase2_last_break_pre = S_low_bef.detach().clone()
                    self._phase2_last_break_post = S.detach().clone()
            consecutive_low_t_steps_t = torch.where(need_jitter, zero_long, consecutive_low_t_steps_t)
            # GOAT node updates are Python-side; one scalar sync only when GOAT is on and jitter fires.
            if (
                self._goat_mgr is not None
                and context_ids is not None
                and bool(torch.any(need_jitter))
            ):
                self._goat_transition_context_ids = context_ids
                self.goat_memory.apply_transition(
                    S[:, -1, :].mean(dim=0), "DORMANT", "ACTIVE"
                )
            high = (T > thigh).any()
            if bool(torch.any(high)):
                break_count_window += 1
                S_h_bef = S
                delta_h = S - S_prev_outer
                T_pre_hm = self.compute_tension_window(S_h_bef).mean().detach()
                cos_pre_h = self._window_row_cos_mean(S_h_bef).detach()
                S_try_h = self._phase2_directional_escape(
                    S_h_bef,
                    delta_h,
                    t_mean,
                    row_renorm=True,
                    legacy_scale=0.01,
                )
                T_post_hm = self.compute_tension_window(S_try_h).mean().detach()
                cos_post_h = self._window_row_cos_mean(S_try_h).detach()
                if self.phase2_config.enable_break_rejection:
                    reject_h = (T_post_hm > T_pre_hm) & (cos_post_h < cos_pre_h)
                    if bool(torch.any(reject_h)):
                        S_try_h = S_h_bef
                S_hi = S_try_h
                if trace:
                    _br_high += 1
                    _t_pre_h = _t_pre_h + t_mean.detach()
                    with torch.no_grad():
                        _acc_p2_dn = _acc_p2_dn + torch.linalg.vector_norm(
                            delta_h, dim=-1
                        ).mean()
                        _acc_p2_al = _acc_p2_al + self._phase2_break_alpha(t_mean)
                        _acc_p2_dt = _acc_p2_dt + (T_post_hm - T_pre_hm)
                        _acc_p2_dc = _acc_p2_dc + (cos_post_h - cos_pre_h)
                        _n_p2_br += 1
                if self.phase2_config.store_break_memory:
                    self._phase2_last_break_pre = S_h_bef.detach().clone()
                    self._phase2_last_break_post = S_hi.detach().clone()
                S = self._normalize_window_state_if_enabled(S_hi)
            if trace and bool(torch.any(high)):
                t_af = self.compute_tension_window(S).mean().detach()
                _t_post_h = _t_post_h + t_af
                _n_h_ev += 1
            if intermediate_states is not None:
                intermediate_states.append(S.clone())
            with torch.no_grad():
                energy_now = float(self._window_energy_per_batch_row(S).mean().item())
            if step + 1 >= min_steps and conv_eps > 0:
                if prev_energy is not None and abs(prev_energy - energy_now) < conv_eps:
                    convergence_triggered = True
                    break
            prev_energy = energy_now
        self._last_convergence_triggered = convergence_triggered
        self._last_attractor_steps_used = steps_used
        self._attractor_steps_cumulative_sum += float(steps_used)
        self._attractor_steps_cumulative_count += 1
        self._last_window_break_count = break_count_window
        self._last_final_window_tension_diag = (
            float(t_mean.detach().item()) if steps_used > 0 else float("nan")
        )
        if trace and _n_trace > 0:
            nf = float(_n_trace)
            if _n_h_ev > 0:
                nh = float(_n_h_ev)
                avg_post = float((_t_post_h / nh).item())
                avg_pre = float((_t_pre_h / nh).item())
            else:
                avg_post = float("nan")
                avg_pre = float("nan")
            self._phase05_last_window_trace = {
                "win_T_mean": float((_acc_T / nf).item()),
                "win_T_energy": float((_acc_Te / nf).item()),
                "win_T_align": float((_acc_Ta / nf).item()),
                "win_T_entropy": float((_acc_Th / nf).item()),
                "state_norm_mean": float((_acc_rnm / nf).item()),
                "state_norm_std": float((_acc_rns / nf).item()),
                "win_delta_l2_mean": float((_acc_dl2 / nf).item()),
                "stagnation_frac": float((_acc_stag / max(_n_trace - 1, 1)).item()),
                "break_high_count": float(_br_high),
                "break_low_jitter_count": float(_br_low),
                "break_avg_t_pre": avg_pre,
                "break_avg_t_post": avg_post,
                "interaction_dt_scale": float(
                    self._phase05_interaction_dt_scale.detach().item()
                ),
                "phase1_interaction_rms": float(
                    (_acc_p1_ir / float(max(_n_p1_ir, 1))).item()
                )
                if _n_p1_ir > 0
                else float("nan"),
                "phase1_tension_heads_mean": float(
                    (_acc_p1_th / float(max(_n_p1_th, 1))).item()
                )
                if _n_p1_th > 0
                else float("nan"),
                "phase2_break_direction_norm_mean": float(
                    (_acc_p2_dn / float(max(_n_p2_br, 1))).item()
                )
                if _n_p2_br > 0
                else float("nan"),
                "phase2_break_applied_alpha_mean": float(
                    (_acc_p2_al / float(max(_n_p2_br, 1))).item()
                )
                if _n_p2_br > 0
                else float("nan"),
                "phase2_break_delta_tension_mean": float(
                    (_acc_p2_dt / float(max(_n_p2_br, 1))).item()
                )
                if _n_p2_br > 0
                else float("nan"),
                "phase2_break_delta_alignment_mean": float(
                    (_acc_p2_dc / float(max(_n_p2_br, 1))).item()
                )
                if _n_p2_br > 0
                else float("nan"),
                "energy_descent_base_mean": float(_acc_e_base / float(_n_energy))
                if _n_energy > 0
                else float("nan"),
                "energy_descent_tension_mean": float(_acc_e_tension / float(_n_energy))
                if _n_energy > 0
                else float("nan"),
                "energy_descent_total_mean": float(_acc_e_total / float(_n_energy))
                if _n_energy > 0
                else float("nan"),
                "energy_per_wave_means": [
                    float(_acc_e_per_wave[i] / float(_n_energy))
                    for i in range(self.num_waves)
                ]
                if _n_energy > 0
                else [float("nan")] * self.num_waves,
                "anchor_energy_mean": float(_acc_e_anchor_mean / float(_n_e_anchor))
                if _n_e_anchor > 0
                else float("nan"),
                "anchor_energy_std": float(_acc_e_anchor_std / float(_n_e_anchor))
                if _n_e_anchor > 0
                else float("nan"),
                "frozen_fraction_mean": float(_acc_ff_mean / float(_n_ff))
                if _n_ff > 0
                else float("nan"),
                "frozen_fraction_std": float(_acc_ff_std / float(_n_ff))
                if _n_ff > 0
                else float("nan"),
                "attractor_dt_mean_over_steps": float(_acc_adt_mean / float(_n_adt))
                if _n_adt > 0
                else float("nan"),
                "attractor_dt_min_seen": float(_acc_adt_min)
                if _n_adt > 0 and math.isfinite(_acc_adt_min)
                else float("nan"),
                "attractor_dt_max_seen": float(_acc_adt_max)
                if _n_adt > 0 and math.isfinite(_acc_adt_max)
                else float("nan"),
                "attractor_dt_mean_final": float(
                    self._attractor_dt_per_batch.mean().detach().item()
                )
                if self._attractor_dt_per_batch is not None
                else float("nan"),
                "attractor_dt_min_final": float(
                    self._attractor_dt_per_batch.min().detach().item()
                )
                if self._attractor_dt_per_batch is not None
                else float("nan"),
                "attractor_dt_max_final": float(
                    self._attractor_dt_per_batch.max().detach().item()
                )
                if self._attractor_dt_per_batch is not None
                else float("nan"),
                "attractor_steps_mean_lifetime": float(
                    self._attractor_steps_cumulative_sum
                    / max(float(self._attractor_steps_cumulative_count), 1.0)
                ),
            }
        elif trace:
            self._phase05_last_window_trace = {}
        if step_logs is not None and step_log_tensors is not None:
            for entry in step_log_tensors:
                step_logs.append({k: float(v.item()) for k, v in entry.items()})
        if tension_curve_tensors is not None:
            self._last_window_tension_curve = [float(x.item()) for x in tension_curve_tensors]
        else:
            self._last_window_tension_curve = []
        if single:
            S = S.squeeze(0)
            if intermediate_states is not None:
                intermediate_states = [s.squeeze(0) for s in intermediate_states]
        return S, step_logs, intermediate_states

    def trajectory_contrastive_loss(
        self, state_a: torch.Tensor, state_b: torch.Tensor
    ) -> torch.Tensor:
        """
        state_a: evolved(context); state_b: evolved(shifted_context + target).
        Same shape (B, W, D).
        """
        cfg = self.phase05_config
        a = state_a.reshape(state_a.size(0), -1)
        b = state_b.reshape(state_b.size(0), -1)
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        B = a.size(0)
        pos = (a * b).sum(dim=-1)
        if cfg.multi_negative and cfg.num_negatives > 1:
            K = max(2, int(cfg.num_negatives))
            scores = torch.rand(K, B, device=a.device, dtype=a.dtype)
            perm = torch.argsort(scores, dim=1)
            b_k = b[perm]
            neg = (a.unsqueeze(0) * b_k).sum(dim=-1).mean(dim=0)
        else:
            b_neg = b[torch.randperm(B, device=b.device)]
            neg = (a * b_neg).sum(dim=-1)
        tau = max(1e-6, float(cfg.trajectory_temperature))
        margin = (0.2 - pos + neg) / tau
        with torch.no_grad():
            self._last_traj_cos_pos = float(pos.mean().item())
            self._last_traj_cos_neg = float(neg.mean().item())
            self._last_traj_margin = float((pos - neg).mean().item())
        return F.relu(margin).mean() * tau

    def anchor_contrastive_loss(
        self, state: torch.Tensor, target_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Pull ``state`` toward target token embeddings and away from random negatives (squared L2).

        ``state``: (B, D), ``target_ids``: (B,) int64. Loss is
        ``mean_b ( ‖s_b - e_{t_b}‖² - mean_k ‖s_b - ñ_{b,k}‖² )`` where ``ñ`` are detached
        embedding rows for vocab indices sampled uniformly from ``{0…V-1} \\ {t_b}``.
        """
        if state.dim() != 2:
            raise ValueError(f"anchor_contrastive_loss expects state (B, D), got {tuple(state.shape)}")
        B, D = state.shape
        if target_ids.shape != (B,):
            raise ValueError(
                f"anchor_contrastive_loss expects target_ids (B,), got {tuple(target_ids.shape)}"
            )
        V = self.vocab_size
        target_anchor = self.embedder(target_ids)
        pos_dist = (state - target_anchor).pow(2).sum(dim=-1)
        if V <= 1:
            return pos_dist.mean()
        K = max(1, int(self.phase05_config.anchor_contrastive_num_negatives))
        r = torch.randint(0, V - 1, (B, K), device=state.device, dtype=torch.long)
        t = target_ids.unsqueeze(1).expand(B, K)
        neg_idx = r + (r >= t).long()
        neg_anchors = self.embedder.weight.detach()[neg_idx]
        neg_dist = (state.unsqueeze(1) - neg_anchors).pow(2).sum(dim=-1)
        return (pos_dist - neg_dist.mean(dim=1)).mean()

    def window_ids_from_sequence(self, seq_ids: list[int]) -> list[int]:
        """Last W token ids; left-pad with seq_ids[0] if shorter than window (full window for dynamics)."""
        W = self.train_window_size
        if not seq_ids:
            seq_ids = [0]
        if len(seq_ids) >= W:
            return seq_ids[-W:]
        pad = seq_ids[0]
        return [pad] * (W - len(seq_ids)) + list(seq_ids)

    def embed_window(self, context_ids: list[int]) -> torch.Tensor:
        assert len(context_ids) == self.train_window_size
        device = self.embedder.weight.device
        dtype = self.embedder.weight.dtype
        ids = torch.tensor(context_ids, device=device, dtype=torch.long)
        emb = self.norm(self.embedder(ids))
        n0 = torch.linalg.vector_norm(emb, dim=-1, keepdim=True).clamp(min=1e-12)
        return emb / n0

    def embed_windows_batch(self, context_tensor: torch.Tensor) -> torch.Tensor:
        """
        (B, W) int64 token ids on any device -> (B, W, D) on embedder device.
        Row b matches ``embed_window(context_tensor[b].tolist())`` (same norm + row L2).
        """
        if context_tensor.dim() != 2:
            raise ValueError("context_tensor must be 2D (B, W)")
        if context_tensor.size(1) != self.train_window_size:
            raise ValueError(
                f"context_tensor W={context_tensor.size(1)} != train_window_size={self.train_window_size}"
            )
        device = self.embedder.weight.device
        ids = context_tensor.to(device=device, dtype=torch.long)
        emb = self.norm(self.embedder(ids))
        n0 = torch.linalg.vector_norm(emb, dim=-1, keepdim=True).clamp(min=1e-12)
        return emb / n0

    def forward_training_window(
        self, context_ids: list[int], collect_dynamics_metrics: bool = False
    ) -> torch.Tensor:
        """
        Embed window tokens to (W, D), run multi-step interacting dynamics, read out vocab logits.
        Training, validation, and generation all use this path.
        """
        assert len(context_ids) == self.train_window_size
        S = self.embed_window(context_ids)
        S, dyn_logs, _ = self.run_window_dynamics(
            S, collect_metrics=collect_dynamics_metrics, context_ids=context_ids
        )
        self._last_dynamics_logs = dyn_logs
        logits = self._readout_window_logits(S)
        if logits.dim() == 2 and logits.size(0) == 1:
            logits = logits.squeeze(0)
        logits = logits / self.effective_temperature()
        return torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=-1e4)

    def shifted_next_window(self, context_ids: list[int], target_id: int) -> list[int]:
        """One-step shift: [x2, …, xW, next_token] — teacher window for trajectory consistency."""
        assert len(context_ids) == self.train_window_size
        return context_ids[1:] + [target_id]

    def trajectory_contrastive_loss_and_logits(
        self,
        contexts: list[list[int]],
        targets: list[int],
        teacher_steps: int | None = None,
        update_repulsion_memory: bool = True,
        target_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Batched trajectory contrastive loss + readout logits from pred state.
        Teacher states are detached (no grad through shifted window).

        When ``phase05_config.trajectory_intermediate_ce_weight > 0``, also adds
        ``weight * mean_k CE(readout_window(S_k), target)`` over post-step states
        from ``run_window_dynamics(..., return_intermediate_states=True)``.

        Adds ``energy_reg_weight * mean(E^2)`` for ``E = _window_energy_per_batch_row(S_pred)``.

        With ``anchor_contrastive_weight > 0``, adds contrastive anchor loss on ``S_pred[:, -1, :]``
        vs the target token embedding and random negative embeddings (see :meth:`anchor_contrastive_loss`).

        ``target_states``: optional ``(B, W, D)`` precomputed trajectory targets (same device/dtype
        as embeddings). Used for nudging inside :meth:`run_window_dynamics` when
        ``trajectory_guidance_nudge_scale > 0``, and for ``trajectory_guidance_mse_weight * MSE``
        vs ``S_pred`` when that weight is positive.
        """
        B = len(contexts)
        assert B == len(targets) and B >= 1
        if self.phase05_config.log_metrics:
            self._phase05_evolve_high_breaks = 0
            self._phase05_evolve_low_exits = 0
            self._phase05_evolve_t_pre_sum = 0.0
            self._phase05_evolve_t_post_sum = 0.0
        if isinstance(self.dynamics, SimpleAttractorDynamics):
            self.dynamics._last_multihead_drifts = None
        device = self.embedder.weight.device
        context_tensor = torch.as_tensor(contexts, dtype=torch.long, device=device)
        targets_tensor_tj = torch.as_tensor(targets, dtype=torch.long, device=device)
        w_ce = float(self.phase05_config.trajectory_intermediate_ce_weight)
        w_tg = float(self.phase05_config.trajectory_guidance_mse_weight)
        beta = float(self.phase05_config.trajectory_guidance_nudge_scale)
        ts_dyn: torch.Tensor | None = None
        if target_states is not None:
            ts_dyn = target_states.to(device=device, dtype=self.embedder.weight.dtype)
            if ts_dyn.shape != (B, self.train_window_size, self.state_dim):
                raise ValueError(
                    f"target_states must be (B, W, D)=({B}, {self.train_window_size}, "
                    f"{self.state_dim}), got {tuple(ts_dyn.shape)}"
                )
            if beta <= 0.0:
                ts_dyn = None
        S_pred = self.embed_windows_batch(context_tensor)
        S_pred, _, states_pred = self.run_window_dynamics(
            S_pred,
            collect_metrics=False,
            record_tension_log=True,
            context_ids=contexts,
            target_states=ts_dyn,
            return_intermediate_states=(w_ce > 0.0),
        )
        al_cfg = float(self.phase05_config.anchor_lambda)
        if al_cfg != 0.0:
            with torch.no_grad():
                nd = self._window_anchor_nearest_distances(S_pred)
                ndf = nd.reshape(-1)
                self._phase05_anchor_energy_mean = float(ndf.mean().item())
                self._phase05_anchor_energy_std = float(ndf.std(unbiased=False).item())
        else:
            self._phase05_anchor_energy_mean = float("nan")
            self._phase05_anchor_energy_std = float("nan")
        self._last_traj_student_tension_mean = self.compute_tension_window(S_pred).mean().detach()
        with torch.no_grad():
            tgt_col = targets_tensor_tj.unsqueeze(1)
            shifted_tensor = torch.cat([context_tensor[:, 1:], tgt_col], dim=1)
            S_tgt = self.embed_windows_batch(shifted_tensor)
            orig_steps = self.max_window_steps
            try:
                if teacher_steps is not None:
                    self.max_window_steps = max(1, int(teacher_steps))
                S_tgt, _, _ = self.run_window_dynamics(
                    S_tgt, collect_metrics=False, record_tension_log=False
                )
            finally:
                self.max_window_steps = orig_steps
        loss_core = self.trajectory_contrastive_loss(S_pred, S_tgt)
        self._phase05_loss_traj_core = float(loss_core.detach().item())
        loss_ce_intermediate: torch.Tensor | None = None
        if w_ce > 0.0 and states_pred:
            per_step_ce: list[torch.Tensor] = []
            for Sk in states_pred:
                Sk_b = Sk.unsqueeze(0) if Sk.dim() == 2 else Sk
                Bk = Sk_b.size(0)
                logits_k = (
                    self._readout_window_logits(Sk_b) / self.effective_temperature()
                )
                logits_k = torch.nan_to_num(
                    logits_k, nan=0.0, posinf=0.0, neginf=-1e4
                )
                per_step_ce.append(
                    F.cross_entropy(
                        logits_k,
                        targets_tensor_tj,
                        label_smoothing=LABEL_SMOOTHING,
                    )
                )
            loss_ce_intermediate = torch.stack(per_step_ce).mean()
        T = self.compute_tension_window(S_pred).mean()
        base_entropy_floor = torch.as_tensor(
            float(ENTROPY_FLOOR), device=S_pred.device, dtype=S_pred.dtype
        )
        entropy_floor = base_entropy_floor * (
            1.0 + 0.5 * (1.0 - torch.sigmoid(8.0 * (T - 0.12)))
        )
        fast_state = S_pred[:, -1, :].mean(dim=0)
        if self._repulsion_prev_states:
            prev_states = torch.stack(self._repulsion_prev_states, dim=0).to(
                device=S_pred.device, dtype=S_pred.dtype
            )
            repulsion = 0.08 * torch.mean(
                torch.cosine_similarity(fast_state.unsqueeze(0), prev_states, dim=-1)
            )
        else:
            repulsion = torch.zeros((), device=S_pred.device, dtype=S_pred.dtype)
        # Contrastive core + optional mean CE over outer dynamics steps + existing aux terms.
        loss_traj = loss_core
        if loss_ce_intermediate is not None:
            loss_traj = loss_traj + w_ce * loss_ce_intermediate
        loss_traj = loss_traj + entropy_floor - repulsion
        div = self.phase1_head_diversity_loss()
        if self.phase1_config.head_diversity_weight > 0.0:
            loss_traj = loss_traj + self.phase1_config.head_diversity_weight * div
        reg_c = self.phase2_interaction_reg_loss()
        if self.phase2_config.interaction_reg_weight > 0.0:
            loss_traj = loss_traj + self.phase2_config.interaction_reg_weight * reg_c
        E_row = self._window_energy_per_batch_row(S_pred)
        energy_reg = (E_row**2).mean()
        w_er = float(self.phase05_config.energy_reg_weight)
        if w_er != 0.0:
            loss_traj = loss_traj + w_er * energy_reg
        traj_mse: torch.Tensor | None = None
        if w_tg > 0.0 and target_states is not None:
            ts_m = target_states.to(device=S_pred.device, dtype=S_pred.dtype)
            traj_mse = F.mse_loss(S_pred, ts_m.detach())
            loss_traj = loss_traj + w_tg * traj_mse
        self._phase05_traj_guidance_mse = (
            float(traj_mse.detach().item()) if traj_mse is not None else float("nan")
        )
        w_ac = float(self.phase05_config.anchor_contrastive_weight)
        anchor_contrast: torch.Tensor | None = None
        if w_ac != 0.0:
            anchor_contrast = self.anchor_contrastive_loss(
                S_pred[:, -1, :], targets_tensor_tj
            )
            loss_traj = loss_traj + w_ac * anchor_contrast
        with torch.no_grad():
            _e_mean = float(E_row.mean().item())
            self._last_batch_energy_mean = _e_mean
            self._last_batch_energy_variance = (
                float(E_row.var(unbiased=False).item()) if B > 1 else 0.0
            )
            self._last_batch_energy_min = float(E_row.min().item())
            self._last_batch_energy_max = float(E_row.max().item())
        self._phase05_energy_potential_mean = _e_mean
        self._phase05_energy_reg_mean_sq = float(energy_reg.detach().item())
        self._phase05_energy_reg_weighted = float((w_er * energy_reg).detach().item())
        if update_repulsion_memory:
            with torch.no_grad():
                self._repulsion_prev_states.append(fast_state.detach().clone())
                if len(self._repulsion_prev_states) > 8:
                    self._repulsion_prev_states.pop(0)
        logits = self._readout_window_logits(S_pred) / self.effective_temperature()
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=-1e4)
        # Store final-position single state for auxiliary readout training (Phase 5).
        self._last_pred_final_state = S_pred[:, -1, :]  # (B, D)
        if anchor_contrast is not None:
            self._phase05_anchor_contrast_loss = float(anchor_contrast.detach().item())
        else:
            self._phase05_anchor_contrast_loss = float("nan")
        if self.phase05_config.log_metrics:
            self._phase05_loss_traj_full = float(loss_traj.detach().item())
            if loss_ce_intermediate is not None:
                self._phase05_loss_traj_intermediate_ce = float(
                    loss_ce_intermediate.detach().item()
                )
            else:
                self._phase05_loss_traj_intermediate_ce = float("nan")
            _, t_te, t_ta, t_th = self.compute_tension_window_components(S_pred)
            self._phase05_last_student_T_components = (
                float(T.detach().item()),
                float(t_te.mean().detach().item()),
                float(t_ta.mean().detach().item()),
                float(t_th.mean().detach().item()),
            )
            self._phase1_head_div_logged = (
                float(div.detach().item())
                if self.phase1_config.head_diversity_weight > 0.0
                else float("nan")
            )
            self._phase2_interaction_reg_logged = (
                float(reg_c.detach().item())
                if self.phase2_config.interaction_reg_weight > 0.0
                else float("nan")
            )
            _dyn = self.dynamics
            _he = (
                getattr(_dyn, "_last_head_weight_entropy", None)
                if _dyn is not None
                else None
            )
            self._phase2_head_weight_entropy_logged = (
                float(_he.detach().item())
                if isinstance(_he, torch.Tensor)
                else float("nan")
            )
        return loss_traj, logits

    def phase05_batch_csv_values(
        self,
        epoch: int,
        batch_idx: int,
        global_step: int,
        batch_ce: float,
    ) -> list[float | int]:
        """Flat row for phase05 batch CSV (caller writes header once)."""
        wt = self._phase05_last_window_trace or {}
        t_tot, t_e, t_a, t_h = getattr(
            self,
            "_phase05_last_student_T_components",
            (float("nan"), float("nan"), float("nan"), float("nan")),
        )
        ev_n = self._phase05_evolve_high_breaks
        ev_pre = self._phase05_evolve_t_pre_sum / ev_n if ev_n > 0 else float("nan")
        ev_post = self._phase05_evolve_t_post_sum / ev_n if ev_n > 0 else float("nan")
        curve = self._last_window_tension_curve
        t_final = float(curve[-1]) if curve else float("nan")
        _epw_trace = wt.get("energy_per_wave_means")
        if isinstance(_epw_trace, list) and len(_epw_trace) > 0:
            _energy_per_wave_csv = ";".join(f"{float(x):.6g}" for x in _epw_trace)
        else:
            _energy_per_wave_csv = ""
        return [
            epoch + 1,
            batch_idx,
            global_step,
            t_final,
            t_tot,
            t_e,
            t_a,
            t_h,
            wt.get("win_T_mean", float("nan")),
            wt.get("win_T_energy", float("nan")),
            wt.get("win_T_align", float("nan")),
            wt.get("win_T_entropy", float("nan")),
            wt.get("state_norm_mean", float("nan")),
            wt.get("state_norm_std", float("nan")),
            wt.get("win_delta_l2_mean", float("nan")),
            wt.get("stagnation_frac", float("nan")),
            self._last_traj_cos_pos,
            self._last_traj_cos_neg,
            self._last_traj_margin,
            wt.get("break_high_count", float("nan")),
            wt.get("break_low_jitter_count", float("nan")),
            wt.get("break_avg_t_pre", float("nan")),
            wt.get("break_avg_t_post", float("nan")),
            wt.get("interaction_dt_scale", float("nan")),
            float(self._phase05_evolve_high_breaks),
            float(self._phase05_evolve_low_exits),
            ev_pre,
            ev_post,
            getattr(self, "_phase05_loss_traj_core", float("nan")),
            getattr(self, "_phase05_loss_traj_full", float("nan")),
            batch_ce,
            wt.get("phase1_interaction_rms", float("nan")),
            wt.get("phase1_tension_heads_mean", float("nan")),
            getattr(self, "_phase1_head_div_logged", float("nan")),
            wt.get("phase2_break_direction_norm_mean", float("nan")),
            wt.get("phase2_break_applied_alpha_mean", float("nan")),
            wt.get("phase2_break_delta_tension_mean", float("nan")),
            wt.get("phase2_break_delta_alignment_mean", float("nan")),
            getattr(self, "_phase2_head_weight_entropy_logged", float("nan")),
            getattr(self, "_phase2_interaction_reg_logged", float("nan")),
            float(getattr(self, "_last_attractor_steps_used", 0)),
            (
                float(
                    getattr(self, "_attractor_steps_cumulative_sum", 0.0)
                    / float(getattr(self, "_attractor_steps_cumulative_count", 0))
                )
                if getattr(self, "_attractor_steps_cumulative_count", 0) > 0
                else float("nan")
            ),
            getattr(self, "_last_final_window_tension_diag", float("nan")),
            float(getattr(self, "_last_window_break_count", 0)),
            float(1 if getattr(self, "_last_convergence_triggered", False) else 0),
            wt.get("energy_descent_base_mean", float("nan")),
            wt.get("energy_descent_tension_mean", float("nan")),
            wt.get("energy_descent_total_mean", float("nan")),
            _energy_per_wave_csv,
            getattr(self, "_phase05_energy_potential_mean", float("nan")),
            getattr(self, "_phase05_energy_reg_mean_sq", float("nan")),
            getattr(self, "_phase05_energy_reg_weighted", float("nan")),
            getattr(self, "_phase05_anchor_energy_mean", float("nan")),
            getattr(self, "_phase05_anchor_energy_std", float("nan")),
            wt.get("frozen_fraction_mean", float("nan")),
            wt.get("frozen_fraction_std", float("nan")),
        ]

    @staticmethod
    def summarize_dynamics_logs(logs: list[dict] | None) -> str:
        if not logs:
            return ""
        nds = [x["norm_delta"] for x in logs]
        tvs = [x["token_var_mean"] for x in logs]
        coss = [x["cosine_to_prev"] for x in logs]
        mns = [x["mean_row_norm"] for x in logs]
        base = (
            f"steps={len(logs)}  "
            f"mean|Δ|={statistics.mean(nds):.4f}  "
            f"mean_var={statistics.mean(tvs):.6f}  "
            f"mean_cos(step)={statistics.mean(coss):.4f}  "
            f"mean||row||={statistics.mean(mns):.4f}"
        )
        if logs and "energy_descent_total" in logs[0]:
            ebs = [x["energy_descent_base"] for x in logs]
            ets = [x["energy_descent_tension"] for x in logs]
            eos = [x["energy_descent_total"] for x in logs]
            finite = [v for v in ebs if math.isfinite(v)]
            base += (
                f"  E_base={statistics.mean(finite):.4f}"
                if finite
                else "  E_base=nan"
            )
            finite_t = [v for v in ets if math.isfinite(v)]
            base += (
                f"  T_step={statistics.mean(finite_t):.4f}"
                if finite_t
                else "  T_step=nan"
            )
            finite_o = [v for v in eos if math.isfinite(v)]
            base += (
                f"  E_tot={statistics.mean(finite_o):.4f}"
                if finite_o
                else "  E_tot=nan"
            )
        if logs:
            _wk = sorted(
                k for k in logs[0] if str(k).startswith("energy_wave_")
            )
            if _wk:
                _parts: list[str] = []
                for _k in _wk:
                    _vs = [x[_k] for x in logs if _k in x]
                    _fin = [v for v in _vs if math.isfinite(v)]
                    if _fin:
                        _parts.append(
                            f"{_k}={statistics.mean(_fin):.4f}"
                        )
                if _parts:
                    base += "  " + " ".join(_parts)
        return base

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Run window dynamics on the trailing context; return converged state (W, D)."""
        self.reset_readout_trajectory()
        if self.tokenizer is not None:
            input_ids = self.tokenizer.encode(prompt)
        else:
            input_ids = [0]
        if not input_ids:
            input_ids = [0]
        wid = self.window_ids_from_sequence(input_ids)
        # no_grad (not inference_mode): window energy descent uses autograd.grad
        # under a nested enable_grad; inference_mode ignores enable_grad and breaks that path.
        with torch.no_grad():
            S = self.embed_window(wid)
            S, _, _ = self.run_window_dynamics(S, collect_metrics=False)
        # Snapshot so callers (e.g. compare_prompts) never alias internal tensors.
        return S.detach().clone()

    def _print_attractor_diversity(self, top_k: int = 5):
        ctr = self._attractor_counts
        total = sum(ctr.values())
        n_unique = len(ctr)
        if total == 0:
            print("[diversity] no attractor samples")
            return
        probs = [c / total for _, c in ctr.most_common()]
        entropy = -sum(p * math.log(p + 1e-30) for p in probs if p > 0)
        top = ctr.most_common(top_k)
        most_common_count = top[0][1] if top else 0
        print(
            f"[diversity] unique={n_unique}  total_tokens={total}  "
            f"most_common_count={most_common_count}  entropy={entropy:.4f}"
        )
        print(f"[diversity] top-{top_k} raw counts: {[c for _, c in top]}")

    def generate(
        self,
        prompt: str,
        max_tokens=40,
        debug_track=False,
        log_dynamics: bool = False,
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        repeat_penalty: float | None = None,
        no_repeat_last_extra: float | None = None,
    ):
        """Autoregressive generation: each step uses last-W context → dynamics → readout_window (training path)."""
        if self.tokenizer is not None:
            input_ids = self.tokenizer.encode(prompt)
        else:
            input_ids = list(range(min(self.train_window_size, 6)))
        if not input_ids:
            input_ids = [0]

        was_training = self.training
        self.eval()
        self.reset_readout_trajectory()
        generated_ids = list(input_ids)
        # no_grad (not inference_mode): see encode_prompt — run_window_dynamics needs
        # torch.enable_grad inside _window_energy_gradient_step to compute ∇E.
        with torch.no_grad():
            if temperature is not None:
                base_gen_temp = float(temperature)
            else:
                base_gen_temp = self.generation_temperature
                if torch.is_tensor(base_gen_temp):
                    base_gen_temp = float(base_gen_temp.detach())
            tk = GEN_TOP_K if top_k is None else int(top_k)
            rp = GEN_REPEAT_LOGIT_PENALTY if repeat_penalty is None else float(repeat_penalty)
            nre = (
                GEN_NO_REPEAT_LAST_EXTRA
                if no_repeat_last_extra is None
                else float(no_repeat_last_extra)
            )
            for ti in range(max_tokens):
                wid = self.window_ids_from_sequence(generated_ids)
                want_metrics = bool(log_dynamics or debug_track)
                logits = self.forward_training_window(wid, collect_dynamics_metrics=want_metrics)
                if want_metrics and self._last_dynamics_logs:
                    show = log_dynamics or (debug_track and ti < 4)
                    if show:
                        summ = self.summarize_dynamics_logs(self._last_dynamics_logs)
                        curve = self._last_window_tension_curve
                        curve_s = (
                            "[" + ", ".join(f"{x:.4f}" for x in curve) + "]"
                            if curve
                            else "[]"
                        )
                        _adc = self._last_attractor_dt_curve
                        adt_s = (
                            "[" + ", ".join(f"{x:.5g}" for x in _adc) + "]"
                            if _adc
                            else "[]"
                        )
                        print(
                            f"  [dyn t={ti}] tension_curve={curve_s}  "
                            f"attractor_dt_mean={adt_s}  "
                            f"steps={self._last_adaptive_window_steps}  {summ}"
                        )
                next_id = sample_next_token_id(
                    logits,
                    base_gen_temp,
                    tk,
                    generated_ids,
                    rp,
                    nre,
                )
                generated_ids.append(next_id)

        if debug_track:
            print("[generate] window path: last step dynamics summary (if logged above).")

        if was_training:
            self.train()
        if self.tokenizer is not None:
            return self.tokenizer.decode(generated_ids)
        return " ".join(str(i) for i in generated_ids)


class WaveDynamics(nn.Module):
    """
    Small residual MLP on one wave channel. Operates on ``(B, W, wave_dim)`` (vectorized over
    batch and window); optional additive ``signal`` with the same shape.
    """

    def __init__(
        self,
        wave_dim: int,
        hidden_mult: int = 2,
        noise_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        self.wave_dim = int(wave_dim)
        h = max(16, self.wave_dim * int(hidden_mult))
        self.mlp = nn.Sequential(
            nn.Linear(self.wave_dim, h),
            nn.GELU(),
            nn.Linear(h, self.wave_dim),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.register_buffer("noise_scale", torch.tensor(float(noise_scale)))
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        signal: torch.Tensor | None = None,
        *,
        noise_scale_mul: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        if signal is not None:
            x = x + signal
        if self.training and float(self.noise_scale) > 0:
            nsm = float(
                noise_scale_mul.detach().item()
                if isinstance(noise_scale_mul, torch.Tensor)
                else noise_scale_mul
            )
            x = x + nsm * self.noise_scale * torch.randn_like(x)
        delta = self.mlp(x)
        return x + torch.tanh(self.residual_scale) * delta


class SimpleAttractorDynamics(nn.Module):
    """
    Phase 1: optional multi-head diffusion — parallel linear drifts concat → W_mix → dim.
    H=1 + identity W_mix reproduces single matrix drift state @ D.
    """

    def __init__(
        self,
        dim=512,
        dt=0.04,
        cubic_scale=0.008,
        beta_init=0.75,
        noise_scale=1e-3,
        lambda_decay=0.1,
        signal_scale=0.5,
        state_norm_eps=1e-8,
        enforce_negative_definite: bool = False,
        phase1: Phase1Config | None = None,
        phase2: Phase2Config | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.dt = dt
        self.cubic_scale = cubic_scale
        self.phase1 = phase1 if phase1 is not None else Phase1Config()
        self.phase2 = phase2 if phase2 is not None else Phase2Config()
        self.num_heads = max(1, int(self.phase1.num_heads))
        self.head_dim_mode = self.phase1.head_dim_mode
        if self.head_dim_mode not in ("shared", "split"):
            raise ValueError("head_dim_mode must be 'shared' or 'split'")
        if self.head_dim_mode == "split" and dim % self.num_heads != 0:
            raise ValueError(f"split mode requires dim % num_heads == 0; got {dim} % {self.num_heads}")
        self.dh = dim // self.num_heads if self.head_dim_mode == "split" else dim
        self._track_head_drifts = (
            self.num_heads > 1 and float(self.phase1.head_diversity_weight) > 0.0
        )
        self._last_multihead_drifts: torch.Tensor | None = None
        self._last_head_weight_entropy: torch.Tensor | None = None
        mg = float(self.phase2.mixing_gate_init)
        mg = min(max(mg, 1e-4), 1.0 - 1e-4)
        self.mixing_gate_raw = nn.Parameter(
            torch.tensor(math.log(mg / (1.0 - mg)))
        )

        heads: list[torch.Tensor] = []
        H = self.num_heads
        if self.head_dim_mode == "split":
            dsub = self.dh
            for h in range(H):
                heads.append(
                    make_diffusion_matrix(
                        dsub,
                        enforce_negative_definite=enforce_negative_definite,
                        seed_offset=h,
                    )
                )
            self.diffusion_heads = nn.Parameter(torch.stack(heads, dim=0))
            mix_in = dim
        else:
            for h in range(H):
                heads.append(
                    make_diffusion_matrix(
                        dim,
                        enforce_negative_definite=enforce_negative_definite,
                        seed_offset=h,
                    )
                )
            self.diffusion_heads = nn.Parameter(torch.stack(heads, dim=0))
            mix_in = H * dim
        self.w_mix = nn.Linear(mix_in, dim, bias=False)
        with torch.no_grad():
            self.w_mix.weight.zero_()
            if self.head_dim_mode == "shared":
                for h in range(H):
                    self.w_mix.weight[:, h * dim : (h + 1) * dim].copy_(
                        torch.eye(dim) / float(H)
                    )
            else:
                self.w_mix.weight.copy_(torch.eye(dim))

        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.register_buffer("noise_scale", torch.tensor(float(noise_scale)))
        self.register_buffer("lambda_decay", torch.tensor(float(lambda_decay)))
        self.register_buffer("signal_scale", torch.tensor(float(signal_scale)))
        self.register_buffer("state_norm_eps", torch.tensor(float(state_norm_eps)))

    def linear_drift(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Multi-head linear drift; optional softmax(-T_slice) head weights; residual mixing."""
        N, dim = state_batch.shape
        H = self.num_heads
        if self.head_dim_mode == "split":
            x = state_batch.reshape(N, H, self.dh)
            drifts = torch.einsum("nhd,hde->nhe", x, self.diffusion_heads)
        else:
            drifts = torch.einsum("nd,hde->nhe", state_batch, self.diffusion_heads)
        if self.training and self._track_head_drifts:
            self._last_multihead_drifts = drifts
        p2 = self.phase2
        use_ht = (
            p2.enable_head_tension_weighting
            and H > 1
            and dim % H == 0
        )
        if use_ht:
            parts = state_batch.reshape(N, H, dim // H)
            T_h = parts.pow(2).mean(dim=-1)
            w = F.softmax(-T_h, dim=-1)
            self._last_head_weight_entropy = (
                -(w * (w.clamp(min=1e-9)).log()).sum(dim=-1).mean()
            )
            mixed = w.unsqueeze(-1) * drifts
            if self.head_dim_mode == "split":
                mixed = mixed.reshape(N, H * self.dh)
            else:
                mixed = mixed.sum(dim=1)
        else:
            self._last_head_weight_entropy = None
            flat = drifts.reshape(N, -1)
            mixed = F.linear(flat, self.w_mix.weight)
        if p2.enable_residual_mixing:
            g = torch.sigmoid(self.mixing_gate_raw)
            return state_batch + g * mixed
        return mixed

    def _step_rows(
        self,
        state_batch: torch.Tensor,
        applied_signal_batch: torch.Tensor,
        noise_scale: torch.Tensor,
    ) -> torch.Tensor:
        c = state_batch - state_batch.mean(dim=-1, keepdim=True)
        cg = self.cubic_scale * float(WINDOW_NONLINEAR_GAIN)
        drift_lin = self.linear_drift(state_batch)
        # Single fused drift (fewer named temporaries than separate nonlinear + drift).
        drift = (
            drift_lin
            + cg * torch.tanh(c)
            + self.beta * (self.signal_scale * applied_signal_batch)
            - self.lambda_decay * state_batch
        )
        s = state_batch + self.dt * drift
        if noise_scale is not None and float(noise_scale) > 0:
            s = s + noise_scale * torch.randn_like(s)
        s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        nrm = torch.linalg.vector_norm(s, dim=-1, keepdim=True)
        eps = self.state_norm_eps
        if torch.is_tensor(eps):
            eps = eps.to(device=s.device, dtype=s.dtype)
        s = s / (nrm + eps)
        return torch.clamp(s, -10.0, 10.0)

    def forward(self, state, signal, noise_scale_mul=1.0):
        ns = self.noise_scale * noise_scale_mul
        drift_lin = self.linear_drift(state.reshape(1, -1)).reshape(-1)
        c = state - state.mean()
        nonlinear = self.cubic_scale * torch.tanh(c)
        scaled_signal = self.signal_scale * signal
        drift = (
            drift_lin
            + nonlinear
            + self.beta * scaled_signal
            - self.lambda_decay * state
        )
        s = state + self.dt * drift
        if ns is not None and float(ns) > 0:
            s = s + ns * torch.randn_like(s)
        s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        nrm = torch.linalg.vector_norm(s)
        eps = self.state_norm_eps
        if torch.is_tensor(eps):
            eps = eps.to(device=s.device, dtype=s.dtype)
        s = s / (nrm + eps)
        return torch.clamp(s, -10.0, 10.0)

    def step(self, S: torch.Tensor, signal: torch.Tensor | None) -> torch.Tensor:
        """Unified batched step (B, W, D) or (N, D)."""
        ns = (
            self.noise_scale.to(device=S.device, dtype=S.dtype)
            if self.training
            else torch.tensor(0.0, device=S.device, dtype=S.dtype)
        )
        if S.dim() == 3:
            B, W, D = S.shape
            flat = S.reshape(B * W, D)
            sig_flat = (
                signal.reshape(B * W, D)
                if signal is not None
                else torch.zeros_like(flat)
            )
            out_flat = self._step_rows(flat, sig_flat, ns)
            return out_flat.reshape(B, W, D)
        sig = signal if signal is not None else torch.zeros_like(S)
        return self._step_rows(S, sig, ns)


def make_diffusion_matrix(
    dim: int,
    enforce_negative_definite: bool = False,
    eps: float = 1e-2,
    seed_offset: int = 0,
):
    """
    Random negative-definite-ish diffusion (legacy) or strict D = -(A^T A) - eps I.
    seed_offset differentiates independent draws for multi-head matrices.
    """
    sk = 42 + int(seed_offset)
    if enforce_negative_definite:
        g = torch.Generator()
        g.manual_seed(sk)
        A = torch.randn(dim, dim, generator=g)
        D = -(A.T @ A) - float(eps) * torch.eye(dim)
        return D.contiguous()
    if int(seed_offset) == 0:
        torch.manual_seed(42)
        q = torch.linalg.qr(torch.randn(dim, dim))[0]
        u = torch.rand(dim)
    else:
        g = torch.Generator()
        g.manual_seed(sk)
        q = torch.linalg.qr(torch.randn(dim, dim, generator=g))[0]
        u = torch.rand(dim, generator=g)
    eigenvalues = -0.2 - (0.05 + 0.3 * u)
    return (q * eigenvalues) @ q.T


def compare_prompts(model: "TorchAttractorLanguageModel", prompt1: str, prompt2: str):
    """Encode two prompts via window dynamics and compare converged window states (flattened)."""
    model.eval()
    with torch.no_grad():
        S1 = model.encode_prompt(prompt1)
        S2 = model.encode_prompt(prompt2)
    v1 = S1.reshape(-1)
    v2 = S2.reshape(-1)
    dist = torch.linalg.vector_norm(v1 - v2).item()
    cos = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0), dim=1).item()
    print(
        f"[compare_prompts] L2(window)={dist:.6f}  cosine={cos:.6f}  "
        f"||S1||={torch.linalg.vector_norm(v1).item():.4f}  ||S2||={torch.linalg.vector_norm(v2).item():.4f}"
    )


def run_quick_window_tests(model: TorchAttractorLanguageModel) -> None:
    """Sanity checks: divergent states for different orderings; dynamics summary (no training)."""
    print("--- quick window / context test ---", flush=True)
    model.eval()
    W = model.train_window_size

    long_a = list(range(W + 5))
    long_b = list(reversed(range(W + 5)))
    wid_a = model.window_ids_from_sequence(long_a)
    wid_b = model.window_ids_from_sequence(long_b)
    with torch.no_grad():
        Sa = model.embed_window(wid_a)
        Sa, _, _ = model.run_window_dynamics(Sa)
        Sb = model.embed_window(wid_b)
        Sb, _, _ = model.run_window_dynamics(Sb)
    va = Sa.reshape(-1)
    vb = Sb.reshape(-1)
    dist = torch.linalg.vector_norm(va - vb).item()
    cos = F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0), dim=1).item()
    print(
        f"  different order (trailing window): L2={dist:.6f}  cosine={cos:.6f}",
        flush=True,
    )
    with torch.no_grad():
        _, logs, _ = model.run_window_dynamics(
            model.embed_window(wid_a), collect_metrics=True
        )
    if logs:
        print(f"  single-window dynamics: {model.summarize_dynamics_logs(logs)}", flush=True)
    print("--- end quick test ---", flush=True)


def step_state(
    state,
    diffusion,
    applied_signal,
    dt,
    cubic_scale,
    beta=1.0,
    noise_scale=0.0,
    lambda_decay=0.1,
    signal_scale=0.5,
    state_norm_eps=1e-8,
):
    c = state - state.mean()
    # Bounded nonlinearity (avoids cubic blow-up at large |c|).
    nonlinear = cubic_scale * torch.tanh(c)
    scaled_signal = signal_scale * applied_signal
    drift = state @ diffusion.T + nonlinear + beta * scaled_signal - lambda_decay * state
    s = state + dt * drift
    if noise_scale is not None and float(noise_scale) > 0:
        s = s + noise_scale * torch.randn_like(s)
    s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    nrm = torch.linalg.vector_norm(s)
    eps = state_norm_eps
    if torch.is_tensor(eps):
        eps = eps.to(device=s.device, dtype=s.dtype)
    s = s / (nrm + eps)
    return torch.clamp(s, -10.0, 10.0)


def step_state_batch(
    state_batch: torch.Tensor,
    diffusion: torch.Tensor,
    applied_signal_batch: torch.Tensor,
    dt: float,
    cubic_scale: float,
    beta: torch.Tensor | float = 1.0,
    noise_scale: torch.Tensor | float = 0.0,
    lambda_decay: torch.Tensor | float = 0.1,
    signal_scale: torch.Tensor | float = 0.5,
    state_norm_eps: torch.Tensor | float = 1e-8,
    nonlinear_gain: float = 1.0,
) -> torch.Tensor:
    """
    Same physics as step_state applied row-wise: each s_i is (D,) like a single state vector.
    c = s_i - mean(s_i) over D (matches step_state on shape (D,)).
    nonlinear_gain scales tanh(c) (window training uses >1 to resist row collapse).
    """
    if state_batch.dim() == 3:
        B, W, D = state_batch.shape
        flat = state_batch.reshape(B * W, D)
        sig_flat = applied_signal_batch.reshape(B * W, D)
        out_flat = step_state_batch(
            flat,
            diffusion,
            sig_flat,
            dt,
            cubic_scale,
            beta=beta,
            noise_scale=noise_scale,
            lambda_decay=lambda_decay,
            signal_scale=signal_scale,
            state_norm_eps=state_norm_eps,
            nonlinear_gain=nonlinear_gain,
        )
        return out_flat.reshape(B, W, D)
    c = state_batch - state_batch.mean(dim=-1, keepdim=True)
    nonlinear = cubic_scale * float(nonlinear_gain) * torch.tanh(c)
    scaled_signal = signal_scale * applied_signal_batch
    drift = state_batch @ diffusion.T + nonlinear + beta * scaled_signal - lambda_decay * state_batch
    s = state_batch + dt * drift
    if noise_scale is not None and float(noise_scale) > 0:
        s = s + noise_scale * torch.randn_like(s)
    s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    nrm = torch.linalg.vector_norm(s, dim=-1, keepdim=True)
    eps = state_norm_eps
    if torch.is_tensor(eps):
        eps = eps.to(device=s.device, dtype=s.dtype)
    s = s / (nrm + eps)
    return torch.clamp(s, -10.0, 10.0)


def positional_coupling_delta(
    S: torch.Tensor,
    position_gamma: torch.Tensor,
    position_asym: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Sum_j w_ij (S_j - S_i), zero diagonal.
    Base: exp(-gamma * |i-j|); optional asymmetry scales left vs right neighbors (sign j-i).
    """
    if S.dim() == 3:
        B, W, D = S.shape
        device = S.device
        dtype = S.dtype
        idx = torch.arange(W, device=device, dtype=dtype)
        rel = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        weights = torch.exp(-position_gamma * rel) * (1.0 - torch.eye(W, device=device, dtype=dtype))
        if position_asym is not None:
            ji = idx.unsqueeze(0) - idx.unsqueeze(1)
            sign_ji = torch.sign(ji)
            sign_ji = torch.where(ji == 0, torch.zeros_like(sign_ji), sign_ji)
            asym_fac = 1.0 + POSITION_ASYM_STRENGTH * torch.tanh(position_asym) * sign_ji
            weights = weights * asym_fac.clamp(min=0.2)
        wsum = weights.sum(dim=1, keepdim=True)
        weighted = torch.einsum("ij,bjd->bid", weights, S)
        return weighted - wsum.unsqueeze(0) * S
    W, _D = S.shape
    device = S.device
    dtype = S.dtype
    idx = torch.arange(W, device=device, dtype=dtype)
    rel = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    weights = torch.exp(-position_gamma * rel) * (1.0 - torch.eye(W, device=device, dtype=dtype))
    if position_asym is not None:
        ji = idx.unsqueeze(0) - idx.unsqueeze(1)
        sign_ji = torch.sign(ji)
        sign_ji = torch.where(ji == 0, torch.zeros_like(sign_ji), sign_ji)
        asym_fac = 1.0 + POSITION_ASYM_STRENGTH * torch.tanh(position_asym) * sign_ji
        # Keep strictly positive (avoid inverted coupling if asym + tanh are large).
        weights = weights * asym_fac.clamp(min=0.2)
    wsum = weights.sum(dim=1, keepdim=True)
    return (weights @ S) - wsum * S


def positional_coupling_weights_static(
    W: int,
    position_gamma: torch.Tensor,
    position_asym: torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Coupling weights for one window (W, W); depends on parameters only, not state.
    Matches :func:`positional_coupling_delta` for the 3D path.
    """
    idx = torch.arange(W, device=device, dtype=dtype)
    rel = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    weights = torch.exp(-position_gamma * rel) * (1.0 - torch.eye(W, device=device, dtype=dtype))
    if position_asym is not None:
        ji = idx.unsqueeze(0) - idx.unsqueeze(1)
        sign_ji = torch.sign(ji)
        sign_ji = torch.where(ji == 0, torch.zeros_like(sign_ji), sign_ji)
        asym_fac = 1.0 + POSITION_ASYM_STRENGTH * torch.tanh(position_asym) * sign_ji
        weights = weights * asym_fac.clamp(min=0.2)
    wsum = weights.sum(dim=1, keepdim=True)
    return weights, wsum


def positional_coupling_delta_from_weights(
    S: torch.Tensor,
    weights: torch.Tensor,
    wsum: torch.Tensor,
) -> torch.Tensor:
    """``sum_j w_ij (S_j - S_i)`` using precomputed ``weights`` (same as :func:`positional_coupling_delta` 3D)."""
    weighted = torch.einsum("ij,bjd->bid", weights, S)
    return weighted - wsum.unsqueeze(0) * S


def _sequence_is_weak_or_repetitive(token_ids):
    """True if all tokens are identical or one token accounts for >50% of the span (anti-repetition training)."""
    if not token_ids:
        return True
    n = len(token_ids)
    counts = Counter(token_ids)
    max_freq = max(counts.values())
    if max_freq / n > 0.5:
        return True
    return False


def build_sequence_dataset(tokens, window_size=6):
    """
    tokens: List[int] (single sentence, order preserved)
    returns: List of (context, target)
    context: List[int] of length window_size
    target: int (next token)
    Skips windows where the (context + target) span is all-one-token or >50% one token.
    """
    data = []
    for i in range(len(tokens) - window_size):
        context = tokens[i : i + window_size]
        target = tokens[i + window_size]
        span = list(context) + [target]
        if _sequence_is_weak_or_repetitive(span):
            continue
        data.append((context, target))
    return data


def load_corpus(path: Path) -> list[str]:
    """Load non-empty lines from a UTF-8 text file; skip blank lines and #-comments."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Corpus file not found: {path}\n"
            "Create it or pass --corpus /path/to/file.txt (one sentence per line)."
        )
    out: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
    return out


def load_corpus_text_stream(path: Path) -> str:
    """
    Load the full corpus as one string (no per-line tokenization).

    Uses ``data_pipeline`` file discovery: single .txt is read whole; .jsonl
    concatenates ``text`` / ``content`` / ``sentence`` fields; directories
    merge all ``.txt`` / ``.jsonl`` files.
    """
    from data_pipeline import _collect_text_files, load_concatenated_corpus_text  # type: ignore[import]

    files = _collect_text_files(path)
    if not files:
        raise FileNotFoundError(
            f"No text files found for corpus path: {path}\n"
            "Pass a .txt file, .jsonl file, or a directory of text/JSONL."
        )
    return load_concatenated_corpus_text(files)


def split_tokens_train_val(
    tokens: list[int],
    val_fraction: float,
    window_size: int,
    min_val_windows: int = MIN_VAL_WINDOWS,
) -> tuple[list[int], list[int], float]:
    """
    Split a token sequence into train / val at token boundaries.

    Inserts a **gap** of ``window_size`` tokens between train and val so no
    sliding window straddles the boundary (avoids train/val leakage).

    Each side must be able to form at least one window (len >= window_size + 1),
    or val is empty. Requires ``len(train_tokens) > window_size`` when val is on
    (equivalently train has at least ``window_size + 1`` tokens).

    If the hold-out has fewer than ``min_val_windows`` sliding windows after the
    initial split, recomputes once using
    ``max(val_fraction, (min_val_windows + window_size) / n)``.

    Returns ``(train_tokens, val_tokens, effective_val_fraction)`` where
    ``effective_val_fraction = len(val_tokens) / n`` (0 if no val).
    """
    n = len(tokens)
    W = window_size
    min_need = W + 1
    gap = W
    if val_fraction <= 0:
        return list(tokens), [], 0.0
    # train segment, gap, val segment — each of train and val needs min_need tokens
    min_n = min_need + gap + min_need
    if n < min_n:
        return list(tokens), [], 0.0

    def _split(vf: float) -> tuple[list[int], list[int]]:
        split_idx = int(n * (1 - vf))
        split_idx = max(min_need + gap, split_idx)
        split_idx = min(split_idx, n - min_need)
        train_end = split_idx - gap
        if train_end < min_need or (n - split_idx) < min_need:
            return [], []
        return tokens[:train_end], tokens[split_idx:]

    train_tok, val_tok = _split(val_fraction)
    val_w = max(0, len(val_tok) - W) if val_tok else 0

    if (not val_tok) or val_w < min_val_windows:
        vf2 = max(val_fraction, (min_val_windows + W) / n)
        train_tok, val_tok = _split(vf2)
        val_w = max(0, len(val_tok) - W) if val_tok else 0

    if val_tok and val_w < min_val_windows:
        print(
            f"Validation set too small ({val_w} windows). Metrics will be noisy.",
            flush=True,
        )

    if not val_tok:
        return list(tokens), [], 0.0

    effective_vf = len(val_tok) / n
    return train_tok, val_tok, effective_vf


def corpus_coverage_report(
    sentences: list[str],
    tokenizer,
    window_size: int,
) -> None:
    """Print token statistics and usable-line count for the corpus."""
    n_lines = len(sentences)
    raw_tokens = 0
    n_too_short = 0
    n_usable = 0
    for s in sentences:
        ids = tokenizer.encode(s) if tokenizer is not None else s.lower().split()
        raw_tokens += len(ids)
        if len(ids) < window_size + 1:
            n_too_short += 1
        else:
            n_usable += 1
    print(
        f"Corpus coverage: {n_lines} lines  |  {n_usable} usable (≥{window_size + 1} tokens)  "
        f"|  {n_too_short} too short  |  total_tokens={raw_tokens}",
        flush=True,
    )


def train_val_split(
    sentences: list[str],
    val_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    if val_fraction <= 0 or len(sentences) < 2:
        return list(sentences), []
    rng = random.Random(seed)
    s = list(sentences)
    rng.shuffle(s)
    n_val = max(1, int(len(s) * val_fraction))
    n_val = min(n_val, len(s) - 1)
    return s[:-n_val], s[-n_val:]


def sentences_with_training_windows(
    sentences: list[str],
    tokenizer,
    window_size: int,
) -> list[str]:
    """Lines that yield at least one (context, target) pair after tokenization."""
    out: list[str] = []
    for s in sentences:
        ids = tokenizer.encode(s) if tokenizer is not None else s.lower().split()
        if len(ids) >= window_size + 1:
            out.append(s)
    return out


def build_dataset_from_sentences(
    sentences: list[str],
    model: TorchAttractorLanguageModel,
    window_size: int,
) -> list:
    """Tokenize sentences via model.tokenizer and build sliding-window (context, target) pairs."""
    tok = model.tokenizer
    dataset = []
    for sentence in sentences:
        ids = tok.encode(sentence) if tok is not None else []
        if len(ids) < window_size + 1:
            continue
        dataset.extend(build_sequence_dataset(ids, window_size=window_size))
    return dataset


def build_dataset_from_token_ids(token_ids: list[int], window_size: int) -> list:
    """Sliding-window (context, target) pairs from one continuous token sequence."""
    if len(token_ids) < window_size + 1:
        return []
    return build_sequence_dataset(token_ids, window_size=window_size)


@torch.no_grad()
def mean_cross_entropy_eval(
    model: TorchAttractorLanguageModel,
    dataset: list,
    batch_size: int = TRAJECTORY_BATCH_SIZE_DEFAULT,
) -> float:
    """Validation CE: batched window eval with the same logit shaping as training, without noise or entropy-floor branch."""
    if not dataset:
        return float("nan")
    was_training = model.training
    model.eval()
    total = 0.0
    device = model.embedder.weight.device
    E = model.embedder.weight
    for i in range(0, len(dataset), max(1, batch_size)):
        chunk = dataset[i : i + max(1, batch_size)]
        contexts = [c for c, _ in chunk]
        targets = [t for _, t in chunk]
        context_tensor = torch.as_tensor(contexts, dtype=torch.long, device=device)
        targets_tensor = torch.as_tensor(targets, dtype=torch.long, device=device)
        S = model.embed_windows_batch(context_tensor)
        S, _, _ = model.run_window_dynamics(
            S, collect_metrics=False, record_tension_log=False, context_ids=contexts
        )
        B = context_tensor.size(0)
        logits = model.readout_window_logits(S)
        logits = logits / model.effective_temperature()
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=-1e4)
        prev_ids = context_tensor[:, -1]
        logits = logits + BIGRAM_TRAIN_WEIGHT * torch.matmul(E[prev_ids], E.T)
        logits = logits.clone()
        row_idx = torch.arange(B, device=device)
        logits[row_idx, prev_ids] -= 2.0
        last_k = min(3, context_tensor.size(1))
        if last_k > 0:
            rep_ids = context_tensor[:, -last_k:]
            rep_rows = row_idx.unsqueeze(1).expand(B, last_k).reshape(-1)
            logits[rep_rows, rep_ids.reshape(-1)] -= 1.0
        loss_ce = F.cross_entropy(
            logits,
            targets_tensor,
            label_smoothing=LABEL_SMOOTHING,
            reduction="sum",
        )
        total += float(loss_ce.detach())
    if was_training:
        model.train()
    return total / len(dataset)


@torch.no_grad()
def precompute_stream_target_states_embed(
    model: TorchAttractorLanguageModel,
    train_token_ids: list[int],
    window_size: int,
    *,
    device: torch.device,
    embed_batch_size: int = 256,
) -> torch.Tensor:
    """
    One target row per sliding window start ``j`` in ``train_token_ids``: embed
    ``ids[j : j + W]`` via :meth:`TorchAttractorLanguageModel.embed_windows_batch`.

    Returns ``(n_windows, W, D)`` float32 CPU.
    """
    W = window_size
    toks = train_token_ids
    n_win = len(toks) - W
    D = model.state_dim
    if n_win <= 0:
        return torch.empty((0, W, D), dtype=torch.float32)
    was_training = model.training
    model.eval()
    outs: list[torch.Tensor] = []
    bs = max(1, int(embed_batch_size))
    try:
        rows: list[list[int]] = []
        for j in range(n_win):
            rows.append(toks[j : j + W])
            if len(rows) >= bs or j == n_win - 1:
                ct = torch.tensor(rows, dtype=torch.long, device=device)
                emb = model.embed_windows_batch(ct)
                outs.append(emb.detach().float().cpu())
                rows = []
    finally:
        if was_training:
            model.train()
    return torch.cat(outs, dim=0)


@torch.no_grad()
def mean_trajectory_contrastive_eval(
    model: TorchAttractorLanguageModel,
    dataset: list,
    batch_size: int = TRAJECTORY_BATCH_SIZE_DEFAULT,
) -> float:
    """Mean trajectory contrastive loss over the dataset (batched)."""
    if not dataset:
        return float("nan")
    was_training = model.training
    model.eval()
    total = 0.0
    n_seen = 0
    i = 0
    while i < len(dataset):
        chunk = dataset[i : i + batch_size]
        if len(chunk) < 2:
            chunk = chunk + chunk
        contexts = [c for c, _t in chunk]
        targets = [t for _c, t in chunk]
        device = model.embedder.weight.device
        context_tensor = torch.as_tensor(contexts, dtype=torch.long, device=device)
        S_pred = model.embed_windows_batch(context_tensor)
        S_pred, _, _ = model.run_window_dynamics(
            S_pred, collect_metrics=False, record_tension_log=False
        )
        tgt_col = torch.as_tensor(targets, dtype=torch.long, device=device).unsqueeze(1)
        shifted_tensor = torch.cat([context_tensor[:, 1:], tgt_col], dim=1)
        S_tgt = model.embed_windows_batch(shifted_tensor)
        S_tgt, _, _ = model.run_window_dynamics(
            S_tgt, collect_metrics=False, record_tension_log=False
        )
        total += float(model.trajectory_contrastive_loss(S_pred, S_tgt).item()) * len(
            contexts
        )
        n_seen += len(contexts)
        i += batch_size
    if was_training:
        model.train()
    return total / max(n_seen, 1)


def _aux_ce_loss_batch(
    model: TorchAttractorLanguageModel,
    logits: torch.Tensor,
    contexts: list[list[int]],
    targets: list[int],
) -> torch.Tensor:
    """Mean per-example CE − entropy bonus (matches single-window training shaping). Fully batched."""
    device = logits.device
    dtype = logits.dtype
    B, V = logits.shape
    context_tensor = torch.as_tensor(contexts, dtype=torch.long, device=device)
    targets_tensor = torch.as_tensor(targets, dtype=torch.long, device=device)
    E = model.embedder.weight
    prev_ids = context_tensor[:, -1]
    e_prev = E[prev_ids]
    lo = logits + BIGRAM_TRAIN_WEIGHT * torch.matmul(e_prev, E.T)
    lo = lo.clone()
    lo[torch.arange(B, device=device), prev_ids] -= 2.0
    W = context_tensor.size(1)
    last_k = min(3, W)
    if last_k > 0:
        tok_blk = context_tensor[:, -last_k:]
        rb = torch.arange(B, device=device).unsqueeze(1).expand(B, last_k).reshape(-1)
        lo[rb, tok_blk.reshape(-1)] -= 1.0
    lo = lo + TRAIN_LOGIT_NOISE * torch.randn_like(lo)
    probs_floor = F.softmax(lo, dim=-1)
    ent_s = -(probs_floor * torch.log(probs_floor + 1e-9)).sum(dim=-1)
    ts = getattr(model, "_last_traj_student_tension_mean", None)
    base_entropy_floor = torch.as_tensor(float(ENTROPY_FLOOR), device=device, dtype=dtype)
    if ts is not None:
        tref = ts.to(device=device, dtype=dtype)
        if tref.dim() > 0:
            tref = tref.mean()
        entropy_floor = base_entropy_floor * (
            1.0 + 0.5 * (1.0 - torch.sigmoid(8.0 * (tref - 0.12)))
        )
    else:
        entropy_floor = base_entropy_floor
    need_floor_noise = ent_s < entropy_floor
    floor_noise = torch.randn_like(lo) * ENTROPY_FLOOR_NOISE
    lo_for_ce = lo + torch.where(need_floor_noise.unsqueeze(-1), floor_noise, torch.zeros_like(lo))
    probs_for_entropy = torch.where(
        need_floor_noise.unsqueeze(-1),
        F.softmax(lo_for_ce, dim=-1),
        probs_floor,
    )
    loss_ce_vec = F.cross_entropy(
        lo_for_ce, targets_tensor, label_smoothing=LABEL_SMOOTHING, reduction="none"
    )
    entropy_vec = -(probs_for_entropy * torch.log(probs_for_entropy + 1e-8)).sum(dim=-1)
    return (loss_ce_vec - ENTROPY_WEIGHT * entropy_vec).mean()


WINDOW_SIZE = 6
NUM_EPOCHS = 3
ENTROPY_WEIGHT = 0.03  # subtracted from CE; keep small vs CE scale or the objective chases flat distributions
CORPUS_EPOCH_COPIES = 2  # duplicate sentence list per epoch for more windows

# Phase 0: fixed prompts for comparable generations across runs (see docs/BASELINE.md).
BASELINE_PROMPT_1 = (
    "the quick brown fox jumps over the lazy dog and then what happens in the system of mind and reason"
)
BASELINE_PROMPT_2 = "mind reason cause effect system"
BASELINE_PROMPT_3 = "effect cause reason mind system"


def _git_short_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _format_phase0_baseline_block(
    *,
    corpus_path: Path,
    seed: int,
    val_fraction: float,
    effective_stream_val_fraction: float | None = None,
    epoch_copies: int,
    num_epochs: int,
    window_size: int,
    num_dynamics_steps: int,
    loss_mode: str,
    token_aux_ce: float,
    last_epoch: int,
    last_mean_loss: float,
    last_train_ce: float,
    last_val_ce: float | None,
    last_train_traj_contrast: float | None,
    last_val_traj_contrast: float | None,
    last_n_windows: int,
    last_epoch_sec: float,
    train_sec_total: float,
    gen1: str,
    gen2: str,
    gen3: str,
) -> str:
    val_s = f"{last_val_ce:.4f}" if last_val_ce is not None else "n/a (no val)"
    traj_train = (
        f"{last_train_traj_contrast:.6f}"
        if last_train_traj_contrast is not None
        else "n/a"
    )
    traj_val = (
        f"{last_val_traj_contrast:.6f}"
        if last_val_traj_contrast is not None
        else "n/a"
    )
    _vf_line = f"seed: {seed}  val_fraction: {val_fraction}"
    if effective_stream_val_fraction is not None:
        _vf_line += f"  effective_stream_val_fraction: {effective_stream_val_fraction:g}"
    _vf_line += f"  epoch_copies: {epoch_copies}"
    return (
        f"--- Phase 0 baseline (copy into docs/BASELINE.md) ---\n"
        f"time_utc: {datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()}\n"
        f"git: {_git_short_hash()}\n"
        f"corpus: {corpus_path}\n"
        f"{_vf_line}\n"
        f"loss_mode: {loss_mode}  token_aux_ce: {token_aux_ce}\n"
        f"window_size: {window_size}  num_dynamics_steps: {num_dynamics_steps}  num_epochs: {num_epochs}\n"
        f"last_epoch: {last_epoch}/{num_epochs}  windows: {last_n_windows}  epoch_sec: {last_epoch_sec:.1f}\n"
        f"train_sec_total: {train_sec_total:.1f}\n"
        f"mean_loss (objective): {last_mean_loss:.4f}\n"
        f"train_CE: {last_train_ce:.4f}  val_CE: {val_s}\n"
        f"train_traj_contrast: {traj_train}  val_traj_contrast: {traj_val}\n"
        f"\n--- generation baseline prompt 1 ---\n{gen1}\n"
        f"\n--- generation baseline prompt 2 ---\n{gen2}\n"
        f"\n--- generation baseline prompt 3 ---\n{gen3}\n"
        f"--- end baseline ---\n"
    )


def _save_checkpoint(
    model: TorchAttractorLanguageModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    args,
    ckpt_dir: Path,
) -> Path:
    """Save model + optimizer state to a numbered checkpoint file."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"ckpt_step{step:07d}.pt"
    _cfg = {
        "state_dim": model.state_dim,
        "train_window_size": model.train_window_size,
        "vocab_size": model.vocab_size,
        "max_window_steps": model.max_window_steps,
        "convergence_epsilon": model.convergence_epsilon,
        "num_waves": model.num_waves,
        "use_readout_fusion": bool(getattr(model, "use_readout_fusion", False)),
    }
    _dyn = model.dynamics
    if hasattr(_dyn, "use_lorentz"):
        _cfg["use_lorentz"] = bool(_dyn.use_lorentz)
    _mhd = getattr(_dyn, "mhd", None)
    if _mhd is not None and hasattr(_mhd, "dt"):
        try:
            _cfg["vectorized_dt"] = float(_mhd.dt)
        except (TypeError, ValueError):
            pass
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "vocab_size": model.vocab_size,
            "tokenizer_mode": getattr(args, "tokenizer", "fallback"),
            "config": _cfg,
        },
        ckpt_path,
    )
    print(f"[ckpt] Saved: {ckpt_path}", flush=True)
    return ckpt_path


def _checkpoint_state_is_vectorized(st: dict) -> bool:
    return any(str(k).startswith("dynamics.mhd.") for k in st)


def _broadcast_legacy_energy_head_to_heads(
    model: "TorchAttractorLanguageModel", st: dict
) -> None:
    """If ``st`` has legacy ``energy_head.*`` but no ``energy_heads.*``, copy into each wave head."""
    if any(str(k).startswith("energy_heads.") for k in st):
        return
    prefix = "energy_head."
    legacy: dict[str, torch.Tensor] = {}
    for k, v in st.items():
        ks = str(k)
        if ks.startswith(prefix):
            legacy[ks[len(prefix) :]] = v
    if not legacy:
        return
    for i in range(model.num_waves):
        model.energy_heads[i].load_state_dict(legacy, strict=True)


def load_torch_attractor_state_dict(model: "TorchAttractorLanguageModel", st: dict) -> None:
    """
    Load ``st`` into ``TorchAttractorLanguageModel``. Checkpoints may omit ``energy_heads.*``
    (random init); legacy ``energy_head.*`` is copied into every ``energy_heads[i]``.
    Older checkpoints may omit ``wave_interaction.*``; any other mismatch is an error.
    """
    incompat = model.load_state_dict(st, strict=False)
    _broadcast_legacy_energy_head_to_heads(model, st)
    missing = list(incompat.missing_keys)
    unexpected = list(incompat.unexpected_keys)
    has_legacy_dynamics = any(str(k).startswith("dynamics.") for k in st)

    def _allowed_missing(k: str) -> bool:
        sk = str(k)
        if sk.startswith("energy_heads."):
            return True
        if sk.startswith("readout_fusion."):
            return True
        if sk.startswith("wave_interaction."):
            return True
        if sk == "wave_interaction_strength":
            return True
        if has_legacy_dynamics and sk.startswith("wave_dynamics."):
            return True
        return False

    bad_missing = [k for k in missing if not _allowed_missing(k)]
    unexpected = [
        k
        for k in unexpected
        if not (has_legacy_dynamics and str(k).startswith("dynamics."))
        and not str(k).startswith("energy_head.")
    ]
    if bad_missing:
        raise RuntimeError(
            f"load_state_dict: missing keys other than allowed optional prefixes ({len(bad_missing)}): "
            f"{bad_missing[:20]}{'...' if len(bad_missing) > 20 else ''}"
        )
    if unexpected:
        raise RuntimeError(
            f"load_state_dict: unexpected keys ({len(unexpected)}): "
            f"{unexpected[:20]}{'...' if len(unexpected) > 20 else ''}"
        )


def _checkpoint_vectorized_head_rank(st: dict) -> tuple[int, int, int] | None:
    u = st.get("dynamics.mhd.U")
    if u is None or not hasattr(u, "dim") or u.dim() != 3:
        return None
    nh, hd, rk = int(u.shape[0]), int(u.shape[1]), int(u.shape[2])
    return nh, hd, rk


def load_model_from_checkpoint(
    ckpt_path: Path | str,
    *,
    tokenizer_mode: str = "tiktoken",
    vocab_cap: int = 32768,
    device: torch.device | None = None,
    map_location: str | torch.device = "cpu",
    print_vocab_mismatch_warning: bool = True,
) -> TorchAttractorLanguageModel:
    """
    Restore ``TorchAttractorLanguageModel`` from ``torch.save`` output matching
    :func:`_save_checkpoint`. Rebuilds ``VectorizedWindowDynamics`` (heads, rank,
    ``dt``, ``use_lorentz``) before ``load_state_dict`` when ``dynamics.mhd.*`` keys exist.
    """
    p = Path(ckpt_path)
    ckpt = torch.load(p, map_location=map_location)
    st = ckpt["model_state"]
    cfg = ckpt.get("config") or {}
    vocab_size = int(cfg.get("vocab_size", ckpt.get("vocab_size", 50257)))
    state_dim = int(cfg.get("state_dim", 512))
    window_size = int(cfg.get("train_window_size", WINDOW_SIZE))
    num_waves = int(cfg.get("num_waves", 1))

    tok = _build_tokenizer(mode=tokenizer_mode, vocab_cap=vocab_cap)
    if print_vocab_mismatch_warning and tok.n_vocab != vocab_size:
        print(
            f"Warning: tokenizer vocab {tok.n_vocab} != checkpoint {vocab_size} "
            "(decode may be wrong; match training --tokenizer / --vocab-cap).",
            flush=True,
        )

    model = TorchAttractorLanguageModel(
        vocab_size,
        state_dim=state_dim,
        train_window_size=window_size,
        max_window_steps=int(cfg.get("max_window_steps", MAX_WINDOW_STEPS)),
        convergence_epsilon=float(cfg.get("convergence_epsilon", 0.0)),
        num_waves=num_waves,
        use_readout_fusion=bool(cfg.get("use_readout_fusion", False)),
    )
    if _checkpoint_state_is_vectorized(st):
        from dynamics_vectorized import VectorizedWindowDynamics  # type: ignore[import]

        inf = _checkpoint_vectorized_head_rank(st)
        if inf is None:
            raise RuntimeError(
                f"Checkpoint {p} has dynamics.mhd.* keys but could not infer heads/rank from dynamics.mhd.U"
            )
        nh, _hd, rk = inf
        use_lorentz = bool(cfg.get("use_lorentz", False))
        vdt = cfg.get("vectorized_dt", 0.09)
        try:
            vectorized_dt = float(vdt)
        except (TypeError, ValueError):
            vectorized_dt = 0.09
        model.dynamics = VectorizedWindowDynamics(
            state_dim=model.wave_dim,
            window_size=window_size,
            num_heads=nh,
            rank=rk,
            max_steps=model.max_window_steps,
            dt=vectorized_dt,
            use_lorentz=use_lorentz,
        )
    load_torch_attractor_state_dict(model, st)
    model.tokenizer = tok
    model.eval()
    if device is not None:
        model = model.to(device)
    return model


def _training_debug(debug: bool, msg: str) -> None:
    if debug:
        print(f"[debug] {msg}", flush=True)


def phase05_config_from_args(args: argparse.Namespace) -> Phase05Config:
    tw: tuple[float, float, float] | None = None
    raw = getattr(args, "phase05_tension_w", None)
    if raw:
        parts = [float(x.strip()) for x in str(raw).split(",")]
        if len(parts) != 3:
            raise SystemExit("--phase05-tension-w requires exactly three comma-separated floats")
        tw = (parts[0], parts[1], parts[2])
    csv_p = getattr(args, "phase05_batch_metrics_csv", None)
    log_m = bool(getattr(args, "phase05_log_metrics", False))
    if csv_p is not None:
        log_m = True
    return Phase05Config(
        log_metrics=log_m,
        batch_metrics_csv=str(csv_p) if csv_p is not None else None,
        enforce_negative_definite_diffusion=bool(
            getattr(args, "phase05_enforce_negdef_diffusion", False)
        ),
        adaptive_window_dt=bool(getattr(args, "phase05_adaptive_window_dt", False)),
        tension_weights=tw,
        tension_lambda=float(getattr(args, "phase05_tension_lambda", 0.0)),
        anchor_lambda=float(getattr(args, "phase05_anchor_lambda", 0.0)),
        anchor_force_strength=float(getattr(args, "phase05_anchor_force_strength", 0.0)),
        anchor_search_topk=max(1, int(getattr(args, "phase05_anchor_search_topk", 64))),
        anchor_contrastive_weight=float(
            getattr(args, "phase05_anchor_contrastive_weight", 0.0)
        ),
        anchor_contrastive_num_negatives=max(
            1, int(getattr(args, "phase05_anchor_contrastive_num_negatives", 8))
        ),
        enable_anchor_freeze=bool(getattr(args, "phase05_enable_anchor_freeze", False)),
        anchor_freeze_threshold=float(
            getattr(args, "phase05_anchor_freeze_threshold", 0.03)
        ),
        anchor_freeze_max_age=max(
            0, int(getattr(args, "phase05_anchor_freeze_max_age", 0))
        ),
        multi_negative=bool(getattr(args, "phase05_multi_negative", False)),
        num_negatives=max(2, int(getattr(args, "phase05_num_negatives", 4))),
        trajectory_temperature=float(getattr(args, "phase05_traj_temperature", 1.0)),
        trajectory_intermediate_ce_weight=float(
            getattr(args, "trajectory_intermediate_ce_weight", 0.0)
        ),
        trajectory_guidance_nudge_scale=float(
            getattr(args, "trajectory_guidance_nudge_scale", 0.0)
        ),
        trajectory_guidance_mse_weight=float(
            getattr(args, "trajectory_guidance_mse_weight", 0.0)
        ),
        enable_state_normalization=not bool(
            getattr(args, "disable_state_normalization", False)
        ),
        enable_adaptive_attractor_dt=bool(
            getattr(args, "phase05_adaptive_attractor_dt", False)
        ),
        energy_reg_weight=float(getattr(args, "phase05_energy_reg_weight", 0.001)),
    )


def phase2_config_from_args(args: argparse.Namespace) -> Phase2Config:
    tau = getattr(args, "phase2_interaction_decay_tau", None)
    tau_f = float(tau) if tau is not None and tau > 0 else None
    return Phase2Config(
        enable_directional_break=not bool(
            getattr(args, "phase2_disable_directional_break", False)
        ),
        break_base_strength=float(getattr(args, "phase2_break_base_strength", 0.1)),
        break_min_scale=float(getattr(args, "phase2_break_min_scale", 0.1)),
        break_max_scale=float(getattr(args, "phase2_break_max_scale", 2.0)),
        break_t_target=float(getattr(args, "phase2_break_t_target", 0.12)),
        enable_break_rejection=bool(getattr(args, "phase2_enable_break_rejection", False)),
        enable_residual_mixing=not bool(
            getattr(args, "phase2_disable_residual_mixing", False)
        ),
        mixing_gate_init=float(getattr(args, "phase2_mixing_gate_init", 0.1)),
        interaction_reg_weight=float(getattr(args, "phase2_interaction_reg_weight", 0.0)),
        interaction_decay_tau=tau_f,
        enable_head_tension_weighting=bool(
            getattr(args, "phase2_enable_head_tension_weighting", False)
        ),
        store_break_memory=bool(getattr(args, "phase2_store_break_memory", False)),
    )


def phase1_config_from_args(args: argparse.Namespace) -> Phase1Config:
    mode = str(getattr(args, "phase1_head_dim_mode", "shared")).lower()
    if mode not in ("shared", "split"):
        raise SystemExit("--phase1-head-dim-mode must be 'shared' or 'split'")
    return Phase1Config(
        num_heads=max(1, int(getattr(args, "phase1_num_heads", 1))),
        head_dim_mode=mode,
        interaction_scale=float(getattr(args, "phase1_interaction_scale", 0.01)),
        enable_window_interaction=bool(
            getattr(args, "phase1_enable_window_interaction", False)
        ),
        head_diversity_weight=float(getattr(args, "phase1_head_diversity_weight", 0.0)),
        enable_per_head_tension=bool(
            getattr(args, "phase1_enable_per_head_tension", False)
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BoggersTheLanguageModel — attractor dynamics training (see README)."
    )
    # ---- existing args ----
    parser.add_argument("--corpus", type=Path, default=None,
        help=f"Training text (one sentence/line). Default: {DEFAULT_CORPUS_PATH}")
    parser.add_argument("--val-fraction", type=float, default=0.05,
        help="Hold-out fraction for validation (token-level in stream mode). "
        f"Use ~0.3 if you need ≥{MIN_VAL_WINDOWS} val windows; 0 = off.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch-copies", type=int, default=CORPUS_EPOCH_COPIES,
        help="Repeat training sentence list N times per epoch.")
    parser.add_argument(
        "--max-epochs",
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        dest="max_epochs",
        metavar="N",
        help=f"Number of training epochs (default: {NUM_EPOCHS}).",
    )
    parser.add_argument("--baseline-out", type=Path, default=None,
        help="Write Phase-0 baseline snapshot to this file.")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE,
        help="Sliding context length W.")
    parser.add_argument(
        "--state-dim",
        type=int,
        default=512,
        dest="state_dim",
        help="Continuous state dimension D (embedding + dynamics).",
    )
    parser.add_argument(
        "--num-waves",
        type=int,
        default=8,
        dest="num_waves",
        help="Split D into independent wave channels; wave_dim = D // num_waves (each channel has its own dynamics).",
    )
    parser.add_argument(
        "--readout-fusion",
        action="store_true",
        dest="readout_fusion",
        help="Linear(D,D) on each window position before readout_window (uses full D = all wave channels).",
    )
    parser.add_argument("--num-dynamics-steps", "--max-window-steps", type=int, default=MAX_WINDOW_STEPS,
        help="Max outer steps per window (tension-adaptive).")
    parser.add_argument(
        "--convergence-epsilon",
        type=float,
        default=0.0,
        help="Early exit when per-step state delta or |ΔT| is below this (0 = run all outer steps; try 1e-3 on GPU).",
    )
    parser.add_argument(
        "--min-attractor-steps",
        type=int,
        default=2,
        help="Minimum outer attractor steps before early convergence may trigger (>=2).",
    )
    parser.add_argument("--trajectory-batch-size", type=int, default=TRAJECTORY_BATCH_SIZE_DEFAULT,
        help="Batch size for trajectory contrastive training (need ≥2).")
    parser.add_argument("--quick-test", action="store_true",
        help="Window/context sanity checks and exit.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Concise diagnostics: setup summary, each epoch head/tail, first-batch grad norm (trajectory mode).",
    )
    parser.add_argument("--loss-mode", choices=("trajectory", "ce"), default="trajectory")
    parser.add_argument("--token-aux-ce", type=float, default=TOKEN_AUX_CE_WEIGHT_DEFAULT,
        help="trajectory mode: aux readout_window CE weight.")
    parser.add_argument(
        "--trajectory-intermediate-ce-weight",
        type=float,
        default=0.0,
        dest="trajectory_intermediate_ce_weight",
        metavar="W",
        help="trajectory mode: add W * mean(CE per outer dynamics step) to trajectory loss.",
    )
    parser.add_argument(
        "--trajectory-guidance-nudge-scale",
        type=float,
        default=0.0,
        dest="trajectory_guidance_nudge_scale",
        metavar="BETA",
        help="trajectory mode: each outer step S <- S + beta * (target - S) before dynamics "
        "(requires precomputed batch target_states from pipeline). 0 = off.",
    )
    parser.add_argument(
        "--trajectory-guidance-mse-weight",
        type=float,
        default=0.0,
        dest="trajectory_guidance_mse_weight",
        metavar="W",
        help="trajectory mode: add W * MSE(S_pred, target_states) when targets are in the batch.",
    )
    parser.add_argument(
        "--trajectory-guidance-states",
        type=Path,
        default=None,
        help="Optional .pt tensor [n_train_windows, W, D] aligned with stream sliding windows.",
    )
    parser.add_argument(
        "--trajectory-guidance-from-embed",
        action="store_true",
        help="Precompute target states from current token embeddings (stream pipeline only).",
    )
    parser.add_argument(
        "--trajectory-guidance-embed-batch-size",
        type=int,
        default=256,
        metavar="N",
        help="Batch size when building targets with --trajectory-guidance-from-embed.",
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        help="Optional gradient clipping max norm (disabled when not set).",
    )
    parser.add_argument("--lr-decay-every", type=int, default=0)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--epoch-metrics-csv", type=Path, default=None)
    parser.add_argument(
        "--metrics-fast-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Append one row per epoch: batch-wise window energy mean/var/min/max "
            "(from _window_energy_per_batch_row), each averaged over training batches "
            "(trajectory mode only; use e.g. metrics_fast.csv)."
        ),
    )
    parser.add_argument(
        "--attractor-steps-metrics-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Append per-epoch attractor step stats (mean/max/min steps, convergence failures).",
    )
    parser.add_argument("--log-hard-batch-loss-above", type=float, default=0.0)
    # ---- Phase 1: tokenizer ----
    parser.add_argument("--tokenizer", choices=("tiktoken", "fallback"), default="fallback",
        help="'tiktoken': BPE gpt2 (--vocab-cap tokens); 'fallback': same BPE capped at 512.")
    parser.add_argument("--vocab-cap", type=int, default=32768,
        help="BPE vocab cap when --tokenizer tiktoken (default 32768).")
    # ---- Phase 2: data pipeline aliases (accept both hyphens and underscores) ----
    parser.add_argument("--dataset-path", "--dataset_path", type=Path, default=None,
        dest="dataset_path", help="Alias for --corpus (takes precedence).")
    parser.add_argument(
        "--dataset-source",
        "--dataset_source",
        choices=("local", "tinystories", "fineweb-edu"),
        default="local",
        dest="dataset_source",
        help="local = use --corpus / --dataset-path (default). "
        "tinystories / fineweb-edu = download a public corpus via Hugging Face (requires: pip install datasets).",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "cache" / "hf",
        help="Cache directory for --dataset-source Hugging Face materialized .txt files.",
    )
    parser.add_argument(
        "--hf-max-rows",
        type=int,
        default=50_000,
        help="Max rows to read from the Hub dataset (TinyStories slice or FineWeb-Edu stream).",
    )
    parser.add_argument(
        "--hf-max-chars",
        type=int,
        default=0,
        help="Optional cap on total UTF-8 characters when materializing HF data (0 = no cap).",
    )
    parser.add_argument(
        "--hf-refresh",
        action="store_true",
        help="Rebuild cached HF corpus file even if it already exists.",
    )
    parser.add_argument(
        "--no-synthetic-fallback",
        action="store_true",
        help="Do not auto-generate synthetic text when the corpus is missing or too small; exit with an error.",
    )
    parser.add_argument(
        "--eval-results-json",
        type=Path,
        default=None,
        metavar="PATH",
        help="After training, write val CE / perplexity and checkpoint path to this JSON file.",
    )
    parser.add_argument("--seq-len", "--seq_len", type=int, default=None,
        dest="seq_len", help="Alias for --window-size.")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=None,
        dest="batch_size", help="Alias for --trajectory-batch-size.")
    parser.add_argument("--shuffle-buffer", type=int, default=2048,
        help="AttractorDataPipeline shuffle buffer size (line-based mode only).")
    parser.add_argument(
        "--no-streaming-dataset",
        action="store_true",
        help="Legacy: filter corpus line-by-line (short lines dropped). "
        "Default is stream-based: whole corpus → one token sequence → sliding windows.",
    )
    # ---- Phase 3: device ----
    parser.add_argument("--device", default="auto",
        help="'auto' | 'cpu' | 'cuda' | 'cuda:N'.")
    # ---- Phase 4: checkpointing ----
    parser.add_argument("--resume-checkpoint", type=Path, default=None,
        help="Resume training from this checkpoint file.")
    parser.add_argument("--save-every", type=int, default=0,
        help="Save checkpoint every N optimizer steps (0 = only final).")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
        help="Checkpoint directory (default: ./checkpoints).")
    # ---- Phase 5: readout consistency ----
    parser.add_argument("--readout-aux-alpha", type=float, default=0.15,
        help="Weight on aux single-state readout CE loss (0 = disabled).")
    # ---- Phase 7: TSCore substrate (accept both hyphens and underscores) ----
    parser.add_argument("--use-substrate", "--use_substrate", action="store_true",
        dest="use_substrate", help="Enable TSCore substrate coupling (LLMSubstrateNode).")
    # ---- Phase 8: GOAT memory ----
    parser.add_argument("--use-goat-memory", "--use_goat_memory", action="store_true",
        dest="use_goat_memory", help="Enable GOAT-TS memory-state transitions (GoatMemoryManager).")
    # ---- Phase 9: dynamics ----
    parser.add_argument("--dynamics", choices=("simple", "vectorized"), default="vectorized",
        help="'simple': per-wave WaveDynamics only; 'vectorized': MultiHeadDynamics (default).")
    parser.add_argument(
        "--use-lorentz",
        action="store_true",
        dest="use_lorentz",
        help="Vectorized dynamics only: Lorentz/hyperbolic step (tangent projection + curvature scaling).",
    )
    parser.add_argument(
        "--vectorized-num-heads",
        type=int,
        default=4,
        dest="vectorized_num_heads",
        help="MultiHeadDynamics head count (wave_dim = state_dim/num_waves must be divisible by this).",
    )
    parser.add_argument(
        "--vectorized-rank",
        type=int,
        default=64,
        dest="vectorized_rank",
        help="Low-rank factor r per head in vectorized diffusion.",
    )
    parser.add_argument(
        "--vectorized-dt",
        type=float,
        default=0.09,
        dest="vectorized_dt",
        help="Inner Euler dt for MultiHeadDynamics (vectorized path).",
    )
    parser.add_argument(
        "--vectorized-coupling",
        type=float,
        default=0.01,
        dest="vectorized_coupling",
        help="Cross-head coupling strength in MultiHeadDynamics.",
    )
    parser.add_argument(
        "--vectorized-strong-diffusion",
        action="store_true",
        dest="vectorized_strong_diffusion",
        help="Initialize per-head diag(A) eigenvalues in [-0.55,-0.25] (stronger linear contraction).",
    )
    # ---- Phase 0.5: instrumentation + stability (toggleable) ----
    parser.add_argument(
        "--phase05-adaptive-attractor-dt",
        action="store_true",
        dest="phase05_adaptive_attractor_dt",
        help="Per-batch-row inner dt: halve if energy rises, grow (cap base dt) on significant drop.",
    )
    parser.add_argument(
        "--phase05-energy-reg-weight",
        type=float,
        default=0.001,
        dest="phase05_energy_reg_weight",
        metavar="W",
        help="Trajectory: add W * mean(E^2) with E = per-row window potential on S_pred (0 disables).",
    )
    parser.add_argument(
        "--no-state-normalization",
        action="store_true",
        dest="disable_state_normalization",
        help="Disable per-row L2 normalize (dim=-1) after each window step and break updates.",
    )
    parser.add_argument(
        "--phase05-log-metrics",
        action="store_true",
        dest="phase05_log_metrics",
        help="Collect window/token tension breakdown, dynamics, trajectory cosines (cheap batch-boundary export).",
    )
    parser.add_argument(
        "--phase05-batch-metrics-csv",
        type=Path,
        default=None,
        dest="phase05_batch_metrics_csv",
        help="Append one CSV row per training batch (implies --phase05-log-metrics).",
    )
    parser.add_argument(
        "--phase05-enforce-negdef-diffusion",
        action="store_true",
        dest="phase05_enforce_negdef_diffusion",
        help="SimpleAttractorDynamics: D = -(A^T A) - eps I (strictly negative definite).",
    )
    parser.add_argument(
        "--phase05-adaptive-window-dt",
        action="store_true",
        dest="phase05_adaptive_window_dt",
        help="EMA scale on window positional dt from per-step tension (clamped, smooth).",
    )
    parser.add_argument(
        "--phase05-tension-w",
        type=str,
        default=None,
        dest="phase05_tension_w",
        metavar="W1,W2,W3",
        help="Tension weights (energy, alignment, entropy). Default: 1,λ,μ from model buffers.",
    )
    parser.add_argument(
        "--phase05-tension-lambda",
        type=float,
        default=0.0,
        dest="phase05_tension_lambda",
        metavar="LAMBDA",
        help="λ in window energy descent: E = mean(sum_i energy_heads[i])+λ·mean(compute_tension_window); 0=head only.",
    )
    parser.add_argument(
        "--phase05-anchor-lambda",
        type=float,
        default=0.0,
        dest="phase05_anchor_lambda",
        metavar="LAMBDA",
        help="Weight on mean nearest-token-embedding L2 distance in window energy (embeddings detached); 0=off.",
    )
    parser.add_argument(
        "--phase05-anchor-force-strength",
        type=float,
        default=0.0,
        dest="phase05_anchor_force_strength",
        metavar="S",
        help="Before energy descent each step: S <- S - strength * (S - nearest_embed); embeddings detached; 0=off.",
    )
    parser.add_argument(
        "--phase05-anchor-search-topk",
        type=int,
        default=64,
        dest="phase05_anchor_search_topk",
        metavar="K",
        help="Anchor search: readout_window top-K tokens per batch row; distances only vs those embeds.",
    )
    parser.add_argument(
        "--phase05-anchor-contrastive-weight",
        type=float,
        default=0.0,
        dest="phase05_anchor_contrastive_weight",
        metavar="W",
        help="Trajectory: W * anchor contrastive loss on last window state vs target vs random neg embeds.",
    )
    parser.add_argument(
        "--phase05-anchor-contrastive-num-negatives",
        type=int,
        default=8,
        dest="phase05_anchor_contrastive_num_negatives",
        metavar="K",
        help="Anchor contrastive: K random negative token embeddings per row (excluding target).",
    )
    parser.add_argument(
        "--phase05-enable-anchor-freeze",
        action="store_true",
        dest="phase05_enable_anchor_freeze",
        help="Zero energy gradients on wave slices within anchor_freeze_threshold of top-k embeds; see max-age.",
    )
    parser.add_argument(
        "--phase05-anchor-freeze-threshold",
        type=float,
        default=0.03,
        dest="phase05_anchor_freeze_threshold",
        metavar="D",
        help="Per-wave L2 distance to nearest top-k anchor slice below which gradient freeze applies.",
    )
    parser.add_argument(
        "--phase05-anchor-freeze-max-age",
        type=int,
        default=0,
        dest="phase05_anchor_freeze_max_age",
        metavar="N",
        help="Max consecutive stable outer steps to keep freezing (0 = no cap).",
    )
    parser.add_argument(
        "--phase05-multi-negative",
        action="store_true",
        dest="phase05_multi_negative",
        help="Trajectory contrastive: average cosine over K random permutations (see --phase05-num-negatives).",
    )
    parser.add_argument(
        "--phase05-num-negatives",
        type=int,
        default=4,
        dest="phase05_num_negatives",
        metavar="K",
        help="Number of shuffled negatives per batch row for trajectory loss (>=2).",
    )
    parser.add_argument(
        "--phase05-traj-temperature",
        type=float,
        default=1.0,
        dest="phase05_traj_temperature",
        help="Divides trajectory margin inside ReLU; 1.0 preserves default scaling.",
    )
    # ---- Phase 1: multi-head diffusion + structured window coupling ----
    parser.add_argument(
        "--phase1-num-heads",
        type=int,
        default=1,
        dest="phase1_num_heads",
        help="SimpleAttractorDynamics: parallel diffusion heads (1 = legacy single-matrix drift).",
    )
    parser.add_argument(
        "--phase1-head-dim-mode",
        choices=("shared", "split"),
        default="shared",
        dest="phase1_head_dim_mode",
        help="shared: H full D×D matrices; split: H blocks of (D/H)×(D/H) (requires D %% H == 0).",
    )
    parser.add_argument(
        "--phase1-interaction-scale",
        type=float,
        default=0.01,
        dest="phase1_interaction_scale",
        help="Scales learnable cross-position coupling delta after local dynamics.",
    )
    parser.add_argument(
        "--phase1-enable-window-interaction",
        action="store_true",
        dest="phase1_enable_window_interaction",
        help="Add einsum('bid,ij->bjd', S, C) after each local dynamics step.",
    )
    parser.add_argument(
        "--phase1-head-diversity-weight",
        type=float,
        default=0.0,
        dest="phase1_head_diversity_weight",
        help="Penalty on mean pairwise cosine similarity of head drift directions (0 = off).",
    )
    parser.add_argument(
        "--phase1-enable-per-head-tension",
        action="store_true",
        dest="phase1_enable_per_head_tension",
        help="When --phase05-log-metrics, log mean per-head geometry tension (split layout on D).",
    )
    # ---- Phase 2: directional breaks, residual mixing, C regularisation, head tension weights ----
    parser.add_argument(
        "--phase2-disable-directional-break",
        action="store_true",
        dest="phase2_disable_directional_break",
        help="Use legacy Gaussian jitter for window/token breaks.",
    )
    parser.add_argument(
        "--phase2-break-base-strength",
        type=float,
        default=0.1,
        dest="phase2_break_base_strength",
    )
    parser.add_argument(
        "--phase2-break-min-scale",
        type=float,
        default=0.1,
        dest="phase2_break_min_scale",
    )
    parser.add_argument(
        "--phase2-break-max-scale",
        type=float,
        default=2.0,
        dest="phase2_break_max_scale",
    )
    parser.add_argument(
        "--phase2-break-t-target",
        type=float,
        default=0.12,
        dest="phase2_break_t_target",
        help="Reference T in α = base * clamp((T_ref−T)/T_ref, …).",
    )
    parser.add_argument(
        "--phase2-enable-break-rejection",
        action="store_true",
        dest="phase2_enable_break_rejection",
    )
    parser.add_argument(
        "--phase2-disable-residual-mixing",
        action="store_true",
        dest="phase2_disable_residual_mixing",
        help="Use Phase-1 linear mix only (no state + gate * mixed).",
    )
    parser.add_argument(
        "--phase2-mixing-gate-init",
        type=float,
        default=0.1,
        dest="phase2_mixing_gate_init",
        help="Initial sigmoid(gate_raw) for residual mixing.",
    )
    parser.add_argument(
        "--phase2-interaction-reg-weight",
        type=float,
        default=0.0,
        dest="phase2_interaction_reg_weight",
        help="||C−I||² weight (requires --phase1-enable-window-interaction).",
    )
    parser.add_argument(
        "--phase2-interaction-decay-tau",
        type=float,
        default=None,
        dest="phase2_interaction_decay_tau",
        metavar="TAU",
        help="If set, multiply C elementwise by exp(−|i−j|/τ).",
    )
    parser.add_argument(
        "--phase2-enable-head-tension-weighting",
        action="store_true",
        dest="phase2_enable_head_tension_weighting",
        help="Head-level softmax(−T_slice) weights on drift heads (not token attention).",
    )
    parser.add_argument(
        "--phase2-store-break-memory",
        action="store_true",
        dest="phase2_store_break_memory",
        help="Keep last pre/post break window states on the model.",
    )

    args = parser.parse_args()

    # ---- resolve aliases ----
    corpus_path = args.dataset_path or args.corpus or DEFAULT_CORPUS_PATH
    if args.dataset_source != "local":
        from data.hf_remote_corpus import ensure_hf_corpus_file  # type: ignore[import]

        corpus_path = ensure_hf_corpus_file(
            args.dataset_source,
            cache_dir=args.hf_cache_dir,
            max_rows=max(1, args.hf_max_rows),
            max_chars=max(0, args.hf_max_chars),
            refresh=args.hf_refresh,
        )
    window_size = args.seq_len if args.seq_len is not None else args.window_size
    traj_batch_size = args.batch_size if args.batch_size is not None else args.trajectory_batch_size

    if window_size < 2:
        raise SystemExit("window size must be >= 2")
    if args.state_dim % args.num_waves != 0:
        raise SystemExit("--state-dim must be divisible by --num-waves")
    if args.loss_mode == "trajectory" and traj_batch_size < 2:
        raise SystemExit("trajectory batch size must be >= 2 for contrastive training")
    num_epochs = args.max_epochs
    if num_epochs < 1:
        raise SystemExit("--max-epochs must be >= 1")

    random.seed(args.seed)

    # ---- Phase 3: device ----
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # ---- Phase 1: tokenizer ----
    tok = _build_tokenizer(mode=args.tokenizer, vocab_cap=args.vocab_cap)
    vocab_size = tok.n_vocab
    print(f"Vocab size: {vocab_size}  tokenizer={args.tokenizer}", flush=True)

    # ---- Build model ----
    _p05 = phase05_config_from_args(args)
    _p1 = phase1_config_from_args(args)
    _p2 = phase2_config_from_args(args)
    model = TorchAttractorLanguageModel(
        vocab_size,
        state_dim=int(args.state_dim),
        train_window_size=window_size,
        max_window_steps=args.num_dynamics_steps,
        convergence_epsilon=float(args.convergence_epsilon),
        min_attractor_steps=max(2, int(args.min_attractor_steps)),
        num_waves=int(args.num_waves),
        use_readout_fusion=bool(getattr(args, "readout_fusion", False)),
        phase05=_p05,
        phase1=_p1,
        phase2=_p2,
    )
    model.tokenizer = tok

    start_epoch = 0
    global_step = 0
    ckpt = None

    if args.resume_checkpoint is not None and args.resume_checkpoint.is_file():
        ckpt = torch.load(args.resume_checkpoint, map_location="cpu")
        load_torch_attractor_state_dict(model, ckpt["model_state"])
        global_step = ckpt.get("step", 0)
        start_epoch = ckpt.get("epoch", 0)
        print(
            f"Resumed from {args.resume_checkpoint} (step={global_step}, epoch={start_epoch})",
            flush=True,
        )
        if args.debug:
            _training_debug(
                True,
                f"resume: continuing from epoch index {start_epoch}  global_step={global_step}",
            )

    # ---- Phase 3: move to device ----
    model = model.to(device)
    # ---- Phase 4: build optimizer (after device placement) ----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    if ckpt is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    _compile_status = "skipped"
    if args.debug:
        _np = sum(p.numel() for p in model.parameters())
        _training_debug(
            True,
            f"model params={_np:,}  state_dim={model.state_dim}  num_waves={model.num_waves}  "
            f"wave_dim={model.wave_dim}  train_window={model.train_window_size}  "
            f"max_window_steps={model.max_window_steps}",
        )

    # ---- Phase 9: dynamics swap (vectorized default; fall back to simple on failure) ----
    _dynamics_backend = str(args.dynamics)
    if _dynamics_backend == "vectorized":
        try:
            from dynamics_vectorized import VectorizedWindowDynamics  # type: ignore[import]

            _vnh = max(1, int(getattr(args, "vectorized_num_heads", 4)))
            if model.wave_dim % _vnh != 0:
                raise ValueError(
                    f"wave_dim={model.wave_dim} (state_dim/num_waves) not divisible by "
                    f"--vectorized-num-heads={_vnh}"
                )
            _dmin, _dmax = (-0.55, -0.25) if bool(
                getattr(args, "vectorized_strong_diffusion", False)
            ) else (-0.4, -0.1)
            model.dynamics = VectorizedWindowDynamics(
                state_dim=model.wave_dim,
                window_size=model.train_window_size,
                num_heads=_vnh,
                rank=max(1, int(getattr(args, "vectorized_rank", 64))),
                max_steps=model.max_window_steps,
                dt=float(getattr(args, "vectorized_dt", 0.09)),
                coupling=float(getattr(args, "vectorized_coupling", 0.01)),
                use_lorentz=bool(args.use_lorentz),
                diag_eigen_min=_dmin,
                diag_eigen_max=_dmax,
            ).to(device)
            print("[phase-9] VectorizedWindowDynamics active", flush=True)
        except Exception as _dyn_err:
            print(
                f"[phase-9] Warning: vectorized dynamics unavailable ({_dyn_err}); "
                "using per-wave WaveDynamics.",
                flush=True,
            )
            _dynamics_backend = "simple"

    if torch.cuda.is_available():
        try:
            dyn = model.dynamics
            if _dynamics_backend == "vectorized" and hasattr(dyn, "_step"):
                try:
                    dyn._step = torch.compile(dyn._step, mode="reduce-overhead")  # type: ignore[assignment]
                    _compile_status = "vectorized _step"
                except Exception as _ve:
                    print(f"[step-2] Warning: vectorized _step compile skipped ({_ve})", flush=True)
                    _compile_status = "vectorized _step failed"
            elif hasattr(dyn, "_step_rows"):
                try:
                    dyn._step_rows = torch.compile(dyn._step_rows, mode="reduce-overhead")  # type: ignore[assignment]
                    _compile_status = "simple _step_rows"
                except Exception as _se:
                    print(f"[step-2] Warning: simple _step_rows compile skipped ({_se})", flush=True)
                    _compile_status = "simple _step_rows failed"
            else:
                _compile_status = "skipped (no inner step)"
        except Exception as _comp_err:
            print(f"[step-2] Warning: torch.compile skipped ({_comp_err})", flush=True)
            _compile_status = f"error ({_comp_err})"

    if args.quick_test:
        if args.debug:
            _training_debug(True, "quick_test: window/context sanity checks then exit")
        run_quick_window_tests(model)
        return

    lr_scheduler = None
    if args.lr_decay_every > 0:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_decay_every, gamma=args.lr_gamma
        )

    # ---- Phase 7: TSCore substrate ----
    substrate = None
    if args.use_substrate:
        try:
            from llm_substrate_node import LLMSubstrateNode  # type: ignore[import]
            substrate = LLMSubstrateNode(model, quiet=True)
            print("[phase-7] TSCore substrate node active", flush=True)
        except Exception as _sub_err:
            print(f"[phase-7] Warning: substrate unavailable ({_sub_err})", flush=True)

    # ---- Phase 8: GOAT memory ----
    if args.use_goat_memory:
        try:
            from goat_memory_transitions import GoatMemoryManager  # type: ignore[import]
            model._goat_mgr = GoatMemoryManager(model, bonus_scale=0.05)
            print("[phase-8] GOAT memory manager active", flush=True)
        except Exception as _goat_err:
            print(f"[phase-8] Warning: GOAT memory unavailable ({_goat_err})", flush=True)

    if args.debug:
        _training_debug(
            True,
            f"dynamics={type(model.dynamics).__name__ if model.dynamics is not None else 'WaveDynamics'}  "
            f"torch.compile={_compile_status}",
        )
        _training_debug(
            True,
            f"integrations: substrate={substrate is not None}  "
            f"goat={getattr(model, '_goat_mgr', None) is not None}",
        )

    # ---- checkpoint dir ----
    ckpt_dir = args.checkpoint_dir or (_REPO_ROOT / "checkpoints")

    # ---- Phase 2: data pipeline (stream-based by default) ----
    streaming_dataset = not args.no_streaming_dataset
    stream_train_ids: list[int] | None = None
    train_sents: list[str] = []
    _tmp_train_path: Path | None = None
    _synthetic_corpus_cleanup: Path | None = None
    val_metrics_unreliable = False
    stream_source_path = corpus_path
    stream_val_fraction_effective: float = 0.0

    if streaming_dataset:
        from data_pipeline import _collect_text_files  # type: ignore[import]

        files = _collect_text_files(corpus_path)
        need_syn = not files
        if files:
            full_text = load_corpus_text_stream(corpus_path)
            tokens = tok.encode(full_text)
            need_syn = len(tokens) < window_size * 20
        if need_syn:
            if args.no_synthetic_fallback:
                raise SystemExit(
                    "Corpus missing or too small after tokenization, and --no-synthetic-fallback is set. "
                    "Provide a larger --corpus / --dataset-path, use --dataset-source tinystories (or fineweb-edu), "
                    "or remove --no-synthetic-fallback."
                )
            print("Corpus too small — generating synthetic corpus...", flush=True)
            import tempfile as _tmpcorpus

            _tf = _tmpcorpus.NamedTemporaryFile(
                mode="w", delete=False, suffix=".txt", encoding="utf-8"
            )
            _tf.close()
            _syn_p = Path(_tf.name)
            from data.generate_corpus import generate_corpus  # type: ignore[import]

            generate_corpus(
                _syn_p,
                target_tokens=max(20_000, window_size * 20 + 2_000),
                seed=args.seed,
            )
            stream_source_path = _syn_p
            _synthetic_corpus_cleanup = _syn_p
            full_text = load_corpus_text_stream(stream_source_path)
            tokens = tok.encode(full_text)

        need = window_size + 1
        if len(tokens) < need:
            raise RuntimeError(
                "Corpus too small after tokenization: need at least "
                f"{need} tokens for window_size={window_size}, got {len(tokens)}."
            )
        train_tok, val_tok, stream_val_fraction_effective = split_tokens_train_val(
            tokens, args.val_fraction, window_size
        )
        if args.val_fraction > 0 and not val_tok:
            _min_stream = (window_size + 1) + window_size + (window_size + 1)
            print(
                f"Warning: token-level val split skipped (need ≥ {_min_stream} tokens "
                f"for window_size={window_size}, including train/val gap); "
                "using all tokens for training.",
                flush=True,
            )
        n_train_win = max(0, len(train_tok) - window_size)
        n_val_win = max(0, len(val_tok) - window_size)
        print(f"total_tokens={len(tokens)}", flush=True)
        print(f"train_tokens={len(train_tok)}", flush=True)
        print(f"val_tokens={len(val_tok)}", flush=True)
        print(f"train_windows={n_train_win}", flush=True)
        print(f"val_windows={n_val_win}", flush=True)
        if args.val_fraction > 0 and val_tok:
            print(
                f"final val_fraction={stream_val_fraction_effective:.6g} "
                f"(requested --val-fraction={args.val_fraction:g})",
                flush=True,
            )
        elif args.val_fraction > 0 and not val_tok:
            print("final val_fraction=0 (hold-out could not be formed)", flush=True)
        elif args.val_fraction <= 0:
            print("final val_fraction=0 (validation off)", flush=True)
        if val_tok and n_val_win < MIN_VAL_WINDOWS:
            print("WARNING: validation unreliable", flush=True)
            val_metrics_unreliable = True
            print(
                f"Warning: val_windows={n_val_win} < {MIN_VAL_WINDOWS} — "
                "treat val CE / perplexity as unreliable. Prefer --val-fraction 0.3 or more text. "
                "Tiny corpora: loss deltas (e.g. GOAT on/off) reflect integration noise, not real gains.",
                flush=True,
            )
        print(
            f"Loaded corpus (stream): {stream_source_path}",
            flush=True,
        )
        print(
            f"  window_size={window_size}  "
            f"(train/val gap={window_size} tokens, no window leakage)",
            flush=True,
        )
        val_dataset = build_dataset_from_token_ids(val_tok, window_size)
        if len(tokens) < 5000 and _synthetic_corpus_cleanup is None:
            print(
                "Note: very small corpus — use training for integration checks only; "
                "do not interpret val metrics or A/B deltas as model quality.",
                flush=True,
            )
        stream_train_ids = train_tok
        if args.epoch_copies > 1:
            print(
                "Note: --epoch-copies is ignored in stream mode; use --max-epochs for more passes.",
                flush=True,
            )
    else:
        sentences = load_corpus(corpus_path)
        print(f"Loaded corpus (line mode): {corpus_path}  ({len(sentences)} lines)", flush=True)
        corpus_coverage_report(sentences, tok, window_size)

        usable = sentences_with_training_windows(sentences, tok, window_size)
        if not usable:
            raise RuntimeError(
                "No corpus lines have enough tokens to form a training window. "
                "Add more text, lower --window-size, or remove --no-streaming-dataset "
                "to use stream-based tokenization."
            )
        n_skip = len(sentences) - len(usable)
        if n_skip:
            print(
                f"Training uses only {len(usable)} of {len(sentences)} lines "
                f"({n_skip} too short for window_size={window_size}).",
                flush=True,
            )

        train_sents, val_sents = train_val_split(usable, args.val_fraction, args.seed)
        if val_sents:
            print(
                f"Train/val split: {len(train_sents)} train, {len(val_sents)} val "
                f"(fraction={args.val_fraction:g})",
                flush=True,
            )
        val_dataset = build_dataset_from_sentences(val_sents, model, window_size)
        n_val_win_line = len(val_dataset)
        if val_sents and n_val_win_line < MIN_VAL_WINDOWS:
            val_metrics_unreliable = True
            print(
                f"Warning: val_windows={n_val_win_line} < {MIN_VAL_WINDOWS} — "
                "treat val CE / perplexity as unreliable. Prefer --val-fraction 0.3 or more text.",
                flush=True,
            )

        import tempfile as _tempfile

        _tmp_f = _tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        for _s in train_sents * args.epoch_copies:
            _tmp_f.write(_s + "\n")
        _tmp_f.close()
        _tmp_train_path = Path(_tmp_f.name)

    train_guidance_tensor: torch.Tensor | None = None
    if streaming_dataset and stream_train_ids is not None:
        tg_path = getattr(args, "trajectory_guidance_states", None)
        tg_embed = bool(getattr(args, "trajectory_guidance_from_embed", False))
        if tg_path is not None and tg_embed:
            raise SystemExit(
                "Use only one of --trajectory-guidance-states and --trajectory-guidance-from-embed."
            )
        n_win_exp = max(0, len(stream_train_ids) - window_size)
        Wm, Dm = window_size, int(model.state_dim)
        if tg_path is not None:
            if not tg_path.is_file():
                raise SystemExit(f"--trajectory-guidance-states not found: {tg_path}")
            try:
                loaded = torch.load(tg_path, map_location="cpu", weights_only=True)
            except TypeError:
                loaded = torch.load(tg_path, map_location="cpu")
            ts = loaded if isinstance(loaded, torch.Tensor) else torch.as_tensor(
                loaded, dtype=torch.float32
            )
            ts = ts.to(dtype=torch.float32).contiguous()
            if ts.dim() != 3 or ts.shape[0] != n_win_exp or ts.shape[1] != Wm or ts.shape[2] != Dm:
                raise SystemExit(
                    f"--trajectory-guidance-states: expected shape ({n_win_exp}, {Wm}, {Dm}), "
                    f"got {tuple(ts.shape)}"
                )
            train_guidance_tensor = ts
            print(
                f"Loaded trajectory guidance tensor {tuple(train_guidance_tensor.shape)} from {tg_path}",
                flush=True,
            )
        elif tg_embed:
            if n_win_exp <= 0:
                raise SystemExit(
                    "--trajectory-guidance-from-embed: no training windows (corpus too short)."
                )
            eb = max(1, int(getattr(args, "trajectory_guidance_embed_batch_size", 256)))
            train_guidance_tensor = precompute_stream_target_states_embed(
                model,
                stream_train_ids,
                window_size,
                device=device,
                embed_batch_size=eb,
            )
            print(
                f"Precomputed trajectory guidance targets embed: shape={tuple(train_guidance_tensor.shape)}",
                flush=True,
            )

    pipeline = None
    try:
        from data_pipeline import AttractorDataPipeline  # type: ignore[import]
        if streaming_dataset:
            pipeline = AttractorDataPipeline(
                sources=[stream_source_path],
                model=model,
                batch_size=traj_batch_size,
                window_size=window_size,
                shuffle_buffer=args.shuffle_buffer,
                tokenizer=tok,
                seed=args.seed,
                streaming_dataset=True,
                train_token_ids=stream_train_ids,
                train_target_states=train_guidance_tensor,
            )
        else:
            assert _tmp_train_path is not None
            pipeline = AttractorDataPipeline(
                sources=[_tmp_train_path],
                model=model,
                batch_size=traj_batch_size,
                window_size=window_size,
                shuffle_buffer=args.shuffle_buffer,
                tokenizer=tok,
                seed=args.seed,
                streaming_dataset=False,
            )
    except Exception as _pipe_err:
        print(
            f"Warning: data pipeline unavailable ({_pipe_err}), "
            "falling back to legacy in-memory loader.",
            flush=True,
        )

    _tg_mse = float(model.phase05_config.trajectory_guidance_mse_weight)
    _tg_nudge = float(model.phase05_config.trajectory_guidance_nudge_scale)
    if (_tg_mse > 0.0 or _tg_nudge > 0.0) and (
        pipeline is None or getattr(pipeline, "_train_target_states", None) is None
    ):
        print(
            "Warning: trajectory guidance MSE/nudge weights are > 0 but the data pipeline has "
            "no train_target_states — guidance is inactive. Use stream mode with "
            "--trajectory-guidance-from-embed or --trajectory-guidance-states PATH.pt.",
            flush=True,
        )

    print(
        f"Pre-training ({num_epochs} epochs, window={window_size}, "
        f"dynamics_steps={args.num_dynamics_steps}, loss={args.loss_mode}, "
        f"aux_ce={args.token_aux_ce}, readout_aux_alpha={args.readout_aux_alpha}, "
        f"batch={traj_batch_size}, lr={args.lr}, device={device})...",
        flush=True,
    )
    if args.debug:
        if pipeline is not None:
            try:
                _est = pipeline.epoch_count_estimate()
            except Exception:
                _est = -1
            _training_debug(
                True,
                f"pipeline streaming={getattr(pipeline, 'streaming_dataset', '?')} "
                f"batch_size={getattr(pipeline, 'batch_size', '?')} ~batches/epoch≈{_est}",
            )
        else:
            _training_debug(True, "pipeline: legacy in-memory iterator (no AttractorDataPipeline)")
    if (
        args.loss_mode == "trajectory"
        and args.token_aux_ce <= 0
        and args.readout_aux_alpha <= 0
        and getattr(args, "trajectory_intermediate_ce_weight", 0.0) <= 0
    ):
        print(
            "Warning: trajectory mode with token_aux_ce=0, readout_aux_alpha=0, and "
            "trajectory_intermediate_ce_weight=0 — readout head gets no gradients. "
            "Use --token-aux-ce, --readout-aux-alpha, or --trajectory-intermediate-ce-weight.",
            flush=True,
        )

    t_train0 = time.perf_counter()
    last_mean_loss = 0.0
    last_train_ce = 0.0
    last_val_ce: float | None = None
    last_train_traj_contrast: float | None = None
    last_val_traj_contrast: float | None = None
    last_n_windows = 0
    last_epoch_sec = 0.0
    last_epoch_num = 0

    phase05_csv_fp = None
    phase05_csv_writer = None
    if (
        model.phase05_config.batch_metrics_csv is not None
        and args.loss_mode == "trajectory"
    ):
        pcsv = Path(model.phase05_config.batch_metrics_csv)
        pcsv.parent.mkdir(parents=True, exist_ok=True)
        new_p05 = not pcsv.exists() or pcsv.stat().st_size == 0
        phase05_csv_fp = pcsv.open("a", newline="", encoding="utf-8")
        phase05_csv_writer = csv.writer(phase05_csv_fp)
        if new_p05:
            phase05_csv_writer.writerow(PHASE05_BATCH_CSV_HEADER)

    if args.debug:
        _training_debug(
            True,
            f"train loop: epochs {start_epoch + 1}..{start_epoch + num_epochs}  "
            f"global_step_start={global_step}  loss_mode={args.loss_mode}",
        )

    for epoch in range(start_epoch, start_epoch + num_epochs):
        t_ep0 = time.perf_counter()
        loss_sum = 0.0
        mean_final_step_tension = float("nan")
        max_batch_loss_epoch = float("nan")
        final_tension_values: list[float] = []
        max_batch_loss = -1.0
        batch_idx = -1
        attractor_steps_hist: list[int] = []
        attractor_conv_hist: list[bool] = []
        energy_fast_sum_mean = 0.0
        energy_fast_sum_var = 0.0
        energy_fast_sum_min = 0.0
        energy_fast_sum_max = 0.0
        energy_fast_n = 0

        if pipeline is not None:
            # ---- Phase 2: streaming data pipeline ----
            batch_iter = pipeline.epoch_batches(epoch_index=epoch)
            n_est = max(1, pipeline.epoch_count_estimate())
            report_every = max(1, n_est // 10)
        else:
            # Legacy in-memory fallback (keeps working if data_pipeline unavailable).
            legacy_dataset: list = []
            _rng_ep = random.Random(args.seed + epoch)
            if stream_train_ids is not None:
                legacy_dataset = build_sequence_dataset(stream_train_ids, window_size)
                _rng_ep.shuffle(legacy_dataset)
            else:
                training_sentences = list(train_sents * args.epoch_copies)
                _rng_ep.shuffle(training_sentences)
                for _s in training_sentences:
                    _ids = tok.encode(_s)
                    if len(_ids) >= window_size + 1:
                        legacy_dataset.extend(build_sequence_dataset(_ids, window_size))
                _rng_ep.shuffle(legacy_dataset)
            _bs = max(2, traj_batch_size)

            def _legacy_batch_iter():
                for _start in range(0, len(legacy_dataset), _bs):
                    chunk = legacy_dataset[_start : _start + _bs]
                    if len(chunk) < 2:
                        chunk = chunk * 2
                    yield [c for c, _ in chunk], [t for _, t in chunk], None

            batch_iter = _legacy_batch_iter()
            n_est = max(1, (len(legacy_dataset) + _bs - 1) // _bs)
            report_every = max(1, n_est // 10)

        print(f"  epoch {epoch + 1}/{start_epoch + num_epochs}", flush=True)
        if args.debug:
            _training_debug(
                True,
                f"epoch {epoch + 1}: ~{n_est} batches  report_every={report_every}  "
                f"lr={optimizer.param_groups[0]['lr']:.6g}",
            )

        # Bug 4: accumulate real train CE from batch readout logits.
        train_ce_sum = 0.0
        train_ce_count = 0
        # Bug 5/6: TSCore per-epoch tracking.
        substrate_evolve_start = substrate.evolve_count if substrate is not None else 0
        substrate_idle_count = 0
        substrate_active_count = 0
        # Last batch data for train_traj_contrast (Bug 3).
        _last_batch_contexts: list[list[int]] = []
        _last_batch_targets: list[int] = []
        _last_batch_target_states: torch.Tensor | None = None

        if args.loss_mode == "trajectory":
            for batch_idx, (contexts, targets, target_states_batch) in tqdm(
                enumerate(batch_iter),
                total=n_est,
                desc=f"epoch {epoch + 1}/{num_epochs}",
            ):
                if len(contexts) < 2:
                    contexts = contexts * 2
                    targets = targets * 2
                    if target_states_batch is not None:
                        target_states_batch = target_states_batch.repeat(2, 1, 1)

                # Phase 3: targets tensor on device
                targets_tensor = torch.tensor(targets, device=device, dtype=torch.long)
                _last_batch_contexts = contexts
                _last_batch_targets = targets
                ts_b: torch.Tensor | None = None
                if target_states_batch is not None:
                    ts_b = target_states_batch.to(
                        device=device, dtype=model.embedder.weight.dtype
                    )
                _last_batch_target_states = (
                    ts_b.detach().cpu() if ts_b is not None else None
                )

                loss_traj, logits = model.trajectory_contrastive_loss_and_logits(
                    contexts, targets, target_states=ts_b
                )
                loss = loss_traj

                # Bug 4: accumulate real batch CE from readout_window logits.
                with torch.no_grad():
                    _bce = float(
                        F.cross_entropy(logits, targets_tensor, label_smoothing=LABEL_SMOOTHING)
                        .detach()
                    )
                    if math.isfinite(_bce):
                        train_ce_sum += _bce
                        train_ce_count += 1

                # Phase 5: auxiliary single-state readout loss
                if args.readout_aux_alpha > 0 and model._last_pred_final_state is not None:
                    # Aux loss only: single-position readout (not the primary readout_window CE path).
                    single_logits = model.readout(model._last_pred_final_state.to(device))
                    single_logits = single_logits / model.effective_temperature()
                    single_logits = torch.nan_to_num(
                        single_logits, nan=0.0, posinf=0.0, neginf=-1e4
                    )
                    aux_r_loss = F.cross_entropy(
                        single_logits, targets_tensor, label_smoothing=LABEL_SMOOTHING
                    )
                    loss = loss + args.readout_aux_alpha * aux_r_loss

                if args.token_aux_ce > 0.0:
                    loss = loss + args.token_aux_ce * _aux_ce_loss_batch(
                        model, logits, contexts, targets
                    )

                curve = model._last_window_tension_curve
                if curve:
                    final_tension_values.append(curve[-1])
                li = float(loss.detach())
                loss_sum += li
                if li > max_batch_loss:
                    max_batch_loss = li
                if args.log_hard_batch_loss_above > 0 and li >= args.log_hard_batch_loss_above:
                    print(
                        f"    [hard batch] bi={batch_idx + 1} loss={li:.4f}  "
                        f"ctx0={contexts[0][:6]}",
                        flush=True,
                    )

                # One backward per batch (no loss.backward(retain_graph=True)).
                # Inner window energy uses autograd.grad(create_graph=True); that subgraph
                # is kept alive per-step via grad()'s default retain_graph=create_graph
                # (see _window_energy_gradient_step — do not override with retain_graph=False).
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if args.debug and batch_idx == 0:
                    _gn = float(
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
                    )
                    _lf = float(loss.detach()) if torch.isfinite(loss.detach()) else float("nan")
                    _ok = bool(torch.isfinite(logits).all().item())
                    _training_debug(
                        True,
                        f"first batch: loss={_lf:.4f}  grad_norm={_gn:.4f}  logits_all_finite={_ok}",
                    )
                optimizer.step()
                global_step += 1

                if args.attractor_steps_metrics_csv is not None:
                    attractor_steps_hist.append(int(model._last_attractor_steps_used))
                    attractor_conv_hist.append(bool(model._last_convergence_triggered))

                if args.metrics_fast_csv is not None:
                    _bem = getattr(model, "_last_batch_energy_mean", float("nan"))
                    if math.isfinite(_bem):
                        energy_fast_sum_mean += _bem
                        energy_fast_sum_var += float(
                            getattr(model, "_last_batch_energy_variance", 0.0)
                        )
                        energy_fast_sum_min += float(
                            getattr(model, "_last_batch_energy_min", float("nan"))
                        )
                        energy_fast_sum_max += float(
                            getattr(model, "_last_batch_energy_max", float("nan"))
                        )
                        energy_fast_n += 1

                if phase05_csv_writer is not None:
                    _bce_log = _bce if math.isfinite(_bce) else float("nan")
                    phase05_csv_writer.writerow(
                        model.phase05_batch_csv_values(
                            epoch, batch_idx, global_step, _bce_log
                        )
                    )
                    if (batch_idx + 1) % 32 == 0 and phase05_csv_fp is not None:
                        phase05_csv_fp.flush()

                # Phase 7: TSCore substrate coupling + idle tracking (Bugs 5/6).
                if substrate is not None:
                    _curve = model._last_window_tension_curve
                    _lang_t = float(_curve[-1]) if _curve else 0.0
                    if _lang_t < substrate.high_tension_threshold:
                        substrate_idle_count += 1
                    else:
                        substrate_active_count += 1
                    try:
                        substrate.on_batch(model)
                    except Exception:
                        pass

                # Phase 8: GOAT memory tick
                if model._goat_mgr is not None:
                    try:
                        model._goat_mgr.tick(contexts)
                    except Exception:
                        pass

                # Phase 4: periodic checkpoint
                if args.save_every > 0 and global_step % args.save_every == 0:
                    _save_checkpoint(model, optimizer, global_step, epoch, args, ckpt_dir)

                ts = model._last_adaptive_window_steps
                if batch_idx % report_every == 0:
                    _wt = model._phase05_last_window_trace or {}
                    _ebar = _wt.get("energy_descent_base_mean", float("nan"))
                    _etn = _wt.get("energy_descent_tension_mean", float("nan"))
                    _etot = _wt.get("energy_descent_total_mean", float("nan"))
                    _estr = ""
                    if model.phase05_config.log_metrics and (
                        math.isfinite(_ebar)
                        or math.isfinite(_etn)
                        or math.isfinite(_etot)
                    ):
                        _estr = (
                            f"  E_base={_ebar:.4f}  T_en={_etn:.4f}  E_tot={_etot:.4f}"
                        )
                    _ffm = _wt.get("frozen_fraction_mean", float("nan"))
                    _ffs = _wt.get("frozen_fraction_std", float("nan"))
                    if model.phase05_config.log_metrics and math.isfinite(_ffm):
                        _estr += f"  ff_m={_ffm:.3f}  ff_s={_ffs:.3f}"
                    _adc = model._last_attractor_dt_curve
                    if _adc and model.phase05_config.log_metrics:
                        _estr += (
                            "  dt_mean=["
                            + ", ".join(f"{x:.5g}" for x in _adc[:10])
                            + ("...]" if len(_adc) > 10 else "]")
                        )
                    if curve:
                        cs = "[" + ", ".join(f"{x:.4f}" for x in curve) + "]"
                        print(
                            f"    [batch {batch_idx + 1}] loss={li:.4f}  T={cs}  steps={ts}"
                            f"{_estr}",
                            flush=True,
                        )
                    else:
                        print(
                            f"    [batch {batch_idx + 1}] loss={li:.4f}{_estr}",
                            flush=True,
                        )

            mean_loss = loss_sum / max(batch_idx + 1, 1)
            if final_tension_values:
                mean_final_step_tension = float(statistics.mean(final_tension_values))
            max_batch_loss_epoch = max_batch_loss if batch_idx >= 0 else float("nan")
            n_windows = (batch_idx + 1) * traj_batch_size

        else:  # CE mode
            for batch_idx, (contexts, targets, _tg_unused) in tqdm(
                enumerate(batch_iter),
                total=n_est,
                desc=f"epoch {epoch + 1}/{num_epochs}",
            ):
                for context, target_id in zip(contexts, targets):
                    logits = model.forward_training_window(context)
                    prev_id = context[-1]
                    logits = logits + BIGRAM_TRAIN_WEIGHT * torch.matmul(
                        model.embedder.weight, model.embedder.weight[prev_id]
                    )
                    lo = logits.clone()
                    lo[prev_id] -= 2.0
                    for t in context[-3:]:
                        lo[t] -= 1.0
                    lo = lo + TRAIN_LOGIT_NOISE * torch.randn_like(lo)
                    probs_floor = F.softmax(lo, dim=-1)
                    ent_s = -(probs_floor * torch.log(probs_floor + 1e-9)).sum()
                    if float(ent_s.detach()) < ENTROPY_FLOOR:
                        lo = lo + torch.randn_like(lo) * ENTROPY_FLOOR_NOISE
                        probs_for_entropy = F.softmax(lo, dim=-1)
                    else:
                        probs_for_entropy = probs_floor
                    target = torch.tensor([target_id], device=device, dtype=torch.long)
                    loss_ce = F.cross_entropy(
                        lo.unsqueeze(0), target, label_smoothing=LABEL_SMOOTHING
                    )
                    entropy = -(probs_for_entropy * torch.log(probs_for_entropy + 1e-8)).sum()
                    step_loss = loss_ce - ENTROPY_WEIGHT * entropy
                    optimizer.zero_grad(set_to_none=True)
                    step_loss.backward()
                    if args.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    loss_sum += float(step_loss.detach())
                    global_step += 1

                    if substrate is not None:
                        try:
                            substrate.on_batch(model)
                        except Exception:
                            pass
                    if model._goat_mgr is not None:
                        try:
                            model._goat_mgr.tick([context])
                        except Exception:
                            pass
                    if args.save_every > 0 and global_step % args.save_every == 0:
                        _save_checkpoint(model, optimizer, global_step, epoch, args, ckpt_dir)

            mean_loss = loss_sum / max((batch_idx + 1) * traj_batch_size, 1)
            max_batch_loss_epoch = float("nan")
            n_windows = (batch_idx + 1) * traj_batch_size

        ep_sec = time.perf_counter() - t_ep0

        # Bug 4: real train CE from batch readout_window logits (not val CE).
        train_ce: float = train_ce_sum / max(train_ce_count, 1) if train_ce_count > 0 else float("nan")

        vce: float | None = None
        val_msg = ""
        if val_dataset:
            vce = mean_cross_entropy_eval(model, val_dataset)
            val_msg = f"  |  val CE={vce:.4f}"
            if val_metrics_unreliable:
                val_msg += "  [val unreliable]"

        # Bug 3: compute val_traj_contrast (on held-out val data) and
        # train_traj_contrast (on last training batch, no extra eval pass).
        train_traj_contrast: float | None = None
        val_traj_contrast: float | None = None
        lr_now = optimizer.param_groups[0]["lr"]

        if args.loss_mode == "trajectory":
            # val_traj_contrast — on the held-out val split.
            if val_dataset:
                val_traj_contrast = mean_trajectory_contrastive_eval(
                    model, val_dataset, batch_size=traj_batch_size
                )
            # train_traj_contrast — on the last batch seen this epoch (cheap, no extra forward pass
            # beyond what's already in memory).
            if _last_batch_contexts:
                _tc_data = list(zip(_last_batch_contexts, _last_batch_targets))
                with torch.no_grad():
                    _tc_contexts = [c for c, _ in _tc_data]
                    _tc_targets = [t for _, t in _tc_data]
                    _ts_cpu = _last_batch_target_states
                    _ts_eval = (
                        _ts_cpu.to(device=device, dtype=model.embedder.weight.dtype)
                        if _ts_cpu is not None
                        else None
                    )
                    _tl, _ = model.trajectory_contrastive_loss_and_logits(
                        _tc_contexts,
                        _tc_targets,
                        update_repulsion_memory=False,
                        target_states=_ts_eval,
                    )
                    train_traj_contrast = float(_tl.detach())

            vtc_s = f"{val_traj_contrast:.6f}" if val_traj_contrast is not None else "n/a"
            mft_s = (
                f"  mean_final_T={mean_final_step_tension:.4f}"
                if math.isfinite(mean_final_step_tension)
                else ""
            )
            mb_s = (
                f"  max_batch_loss={max_batch_loss_epoch:.4f}"
                if math.isfinite(max_batch_loss_epoch)
                else ""
            )
            print(
                f"  epoch {epoch + 1} done  |  {ep_sec:.1f}s  |  lr={lr_now:g}  |  "
                f"mean loss={mean_loss:.4f}  |  val_traj={vtc_s}"
                f"  train CE={train_ce:.4f}{mft_s}{mb_s}{val_msg}",
                flush=True,
            )
        else:
            print(
                f"  epoch {epoch + 1} done  |  {ep_sec:.1f}s  |  lr={lr_now:g}  |  "
                f"mean loss={mean_loss:.4f}  |  train CE={train_ce:.4f}{val_msg}",
                flush=True,
            )
        if args.debug and batch_idx >= 0:
            _bps = (batch_idx + 1) / max(ep_sec, 1e-9)
            _training_debug(
                True,
                f"epoch {epoch + 1} summary: batches={batch_idx + 1}  "
                f"window_updates≈{n_windows}  throughput={_bps:.2f} batch/s",
            )

        # Bug 5: log TSCore evolve_count and last_ts_tension.
        ep_ts_tension: float = 0.0
        ep_evolves: int = 0
        if substrate is not None:
            ep_evolves = substrate.evolve_count - substrate_evolve_start
            ep_ts_tension = substrate.last_ts_tension
            print(
                f"  [tscore] evolves={ep_evolves}  last_ts_tension={ep_ts_tension:.4f}"
                f"  active_batches={substrate_active_count}  idle_batches={substrate_idle_count}",
                flush=True,
            )
            # Bug 6: warn once per epoch when TSCore was idle the entire epoch.
            if substrate_active_count == 0 and substrate_idle_count > 0:
                _max_tension = max(final_tension_values) if final_tension_values else 0.0
                print(
                    f"  [tscore] WARNING: inactive all epoch "
                    f"(max lang_tension={_max_tension:.4f} < threshold={substrate.high_tension_threshold:.3f})",
                    flush=True,
                )

        # Phase 12: CSV metrics logging (updated headers for Bug 4/5).
        if args.epoch_metrics_csv is not None:
            mpath = Path(args.epoch_metrics_csv)
            mpath.parent.mkdir(parents=True, exist_ok=True)
            new_file = not mpath.exists() or mpath.stat().st_size == 0
            row = [
                epoch + 1,
                args.loss_mode,
                f"{mean_loss:.6f}",
                f"{train_ce:.6f}",
                f"{vce:.6f}" if vce is not None else "",
                f"{train_traj_contrast:.6f}" if train_traj_contrast is not None else "",
                f"{val_traj_contrast:.6f}" if val_traj_contrast is not None else "",
                f"{mean_final_step_tension:.6f}" if math.isfinite(mean_final_step_tension) else "",
                f"{max_batch_loss_epoch:.6f}" if math.isfinite(max_batch_loss_epoch) else "",
                f"{lr_now:.8f}",
                str(global_step),
                str(ep_evolves),
                f"{ep_ts_tension:.6f}",
            ]
            with mpath.open("a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow([
                        "epoch", "loss_mode", "mean_loss", "train_ce", "val_ce",
                        "train_traj_contrast", "val_traj_contrast",
                        "mean_final_step_tension", "max_batch_loss", "lr", "global_step",
                        "tscore_evolves", "tscore_last_tension",
                    ])
                w.writerow(row)

        if args.metrics_fast_csv is not None:
            mfpath = Path(args.metrics_fast_csv)
            mfpath.parent.mkdir(parents=True, exist_ok=True)
            mf_new = not mfpath.exists() or mfpath.stat().st_size == 0
            if energy_fast_n > 0:
                inv = 1.0 / float(energy_fast_n)
                ep_mean_e = energy_fast_sum_mean * inv
                ep_mean_var = energy_fast_sum_var * inv
                ep_mean_min = energy_fast_sum_min * inv
                ep_mean_max = energy_fast_sum_max * inv
            else:
                ep_mean_e = ep_mean_var = ep_mean_min = ep_mean_max = float("nan")
            with mfpath.open("a", newline="", encoding="utf-8") as mff:
                mw = csv.writer(mff)
                if mf_new:
                    mw.writerow(
                        [
                            "epoch",
                            "mean_energy",
                            "energy_variance",
                            "energy_min",
                            "energy_max",
                            "global_step",
                        ]
                    )
                mw.writerow(
                    [
                        epoch + 1,
                        f"{ep_mean_e:.8f}" if math.isfinite(ep_mean_e) else "",
                        f"{ep_mean_var:.8f}" if math.isfinite(ep_mean_var) else "",
                        f"{ep_mean_min:.8f}" if math.isfinite(ep_mean_min) else "",
                        f"{ep_mean_max:.8f}" if math.isfinite(ep_mean_max) else "",
                        str(global_step),
                    ]
                )

        if args.attractor_steps_metrics_csv is not None and attractor_steps_hist:
            apath = Path(args.attractor_steps_metrics_csv)
            apath.parent.mkdir(parents=True, exist_ok=True)
            anew = not apath.exists() or apath.stat().st_size == 0
            mxw = int(model.max_window_steps)
            mean_s = statistics.mean(attractor_steps_hist)
            max_s = max(attractor_steps_hist)
            min_s = min(attractor_steps_hist)
            conv_fail = sum(
                1
                for s, c in zip(attractor_steps_hist, attractor_conv_hist)
                if s >= mxw and not c
            )
            early = sum(1 for c in attractor_conv_hist if c)
            with apath.open("a", newline="", encoding="utf-8") as af:
                aw = csv.writer(af)
                if anew:
                    aw.writerow([
                        "epoch",
                        "mean_steps",
                        "max_steps",
                        "min_steps",
                        "convergence_failures",
                        "early_convergence_batches",
                        "batches",
                    ])
                aw.writerow([
                    epoch + 1,
                    f"{mean_s:.4f}",
                    str(max_s),
                    str(min_s),
                    str(conv_fail),
                    str(early),
                    str(len(attractor_steps_hist)),
                ])

        if lr_scheduler is not None:
            lr_scheduler.step()

        last_mean_loss = mean_loss
        last_train_ce = train_ce
        last_val_ce = vce
        last_train_traj_contrast = train_traj_contrast if args.loss_mode == "trajectory" else None
        last_val_traj_contrast = val_traj_contrast if args.loss_mode == "trajectory" else None
        last_n_windows = n_windows
        last_epoch_sec = ep_sec
        last_epoch_num = epoch + 1

    if phase05_csv_fp is not None:
        try:
            phase05_csv_fp.flush()
            phase05_csv_fp.close()
        except OSError:
            pass

    # Clean up temp files (line-based train list, synthetic corpus fallback)
    if _tmp_train_path is not None:
        try:
            os.unlink(_tmp_train_path)
        except OSError:
            pass
    if _synthetic_corpus_cleanup is not None:
        try:
            os.unlink(_synthetic_corpus_cleanup)
        except OSError:
            pass

    train_sec_total = time.perf_counter() - t_train0
    print(f"Pre-training done in {train_sec_total:.1f}s total.", flush=True)
    if args.debug:
        _training_debug(
            True,
            f"finished global_step={global_step}  last_epoch={last_epoch_num}  "
            f"last_mean_loss={last_mean_loss:.4f}  last_train_ce={last_train_ce:.4f}",
        )

    # Phase 4: final checkpoint
    final_ckpt_path = _save_checkpoint(
        model, optimizer, global_step, last_epoch_num, args, ckpt_dir
    )

    if args.eval_results_json is not None:
        model.eval()
        vce_final: float | None = None
        vtc_final: float | None = None
        if val_dataset:
            vce_final = mean_cross_entropy_eval(model, val_dataset)
            if args.loss_mode == "trajectory":
                vtc_final = mean_trajectory_contrastive_eval(
                    model, val_dataset, batch_size=traj_batch_size
                )
        ppl_final = (
            math.exp(vce_final)
            if vce_final is not None and math.isfinite(vce_final)
            else float("nan")
        )

        def _json_num(x: float | None) -> float | None:
            if x is None:
                return None
            return x if math.isfinite(x) else None

        eval_payload = {
            "base": {
                "val_ce": _json_num(vce_final),
                "val_ppl": _json_num(ppl_final),
                "val_traj_contrast": _json_num(vtc_final),
                "val_windows": len(val_dataset),
                "val_metrics_reliable": not val_metrics_unreliable,
            },
            "checkpoint": str(final_ckpt_path),
            "config": {
                "dataset_source": args.dataset_source,
                "corpus_path": str(corpus_path),
                "stream_source_path": str(stream_source_path)
                if streaming_dataset
                else str(corpus_path),
                "tokenizer": args.tokenizer,
                "vocab_cap": args.vocab_cap,
                "window_size": window_size,
                "max_epochs": num_epochs,
                "val_fraction": args.val_fraction,
                "loss_mode": args.loss_mode,
                "use_goat_memory": bool(args.use_goat_memory),
                "use_substrate": bool(args.use_substrate),
                "seed": args.seed,
            },
        }
        args.eval_results_json.parent.mkdir(parents=True, exist_ok=True)
        args.eval_results_json.write_text(
            json.dumps(eval_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Wrote eval summary to {args.eval_results_json}", flush=True)

    # Phase 12: sample generations (decode via tokenizer)
    print("\nPrompt 1:", flush=True)
    gen_baseline_1 = model.generate(BASELINE_PROMPT_1)
    print(gen_baseline_1, flush=True)
    print("\nPrompt 2:", flush=True)
    gen_baseline_2 = model.generate(BASELINE_PROMPT_2)
    print(gen_baseline_2, flush=True)
    print("\n(Order sensitivity check:)", flush=True)
    gen_baseline_3 = model.generate(BASELINE_PROMPT_3)
    print(gen_baseline_3, flush=True)

    baseline_block = _format_phase0_baseline_block(
        corpus_path=corpus_path,
        seed=args.seed,
        val_fraction=args.val_fraction,
        effective_stream_val_fraction=stream_val_fraction_effective
        if streaming_dataset
        else None,
        epoch_copies=args.epoch_copies,
        num_epochs=num_epochs,
        window_size=window_size,
        num_dynamics_steps=args.num_dynamics_steps,
        loss_mode=args.loss_mode,
        token_aux_ce=args.token_aux_ce,
        last_epoch=last_epoch_num,
        last_mean_loss=last_mean_loss,
        last_train_ce=last_train_ce,
        last_val_ce=last_val_ce,
        last_train_traj_contrast=last_train_traj_contrast,
        last_val_traj_contrast=last_val_traj_contrast,
        last_n_windows=last_n_windows,
        last_epoch_sec=last_epoch_sec,
        train_sec_total=train_sec_total,
        gen1=gen_baseline_1,
        gen2=gen_baseline_2,
        gen3=gen_baseline_3,
    )
    print("\n" + baseline_block, flush=True)
    if args.baseline_out is not None:
        args.baseline_out.parent.mkdir(parents=True, exist_ok=True)
        args.baseline_out.write_text(baseline_block, encoding="utf-8")
        print(f"Wrote baseline snapshot to {args.baseline_out}", flush=True)

    print("\nDebug attractor tracking:", flush=True)
    model.generate(
        "the system stays stable because the reason is clear",
        max_tokens=12,
        debug_track=True,
    )
    print("\nTrajectory sensitivity:", flush=True)
    compare_prompts(model, "mind reason cause effect system", "effect cause reason mind system")
    compare_prompts(
        model,
        "the quick brown fox jumps over the lazy dog",
        "the lazy dog jumps over the quick brown fox",
    )


if __name__ == "__main__":
    main()
