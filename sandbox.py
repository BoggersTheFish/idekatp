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
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.state_dim = state_dim
        self.train_window_size = train_window_size
        self.max_window_steps = max_window_steps
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
        self.dynamics = SimpleAttractorDynamics(state_dim)
        self.embedder = nn.Embedding(self.vocab_size, state_dim)
        self.norm = nn.LayerNorm(state_dim, elementwise_affine=False)
        self.readout = nn.Linear(self.state_dim, self.vocab_size, bias=False)
        # Training: readout from full converged window tensor flattened (context interaction path).
        self.readout_window = nn.Linear(
            train_window_size * state_dim, self.vocab_size, bias=False
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
        T = de + self.tension_lambda * div + self.tension_mu * H
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
            t_prev = self._last_tension_val
            noise_mul = (1.0 + F.softplus(self.tension_noise_gain) * min(t_prev, 3.0)).detach()
            fast_state = self.dynamics(fast_state, signal, noise_scale_mul=noise_mul)
            logits_t = self._logits_for_tension(fast_state, slow_state)
            T, _de, _div, _H = self.compute_tension(
                fast_state, slow_state, logits_t, prev_energy
            )
            prev_energy = self._state_energy(fast_state)
            t_item = T.detach().item()
            self._last_tension_val = t_item
            if t_item > brk:
                fast_state = fast_state + 0.02 * torch.randn_like(fast_state)
                nrm = torch.linalg.vector_norm(fast_state)
                fast_state = fast_state / (nrm + 1e-8)
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

    def compute_window_tension(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: (B, W, D) normalized states

        Returns:
            scalar tension per batch (B,)
        """
        assert state.dim() == 3 and state.size(1) >= 2
        lam = self.tension_lambda.to(device=state.device, dtype=state.dtype)
        delta = state[:, 1:] - state[:, :-1]
        energy = delta.pow(2).mean(dim=(1, 2))
        cos = F.cosine_similarity(state[:, 1:], state[:, :-1], dim=-1)
        misalign = (1 - cos).mean(dim=1)
        T = energy + lam * misalign
        if WINDOW_TENSION_USE_ENTROPY:
            mu = self.tension_mu.to(device=state.device, dtype=state.dtype)
            flat = state.reshape(state.size(0), -1)
            logits = self.readout_window(flat)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
            T = T + mu * entropy
        return T

    def _single_window_step(
        self,
        S: torch.Tensor,
        context_ids: list[list[int]] | None = None,
        context_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One coupling + relaxation step; S is (B, W, D).

        Uses the unified dynamics.step() interface so both SimpleAttractorDynamics
        and VectorizedWindowDynamics work without attribute access.

        If GOAT memory is active and context_tensor is provided (B, W) long), an
        activation-bonus signal is gathered on GPU (no per-step Python over B×W).
        """
        B, W, D = S.shape
        assert W == self.train_window_size
        pos_g = F.softplus(self.position_gamma_raw) + 1e-6
        isc = F.softplus(self.interaction_scale_raw)
        idt = F.softplus(self.interaction_dt_raw)
        delta = positional_coupling_delta(S, pos_g, self.position_asym)
        S = S + idt * isc * delta

        # Build GOAT activation-bonus signal (Bug 1 fix).
        # bonuses[b, t] is a scalar; we broadcast it across the D dimension.
        if self._goat_mgr is not None and context_tensor is not None:
            V = len(self._goat_mgr.nodes)
            valid = (context_tensor >= 0) & (context_tensor < V)
            safe = context_tensor.clamp(min=0, max=max(V - 1, 0))
            bv = self._goat_mgr.bonus_tensor(S.device, S.dtype)
            bonuses_2d = bv[safe] * valid.to(dtype=S.dtype)
            signal = bonuses_2d.unsqueeze(-1).expand(B, W, D)
        else:
            signal = torch.zeros(B, W, D, device=S.device, dtype=S.dtype)

        # Unified interface: both SimpleAttractorDynamics and
        # VectorizedWindowDynamics implement step(S, signal) → S (Bug 2 fix).
        S = self.dynamics.step(S, signal)
        return S

    def run_window_dynamics(
        self,
        S: torch.Tensor,
        collect_metrics: bool = False,
        record_tension_log: bool = True,
        context_ids: list[list[int]] | None = None,
    ) -> tuple[torch.Tensor, list[dict] | None]:
        """
        Tension-adaptive evolution: (W, D) or (B, W, D). Gradients flow through all steps.
        If record_tension_log, fills _last_window_tension_curve with mean(T) after each step
        (use False on teacher-only passes so the student curve is preserved).

        context_ids: list of per-batch token ID windows (B × W) used to compute
        GOAT activation bonuses inside each dynamics step.
        """
        single = S.dim() == 2
        if single:
            S = S.unsqueeze(0)
            # Wrap single window in a batch list so _single_window_step can index it.
            if context_ids is not None and len(context_ids) > 0 and not isinstance(context_ids[0], list):
                context_ids = [context_ids]  # type: ignore[list-item]
        B, W, D = S.shape
        assert W == self.train_window_size
        step_logs: list[dict] | None = [] if collect_metrics else None
        tension_curve_tensors: list[torch.Tensor] = []
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
        for step in range(self.max_window_steps):
            S0 = S.detach().clone() if collect_metrics else None
            S = self._single_window_step(
                S, context_ids=context_ids, context_tensor=context_tensor
            )
            T = self.compute_window_tension(S)
            t_mean = T.mean()
            self._last_window_tension_mean = t_mean.detach()
            self._last_adaptive_window_steps = step + 1
            if record_tension_log:
                tension_curve_tensors.append(t_mean.detach())
            if collect_metrics and S0 is not None and step_logs is not None:
                with torch.no_grad():
                    diff = S - S0
                    nd = float(torch.linalg.vector_norm(diff).item())
                    tok_var = float(
                        S.var(dim=(0, 1), unbiased=False).mean().item()
                    )
                    cos = float(
                        F.cosine_similarity(S.flatten(), S0.flatten(), dim=0).item()
                    )
                    mn = float(torch.linalg.vector_norm(S, dim=-1).mean().item())
                    step_logs.append(
                        {
                            "norm_delta": nd,
                            "token_var_mean": tok_var,
                            "cosine_to_prev": cos,
                            "mean_row_norm": mn,
                        }
                    )
            is_low = t_mean < 0.08
            consecutive_low_t_steps_t = torch.where(
                is_low,
                consecutive_low_t_steps_t + 1,
                torch.zeros((), device=S.device, dtype=torch.long),
            )
            need_jitter = is_low & (consecutive_low_t_steps_t >= 4)
            # Extra stochastic break for stuck low-tension attractors (GOAT DORMANT→ACTIVE jitter)
            S = S + torch.randn_like(S) * (need_jitter.to(dtype=S.dtype) * 0.015)
            consecutive_low_t_steps_t = torch.where(need_jitter, zero_long, consecutive_low_t_steps_t)
            # GOAT node updates are Python-side; one scalar sync only when GOAT is on and jitter fires.
            if (
                self._goat_mgr is not None
                and context_ids is not None
                and bool(need_jitter.item())
            ):
                self._goat_transition_context_ids = context_ids
                self.goat_memory.apply_transition(
                    S[:, -1, :].mean(dim=0), "DORMANT", "ACTIVE"
                )
            high = (T > thigh).any()
            S_hi = S + 0.01 * torch.randn_like(S)
            S_hi = S_hi / (torch.linalg.vector_norm(S_hi, dim=-1, keepdim=True) + 1e-8)
            S = torch.where(high.view(1, 1, 1), S_hi, S)
        if record_tension_log:
            self._last_window_tension_curve = [
                float(x) for x in tension_curve_tensors
            ]
        if single:
            S = S.squeeze(0)
        return S, step_logs

    def trajectory_contrastive_loss(
        self, state_a: torch.Tensor, state_b: torch.Tensor
    ) -> torch.Tensor:
        """
        state_a: evolved(context); state_b: evolved(shifted_context + target).
        Same shape (B, W, D).
        """
        a = state_a.reshape(state_a.size(0), -1)
        b = state_b.reshape(state_b.size(0), -1)
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        pos = (a * b).sum(dim=-1)
        b_neg = b[torch.randperm(b.size(0), device=b.device)]
        neg = (a * b_neg).sum(dim=-1)
        return F.relu(0.2 - pos + neg).mean()

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

    def forward_training_window(
        self, context_ids: list[int], collect_dynamics_metrics: bool = False
    ) -> torch.Tensor:
        """
        Embed window tokens to (W, D), run multi-step interacting dynamics, read out vocab logits.
        Training, validation, and generation all use this path.
        """
        assert len(context_ids) == self.train_window_size
        S = self.embed_window(context_ids)
        S, dyn_logs = self.run_window_dynamics(
            S, collect_metrics=collect_dynamics_metrics, context_ids=context_ids
        )
        self._last_dynamics_logs = dyn_logs
        logits = self.readout_window(S.reshape(-1))
        logits = logits / self.effective_temperature()
        return torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=-1e4)

    def shifted_next_window(self, context_ids: list[int], target_id: int) -> list[int]:
        """One-step shift: [x2, …, xW, next_token] — teacher window for trajectory consistency."""
        assert len(context_ids) == self.train_window_size
        return context_ids[1:] + [target_id]

    def trajectory_contrastive_loss_and_logits(
        self, contexts: list[list[int]], targets: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Batched trajectory contrastive loss + readout logits from pred state.
        Teacher states are detached (no grad through shifted window).
        """
        B = len(contexts)
        assert B == len(targets) and B >= 1
        S_pred = torch.stack([self.embed_window(c) for c in contexts], dim=0)
        S_pred, _ = self.run_window_dynamics(
            S_pred, collect_metrics=False, record_tension_log=True, context_ids=contexts
        )
        self._last_traj_student_tension_mean = self.compute_window_tension(S_pred).mean().detach()
        with torch.no_grad():
            S_tgt = torch.stack(
                [
                    self.embed_window(self.shifted_next_window(c, t))
                    for c, t in zip(contexts, targets, strict=True)
                ],
                dim=0,
            )
            S_tgt, _ = self.run_window_dynamics(
                S_tgt, collect_metrics=False, record_tension_log=False
            )
        loss_traj = self.trajectory_contrastive_loss(S_pred, S_tgt)
        T = self.compute_window_tension(S_pred).mean()
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
        # Repulsion term to push away from previous states and break basin collapse
        loss_traj = loss_traj + entropy_floor - repulsion
        with torch.no_grad():
            self._repulsion_prev_states.append(fast_state.detach().clone())
            if len(self._repulsion_prev_states) > 8:
                self._repulsion_prev_states.pop(0)
        logits = self.readout_window(S_pred.reshape(B, -1)) / self.effective_temperature()
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=-1e4)
        # Store final-position single state for auxiliary readout training (Phase 5).
        self._last_pred_final_state = S_pred[:, -1, :]  # (B, D)
        return loss_traj, logits

    @staticmethod
    def summarize_dynamics_logs(logs: list[dict] | None) -> str:
        if not logs:
            return ""
        nds = [x["norm_delta"] for x in logs]
        tvs = [x["token_var_mean"] for x in logs]
        coss = [x["cosine_to_prev"] for x in logs]
        mns = [x["mean_row_norm"] for x in logs]
        return (
            f"steps={len(logs)}  "
            f"mean|Δ|={statistics.mean(nds):.4f}  "
            f"mean_var={statistics.mean(tvs):.6f}  "
            f"mean_cos(step)={statistics.mean(coss):.4f}  "
            f"mean||row||={statistics.mean(mns):.4f}"
        )

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
        with torch.inference_mode():
            S = self.embed_window(wid)
            S, _ = self.run_window_dynamics(S, collect_metrics=False)
        return S

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
    ):
        """Autoregressive generation: each step uses last-W context → dynamics → readout (same as training)."""
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
        with torch.inference_mode():
            base_gen_temp = self.generation_temperature
            if torch.is_tensor(base_gen_temp):
                base_gen_temp = float(base_gen_temp.detach())
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
                        print(
                            f"  [dyn t={ti}] tension_curve={curve_s}  "
                            f"steps={self._last_adaptive_window_steps}  {summ}"
                        )
                next_id = sample_next_token_id(
                    logits,
                    base_gen_temp,
                    GEN_TOP_K,
                    generated_ids,
                    GEN_REPEAT_LOGIT_PENALTY,
                    GEN_NO_REPEAT_LAST_EXTRA,
                )
                generated_ids.append(next_id)

        if debug_track:
            print("[generate] window path: last step dynamics summary (if logged above).")

        if was_training:
            self.train()
        if self.tokenizer is not None:
            return self.tokenizer.decode(generated_ids)
        return " ".join(str(i) for i in generated_ids)


class SimpleAttractorDynamics(nn.Module):
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
    ):
        super().__init__()
        self.dim = dim
        self.dt = dt
        self.cubic_scale = cubic_scale
        self.diffusion = nn.Parameter(make_diffusion_matrix(dim))
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.register_buffer("noise_scale", torch.tensor(float(noise_scale)))
        self.register_buffer("lambda_decay", torch.tensor(float(lambda_decay)))
        self.register_buffer("signal_scale", torch.tensor(float(signal_scale)))
        self.register_buffer("state_norm_eps", torch.tensor(float(state_norm_eps)))

    def forward(self, state, signal, noise_scale_mul=1.0):
        ns = self.noise_scale * noise_scale_mul
        return step_state(
            state,
            self.diffusion,
            signal,
            self.dt,
            self.cubic_scale,
            beta=self.beta,
            noise_scale=ns,
            lambda_decay=self.lambda_decay,
            signal_scale=self.signal_scale,
            state_norm_eps=self.state_norm_eps,
        )

    def step(self, S: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:
        """Unified batched step interface for the window training path (B, W, D).

        Noise is enabled when the module is in training mode, disabled in eval.
        """
        ns = (
            self.noise_scale.to(device=S.device, dtype=S.dtype)
            if self.training
            else torch.tensor(0.0, device=S.device, dtype=S.dtype)
        )
        return step_state_batch(
            S,
            self.diffusion,
            signal,
            self.dt,
            self.cubic_scale,
            beta=self.beta,
            noise_scale=ns,
            lambda_decay=self.lambda_decay,
            signal_scale=self.signal_scale,
            state_norm_eps=self.state_norm_eps,
            nonlinear_gain=WINDOW_NONLINEAR_GAIN,
        )


def make_diffusion_matrix(dim):
    torch.manual_seed(42)
    q = torch.linalg.qr(torch.randn(dim, dim))[0]
    u = torch.rand(dim)
    eigenvalues = -0.2 - (0.05 + 0.3 * u)
    return (q * eigenvalues) @ q.T


def compare_prompts(model: "TorchAttractorLanguageModel", prompt1: str, prompt2: str):
    """Encode two prompts via window dynamics and compare converged window states (flattened)."""
    model.eval()
    with torch.inference_mode():
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
    with torch.inference_mode():
        Sa = model.embed_window(wid_a)
        Sa, _ = model.run_window_dynamics(Sa)
        Sb = model.embed_window(wid_b)
        Sb, _ = model.run_window_dynamics(Sb)
    va = Sa.reshape(-1)
    vb = Sb.reshape(-1)
    dist = torch.linalg.vector_norm(va - vb).item()
    cos = F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0), dim=1).item()
    print(
        f"  different order (trailing window): L2={dist:.6f}  cosine={cos:.6f}",
        flush=True,
    )
    with torch.inference_mode():
        _, logs = model.run_window_dynamics(model.embed_window(wid_a), collect_metrics=True)
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
) -> float:
    """Validation CE: same logit shaping as training, without noise or entropy-floor branch."""
    if not dataset:
        return float("nan")
    was_training = model.training
    model.eval()
    total = 0.0
    for context, target_id in dataset:
        logits = model.forward_training_window(context)
        prev_id = context[-1]
        logits = logits + BIGRAM_TRAIN_WEIGHT * torch.matmul(
            model.embedder.weight, model.embedder.weight[prev_id]
        )
        logits[prev_id] -= 2.0
        for t in context[-3:]:
            logits[t] -= 1.0
        target = torch.tensor([target_id], device=logits.device, dtype=torch.long)
        loss_ce = F.cross_entropy(
            logits.unsqueeze(0), target, label_smoothing=LABEL_SMOOTHING
        )
        total += float(loss_ce)
    if was_training:
        model.train()
    return total / len(dataset)


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
        S_pred = torch.stack([model.embed_window(c) for c in contexts], dim=0)
        S_pred, _ = model.run_window_dynamics(
            S_pred, collect_metrics=False, record_tension_log=False
        )
        S_tgt = torch.stack(
            [
                model.embed_window(model.shifted_next_window(c, t))
                for c, t in zip(contexts, targets, strict=True)
            ],
            dim=0,
        )
        S_tgt, _ = model.run_window_dynamics(
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
    """Mean per-example CE − entropy bonus (matches single-window training shaping)."""
    device = logits.device
    acc = torch.zeros((), device=device, dtype=logits.dtype)
    B = logits.size(0)
    for bi in range(B):
        lo = logits[bi]
        prev_id = contexts[bi][-1]
        lo = lo + BIGRAM_TRAIN_WEIGHT * torch.matmul(
            model.embedder.weight, model.embedder.weight[prev_id]
        )
        lo = lo.clone()
        lo[prev_id] -= 2.0
        for t in contexts[bi][-3:]:
            lo[t] -= 1.0
        lo = lo + TRAIN_LOGIT_NOISE * torch.randn_like(lo)
        probs_floor = F.softmax(lo, dim=-1)
        ent_s = -(probs_floor * torch.log(probs_floor + 1e-9)).sum()
        ts = getattr(model, "_last_traj_student_tension_mean", None)
        base_entropy_floor = torch.as_tensor(float(ENTROPY_FLOOR), device=device, dtype=lo.dtype)
        if ts is not None:
            tref = ts.to(device=device, dtype=lo.dtype)
            if tref.dim() > 0:
                tref = tref.mean()
            entropy_floor = base_entropy_floor * (
                1.0 + 0.5 * (1.0 - torch.sigmoid(8.0 * (tref - 0.12)))
            )
        else:
            entropy_floor = base_entropy_floor
        if float(ent_s.detach()) < float(entropy_floor.detach()):
            lo = lo + torch.randn_like(lo) * ENTROPY_FLOOR_NOISE
            probs_for_entropy = F.softmax(lo, dim=-1)
        else:
            probs_for_entropy = probs_floor
        tgt = torch.tensor([targets[bi]], device=device, dtype=torch.long)
        loss_ce = F.cross_entropy(
            lo.unsqueeze(0), tgt, label_smoothing=LABEL_SMOOTHING
        )
        entropy = -(probs_for_entropy * torch.log(probs_for_entropy + 1e-8)).sum()
        acc = acc + (loss_ce - ENTROPY_WEIGHT * entropy)
    return acc / B


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
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "vocab_size": model.vocab_size,
            "tokenizer_mode": getattr(args, "tokenizer", "fallback"),
            "config": {
                "state_dim": model.state_dim,
                "train_window_size": model.train_window_size,
                "vocab_size": model.vocab_size,
            },
        },
        ckpt_path,
    )
    print(f"[ckpt] Saved: {ckpt_path}", flush=True)
    return ckpt_path


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
    parser.add_argument("--num-dynamics-steps", type=int, default=MAX_WINDOW_STEPS,
        help="Max outer steps per window (tension-adaptive).")
    parser.add_argument("--trajectory-batch-size", type=int, default=TRAJECTORY_BATCH_SIZE_DEFAULT,
        help="Batch size for trajectory contrastive training (need ≥2).")
    parser.add_argument("--quick-test", action="store_true",
        help="Window/context sanity checks and exit.")
    parser.add_argument("--loss-mode", choices=("trajectory", "ce"), default="trajectory")
    parser.add_argument("--token-aux-ce", type=float, default=TOKEN_AUX_CE_WEIGHT_DEFAULT,
        help="trajectory mode: aux readout_window CE weight.")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr-decay-every", type=int, default=0)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--epoch-metrics-csv", type=Path, default=None)
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
    parser.add_argument("--dynamics", choices=("simple", "vectorized"), default="simple",
        help="'simple': SimpleAttractorDynamics; 'vectorized': MultiHeadDynamics.")

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

    # ---- Phase 1: tokenizer ----
    tok = _build_tokenizer(mode=args.tokenizer, vocab_cap=args.vocab_cap)
    vocab_size = tok.n_vocab
    print(f"Vocab size: {vocab_size}  tokenizer={args.tokenizer}", flush=True)

    # ---- Build model ----
    model = TorchAttractorLanguageModel(
        vocab_size,
        train_window_size=window_size,
        max_window_steps=args.num_dynamics_steps,
    )
    model.tokenizer = tok

    # ---- Phase 4: build optimizer (before checkpoint load) ----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    start_epoch = 0
    global_step = 0

    if args.resume_checkpoint is not None and args.resume_checkpoint.is_file():
        ckpt = torch.load(args.resume_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        global_step = ckpt.get("step", 0)
        start_epoch = ckpt.get("epoch", 0)
        print(
            f"Resumed from {args.resume_checkpoint} (step={global_step}, epoch={start_epoch})",
            flush=True,
        )

    # ---- Phase 3: move to device ----
    model = model.to(device)

    if args.quick_test:
        run_quick_window_tests(model)
        return

    lr_scheduler = None
    if args.lr_decay_every > 0:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_decay_every, gamma=args.lr_gamma
        )

    # ---- Phase 9: dynamics swap ----
    if args.dynamics == "vectorized":
        try:
            from dynamics_vectorized import VectorizedWindowDynamics  # type: ignore[import]
            model.dynamics = VectorizedWindowDynamics(model.state_dim).to(device)
            print("[phase-9] VectorizedWindowDynamics active", flush=True)
        except Exception as _dyn_err:
            print(f"[phase-9] Warning: vectorized dynamics unavailable ({_dyn_err})", flush=True)

    if torch.cuda.is_available():
        try:
            # Step 2: torch.compile for speed (Phase 1 scaling)
            model.dynamics = torch.compile(model.dynamics, mode="reduce-overhead")
        except Exception as _comp_err:
            print(f"[step-2] Warning: torch.compile skipped ({_comp_err})", flush=True)

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

    print(
        f"Pre-training ({num_epochs} epochs, window={window_size}, "
        f"dynamics_steps={args.num_dynamics_steps}, loss={args.loss_mode}, "
        f"aux_ce={args.token_aux_ce}, readout_aux_alpha={args.readout_aux_alpha}, "
        f"batch={traj_batch_size}, lr={args.lr}, device={device})...",
        flush=True,
    )
    if args.loss_mode == "trajectory" and args.token_aux_ce <= 0 and args.readout_aux_alpha <= 0:
        print(
            "Warning: trajectory mode with token_aux_ce=0 and readout_aux_alpha=0 — "
            "readout head gets no gradients. Use --token-aux-ce or --readout-aux-alpha.",
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

    for epoch in range(start_epoch, start_epoch + num_epochs):
        t_ep0 = time.perf_counter()
        loss_sum = 0.0
        mean_final_step_tension = float("nan")
        max_batch_loss_epoch = float("nan")
        final_tension_values: list[float] = []
        max_batch_loss = -1.0
        batch_idx = -1

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
                    yield [c for c, _ in chunk], [t for _, t in chunk]

            batch_iter = _legacy_batch_iter()
            n_est = max(1, (len(legacy_dataset) + _bs - 1) // _bs)
            report_every = max(1, n_est // 10)

        print(f"  epoch {epoch + 1}/{start_epoch + num_epochs}", flush=True)

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

        if args.loss_mode == "trajectory":
            for batch_idx, (contexts, targets) in tqdm(
                enumerate(batch_iter),
                total=n_est,
                desc=f"epoch {epoch + 1}/{num_epochs}",
            ):
                if len(contexts) < 2:
                    contexts = contexts * 2
                    targets = targets * 2

                # Phase 3: targets tensor on device
                targets_tensor = torch.tensor(targets, device=device, dtype=torch.long)
                _last_batch_contexts = contexts
                _last_batch_targets = targets

                loss_traj, logits = model.trajectory_contrastive_loss_and_logits(
                    contexts, targets
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

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                global_step += 1

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
                    if curve:
                        cs = "[" + ", ".join(f"{x:.4f}" for x in curve) + "]"
                        print(
                            f"    [batch {batch_idx + 1}] loss={li:.4f}  T={cs}  steps={ts}",
                            flush=True,
                        )
                    else:
                        print(f"    [batch {batch_idx + 1}] loss={li:.4f}", flush=True)

            mean_loss = loss_sum / max(batch_idx + 1, 1)
            if final_tension_values:
                mean_final_step_tension = float(statistics.mean(final_tension_values))
            max_batch_loss_epoch = max_batch_loss if batch_idx >= 0 else float("nan")
            n_windows = (batch_idx + 1) * traj_batch_size

        else:  # CE mode
            for batch_idx, (contexts, targets) in tqdm(
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
                    _tl, _ = model.trajectory_contrastive_loss_and_logits(_tc_contexts, _tc_targets)
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
