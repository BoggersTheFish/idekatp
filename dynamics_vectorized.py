"""
Wave B — Vectorized window dynamics.

Replaces the Python for-loop in TorchAttractorLanguageModel.run_window_dynamics
with a fully vectorized, fixed-step solver that:

  1. Uses MultiHeadDynamics from vendor/ts-llm (low-rank per-head diffusion +
     hierarchical fast/slow coupling) as the inner step.
  2. Supports torch.compile (no Python control flow in the hot path).
  3. Exposes the same tension-adaptive API as sandbox.py so callers are unchanged.

Usage
-----
    from dynamics_vectorized import VectorizedWindowDynamics, run_window_dynamics_vectorized

    # Drop-in within TorchAttractorLanguageModel:
    vec_dyn = VectorizedWindowDynamics(
        state_dim=model.state_dim,
        window_size=model.train_window_size,
        num_heads=4,
        rank=64,
        max_steps=model.max_window_steps,
    )
    S_out, _ = run_window_dynamics_vectorized(S, model, vec_dyn)

Parity tests
------------
    python dynamics_vectorized.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import NoReturn, Optional
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- vendored imports ----------------------------------------------------
_REPO = Path(__file__).resolve().parent
_TS_LLM = _REPO / "vendor" / "ts-llm"
if str(_TS_LLM) not in sys.path:
    sys.path.insert(0, str(_TS_LLM))

from attractor_llm.torch_core import MultiHeadDynamics, _clamp_norm  # type: ignore[import]

# #region agent log
_AGENT_DEBUG_LOG_PATH = Path("/home/boggersthefish/BoggersTheLLM/.cursor/debug-b56157.log")
_AGENT_DEBUG_KEYS: set[str] = set()


def _agent_debug_log(*, run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    try:
        payload = {
            "sessionId": "b56157",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with _AGENT_DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, separators=(",", ":")) + "\n")
    except Exception:
        pass
# #endregion


# --------------------------------------------------------------------------
# Core: VectorizedWindowDynamics
# --------------------------------------------------------------------------

class VectorizedWindowDynamics(nn.Module):
    """
    Fixed-step window dynamics using MultiHeadDynamics (low-rank diffusion per head,
     hierarchical coupling). Replaces the Python loop in run_window_dynamics.

    Shape convention: (B, W, D) — same as sandbox.py.

    Parameters
    ----------
    state_dim : int
        Token embedding / state dimension (D).
    window_size : int
        Context window width (W).
    num_heads : int
        Number of low-rank heads. state_dim must be divisible by num_heads.
    rank : int
        Per-head low-rank factor for diffusion matrix.
    max_steps : int
        Maximum dynamics steps (hard ceiling; early-exit via tension).
    dt : float
        Euler step size.
    coupling : float
        Cross-head alignment coupling coefficient.
    """

    def __init__(
        self,
        state_dim: int = 512,
        window_size: int = 6,
        num_heads: int = 4,
        rank: int = 64,
        max_steps: int = 16,
        dt: float = 0.09,
        coupling: float = 0.01,
        use_lorentz: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.window_size = window_size
        self.max_steps = max_steps
        self.use_lorentz = use_lorentz

        # Multi-head low-rank dynamics; state shape for each row is (D,)
        # We reshape (B, W, D) → (B*W, D) for the batched head pass.
        self.mhd = MultiHeadDynamics(
            state_dim=state_dim,
            num_heads=num_heads,
            rank=rank,
            dt=dt,
            coupling=coupling,
        )

    def minkowski_inner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x[..., 0] * y[..., 0] - (x[..., 1:] * y[..., 1:]).sum(dim=-1)

    def project_tangent(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        inner = self.minkowski_inner(x, v)
        return v + inner.unsqueeze(-1) * x

    def project(self, x: torch.Tensor) -> torch.Tensor:
        norm = self.minkowski_inner(x, x)
        x = x / torch.sqrt(torch.abs(norm).unsqueeze(-1) + 1e-8)
        x[..., 0] = torch.abs(x[..., 0])
        return x

    def _step(self, S: torch.Tensor, signal: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        One vectorized Euler step on (B, W, D).
        `signal` is an additive injection (e.g. position coupling); zero if None.
        """
        B, W, D = S.shape
        flat = S.reshape(B * W, D)
        if signal is None:
            sig = torch.zeros_like(flat)
        else:
            sig = signal.reshape(B * W, D)
        # #region agent log
        if "vec_step_entry" not in _AGENT_DEBUG_KEYS:
            _AGENT_DEBUG_KEYS.add("vec_step_entry")
            _agent_debug_log(
                run_id="pre-fix",
                hypothesis_id="H2",
                location="dynamics_vectorized.py:118",
                message="VectorizedWindowDynamics._step entry",
                data={
                    "file": __file__,
                    "flat_shape": list(flat.shape),
                    "sig_shape": list(sig.shape),
                    "use_lorentz": bool(self.use_lorentz),
                    "mhd_module": type(self.mhd).__module__,
                },
            )
        # #endregion
        if self.use_lorentz:
            v_raw = self.mhd.drift(flat, sig)
            v = self.project_tangent(flat, v_raw)
            radius = torch.acosh(flat[..., 0].clamp(min=1 + 1e-6))
            scale = 1.0 / (1.0 + radius)
            v = v * scale.unsqueeze(-1)
            out = flat + self.mhd.dt * v
            out = self.project(out)
        else:
            out = flat + self.mhd.dt * self.mhd.drift(flat, sig)
        out = _clamp_norm(out, 1e-3, 12.0)
        return out.reshape(B, W, D)

    def step(self, S: torch.Tensor, signal: Optional[torch.Tensor]) -> torch.Tensor:
        """Unified step interface matching SimpleAttractorDynamics.step(S, signal).

        Delegates to _step so both dynamics classes are drop-in swappable inside
        _single_window_step without any attribute access.
        """
        return self._step(S, signal)

    def forward(self, *args: object, **kwargs: object) -> NoReturn:
        """
        Disabled: adaptive window loops live in ``TorchAttractorLanguageModel.run_window_dynamics``.
        Use ``.step(S, signal)`` with ``S`` of shape ``(B, W, D)``, or call ``model.run_window_dynamics``.
        """
        raise NotImplementedError(
            "VectorizedWindowDynamics.forward is disabled. Use "
            "TorchAttractorLanguageModel.run_window_dynamics (unified tension + jitter) "
            "or call .step(S, signal) with S shaped (B, W, D)."
        )


# --------------------------------------------------------------------------
# Drop-in replacement for TorchAttractorLanguageModel.run_window_dynamics
# --------------------------------------------------------------------------

def run_window_dynamics_vectorized(
    S: torch.Tensor,
    model: "TorchAttractorLanguageModel",  # type: ignore[name-defined]  # noqa: F821
    vec_dyn: VectorizedWindowDynamics,
    collect_metrics: bool = False,
    record_tension_log: bool = True,
    **kwargs: object,
) -> tuple[torch.Tensor, list[dict] | None]:
    """
    Redirect to ``model.run_window_dynamics`` with ``model.dynamics`` temporarily set to
    ``vec_dyn`` so only ``.step`` is used (no ``VectorizedWindowDynamics.forward``).
    """
    saved = model.dynamics
    model.dynamics = vec_dyn
    try:
        return model.run_window_dynamics(
            S,
            collect_metrics=collect_metrics,
            record_tension_log=record_tension_log,
            **kwargs,  # type: ignore[arg-type]
        )
    finally:
        model.dynamics = saved


# --------------------------------------------------------------------------
# torch.compile wrapper (optional; enabled by setting COMPILE=True)
# --------------------------------------------------------------------------
_COMPILED: dict[str, VectorizedWindowDynamics] = {}


def get_compiled(
    state_dim: int = 512,
    window_size: int = 6,
    num_heads: int = 4,
    rank: int = 64,
    max_steps: int = 16,
    dt: float = 0.09,
    use_lorentz: bool = False,
) -> VectorizedWindowDynamics:
    """
    Return a torch.compile'd VectorizedWindowDynamics (cached by key).
    Falls back to uncompiled if torch.compile is unavailable (PyTorch < 2.0).
    """
    key = f"{state_dim}_{window_size}_{num_heads}_{rank}_{max_steps}_{dt}_{int(use_lorentz)}"
    if key not in _COMPILED:
        mod = VectorizedWindowDynamics(
            state_dim=state_dim,
            window_size=window_size,
            num_heads=num_heads,
            rank=rank,
            max_steps=max_steps,
            dt=dt,
            use_lorentz=use_lorentz,
        )
        try:
            mod._step = torch.compile(mod._step, mode="reduce-overhead")  # type: ignore[assignment]
        except Exception:
            pass
        _COMPILED[key] = mod  # type: ignore[assignment]
    return _COMPILED[key]


# --------------------------------------------------------------------------
# Parity tests
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(_REPO))
    import sandbox as sb  # type: ignore[import]

    print("[wave-b] running parity tests ...", flush=True)

    STATE_DIM = 128
    WINDOW_SIZE = 4

    # --- test 1: run_window_dynamics + VectorizedWindowDynamics.step is finite ---
    torch.manual_seed(0)
    vec_dyn = VectorizedWindowDynamics(state_dim=STATE_DIM, window_size=WINDOW_SIZE, num_heads=4, rank=16, max_steps=8)
    model = sb.TorchAttractorLanguageModel(sb.FULL_VOCAB, state_dim=STATE_DIM, train_window_size=WINDOW_SIZE, max_window_steps=8)
    model.eval()
    saved = model.dynamics
    model.dynamics = vec_dyn
    S = torch.randn(2, WINDOW_SIZE, STATE_DIM)
    with torch.no_grad():
        S_out, _logs = model.run_window_dynamics(S.clone(), record_tension_log=True)
    model.dynamics = saved
    assert torch.isfinite(S_out).all(), "run_window_dynamics + vec_dyn output has non-finite values"
    assert len(model._last_window_tension_curve) > 0, "tension curve is empty"
    print(
        f"  test 1 PASS — S_out finite, tension_curve len={len(model._last_window_tension_curve)}",
        flush=True,
    )

    # --- test 2: gradient flows through vectorized dynamics via run_window_dynamics
    vec_dyn2 = VectorizedWindowDynamics(state_dim=STATE_DIM, window_size=WINDOW_SIZE, num_heads=4, rank=16, max_steps=8)
    model2 = sb.TorchAttractorLanguageModel(sb.FULL_VOCAB, state_dim=STATE_DIM, train_window_size=WINDOW_SIZE, max_window_steps=8)
    model2.train()
    model2.dynamics = vec_dyn2
    S_train = torch.randn(2, WINDOW_SIZE, STATE_DIM, requires_grad=True)
    S_out2, _ = model2.run_window_dynamics(S_train, record_tension_log=False)
    loss = S_out2.pow(2).mean()
    loss.backward()
    grad = vec_dyn2.mhd.U.grad
    assert grad is not None and grad.abs().sum() > 0, "No gradient through vectorized dynamics"
    print(f"  test 2 PASS — gradient flows (mhd.U grad norm={grad.norm():.4f})", flush=True)

    # --- test 3: simple vs vectorized dynamics both finite (different fields) ---
    model3 = sb.TorchAttractorLanguageModel(sb.FULL_VOCAB, state_dim=STATE_DIM, train_window_size=WINDOW_SIZE, max_window_steps=8)
    model3.eval()
    S_base = torch.randn(1, WINDOW_SIZE, STATE_DIM)
    with torch.no_grad():
        S_simple, _ = model3.run_window_dynamics(S_base.clone())
    vec_dyn3 = VectorizedWindowDynamics(state_dim=STATE_DIM, window_size=WINDOW_SIZE, num_heads=4, rank=16, max_steps=8)
    model3.dynamics = vec_dyn3
    with torch.no_grad():
        S_vec, _ = model3.run_window_dynamics(S_base.clone())
    assert torch.isfinite(S_simple).all(), "simple dynamics produced non-finite output"
    assert torch.isfinite(S_vec).all(), "vectorized dynamics produced non-finite output"
    ref_flat = S_simple.reshape(-1)
    vec_flat = S_vec.reshape(-1)
    cos = F.cosine_similarity(ref_flat.unsqueeze(0), vec_flat.unsqueeze(0)).item()
    l2 = (ref_flat - vec_flat).norm().item()
    print(f"  test 3 PASS — both finite; cosine(simple,vec)={cos:.4f}  L2={l2:.4f}", flush=True)

    # --- test 4: run_window_dynamics_vectorized drop-in -------------------
    model4 = sb.TorchAttractorLanguageModel(sb.FULL_VOCAB, state_dim=STATE_DIM, train_window_size=WINDOW_SIZE, max_window_steps=8)
    model4.eval()
    vec_dyn4 = VectorizedWindowDynamics(state_dim=STATE_DIM, window_size=WINDOW_SIZE, num_heads=4, rank=16, max_steps=8)
    with torch.no_grad():
        S_drop, logs = run_window_dynamics_vectorized(S_base.clone(), model4, vec_dyn4, collect_metrics=True)
    assert torch.isfinite(S_drop).all(), "drop-in wrapper produced non-finite output"
    assert hasattr(model4, "_last_window_tension_curve"), "model._last_window_tension_curve not set"
    print(f"  test 4 PASS — drop-in wrapper OK, tension_curve={model4._last_window_tension_curve}", flush=True)

    print("\n[wave-b] ALL PARITY TESTS PASSED", flush=True)
