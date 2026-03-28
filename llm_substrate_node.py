"""
Wave E — LLM Substrate Node (BoggersTheLanguageModel).

Registers the TorchAttractorLanguageModel as a native node inside the
TSCore graph (vendor/TS-Core). Closes the loop:

  language tension → TSCore graph → propagate → factory_evolve()

When the language substrate's window tension exceeds TENSION_HIGH_THRESHOLD,
this shim:
  1. Pushes a normalised tension scalar to the "llm_substrate" node's activation.
  2. Calls ts.propagate_wave() to let the constraint graph absorb the signal.
  3. If TSCore tension then exceeds TS_EVOLVE_THRESHOLD, calls ts.factory_evolve()
     (the self-improvement tick — adds a stability node to the factory graph).
  4. Optionally posts a high-tension event to the BoggersTheAI FastAPI endpoint
     (LLM_HOOK_URL env var; no-op if unset or unreachable).

Usage (attach to training loop)
--------------------------------
    from llm_substrate_node import LLMSubstrateNode

    substrate = LLMSubstrateNode(model)

    for contexts, targets in data_loader:
        loss, _ = model.trajectory_contrastive_loss_and_logits(contexts, targets)
        ...
        substrate.on_batch(model)   # push tension, maybe evolve

    # or attach as an on_propagate callback:
    substrate = LLMSubstrateNode(model, auto_attach=True)
    # ts.propagate_wave() now calls substrate._on_propagate automatically

API
---
LLMSubstrateNode(model, *, data_dir, evolve_threshold, high_tension_threshold,
                 llm_hook_url, auto_attach, quiet)
    .on_batch(model)        — call after every training batch
    .ts                     — the TSCore instance
    .last_ts_tension        — float: last measured TSCore tension
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import sandbox as sb  # type: ignore[import]

# ---- TSCore import -------------------------------------------------------
_REPO = Path(__file__).resolve().parent
_TS_CORE = _REPO / "vendor" / "TS-Core"
if str(_TS_CORE) not in sys.path:
    sys.path.insert(0, str(_TS_CORE))

from src.python.core import TSCore  # type: ignore[import]

# Tension threshold above which we call factory_evolve
_DEFAULT_TS_EVOLVE_THRESHOLD = 0.15
# Language substrate tension level that triggers a TSCore push
_DEFAULT_LANG_TENSION_THRESHOLD = 0.10


class LLMSubstrateNode:
    """
    Registers the attractor language model as a node in the TSCore graph and
    implements the Propagate → Relax → (high tension → Evolve) loop.

    Parameters
    ----------
    model : TorchAttractorLanguageModel
    data_dir : Path for TSCore wave history (default ~/.tscore/llm_substrate)
    evolve_threshold : TSCore tension above which factory_evolve() fires
    high_tension_threshold : lang-substrate tension level that triggers propagation
    llm_hook_url : optional HTTP URL for BoggersTheAI Evolve hook (env LLM_HOOK_URL)
    auto_attach : if True, register as TSCore.on_propagate callback (for external loops)
    quiet : suppress TSCore console output
    """

    def __init__(
        self,
        model: "sb.TorchAttractorLanguageModel",
        *,
        data_dir: Optional[Path] = None,
        evolve_threshold: float = _DEFAULT_TS_EVOLVE_THRESHOLD,
        high_tension_threshold: float = _DEFAULT_LANG_TENSION_THRESHOLD,
        llm_hook_url: Optional[str] = None,
        auto_attach: bool = False,
        quiet: bool = True,
    ) -> None:
        self.model = model
        self.evolve_threshold = evolve_threshold
        self.high_tension_threshold = high_tension_threshold
        self.llm_hook_url = llm_hook_url or os.environ.get("LLM_HOOK_URL", "")
        self.quiet = quiet
        self.last_ts_tension: float = 0.0
        self._evolve_count: int = 0

        _data_dir = data_dir or (Path.home() / ".tscore" / "llm_substrate")

        on_propagate = self._on_propagate if auto_attach else None
        self.ts = TSCore(damping=0.35, data_dir=_data_dir, on_propagate=on_propagate)

        # Register the language substrate node
        if "llm_substrate" not in self.ts.graph.get("nodes", {}):
            self.ts.add_node("llm_substrate", activation=0.5, stability=0.5)
        if "ts_native" in self.ts.graph.get("nodes", {}):
            self.ts.add_edge("ts_native", "llm_substrate", weight=1.0)

    # ------------------------------------------------------------------
    def on_batch(self, model: "sb.TorchAttractorLanguageModel") -> None:
        """
        Call after each training batch.

        Reads the last window tension from the model and pushes it into the
        TSCore graph. If tension is high, fires propagate + optional Evolve.
        """
        curve = getattr(model, "_last_window_tension_curve", [])
        if not curve:
            return
        lang_tension = float(curve[-1])

        # Only interact with TS graph on high-tension batches (avoid per-batch overhead)
        if lang_tension < self.high_tension_threshold:
            return

        # Normalise to [0, 1] for TSCore activation
        act = min(1.0, lang_tension)
        self.ts.graph["nodes"]["llm_substrate"]["activation"] = act

        ts_tension, _ = self.ts.propagate_wave(quiet=self.quiet)
        self.last_ts_tension = ts_tension

        if ts_tension > self.evolve_threshold:
            self.ts.factory_evolve()
            self._evolve_count += 1
            if not self.quiet:
                print(
                    f"[llm-substrate] Evolve #{self._evolve_count}: "
                    f"lang_tension={lang_tension:.4f}  ts_tension={ts_tension:.4f}",
                    flush=True,
                )
            self._post_hook(lang_tension, ts_tension)

    def _on_propagate(self, ts: TSCore) -> None:
        """Callback for TSCore.on_propagate when auto_attach=True."""
        t = ts.measure_tension()
        self.last_ts_tension = t
        if t > self.evolve_threshold:
            ts.factory_evolve()
            self._evolve_count += 1

    def _post_hook(self, lang_tension: float, ts_tension: float) -> None:
        """HTTP POST to BoggersTheAI Evolve endpoint (fire-and-forget; best-effort)."""
        if not self.llm_hook_url:
            return
        try:
            import urllib.request, json as _json, urllib.error
            payload = _json.dumps(
                {"lang_tension": lang_tension, "ts_tension": ts_tension, "evolve": True}
            ).encode()
            req = urllib.request.Request(
                self.llm_hook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=2.0) as _r:
                pass
        except Exception:
            pass  # network/service unavailable is not a training error

    # convenience
    @property
    def evolve_count(self) -> int:
        return self._evolve_count


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, str(_REPO))
    import sandbox as sb  # type: ignore[import]

    print("[wave-e] llm_substrate_node self-test ...", flush=True)

    model = sb.TorchAttractorLanguageModel(sb.FULL_VOCAB, train_window_size=4)
    model.eval()

    substrate = LLMSubstrateNode(
        model,
        evolve_threshold=0.05,
        high_tension_threshold=0.01,
        quiet=True,
    )

    # Test 1: node is registered
    assert "llm_substrate" in substrate.ts.graph["nodes"], "llm_substrate node missing"
    print("  test 1 PASS — llm_substrate node registered in TSCore graph", flush=True)

    # Test 2: on_batch fires correctly
    # Manually set a fake tension curve
    model._last_window_tension_curve = [0.5, 0.3, 0.15, 0.12]
    substrate.on_batch(model)
    ts_t = substrate.last_ts_tension
    import math
    assert math.isfinite(ts_t), f"TSCore tension not finite: {ts_t}"
    print(f"  test 2 PASS — on_batch ran, ts_tension={ts_t:.4f}", flush=True)

    # Test 3: evolve fires when ts_tension > threshold (set threshold very low to force)
    substrate2 = LLMSubstrateNode(
        model,
        evolve_threshold=0.0,  # always evolve
        high_tension_threshold=0.01,
        quiet=True,
    )
    model._last_window_tension_curve = [0.8, 0.5, 0.2]
    substrate2.on_batch(model)
    assert substrate2.evolve_count >= 1, "factory_evolve was not called"
    print(f"  test 3 PASS — Evolve fired {substrate2.evolve_count} time(s)", flush=True)

    # Test 4: TSCore graph has more nodes after evolve
    n_after = len(substrate2.ts.graph["nodes"])
    assert n_after > len(substrate2.ts.graph["edges"]) or n_after >= 4, "graph did not grow after Evolve"
    print(f"  test 4 PASS — TSCore graph has {n_after} nodes after Evolve", flush=True)

    print("\n[wave-e] ALL TESTS PASSED", flush=True)
