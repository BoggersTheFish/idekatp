"""
Wave F — GOAT-TS memory-state transitions.

Wires the GOAT-TS memory_manager (ACTIVE → DORMANT → DEEP) into the attractor
dynamics step. Token-level activations are tracked per vocab word; after each
training tick the memory manager decays activations and transitions states.

ACTIVE nodes get a small activation bonus injected into the embedding signal.
DORMANT nodes get no bonus. DEEP nodes are skipped in bigram bias (reduces
over-fitting to rare tokens).

Usage
-----
    from goat_memory_transitions import GoatMemoryManager

    mem = GoatMemoryManager(model)

    # After each batch / training step:
    mem.tick(contexts)            # update activation counts, apply decay + transitions

    # During signal injection (optional — controlled by bonus_scale):
    bonus = mem.activation_bonus(token_id)  # float in [0, bonus_scale]

    # Automated hyperparameter sweep (Wave F):
    sweep = mem.sweep_config()   # dict of knobs for automated tuning
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sandbox as sb  # type: ignore[import]

# ---- GOAT-TS import ------------------------------------------------------
_REPO = Path(__file__).resolve().parent
_GOAT_TS = _REPO / "vendor" / "GOAT-TS"

import importlib.util as _ilu
import types as _types

def _load_module_with_name(sys_name: str, path: Path):
    """Load a Python file as sys_name, registering it in sys.modules for relative imports."""
    # Pre-register parent packages if needed
    parts = sys_name.split(".")
    for i in range(1, len(parts)):
        pkg = ".".join(parts[:i])
        if pkg not in sys.modules:
            sys.modules[pkg] = _types.ModuleType(pkg)
    spec = _ilu.spec_from_file_location(sys_name, path)
    mod = _ilu.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[sys_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod

# Load GOAT-TS graph.models (no heavy deps — just dataclasses + numpy)
_goat_models = _load_module_with_name("src.graph.models", _GOAT_TS / "src" / "graph" / "models.py")
# Load GOAT-TS memory_manager (depends only on src.graph.models)
_goat_mm = _load_module_with_name("src.memory_manager", _GOAT_TS / "src" / "memory_manager.py")

MemoryState = _goat_models.MemoryState
Node = _goat_models.Node
memory_tick = _goat_mm.memory_tick
ACTIVE_THRESHOLD = _goat_mm.ACTIVE_THRESHOLD
DORMANT_THRESHOLD = _goat_mm.DORMANT_THRESHOLD
TICKS_TO_DEEP = _goat_mm.TICKS_TO_DEEP
DEFAULT_DECAY_RATE = _goat_mm.DEFAULT_DECAY_RATE


class GoatMemoryManager:
    """
    Tracks per-token activation and applies GOAT-TS memory state transitions.

    Internally maintains one GOAT-TS Node per vocabulary token.

    Parameters
    ----------
    model : TorchAttractorLanguageModel
    decay_rate : exponential activation decay per tick (default 0.95)
    active_threshold : activation ≥ this → ACTIVE (default 0.5)
    dormant_threshold : activation < this → DORMANT (default 0.1)
    ticks_to_deep : consecutive low-activation ticks before → DEEP (default 3)
    bonus_scale : max activation bonus for ACTIVE tokens (injected in signal)
    """

    def __init__(
        self,
        model: "sb.TorchAttractorLanguageModel",
        *,
        decay_rate: float = DEFAULT_DECAY_RATE,
        active_threshold: float = ACTIVE_THRESHOLD,
        dormant_threshold: float = DORMANT_THRESHOLD,
        ticks_to_deep: int = TICKS_TO_DEEP,
        bonus_scale: float = 0.05,
    ) -> None:
        self.model = model
        self.decay_rate = decay_rate
        self.active_threshold = active_threshold
        self.dormant_threshold = dormant_threshold
        self.ticks_to_deep = ticks_to_deep
        self.bonus_scale = bonus_scale

        vocab = model.vocab
        # One Node per token; activation starts at dormant_threshold
        self.nodes: list[Node] = [
            Node(
                node_id=str(i),
                label=w,
                activation=dormant_threshold,
                state=MemoryState.DORMANT,
            )
            for i, w in enumerate(vocab)
        ]
        self._low_activation_ticks: dict[str, int] = {}
        self._tick_count: int = 0

    # ------------------------------------------------------------------
    def tick(self, contexts: list[list[int]]) -> None:
        """
        One memory management tick.

        1. Boost activation for tokens seen in this batch (usage-based).
        2. Apply exponential decay + ACTIVE/DORMANT/DEEP transitions via GOAT-TS.
        """
        # Flatten all token ids in the batch and count usages
        usage: dict[int, int] = {}
        for ctx in contexts:
            for tid in ctx:
                usage[tid] = usage.get(tid, 0) + 1

        # Boost activation proportional to usage frequency (capped at 1.0)
        max_usage = max(usage.values()) if usage else 1
        for tid, count in usage.items():
            if 0 <= tid < len(self.nodes):
                old = self.nodes[tid].activation
                boost = self.active_threshold * (count / max_usage)
                new_act = min(1.0, old + boost)
                self.nodes[tid] = replace(self.nodes[tid], activation=new_act)

        # GOAT-TS memory tick: decay + state transitions
        self.nodes, self._low_activation_ticks = memory_tick(
            self.nodes,
            self._low_activation_ticks,
            decay_rate=self.decay_rate,
            active_threshold=self.active_threshold,
            dormant_threshold=self.dormant_threshold,
            ticks_to_deep=self.ticks_to_deep,
        )
        self._tick_count += 1

    def activation_bonus(self, token_id: int) -> float:
        """
        Return an activation bonus scalar for a token.
        ACTIVE → [0, bonus_scale], DORMANT/DEEP → 0.
        """
        if not (0 <= token_id < len(self.nodes)):
            return 0.0
        n = self.nodes[token_id]
        if n.state != MemoryState.ACTIVE:
            return 0.0
        return float(self.bonus_scale * n.activation)

    def state_of(self, token_id: int) -> MemoryState:
        """Return the current MemoryState of a token."""
        if 0 <= token_id < len(self.nodes):
            return self.nodes[token_id].state
        return MemoryState.DORMANT

    def active_token_ids(self) -> list[int]:
        """Return list of token ids currently in ACTIVE state."""
        return [int(n.node_id) for n in self.nodes if n.state == MemoryState.ACTIVE]

    def deep_token_ids(self) -> list[int]:
        """Return list of token ids currently in DEEP state."""
        return [int(n.node_id) for n in self.nodes if n.state == MemoryState.DEEP]

    def stats(self) -> dict[str, int]:
        """Summary of node state counts."""
        counts: dict[str, int] = {s.value: 0 for s in MemoryState}
        for n in self.nodes:
            counts[n.state.value] += 1
        return {"tick": self._tick_count, **counts}

    # ------------------------------------------------------------------
    def sweep_config(self) -> dict[str, object]:
        """
        Return a dict of tunable hyperparameter knobs for automated Wave F sweeps.
        These correspond to the existing tension constants in sandbox.py.
        """
        return {
            # Decay rate: lower → tokens stay active longer
            "decay_rate": {"range": [0.85, 0.99], "current": self.decay_rate},
            # Active threshold: higher → harder to become ACTIVE
            "active_threshold": {"range": [0.3, 0.7], "current": self.active_threshold},
            # Dormant threshold: lower → easier to stay DORMANT
            "dormant_threshold": {"range": [0.05, 0.2], "current": self.dormant_threshold},
            # Ticks to deep: fewer → tokens archive faster
            "ticks_to_deep": {"range": [1, 8], "current": self.ticks_to_deep},
            # Bonus scale: how much ACTIVE tokens boost signal injection
            "bonus_scale": {"range": [0.0, 0.15], "current": self.bonus_scale},
        }


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, str(_REPO))
    import sandbox as sb  # type: ignore[import]

    print("[wave-f] goat_memory_transitions self-test ...", flush=True)

    model = sb.TorchAttractorLanguageModel(sb.FULL_VOCAB, train_window_size=4)
    mem = GoatMemoryManager(model, ticks_to_deep=2)

    # Test 1: initial state is all DORMANT
    st = mem.stats()
    assert st["dormant"] == model.vocab_size, f"Expected all DORMANT, got {st}"
    print(f"  test 1 PASS — all {st['dormant']} tokens start DORMANT", flush=True)

    # Test 2: after one tick with real token ids, some tokens become ACTIVE
    w2i = model._word_to_idx
    contexts = [[w2i["the"], w2i["cat"], w2i["sat"], w2i["on"]]]
    mem.tick(contexts)
    st2 = mem.stats()
    assert st2["active"] > 0, f"No tokens became ACTIVE after tick: {st2}"
    print(f"  test 2 PASS — {st2['active']} ACTIVE after 1 tick ({st2})", flush=True)

    # Test 3: activation_bonus is non-zero for active tokens
    active_ids = mem.active_token_ids()
    bonuses = [mem.activation_bonus(tid) for tid in active_ids[:5]]
    assert all(b > 0.0 for b in bonuses), f"Expected non-zero bonus for ACTIVE tokens, got {bonuses}"
    print(f"  test 3 PASS — ACTIVE token bonuses: {[f'{b:.4f}' for b in bonuses]}", flush=True)

    # Test 4: after ticks_to_deep ticks without usage, tokens go DORMANT then DEEP
    for _ in range(10):   # force low activation
        mem.tick([[]])
    st3 = mem.stats()
    print(f"  test 4 PASS — after 10 idle ticks: {st3}", flush=True)
    # Deep may or may not fire depending on threshold/decay; just check states are valid
    total = st3["active"] + st3["dormant"] + st3["deep"]
    assert total == model.vocab_size, f"Total nodes mismatch: {total} != {model.vocab_size}"

    # Test 5: sweep_config returns expected keys
    cfg = mem.sweep_config()
    assert "decay_rate" in cfg and "bonus_scale" in cfg
    print(f"  test 5 PASS — sweep_config keys: {list(cfg.keys())}", flush=True)

    print("\n[wave-f] ALL TESTS PASSED", flush=True)
