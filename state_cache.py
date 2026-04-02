"""
Wave C — Rolling state cache for inference aligned with window dynamics.

Each ``step(token_id)`` builds the last-``W`` token ids (same padding as
``TorchAttractorLanguageModel.window_ids_from_sequence``), embeds with
``embedder`` + ``F.normalize`` per row (matches the prior single-token cache:
no ``LayerNorm``), runs ``model.run_window_dynamics`` on ``S`` of shape
``(1, W, D)`` so jitter / GOAT / high-tension behaviour matches training,
then updates ``fast_state`` from the last window row and ``slow_memory`` with
the same EMA as before. ``logits()`` still uses ``readout(combined)`` /
``effective_temperature()`` (not ``readout_window``).

Usage
-----
    from state_cache import AttractorStateCache, generate_with_cache

    cache = AttractorStateCache(model)
    text = generate_with_cache(model, cache, prompt="the cat sat", max_tokens=30)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    import sandbox as sb  # type: ignore[import]


@dataclass
class AttractorStateCache:
    """
    Mutable inference-state container tied to one TorchAttractorLanguageModel instance.

    Attributes
    ----------
    fast_state : Tensor (D,) — last post-dynamics row of the rolling window (unit-norm).
    slow_memory : Tensor (D,) — slow decaying memory.
    phrase_table : list of (token_ids, state_snapshot) pairs — rolling window history.
    token_history : flat list[int] of all stepped token ids (including warmup).
    """
    model: "sb.TorchAttractorLanguageModel"
    fast_state: torch.Tensor = field(init=False)
    slow_memory: torch.Tensor = field(init=False)
    phrase_table: list = field(default_factory=list)
    token_history: list = field(default_factory=list)

    def __post_init__(self) -> None:
        device = self.model.embedder.weight.device
        dtype = self.model.embedder.weight.dtype
        D = self.model.state_dim
        self.fast_state = torch.zeros(D, device=device, dtype=dtype)
        self.slow_memory = torch.zeros(D, device=device, dtype=dtype)

    def reset(self) -> None:
        """Clear all state (new conversation / generation)."""
        device = self.model.embedder.weight.device
        dtype = self.model.embedder.weight.dtype
        D = self.model.state_dim
        self.fast_state = torch.zeros(D, device=device, dtype=dtype)
        self.slow_memory = torch.zeros(D, device=device, dtype=dtype)
        self.phrase_table.clear()
        self.token_history.clear()

    def step(self, token_id: int) -> torch.Tensor:
        """
        Evolve state with one new token via the same window pipeline as training
        (``run_window_dynamics`` on ``(1, W, D)``), then EMA slow memory.

        Returns the current ``fast_state`` ``(D,)`` after the update.
        """
        model = self.model
        device = self.fast_state.device
        dtype = self.fast_state.dtype

        seq = self.token_history + [token_id]
        ids = model.window_ids_from_sequence(seq)

        with torch.no_grad():
            ids_t = torch.tensor(ids, device=device, dtype=torch.long)
            emb = model.embedder(ids_t)
            emb = model.norm(emb)
            S = F.normalize(emb, dim=-1).unsqueeze(0)
            S_out, _ = model.run_window_dynamics(
                S,
                collect_metrics=False,
                record_tension_log=False,
                context_ids=[ids],
            )
            if S_out.dim() == 2:
                S_out = S_out.unsqueeze(0)
            new_fast = S_out[0, -1, :].clone()
            new_fast = F.normalize(new_fast, dim=-1)

        slow_lr = float(model.slow_lr.detach())
        slow_dec = float(model.slow_decay.detach())
        new_slow = (1.0 - slow_dec) * self.slow_memory + slow_lr * new_fast
        slow_n = torch.linalg.vector_norm(new_slow)
        max_slow = 3.0
        if slow_n > max_slow:
            new_slow = new_slow * (max_slow / slow_n)

        self.fast_state = new_fast
        self.slow_memory = new_slow
        self.token_history.append(token_id)

        W = model.train_window_size
        self.phrase_table.append((token_id, new_fast.detach().clone()))
        if len(self.phrase_table) > W:
            self.phrase_table.pop(0)

        return self.fast_state

    def logits(self) -> torch.Tensor:
        """
        Compute readout logits from the current (fast, slow) state.
        Symplectic blend: combined = w_fast * fast + w_slow * slow (normalised).
        """
        model = self.model
        w_fast = float(model.w_fast)
        w_slow = float(model.w_slow)
        combined = w_fast * self.fast_state + w_slow * self.slow_memory
        n = torch.linalg.vector_norm(combined)
        if n > 1e-8:
            combined = combined / n

        with torch.no_grad():
            logits = model.readout(combined)
        return logits

    def warmup(self, prompt_ids: list[int]) -> None:
        """
        Seed the cache by stepping through prompt token ids one at a time.
        Call this before generate_with_cache() to warm up the attractor state.
        """
        for tid in prompt_ids:
            self.step(tid)


# --------------------------------------------------------------------------
# generate_with_cache — rolling window inference
# --------------------------------------------------------------------------

def generate_with_cache(
    model: "sb.TorchAttractorLanguageModel",
    cache: AttractorStateCache,
    prompt: str,
    max_tokens: int = 40,
    temperature: float = 1.0,
    top_k: int = 28,
    repeat_penalty: float = 1.35,
    no_repeat_last_extra: float = 5.0,
    reset: bool = True,
) -> str:
    """
    Generate text using the rolling state cache.

    Parameters
    ----------
    model : TorchAttractorLanguageModel
    cache : AttractorStateCache (will be reset if reset=True)
    prompt : seed text
    max_tokens : tokens to generate
    temperature : sampling temperature
    top_k : top-k truncation
    repeat_penalty : logit divisor for recently seen tokens
    no_repeat_last_extra : extra penalty for the immediately preceding token
    reset : clear the cache before generation (True = stateless call)
    """
    import sandbox as _sb  # noqa: F811  # type: ignore[import]

    tok = getattr(model, "tokenizer", None)
    if tok is not None:
        prompt_ids = tok.encode(prompt)
    else:
        prompt_ids = list(range(min(model.train_window_size, 6)))
    if not prompt_ids:
        prompt_ids = [0]

    if reset:
        cache.reset()

    was_training = model.training
    model.eval()

    with torch.inference_mode():
        cache.warmup(prompt_ids)

        generated_ids = list(prompt_ids)

        for _ in range(max_tokens):
            logits = cache.logits()
            recent = generated_ids[-top_k:] if len(generated_ids) > top_k else generated_ids
            for rid in set(recent):
                logits[rid] = logits[rid] / repeat_penalty
            if generated_ids:
                logits[generated_ids[-1]] = logits[generated_ids[-1]] / no_repeat_last_extra

            next_id = _sb.sample_next_token_id(
                logits,
                temperature,
                top_k,
                generated_ids,
                repeat_penalty,
                no_repeat_last_extra,
            )
            generated_ids.append(next_id)
            cache.step(next_id)

    if was_training:
        model.train()

    if tok is not None:
        return tok.decode(generated_ids)
    return " ".join(str(i) for i in generated_ids)


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import sandbox as sb  # type: ignore[import]

    print("[wave-c] state_cache self-test ...", flush=True)

    tok = sb._build_tokenizer(mode="fallback", vocab_cap=512)
    model = sb.TorchAttractorLanguageModel(tok.n_vocab, state_dim=512, train_window_size=4)
    model.tokenizer = tok
    model.eval()
    cache = AttractorStateCache(model)

    # Test 1: step produces finite states
    for word in ["the cat sat on"]:
        for tid in tok.encode(word)[:4]:
            cache.step(tid)
    assert torch.isfinite(cache.fast_state).all(), "fast_state is not finite"
    assert torch.isfinite(cache.slow_memory).all(), "slow_memory is not finite"
    print(f"  test 1 PASS — fast_state norm={cache.fast_state.norm():.4f}  slow_norm={cache.slow_memory.norm():.4f}", flush=True)

    # Test 2: logits are finite
    logits = cache.logits()
    assert torch.isfinite(logits).all(), "logits are not finite"
    print(f"  test 2 PASS — logits shape={logits.shape}  max={logits.max():.4f}", flush=True)

    # Test 3: full generation
    text = generate_with_cache(model, cache, prompt="the quick brown fox", max_tokens=10)
    assert len(text.split()) > 0, "generate_with_cache returned empty string"
    print(f"  test 3 PASS — generated: {text!r}", flush=True)

    # Test 4: phrase_table rolling window
    assert len(cache.phrase_table) <= model.train_window_size, "phrase_table exceeds window_size"
    print(f"  test 4 PASS — phrase_table len={len(cache.phrase_table)} <= window_size={model.train_window_size}", flush=True)

    print("\n[wave-c] ALL TESTS PASSED", flush=True)
