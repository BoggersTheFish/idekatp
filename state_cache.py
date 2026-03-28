"""
Wave C — Rolling state cache for O(1)-per-token inference.

Replaces the re-embedding of the full W-token window every generation step with
a rolling cache that maintains:
  - fast_state : (D,) — most-recent evolved fast state
  - slow_memory : (D,) — accumulated slow memory (decaying average)
  - phrase_table : list[(ids: list[int], state: Tensor)] — sliding phrase memory

On each new token:
1. Update fast_state via one `step_state_batch` call on the new embedding.
2. Update slow_memory with exponential decay: slow = (1 - slow_decay)*slow + slow_lr*fast.
3. Append the new state to phrase_table and evict the oldest entry if len > window_size.
4. Compute readout logits from the cached (fast, slow) without re-running dynamics.

This keeps inference at O(1) per token while the full wave cycle continues in the
background (long context = more phrase_table entries, not more per-step cost).

Usage
-----
    from state_cache import AttractorStateCache, generate_with_cache

    cache = AttractorStateCache(model)
    text = generate_with_cache(model, cache, prompt="the cat sat", max_tokens=30)

The full wave-cycle dynamics still run during training via run_window_dynamics.
The cache is inference-only and does not change any training code.
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
    fast_state : Tensor (D,) — current fast attractor state.
    slow_memory : Tensor (D,) — slow decaying memory.
    phrase_table : list of (token_ids, state_snapshot) pairs — rolling window history.
    token_history : flat list[int] of all generated token ids.
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
        Evolve state with one new token (O(1)).

        Returns the current fast_state after the update.
        """
        model = self.model
        device = self.fast_state.device
        dtype = self.fast_state.dtype

        # 1. Embed the new token
        with torch.no_grad():
            emb = model.embedder(torch.tensor([token_id], device=device))[0]
            emb = F.normalize(emb, dim=-1)

        # 2. Context-conditioned signal (fast + slow → direction)
        slow_dec = float(model.slow_decay)
        fast_n = torch.linalg.vector_norm(self.fast_state)
        if fast_n > 1e-6:
            ctx = self.fast_state / (fast_n + 1e-8)
        else:
            ctx = emb
        gamma = float(model.gamma.detach())
        signal = emb + gamma * ctx
        n = torch.linalg.vector_norm(signal)
        if n > 1e-12:
            signal = signal / n

        # 3. One dynamics step (reuse sandbox SimpleAttractorDynamics)
        with torch.no_grad():
            new_fast = model.dynamics(self.fast_state.unsqueeze(0), signal.unsqueeze(0)).squeeze(0)
        new_fast = F.normalize(new_fast, dim=-1)

        # 4. Slow memory update
        slow_lr = float(model.slow_lr)
        new_slow = (1.0 - slow_dec) * self.slow_memory + slow_lr * new_fast
        slow_n = torch.linalg.vector_norm(new_slow)
        max_slow = 3.0
        if slow_n > max_slow:
            new_slow = new_slow * (max_slow / slow_n)

        self.fast_state = new_fast
        self.slow_memory = new_slow
        self.token_history.append(token_id)

        # 5. Update phrase table (sliding window of model.train_window_size states)
        W = model.train_window_size
        self.phrase_table.append((token_id, new_fast.detach().clone()))
        if len(self.phrase_table) > W:
            self.phrase_table.pop(0)

        return self.fast_state

    def logits(self) -> torch.Tensor:
        """
        Compute readout logits from the current (fast, slow) state (O(1)).
        Matches symplectic readout: combined = w_fast * fast + w_slow * slow (normalised).
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
# generate_with_cache — O(1)-per-token inference
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
    Generate text using the rolling state cache (O(1) per token).

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

    w2i = model._word_to_idx
    prompt_words = [w for w in prompt.lower().split() if w in w2i] or ["the"]
    prompt_ids = [w2i[w] for w in prompt_words]

    if reset:
        cache.reset()

    was_training = model.training
    model.eval()

    with torch.inference_mode():
        # Warm up the cache on the prompt
        cache.warmup(prompt_ids)

        generated_ids = list(prompt_ids)
        generated_words = list(prompt_words)

        for _ in range(max_tokens):
            logits = cache.logits()
            # Apply repeat penalty on recent tokens
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
            generated_words.append(model.vocab[next_id])
            cache.step(next_id)

    if was_training:
        model.train()

    return " ".join(generated_words)


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import sandbox as sb  # type: ignore[import]

    print("[wave-c] state_cache self-test ...", flush=True)

    model = sb.TorchAttractorLanguageModel(sb.FULL_VOCAB, state_dim=512, train_window_size=4)
    model.eval()
    cache = AttractorStateCache(model)

    # Test 1: step produces finite states
    w2i = model._word_to_idx
    for word in ["the", "cat", "sat", "on"]:
        if word in w2i:
            fs = cache.step(w2i[word])
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
