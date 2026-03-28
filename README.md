# woke-baby-llm

A **minimal, single-file** PyTorch experiment: a small-vocabulary “language model” built from **continuous dynamics** instead of attention or transformers. State evolves along a trajectory; meaning is **path-dependent** (order and context matter).

Repository: [github.com/BoggersTheFish/woke-baby-llm](https://github.com/BoggersTheFish/woke-baby-llm)

## What it does

- **Token embeddings** are normalized vectors; each step feeds a **context-conditioned signal** into a dynamical system.
- **Fast + slow memory**: a fast state reacts to the current input; a slow state tracks longer context via exponential blending.
- **Partial convergence** per token (a few dynamics steps), so the trajectory is not reset to a full attractor every time.
- **Dynamics**: learned diffusion matrix, cubic nonlinearity, scaled input (`beta`), and small Gaussian noise for exploration.
- **Decoding**: negative distance from a **weighted combined state** to context-aware candidate directions, scaled by a learnable temperature.
- **Diagnostics**: optional attractor hashing, diversity metrics (entropy, top‑k counts), and `compare_prompts()` to measure sensitivity to word order.

There is **no** attention, no transformer blocks, and no external model—only `sandbox.py` plus `requirements.txt`.

## Requirements

- Python 3.10+ recommended  
- [PyTorch](https://pytorch.org/) (see `requirements.txt`)

Install:

```bash
pip install -r requirements.txt
```

## Run

```bash
python sandbox.py
```

This trains for a few epochs on a tiny bundled corpus, then prints sample generations, debug attractor stats (when enabled), and trajectory comparisons between prompts.

## Main knobs (in code)

| Idea | Where |
|------|--------|
| Steps per token | `convergence_steps` (default 4); use `step_token()` for one step |
| Slow memory retention | `alpha` (~0.97): `slow = α·slow + (1−α)·fast` |
| Readout mix | `w_fast`, `w_slow` (default 1.0 / 0.5): `combined = w_fast·fast + w_slow·slow` |
| Context in the signal | `gamma` (learnable): `signal ∝ base_embedding + γ·context`, then renormalized |
| Input gain / noise | `beta`, `noise_scale` in `SimpleAttractorDynamics` |

## API sketch

- `TorchAttractorLanguageModel` — embed, `get_signal`, `evolve_token`, `next_token_logits`, `generate`, `encode_prompt`
- `compare_prompts(model, prompt_a, prompt_b)` — L2 / cosine between final combined states

## License

[MIT](LICENSE)
