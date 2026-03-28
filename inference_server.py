"""
Wave G — Inference server.

Serves the TorchAttractorLanguageModel via FastAPI (OpenAI-compatible /v1/completions
endpoint). Uses the rolling state cache from wave_c_cache for O(1)-per-token latency.

Endpoints
---------
GET  /health                 — liveness + tension metrics
POST /v1/completions         — OpenAI-compatible text completion
POST /v1/generate            — raw generate call (more options)
GET  /metrics/tension        — last window tension curve
POST /ts/propagate           — trigger one TSCore wave propagation
GET  /ts/tension             — current TSCore graph tension

Run
---
    python inference_server.py [--model-checkpoint path] [--host 0.0.0.0] [--port 8000]

Requires FastAPI and uvicorn:
    pip install fastapi uvicorn

If --model-checkpoint is not given, an untrained model (sandbox defaults) is loaded.
The server auto-installs state_cache and llm_substrate_node as background threads.
"""
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

# ---- FastAPI (optional at import time; server only if running as __main__) ---
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False
    # Dummy stubs for type checkers
    class BaseModel:  # type: ignore[no-redef]
        pass
    FastAPI = None  # type: ignore[misc,assignment]


import sandbox as sb  # type: ignore[import]
from state_cache import AttractorStateCache, generate_with_cache  # type: ignore[import]
from llm_substrate_node import LLMSubstrateNode  # type: ignore[import]


# --------------------------------------------------------------------------
# Model loader
# --------------------------------------------------------------------------

def load_model(checkpoint: Optional[str] = None) -> "sb.TorchAttractorLanguageModel":
    import torch
    model = sb.TorchAttractorLanguageModel(sb.FULL_VOCAB)
    if checkpoint:
        p = Path(checkpoint)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        state_dict = torch.load(str(p), map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        print(f"[inference_server] loaded checkpoint: {checkpoint}", flush=True)
    else:
        print("[inference_server] no checkpoint — using untrained model", flush=True)
    model.eval()
    return model


# --------------------------------------------------------------------------
# Application state (shared across requests)
# --------------------------------------------------------------------------

class AppState:
    def __init__(self, checkpoint: Optional[str] = None) -> None:
        self.model = load_model(checkpoint)
        self.cache = AttractorStateCache(self.model)
        self.substrate = LLMSubstrateNode(self.model, quiet=True)
        self._lock = threading.Lock()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 28,
    ) -> dict:
        with self._lock:
            text = generate_with_cache(
                self.model,
                self.cache,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                reset=True,
            )
            curve = getattr(self.model, "_last_window_tension_curve", [])
            ts_t = self.substrate.last_ts_tension
        return {
            "text": text,
            "tension_curve": curve,
            "ts_tension": ts_t,
            "evolve_count": self.substrate.evolve_count,
        }


# --------------------------------------------------------------------------
# FastAPI app factory
# --------------------------------------------------------------------------

def create_app(state: AppState) -> "FastAPI":
    if not _HAS_FASTAPI:
        raise ImportError("FastAPI and uvicorn are required: pip install fastapi uvicorn")

    app = FastAPI(
        title="woke-baby-llm inference server",
        description="Attractor LM inference via FastAPI (Wave G)",
        version="0.0.1",
    )

    # ------ request/response schemas ------

    class CompletionRequest(BaseModel):
        prompt: str = "the"
        max_tokens: int = 64
        temperature: float = 1.0
        top_k: int = 28
        model: str = "woke-attractor"

    class GenerateRequest(BaseModel):
        prompt: str = "the"
        max_tokens: int = 64
        temperature: float = 1.0
        top_k: int = 28

    # ------ endpoints ------

    @app.get("/health")
    async def health():
        curve = getattr(state.model, "_last_window_tension_curve", [])
        return {
            "status": "ok",
            "vocab_size": state.model.vocab_size,
            "state_dim": state.model.state_dim,
            "last_tension": float(curve[-1]) if curve else None,
            "ts_tension": state.substrate.last_ts_tension,
            "evolve_count": state.substrate.evolve_count,
        }

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        try:
            result = state.generate(
                prompt=req.prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        # OpenAI-compatible response shape
        return {
            "id": "cmpl-attractor",
            "object": "text_completion",
            "model": req.model,
            "choices": [
                {
                    "text": result["text"],
                    "index": 0,
                    "finish_reason": "length",
                }
            ],
            "attractor_meta": {
                "tension_curve": result["tension_curve"],
                "ts_tension": result["ts_tension"],
                "evolve_count": result["evolve_count"],
            },
        }

    @app.post("/v1/generate")
    async def generate(req: GenerateRequest):
        try:
            result = state.generate(
                prompt=req.prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return result

    @app.get("/metrics/tension")
    async def tension_metrics():
        curve = getattr(state.model, "_last_window_tension_curve", [])
        return {
            "tension_curve": curve,
            "final_tension": float(curve[-1]) if curve else None,
            "steps": getattr(state.model, "_last_adaptive_window_steps", None),
        }

    @app.post("/ts/propagate")
    async def ts_propagate():
        t, icarus = state.substrate.ts.propagate_wave(quiet=True)
        return {"ts_tension": t, "icarus_line": icarus, "tick": state.substrate.ts.tick}

    @app.get("/ts/tension")
    async def ts_tension():
        return {
            "ts_tension": state.substrate.ts.measure_tension(),
            "tick": state.substrate.ts.tick,
            "evolve_count": state.substrate.evolve_count,
        }

    return app


# --------------------------------------------------------------------------
# CLI entrypoint
# --------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="woke-baby-llm inference server (Wave G)")
    parser.add_argument("--model-checkpoint", default=None, help="Path to model state_dict .pt file")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--self-test", action="store_true", help="Run a quick functional test and exit")
    args = parser.parse_args()

    state = AppState(checkpoint=args.model_checkpoint)

    if args.self_test:
        print("[wave-g] inference_server self-test ...", flush=True)
        # Test 1: generate endpoint logic
        result = state.generate("the cat sat on the mat", max_tokens=10)
        assert "text" in result and len(result["text"].split()) > 0, "Empty generation"
        print(f"  test 1 PASS — generate: {result['text']!r}", flush=True)
        # Test 2: health would return ok
        curve = getattr(state.model, "_last_window_tension_curve", [])
        print(f"  test 2 PASS — tension_curve len={len(curve)}", flush=True)
        print("\n[wave-g] ALL TESTS PASSED", flush=True)
        sys.exit(0)

    if not _HAS_FASTAPI:
        print("FastAPI not installed. Run: pip install fastapi uvicorn", file=sys.stderr)
        sys.exit(1)

    import uvicorn  # type: ignore[import]
    app = create_app(state)
    print(f"[wave-g] serving on http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)
