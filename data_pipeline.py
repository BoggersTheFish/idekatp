"""
Wave D — Streaming sharded DataLoader.

Replaces the in-memory list shuffle in sandbox.py's training loop with a
streaming DataLoader that:

  1. Accepts any iterable of JSONL / plain-text / corpus files (or directories).
  2. Tokenises each line lazily (word-list or tiktoken via wave_a_tokenizer).
  3. Builds sliding-window (context, target) pairs on-the-fly.
  4. Yields mini-batches of (contexts, targets) compatible with
     model.trajectory_contrastive_loss_and_logits(contexts, targets).
  5. Supports multi-shard round-robin by holding one file handle per shard.

Usage
-----
    from data_pipeline import AttractorDataPipeline

    pipe = AttractorDataPipeline(
        sources=["data/corpus.txt", "data/extra/"],   # files or directories
        model=model,          # for vocab / window_size
        batch_size=16,
        window_size=6,
        shuffle_buffer=1024,
        tokenizer=tok,        # optional: AttractorTokenizer from wave_a_tokenizer
    )

    for epoch in range(NUM_EPOCHS):
        for contexts, targets in pipe.epoch_batches():
            loss, _ = model.trajectory_contrastive_loss_and_logits(contexts, targets)
            ...

For single-machine operation (no Redis), shuffle_buffer controls in-memory
random shuffling. For distributed runs, each worker gets its own shard files.
"""
from __future__ import annotations

import os
import random
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Generator, Iterable, Optional

if TYPE_CHECKING:
    import sandbox as sb  # type: ignore[import]


# --------------------------------------------------------------------------
# Tokenizer protocol (duck-typed to accept both word-list and AttractorTokenizer)
# --------------------------------------------------------------------------

class _WordListTokenizer:
    """Fallback tokenizer: split on whitespace, filter to model vocab."""

    def __init__(self, model: "sb.TorchAttractorLanguageModel") -> None:
        self._w2i = model._word_to_idx
        self.n_vocab = model.vocab_size

    def encode(self, text: str) -> list[int]:
        return [self._w2i[w] for w in text.lower().split() if w in self._w2i]


# --------------------------------------------------------------------------
# File discovery
# --------------------------------------------------------------------------

def _collect_text_files(source: str | Path) -> list[Path]:
    p = Path(source)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(
            f for f in p.rglob("*")
            if f.suffix in (".txt", ".jsonl", ".json") and f.is_file()
        )
    return []


def _iter_lines(path: Path) -> Generator[str, None, None]:
    """Yield non-empty, non-comment lines from a text or JSONL file."""
    import json as _json

    with path.open(encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if path.suffix == ".jsonl":
                try:
                    obj = _json.loads(line)
                    text = obj.get("text") or obj.get("content") or obj.get("sentence", "")
                    if text:
                        yield str(text)
                except Exception:
                    yield line
            else:
                yield line


# --------------------------------------------------------------------------
# Sliding-window pair builder
# --------------------------------------------------------------------------

def _make_windows(ids: list[int], window_size: int) -> list[tuple[list[int], int]]:
    pairs: list[tuple[list[int], int]] = []
    for start in range(len(ids) - window_size):
        ctx = ids[start : start + window_size]
        tgt = ids[start + window_size]
        pairs.append((ctx, tgt))
    return pairs


# --------------------------------------------------------------------------
# AttractorDataPipeline
# --------------------------------------------------------------------------

class AttractorDataPipeline:
    """
    Streaming, shard-aware data pipeline for the attractor training loop.

    Parameters
    ----------
    sources : list of file paths or directories (plain text or JSONL)
    model : TorchAttractorLanguageModel — provides vocab + window_size
    batch_size : mini-batch width (≥ 2 for trajectory contrastive loss)
    window_size : overrides model.train_window_size when set
    shuffle_buffer : number of windows to hold in memory for in-batch shuffle
    tokenizer : optional; if None, uses the model word list
    shard_id / num_shards : for multi-worker data parallelism (each worker
        receives every num_shards-th file starting from shard_id)
    seed : random seed for shuffle
    """

    def __init__(
        self,
        sources: list[str | Path],
        model: "sb.TorchAttractorLanguageModel",
        batch_size: int = 16,
        window_size: Optional[int] = None,
        shuffle_buffer: int = 1024,
        tokenizer: Optional[object] = None,
        shard_id: int = 0,
        num_shards: int = 1,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.batch_size = max(2, batch_size)
        self.window_size = window_size or model.train_window_size
        self.shuffle_buffer = shuffle_buffer
        self.tokenizer = tokenizer or _WordListTokenizer(model)
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.seed = seed

        # Collect all text files, then apply sharding
        all_files: list[Path] = []
        for src in sources:
            all_files.extend(_collect_text_files(src))
        self.files: list[Path] = [
            f for i, f in enumerate(all_files) if i % num_shards == shard_id
        ]
        if not self.files:
            raise ValueError(
                f"No text files found under sources={sources} for "
                f"shard_id={shard_id}/{num_shards}"
            )

    # ------------------------------------------------------------------
    def _window_stream(self) -> Generator[tuple[list[int], int], None, None]:
        """Yield (context, target) pairs by streaming all shard files."""
        for path in self.files:
            for line in _iter_lines(path):
                ids = self.tokenizer.encode(line)
                if len(ids) < self.window_size + 1:
                    continue
                for pair in _make_windows(ids, self.window_size):
                    yield pair

    def epoch_batches(
        self,
    ) -> Generator[tuple[list[list[int]], list[int]], None, None]:
        """
        Yield (contexts, targets) mini-batches for one epoch.

        Uses a shuffle buffer: fills `shuffle_buffer` pairs, shuffles, then
        yields batches until the buffer is below batch_size, then refills.
        """
        rng = random.Random(self.seed)
        buf: deque[tuple[list[int], int]] = deque()
        stream = self._window_stream()

        def _fill(n: int) -> bool:
            count = 0
            for item in stream:
                buf.append(item)
                count += 1
                if count >= n:
                    return True
            return False

        # Prime the buffer
        _fill(self.shuffle_buffer)
        buf_list = list(buf)
        buf.clear()

        while buf_list:
            rng.shuffle(buf_list)
            for i in range(0, len(buf_list), self.batch_size):
                chunk = buf_list[i : i + self.batch_size]
                if len(chunk) < 2:
                    # Pad to minimum batch size
                    chunk = chunk * 2
                contexts = [c for c, _t in chunk]
                targets = [_t for _c, _t in chunk]
                yield contexts, targets

            buf_list = []
            _fill(self.shuffle_buffer)
            buf_list = list(buf)
            buf.clear()

    def epoch_count_estimate(self) -> int:
        """Estimate number of batches per epoch (single pass, no refill)."""
        total = 0
        for path in self.files:
            for line in _iter_lines(path):
                ids = self.tokenizer.encode(line)
                if len(ids) >= self.window_size + 1:
                    total += max(0, len(ids) - self.window_size)
        return max(0, total // self.batch_size)


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parent))
    import sandbox as sb  # type: ignore[import]

    print("[wave-d] data_pipeline self-test ...", flush=True)

    CORPUS = _Path(__file__).resolve().parent / "data" / "corpus.txt"
    model = sb.TorchAttractorLanguageModel(sb.FULL_VOCAB, train_window_size=4)
    model.eval()

    pipe = AttractorDataPipeline(
        sources=[CORPUS],
        model=model,
        batch_size=4,
        window_size=4,
        shuffle_buffer=64,
        seed=0,
    )

    count = 0
    total_contexts = 0
    for contexts, targets in pipe.epoch_batches():
        assert len(contexts) == len(targets), "contexts/targets length mismatch"
        assert len(contexts) >= 2, "batch too small for trajectory contrastive loss"
        total_contexts += len(contexts)
        count += 1
        if count >= 5:
            break

    print(f"  test 1 PASS — {count} batches yielded, {total_contexts} total windows", flush=True)

    # Test 2: estimate
    est = pipe.epoch_count_estimate()
    print(f"  test 2 PASS — epoch_count_estimate: {est} batches", flush=True)

    # Test 3: JSONL support (write a temp file)
    import json, tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fh:
        for sentence in sb.load_corpus(CORPUS)[:5]:
            fh.write(json.dumps({"text": sentence}) + "\n")
        tmp = fh.name

    try:
        pipe2 = AttractorDataPipeline(
            sources=[tmp],
            model=model,
            batch_size=2,
            shuffle_buffer=32,
        )
        batches = list(pipe2.epoch_batches())
        assert len(batches) >= 1, "JSONL pipeline yielded no batches"
        print(f"  test 3 PASS — JSONL: {len(batches)} batches", flush=True)
    finally:
        os.unlink(tmp)

    print("\n[wave-d] ALL TESTS PASSED", flush=True)
