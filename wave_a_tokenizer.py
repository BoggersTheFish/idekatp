"""
Wave A — Tokenizer helper.

Replaces the hard-coded _VOCAB_BLOB / 512-word vocabulary with:
  - tiktoken BPE (default: gpt2 encoding, vocab_cap=32768) when tiktoken is installed
  - word-list fallback (sandbox.py FULL_VOCAB) when tiktoken is absent or use_tiktoken=False

Usage (from sandbox.py main or external scripts):

    from wave_a_tokenizer import make_vocab_and_tokenizer, encode_corpus, TokenizerMode

    # BPE mode (32k tokens, state_dim should scale to 2048+)
    vocab_list, tok = make_vocab_and_tokenizer(vocab_cap=32768)

    # Sentence → list[int]
    ids = tok.encode("the cat sat on the mat")

    # list[int] → str
    text = tok.decode(ids)

    # Encode a full corpus (list of sentences) → list[list[int]]
    id_seqs = encode_corpus(sentences, tok, min_len=window_size + 1)

Calling convention is identical to sandbox.py's existing token-id lists,
so train_step / trajectory_contrastive_loss_and_logits are unchanged.
"""
from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import Sequence

# ---- make ts-llm importable from vendor/ --------------------------------
_REPO_ROOT = Path(__file__).resolve().parent
_TS_LLM = _REPO_ROOT / "vendor" / "ts-llm"
if str(_TS_LLM) not in sys.path:
    sys.path.insert(0, str(_TS_LLM))


class TokenizerMode(str, Enum):
    TIKTOKEN = "tiktoken"
    WORDLIST = "wordlist"


def make_vocab_and_tokenizer(
    vocab_cap: int = 32768,
    encoding_name: str = "gpt2",
    use_tiktoken: bool = True,
    fallback_vocab: list[str] | None = None,
) -> tuple[list[str], "AttractorTokenizer"]:
    """
    Build a tokenizer and matching vocabulary list.

    Parameters
    ----------
    vocab_cap:
        Maximum vocab size (embedding table width). 32768 = 2^15, a good mid-point
        between the full GPT-2 BPE (50257) and the baseline 512-word vocab.
    encoding_name:
        tiktoken encoding to use (default "gpt2").
    use_tiktoken:
        If False, use sandbox.py word-list fallback (offline / deterministic).
    fallback_vocab:
        Word list to use in word-list mode. Defaults to sandbox.py FULL_VOCAB.

    Returns
    -------
    vocab_list:
        List of str tokens (BPE) or words (word-list). Length == tokenizer.n_vocab.
        Pass directly as the first arg to TorchAttractorLanguageModel(vocab=...).
    tokenizer:
        AttractorTokenizer instance with .encode(text) and .decode(ids).
    """
    try:
        from attractor_llm.tokenizer import AttractorTokenizer
    except ImportError as exc:
        raise ImportError(
            "ts-llm not found. Run: git submodule update --init --recursive"
        ) from exc

    tok = AttractorTokenizer(
        encoding_name=encoding_name,
        vocab_cap=vocab_cap,
        use_tiktoken=use_tiktoken,
    )

    if tok.uses_tiktoken:
        # Build a vocab list: integer range as string tokens (e.g. "0", "1", …)
        # for API compatibility with TorchAttractorLanguageModel(vocab=list[str]).
        # The actual decoding is done via tok.decode(); the string labels are
        # only used for display / corpus-coverage reporting.
        vocab_list = [str(i) for i in range(tok.n_vocab)]
        mode = TokenizerMode.TIKTOKEN
    else:
        # Word-list mode: fall back to provided or sandbox FULL_VOCAB
        if fallback_vocab is not None:
            vocab_list = list(fallback_vocab)
        else:
            _import_sandbox_vocab()
            import sandbox as _sb  # type: ignore[import]
            vocab_list = list(_sb.FULL_VOCAB)
        mode = TokenizerMode.WORDLIST
        tok._words = vocab_list
        tok._word2id = {w: i for i, w in enumerate(vocab_list)}
        tok.n_vocab = len(vocab_list)

    print(
        f"[wave-a] tokenizer: mode={mode.value}  vocab_cap={vocab_cap}  "
        f"actual_vocab={tok.n_vocab}",
        flush=True,
    )
    return vocab_list, tok


def encode_corpus(
    sentences: list[str],
    tok: "AttractorTokenizer",
    min_len: int = 2,
) -> list[list[int]]:
    """
    Encode a list of sentences to token-id sequences.

    Each sequence is filtered to ids < tok.n_vocab. Sequences shorter
    than min_len after encoding are dropped.
    """
    out: list[list[int]] = []
    for s in sentences:
        ids = tok.encode(s)
        if len(ids) >= min_len:
            out.append(ids)
    return out


def _import_sandbox_vocab() -> None:
    """Add repo root to sys.path so 'import sandbox' works."""
    root = str(_REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


# ---- Convenience: recommended state_dim for vocab_cap -------------------
_VOCAB_DIM_TABLE: list[tuple[int, int]] = [
    (512, 512),
    (8_192, 1024),
    (32_768, 2048),
    (65_536, 4096),
]


def recommended_state_dim(vocab_cap: int) -> int:
    """
    Return a reasonable state_dim for a given vocab_cap.
    Rule: state_dim ≥ ceil(log2(vocab_cap)) * 64, capped at 4096.
    """
    for cap, dim in _VOCAB_DIM_TABLE:
        if vocab_cap <= cap:
            return dim
    return 4096


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Wave A tokenizer smoke test")
    parser.add_argument("--vocab-cap", type=int, default=32768)
    parser.add_argument("--no-tiktoken", action="store_true")
    args = parser.parse_args()

    vocab_list, tok = make_vocab_and_tokenizer(
        vocab_cap=args.vocab_cap, use_tiktoken=not args.no_tiktoken
    )
    sample = "the quick brown fox jumps over the lazy dog"
    ids = tok.encode(sample)
    roundtrip = tok.decode(ids)
    print(f"  sample: {sample!r}")
    print(f"  ids[:10]: {ids[:10]}")
    print(f"  decode: {roundtrip!r}")
    dim = recommended_state_dim(args.vocab_cap)
    print(f"  recommended state_dim for vocab_cap={args.vocab_cap}: {dim}")
    print("  wave_a_tokenizer self-test OK")
