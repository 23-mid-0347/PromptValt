"""Universal token counter — one consistent proxy across providers.

See plan §9 for why exact per-model tokenization isn't required for v1:
cross-provider tokenizer differences are typically within 5-10%, which
essentially never changes tier assignment since relevance ranking
dominates the allocation decision far more than a small counting error.
"""

import tiktoken

_encoder = None


def _get_encoder():
    # Lazy-loaded: tiktoken fetches the cl100k_base BPE file from
    # openaipublic.blob.core.windows.net on first use (then caches it
    # locally). Loading lazily means simply importing promptvault
    # doesn't require network access — only actually counting tokens
    # does, and after the first call it's cached on disk.
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def count_tokens(text: str) -> int:
    """Count tokens using the cl100k_base BPE encoding.

    Applied identically everywhere in the pipeline (scoring, budget
    allocation, compression-ratio reporting) so numbers stay internally
    consistent even if they're not exact for every target model's real
    tokenizer (see plan §9).
    """
    return len(_get_encoder().encode(text))
