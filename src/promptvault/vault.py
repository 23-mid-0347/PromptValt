"""The PromptVault orchestrator class (plan §10's public API).

Wires the pipeline built in tokenizer.py -> scorer.py -> allocator.py
-> compressor.py -> assembler.py into the single `compress()` entry
point users actually call, plus a `score()` escape hatch for advanced
users who want the relevance scores alone.

PIPELINE ORDER (not made explicit in §10's draft, decided here):
  1. Convert plain {"role", "content"} dicts into Turn objects
     (turn_id = list index, so chronological order = input order).
  2. semantic_dedup() — collapse near-duplicate turns FIRST, so
     relevance scoring only runs once per surviving turn rather than
     once per near-duplicate copy.
  3. RelevanceScorer.score() — score only the deduped survivors.
  4. allocate_tiers() — assign KEEP_RECENT/KEEP_FULL/COMPRESS/DROP
     under the (safety-margin-adjusted) token budget.
  5. assemble() — apply each tier's treatment and reassemble into the
     final chronologically-ordered CompressionResult.

DEVIATIONS FROM §10's DRAFT (documented, not silent):
  - `CompressionResult.tier_assignment` values are the real `Tier`
    strings ("KEEP_RECENT"/"KEEP_FULL"/"COMPRESS"/"DROP"), not the
    draft's placeholder ("full"/"shrunk"/"merged"/"dropped"). There is
    no "merged" tier value: semantic_dedup runs before tiering, so a
    merged-away turn never gets its own tier — only the survivor does.
    Merge history lives entirely in `merged_groups`.
  - `preserve_recent_n` (kept as the public constructor name from the
    draft) maps internally to allocate_tiers's `recency_floor`.
  - The draft's "~90% safety margin" on `token_budget` is implemented
    literally via `budget_safety_margin` (default 0.9): the *effective*
    budget handed to allocate_tiers is `token_budget * budget_safety_margin`,
    reserving headroom for whatever the caller adds afterward (system
    prompt, the model's own response, etc).
  - `as_langgraph_node` is a documented stub (raises NotImplementedError)
    until integrations/langgraph.py is built — the method exists per
    the draft's public shape, but isn't functional yet.
"""

from __future__ import annotations

from collections.abc import Callable

from promptvault.allocator import allocate_tiers
from promptvault.assembler import assemble
from promptvault.compressor import EntropyPruner, semantic_dedup
from promptvault.scorer import RelevanceScorer
from promptvault.tokenizer import count_tokens
from promptvault.types import CompressionResult, Turn


def _history_to_turns(
    history: list[dict], token_counter: Callable[[str], int] = count_tokens
) -> list[Turn]:
    """Convert plain {"role", "content"} dicts into Turn objects.
    turn_id is assigned by list position, so input order IS
    chronological order for every downstream module.

    token_counter is injectable (defaults to the real count_tokens) so
    tests can avoid a real tiktoken dependency, matching the same
    pattern used throughout scorer.py/compressor.py/assembler.py.
    """
    turns = []
    for i, msg in enumerate(history):
        try:
            role, content = msg["role"], msg["content"]
        except KeyError as e:
            raise ValueError(
                f"history[{i}] is missing required key {e}; "
                'each entry must be {"role": ..., "content": ...}'
            ) from e
        turns.append(Turn(turn_id=i, role=role, content=content, token_count=token_counter(content)))
    return turns


class PromptVault:
    """Query-aware relevance compression for LLM conversation history.

    See plan §10 for the original API sketch and this module's
    docstring for the pipeline order and documented deviations from
    that draft.
    """

    def __init__(
        self,
        token_budget: int = 4000,
        embedding_model: str = "all-MiniLM-L6-v2",
        semantic_weight: float = 0.6,
        lexical_weight: float = 0.4,
        preserve_recent_n: int = 2,
        compression_ratio: float = 0.3,
        max_recency_fraction: float = 0.5,
        dedup_similarity_threshold: float = 0.92,
        entropy_model: str = "distilgpt2",
        budget_safety_margin: float = 0.9,
        _scorer: RelevanceScorer | None = None,
        _entropy_pruner: EntropyPruner | None = None,
        _token_counter: Callable[[str], int] | None = None,
    ):
        """
        Args:
            token_budget: target ceiling on compressed output tokens.
            embedding_model: sentence-transformers model name for the
                relevance scorer's semantic component.
            semantic_weight, lexical_weight: RelevanceScorer's hybrid
                scoring weights (plan §2.1).
            preserve_recent_n: number of most recent turns always kept
                verbatim regardless of relevance (maps to
                allocate_tiers's `recency_floor` — see module note).
            compression_ratio: estimated/target fraction of a turn's
                size after compression (plan §2, used consistently by
                both allocate_tiers and assemble).
            max_recency_fraction: caps how much of the budget the
                recency floor may claim (allocate_tiers's starvation
                guard — see allocator.py).
            dedup_similarity_threshold: cosine similarity above which
                turns are treated as near-duplicates (semantic_dedup).
            entropy_model: causal LM used for entropy-based token
                pruning (compressor.py's EntropyPruner).
            budget_safety_margin: fraction of `token_budget` actually
                used as the allocation ceiling, in (0, 1] — see module
                docstring's note on the draft's "~90% safety margin".
            _scorer, _entropy_pruner: advanced/testing hook to inject
                pre-configured instances (e.g. with a fake embedder in
                tests) instead of the real defaults above. Not part of
                the documented public API surface.
            _token_counter: advanced/testing hook to inject an offline
                token-counting function instead of the real
                tiktoken-based count_tokens. Not part of the
                documented public API surface.
        """
        if not 0 < budget_safety_margin <= 1:
            raise ValueError("budget_safety_margin must be in (0, 1]")

        self.token_budget = token_budget
        self.preserve_recent_n = preserve_recent_n
        self.compression_ratio = compression_ratio
        self.max_recency_fraction = max_recency_fraction
        self.dedup_similarity_threshold = dedup_similarity_threshold
        self.budget_safety_margin = budget_safety_margin

        self._scorer = _scorer or RelevanceScorer(
            embedding_model=embedding_model,
            semantic_weight=semantic_weight,
            lexical_weight=lexical_weight,
        )
        self._entropy_pruner = _entropy_pruner or EntropyPruner(model_name=entropy_model)
        self._token_counter = _token_counter or count_tokens

    def score(self, history: list[dict], query: str) -> dict[int, float]:
        """Exposes the relevance scorer alone — useful for debugging,
        the evaluation harness, and advanced users who want custom
        allocation logic instead of the built-in one.

        Returns turn_id -> relevance score, where turn_id is the
        position of that message in `history` (0-indexed).
        """
        turns = _history_to_turns(history, self._token_counter)
        return self._scorer.score(turns, query)

    def compress(self, history: list[dict], query: str) -> CompressionResult:
        """Pattern A entry point. `history`/message dicts use the same
        {"role", "content"} shape every LLM API already uses, so
        there's zero translation layer on either side.

        See this module's docstring for the exact pipeline order.
        """
        turns = _history_to_turns(history, self._token_counter)

        deduped_turns, merged_groups = semantic_dedup(
            turns,
            embedder=self._scorer.embedder,
            similarity_threshold=self.dedup_similarity_threshold,
        )

        scores = self._scorer.score(deduped_turns, query)

        effective_budget = int(self.token_budget * self.budget_safety_margin)
        tiers = allocate_tiers(
            deduped_turns,
            scores,
            token_budget=effective_budget,
            recency_floor=self.preserve_recent_n,
            compression_ratio=self.compression_ratio,
            max_recency_fraction=self.max_recency_fraction,
        )

        return assemble(
            deduped_turns,
            tiers,
            query,
            self._scorer,
            self._entropy_pruner,
            merged_groups=merged_groups,
            compression_ratio=self.compression_ratio,
            token_counter=self._token_counter,
        )

    def as_langgraph_node(self, query_key: str = "messages"):
        """Returns a ready-to-use LangGraph node implementing Pattern B
        (plan §6.1).

        NOT YET IMPLEMENTED — this is a documented stub matching §10's
        public API shape; the real implementation lands with
        integrations/langgraph.py.
        """
        raise NotImplementedError(
            "as_langgraph_node is planned but not yet implemented "
            "(see integrations/langgraph.py in the project plan's next steps)"
        )