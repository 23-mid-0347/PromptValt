"""Core data types shared across the pipeline (see plan §10)."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Turn:
    """A single conversation turn, cached with its token count and
    (lazily computed) embedding."""

    turn_id: int
    role: Literal["user", "assistant"]
    content: str
    token_count: int
    embedding: list[float] | None = None


@dataclass
class CompressionResult:
    """Output of PromptVault.compress() — a report, not just compressed
    text, so the evaluation harness can compute evidence recall and
    rate-distortion curves without re-deriving metadata (see plan §10)."""

    messages: list[dict]
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    tier_assignment: dict[int, str] = field(default_factory=dict)
    dropped_turn_ids: list[int] = field(default_factory=list)
    merged_groups: list[list[int]] = field(default_factory=list)
