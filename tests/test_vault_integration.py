"""Full end-to-end test of PromptVault with real models on every
stage — the real RelevanceScorer, real EntropyPruner (distilgpt2),
and real semantic_dedup embedder. Requires network access to
huggingface.co."""

import pytest

from promptvault.vault import PromptVault

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def real_vault() -> PromptVault:
    vault = PromptVault(token_budget=200, preserve_recent_n=1)
    try:
        vault.score([{"role": "user", "content": "warmup"}], "warmup")
    except Exception as e:  # noqa: BLE001 - network failure types vary by library
        pytest.skip(f"real models unavailable in this environment: {e}")
    return vault


def test_end_to_end_compress_with_real_models(real_vault):
    history = [
        {"role": "user", "content": "I'd like to request a refund for order #4521."},
        {"role": "assistant", "content": "Sure, let me look into that for you."},
        {"role": "user", "content": "By the way, the weather has been lovely lately."},
        {"role": "assistant", "content": "It really has! Great weekend for a walk."},
        {"role": "user", "content": "So what's the status of my refund request?"},
    ]
    result = real_vault.compress(history, "what is my refund status")

    assert result.messages
    assert result.original_tokens > 0
    assert result.compressed_tokens <= result.original_tokens
    assert 0 <= result.compression_ratio <= 1

    # The refund-related turns should be represented somewhere in the
    # final output; the weather aside is the clear candidate for
    # compression or dropping given the query.
    combined = " ".join(m["content"] for m in result.messages).lower()
    assert "refund" in combined