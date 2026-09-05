import pytest

from promptvault.tokenizer import count_tokens

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _skip_if_tiktoken_unreachable():
    """tiktoken downloads its BPE file from openaipublic.blob.core.windows.net
    on first use (then caches it). Skip gracefully in network-restricted
    sandboxes rather than failing over an unrelated infra limitation."""
    try:
        count_tokens("warmup")
    except Exception as e:  # noqa: BLE001 - network failure types vary by library
        pytest.skip(f"tiktoken encoding file unreachable in this environment: {e}")


def test_empty_string_has_zero_tokens():
    assert count_tokens("") == 0


def test_longer_text_has_more_tokens():
    assert count_tokens("hello world") < count_tokens(
        "hello world, this is a much longer sentence with many more tokens"
    )


def test_deterministic():
    text = "The quick brown fox jumps over the lazy dog."
    assert count_tokens(text) == count_tokens(text)
