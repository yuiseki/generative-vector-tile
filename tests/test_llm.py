from unittest.mock import MagicMock, patch

import pytest

from generative_vector_tile.datasets import get_dataset
from generative_vector_tile.llm import (
    FilterTranslation,
    LlmTranslationError,
    LlmUnavailable,
    build_system_prompt,
    translate_q,
)


def _fake_client_returning(where_clause: str | None, refusal: str | None = None) -> MagicMock:
    fake_choice = MagicMock()
    fake_choice.message.parsed = (
        FilterTranslation(where_clause=where_clause) if where_clause is not None else None
    )
    fake_choice.message.refusal = refusal
    fake_response = MagicMock()
    fake_response.choices = [fake_choice]
    fake_client = MagicMock()
    fake_client.chat.completions.parse.return_value = fake_response
    return fake_client


def _reset_client_cache():
    from generative_vector_tile.llm import _client
    _client.cache_clear()


def test_build_system_prompt_lists_filterable_columns():
    ds = get_dataset("buildings")
    prompt = build_system_prompt(ds)
    assert "buildings" in prompt
    assert "height" in prompt
    assert "class" in prompt
    # geometry column should not appear in the filterable list
    assert "geom (geometry)" not in prompt


def test_build_system_prompt_includes_japanese_aliases():
    ds = get_dataset("buildings")
    prompt = build_system_prompt(ds)
    assert "高さ" in prompt
    assert "→ height" in prompt


def test_build_system_prompt_places_has_category():
    ds = get_dataset("places")
    prompt = build_system_prompt(ds)
    assert "category" in prompt
    assert "カテゴリ" in prompt


def test_translate_q_raises_when_no_backend_configured(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    _reset_client_cache()
    ds = get_dataset("buildings")
    with pytest.raises(LlmUnavailable):
        translate_q(ds, "高さ100m以上")


def test_translate_q_works_with_base_url_only(monkeypatch):
    # llama-server style: no API key, just a local endpoint.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:18091/v1")
    _reset_client_cache()
    ds = get_dataset("buildings")
    fake_client = _fake_client_returning("height >= 100")
    with patch("generative_vector_tile.llm.OpenAI", return_value=fake_client):
        sql = translate_q(ds, "高さ100m以上")
    assert sql == "height >= 100"


def test_translate_q_parses_structured_response(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _reset_client_cache()
    ds = get_dataset("buildings")
    fake_client = _fake_client_returning("height >= 100")
    with patch("generative_vector_tile.llm.OpenAI", return_value=fake_client):
        sql = translate_q(ds, "高さ100m以上")
    assert sql == "height >= 100"


def test_translate_q_strips_whitespace(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _reset_client_cache()
    ds = get_dataset("buildings")
    fake_client = _fake_client_returning("  height >= 100  ")
    with patch("generative_vector_tile.llm.OpenAI", return_value=fake_client):
        sql = translate_q(ds, "高さ100m以上")
    assert sql == "height >= 100"


def test_translate_q_empty_output_raises(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _reset_client_cache()
    ds = get_dataset("buildings")
    fake_client = _fake_client_returning("   ")
    with patch("generative_vector_tile.llm.OpenAI", return_value=fake_client), \
            pytest.raises(LlmTranslationError):
        translate_q(ds, "高さ100m以上")


def test_translate_q_refusal_raises(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _reset_client_cache()
    ds = get_dataset("buildings")
    fake_client = _fake_client_returning(None, refusal="I cannot answer")
    with patch("generative_vector_tile.llm.OpenAI", return_value=fake_client), \
            pytest.raises(LlmTranslationError):
        translate_q(ds, "高さ100m以上")


def test_translate_q_provider_exception_raises(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _reset_client_cache()
    ds = get_dataset("buildings")
    fake_client = MagicMock()
    fake_client.chat.completions.parse.side_effect = TimeoutError("LLM took too long")
    with patch("generative_vector_tile.llm.OpenAI", return_value=fake_client), \
            pytest.raises(LlmTranslationError):
        translate_q(ds, "高さ100m以上")


def test_client_passes_base_url_when_set(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:18091/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    _reset_client_cache()
    fake_client = _fake_client_returning("height >= 100")
    with patch("generative_vector_tile.llm.OpenAI", return_value=fake_client) as openai_ctor:
        translate_q(get_dataset("buildings"), "高さ100m以上")
    openai_ctor.assert_called_once()
    kwargs = openai_ctor.call_args.kwargs
    assert kwargs["base_url"] == "http://127.0.0.1:18091/v1"
    assert kwargs["api_key"] == "dummy"
