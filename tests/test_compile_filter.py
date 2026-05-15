from unittest.mock import patch

import pytest

from generative_vector_tile.cache import FilterCache
from generative_vector_tile.datasets import get_dataset
from generative_vector_tile.filters import CompileFilterError, compile_filter
from generative_vector_tile.llm import LlmTranslationError, LlmUnavailable


def test_empty_q_returns_none():
    ds = get_dataset("buildings")
    assert compile_filter(ds, None) is None
    assert compile_filter(ds, "") is None
    assert compile_filter(ds, "   ") is None


def test_llm_called_on_cache_miss():
    ds = get_dataset("buildings")
    cache = FilterCache(maxsize=10)
    with patch(
        "generative_vector_tile.filters.translate_q",
        return_value="height >= 100",
    ) as mock_translate:
        sql = compile_filter(ds, "高さ100m以上", cache=cache)
    assert sql == "height >= 100"
    mock_translate.assert_called_once_with(ds, "高さ100m以上")


def test_cache_hit_avoids_llm():
    ds = get_dataset("buildings")
    cache = FilterCache(maxsize=10)
    cache.put(ds.id, "高さ100m以上", "height >= 100")
    with patch("generative_vector_tile.filters.translate_q") as mock_translate:
        sql = compile_filter(ds, "高さ100m以上", cache=cache)
    assert sql == "height >= 100"
    mock_translate.assert_not_called()


def test_llm_result_is_cached():
    ds = get_dataset("buildings")
    cache = FilterCache(maxsize=10)
    with patch(
        "generative_vector_tile.filters.translate_q",
        return_value="height >= 100",
    ) as mock_translate:
        compile_filter(ds, "高さ100m以上", cache=cache)
        compile_filter(ds, "高さ100m以上", cache=cache)
    # Second call hits cache, so translate_q is only invoked once.
    assert mock_translate.call_count == 1


def test_llm_unavailable_yields_compile_filter_error():
    ds = get_dataset("buildings")
    cache = FilterCache(maxsize=10)
    with patch(
        "generative_vector_tile.filters.translate_q",
        side_effect=LlmUnavailable("no key"),
    ), pytest.raises(CompileFilterError) as exc_info:
        compile_filter(ds, "高さ100m以上", cache=cache)
    assert isinstance(exc_info.value.__cause__, LlmUnavailable)


def test_llm_translation_error_yields_compile_filter_error():
    ds = get_dataset("buildings")
    cache = FilterCache(maxsize=10)
    with patch(
        "generative_vector_tile.filters.translate_q",
        side_effect=LlmTranslationError("timeout"),
    ), pytest.raises(CompileFilterError) as exc_info:
        compile_filter(ds, "高さ100m以上", cache=cache)
    assert isinstance(exc_info.value.__cause__, LlmTranslationError)


def test_failed_translation_is_not_cached():
    ds = get_dataset("buildings")
    cache = FilterCache(maxsize=10)
    with patch(
        "generative_vector_tile.filters.translate_q",
        side_effect=LlmTranslationError("timeout"),
    ), pytest.raises(CompileFilterError):
        compile_filter(ds, "高さ100m以上", cache=cache)
    assert cache.get(ds.id, "高さ100m以上") is None
