from generative_vector_tile.cache import FilterCache


def test_get_miss_returns_none():
    c = FilterCache(maxsize=10)
    assert c.get("buildings", "高さ100m以上") is None


def test_put_then_get_hit():
    c = FilterCache(maxsize=10)
    c.put("buildings", "高さ100m以上", "height >= 100")
    assert c.get("buildings", "高さ100m以上") == "height >= 100"


def test_dataset_isolation():
    c = FilterCache(maxsize=10)
    c.put("buildings", "高さ100m以上", "height >= 100")
    assert c.get("places", "高さ100m以上") is None


def test_q_whitespace_normalised():
    c = FilterCache(maxsize=10)
    c.put("buildings", "高さ100m以上", "height >= 100")
    assert c.get("buildings", "  高さ100m以上  ") == "height >= 100"


def test_lru_eviction():
    c = FilterCache(maxsize=2)
    c.put("buildings", "a", "sql_a")
    c.put("buildings", "b", "sql_b")
    c.put("buildings", "c", "sql_c")
    assert c.get("buildings", "a") is None
    assert c.get("buildings", "b") == "sql_b"
    assert c.get("buildings", "c") == "sql_c"


def test_clear():
    c = FilterCache(maxsize=10)
    c.put("buildings", "a", "sql_a")
    c.clear()
    assert c.get("buildings", "a") is None
    assert len(c) == 0
