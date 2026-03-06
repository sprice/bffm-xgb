"""Tests for lib/norms.py."""

from __future__ import annotations

import json

import pytest

from lib.constants import DOMAINS
from lib.norms import clear_norms_cache, load_mini_ipip_norms, load_norms


def _norm_payload(mean: float = 3.0, sd: float = 0.7) -> dict:
    return {
        "norms": {
            domain: {"mean": mean + i * 0.01, "sd": sd + i * 0.01}
            for i, domain in enumerate(DOMAINS)
        },
        "mini_ipip_norms": {
            domain: {"mean": mean + 0.1 + i * 0.02, "sd": sd + 0.1 + i * 0.02}
            for i, domain in enumerate(DOMAINS)
        },
    }


def test_load_norms_reads_valid_payload(tmp_path) -> None:
    path = tmp_path / "norms.json"
    with open(path, "w") as f:
        json.dump(_norm_payload(), f)

    clear_norms_cache()
    loaded = load_norms(path)
    assert set(loaded.keys()) == set(DOMAINS)
    for domain in DOMAINS:
        assert loaded[domain]["sd"] > 0


def test_load_norms_missing_file_fails_closed(tmp_path) -> None:
    clear_norms_cache()
    with pytest.raises(FileNotFoundError):
        load_norms(tmp_path / "missing.json")


def test_clear_norms_cache_forces_reload(tmp_path) -> None:
    path = tmp_path / "norms.json"
    with open(path, "w") as f:
        json.dump(_norm_payload(mean=3.0), f)

    clear_norms_cache()
    first = load_norms(path)

    with open(path, "w") as f:
        json.dump(_norm_payload(mean=4.0), f)

    second_cached = load_norms(path)
    assert second_cached["ext"]["mean"] == first["ext"]["mean"]

    clear_norms_cache()
    reloaded = load_norms(path)
    assert reloaded["ext"]["mean"] != first["ext"]["mean"]


def test_load_mini_ipip_norms_reads_valid_payload(tmp_path) -> None:
    path = tmp_path / "norms.json"
    with open(path, "w") as f:
        json.dump(_norm_payload(), f)

    clear_norms_cache()
    loaded = load_mini_ipip_norms(path)
    assert set(loaded.keys()) == set(DOMAINS)
    for domain in DOMAINS:
        assert loaded[domain]["sd"] > 0


def test_load_mini_ipip_norms_missing_block_fails_closed(tmp_path) -> None:
    path = tmp_path / "norms.json"
    with open(path, "w") as f:
        json.dump({"norms": _norm_payload()["norms"]}, f)

    clear_norms_cache()
    with pytest.raises(ValueError):
        load_mini_ipip_norms(path)
