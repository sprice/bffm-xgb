from __future__ import annotations

import argparse
import hashlib
import json

from lib import provenance


def _parse_provenance_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    provenance.add_provenance_args(parser)
    return parser.parse_args(argv)


def test_build_provenance_uses_cli_overrides(monkeypatch) -> None:
    monkeypatch.delenv(provenance.DATA_SNAPSHOT_ID_ENV, raising=False)
    monkeypatch.delenv(provenance.NORMS_PATH_ENV, raising=False)

    args = _parse_provenance_args(
        [
            "--data-snapshot-id",
            "snapshot-custom",
            "--preprocess-tag",
            "prep-v1",
        ]
    )
    prov = provenance.build_provenance("test_script.py", args=args)

    assert prov["data_snapshot_id"] == "snapshot-custom"
    assert prov["preprocessing_version"] == "prep-v1"


def test_build_provenance_defaults_to_git_snapshot_id_without_sources(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(provenance.DATA_SNAPSHOT_ID_ENV, raising=False)
    monkeypatch.delenv(provenance.NORMS_PATH_ENV, raising=False)
    monkeypatch.setattr(
        provenance,
        "_resolve_norms_lock_path",
        lambda: tmp_path / "missing_norms.json",
    )

    prov = provenance.build_provenance("test_script.py")

    assert "data_snapshot_date" not in prov
    assert isinstance(prov["data_snapshot_id"], str)
    assert prov["data_snapshot_id"].startswith("git:")


def test_build_provenance_uses_norms_hash_for_snapshot_id(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(provenance.DATA_SNAPSHOT_ID_ENV, raising=False)
    monkeypatch.delenv(provenance.NORMS_PATH_ENV, raising=False)

    norms_path = tmp_path / "ipip_bffm_norms.json"
    norms_path.write_text(json.dumps({"norms": {}}), encoding="utf-8")
    monkeypatch.setattr(provenance, "_resolve_norms_lock_path", lambda: norms_path)

    expected_sha = hashlib.sha256(norms_path.read_bytes()).hexdigest()
    prov = provenance.build_provenance("test_script.py")

    assert prov["data_snapshot_id"] == f"norms_sha256:{expected_sha}"


def test_build_provenance_honors_norms_env_override_for_snapshot_id(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.delenv(provenance.DATA_SNAPSHOT_ID_ENV, raising=False)

    norms_path = tmp_path / "custom_norms.json"
    norms_path.write_text(json.dumps({"norms": {"note": "custom"}}), encoding="utf-8")
    monkeypatch.setenv(provenance.NORMS_PATH_ENV, str(norms_path))

    expected_sha = hashlib.sha256(norms_path.read_bytes()).hexdigest()
    prov = provenance.build_provenance("test_script.py")

    assert prov["data_snapshot_id"] == f"norms_sha256:{expected_sha}"
