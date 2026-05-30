from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform

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


def test_build_provenance_records_environment() -> None:
    """A5.3: every artifact's provenance records the runtime toolchain."""
    prov = provenance.build_provenance("test_script.py")
    env = prov["environment"]
    assert env["python_version"] == platform.python_version()
    assert env["python_implementation"] == platform.python_implementation()
    assert isinstance(env["platform"], str) and env["platform"]
    libs = env["libraries"]
    assert isinstance(libs, dict)
    assert set(libs) == set(provenance._PROVENANCE_PACKAGES)
    # numpy is a hard dependency, so a real version (not None) is captured.
    assert libs["numpy"] == importlib.metadata.version("numpy")
    for value in libs.values():
        assert value is None or isinstance(value, str)


def test_safe_version_returns_none_for_missing_package(monkeypatch) -> None:
    """A5.3: a missing distribution records None rather than raising."""
    def _raise(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(_name)

    monkeypatch.setattr(provenance.importlib_metadata, "version", _raise)
    assert provenance._safe_version("definitely-not-installed") is None


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
