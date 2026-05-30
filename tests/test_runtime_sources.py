"""Source-level drift guards for the duplicated TypeScript runtime files.

The standalone TypeScript package (`typescript/`) and the web server
(`web/src/server/`) are isolated npm packages, so the shared error-function /
CDF implementation is kept as a verbatim copy in each rather than a
cross-package import. These tests fail if the copies drift apart, which would
silently break cross-runtime numerical parity.
"""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def _read(rel: str) -> str:
    return (PACKAGE_ROOT / rel).read_text(encoding="utf-8")


def test_erf_module_is_byte_identical_across_packages():
    ts = _read("typescript/erf.ts")
    web = _read("web/src/server/erf.ts")
    assert ts == web, (
        "typescript/erf.ts and web/src/server/erf.ts have drifted. They must be "
        "verbatim copies so the TS and web runtimes use the identical CDF."
    )


def test_erf_test_is_byte_identical_across_packages():
    ts = _read("typescript/erf.test.ts")
    web = _read("web/src/server/erf.test.ts")
    assert ts == web, (
        "typescript/erf.test.ts and web/src/server/erf.test.ts have drifted; "
        "keep the erf reference-table tests in sync across packages."
    )


def test_no_abramowitz_stegun_approximation_remains():
    """The A&S polynomial was replaced by the exact erf; guard the regression."""
    for rel in ("typescript/inference.ts", "web/src/server/predictor.ts"):
        src = _read(rel)
        assert "0.2316419" not in src, (
            f"{rel} still contains the Abramowitz-Stegun CDF constant; it should "
            "import standardNormalCDF from erf.ts instead."
        )
        assert 'from "./erf.js"' in src, f"{rel} should import the shared erf module"
