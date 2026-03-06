#!/usr/bin/env python3
"""Generate publication-quality figures for the IPIP-BFFM adaptive assessment paper.

Reads data from JSON/CSV artifacts and saves figures to figures/.

Figures produced:
1. Efficiency Curves -- r vs K for all item selection strategies
2. Domain Starvation Heatmap -- items per domain at K=5,10,15,20 by strategy
3. ML vs Simple Averaging -- scoring method comparison
4. Per-Domain Accuracy at K=20 -- bar chart of domain-level r

Usage:
    python pipeline/12_generate_figures.py
    python pipeline/12_generate_figures.py --artifacts-dir artifacts/variants/reference --output-dir figures/
"""

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

import argparse
import json
import logging
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from lib.constants import DOMAINS
from lib.provenance import build_provenance, relative_to_root, file_sha256

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared style helpers
# ---------------------------------------------------------------------------

# Colorblind-friendly palette (Wong, 2011 -- Nature Methods)
COLORS = {
    "domain_balanced": "#0072B2",  # blue
    "random": "#009E73",  # green
    "first_n": "#E69F00",  # orange
    "adaptive_topk": "#D55E00",  # vermillion
    "greedy_balanced": "#CC79A7",  # reddish purple
    "worst_k": "#7F7F7F",  # neutral gray (negative control)
    "mini_ipip": "#56B4E9",  # sky blue
}

STRATEGY_LABELS = {
    "domain_balanced": "Domain-Balanced (correlation-ranked)",
    "random": "Random",
    "first_n": "First-N (item order)",
    "adaptive_topk": "Greedy Top-K (correlation utility)",
    "greedy_balanced": "Greedy-Balanced (coverage-constrained)",
    "worst_k": "Worst-K (lowest own-domain correlation)",
    "mini_ipip": "Mini-IPIP",
}

STRATEGY_ORDER = [
    "domain_balanced",
    "first_n",
    "random",
    "greedy_balanced",
    "adaptive_topk",
    "worst_k",
]

DOMAIN_LABELS_SHORT = {
    "ext": "Extraversion",
    "agr": "Agreeableness",
    "csn": "Conscientiousness",
    "est": "Emot. Stability",
    "opn": "Intellect",
}

DOMAIN_ORDER = ["ext", "agr", "csn", "est", "opn"]
EXPECTED_K_VALUES = [5, 10, 15, 20, 25, 30, 40, 50]


def require_file(path: Path, label: str) -> None:
    """Raise a clear error if an expected artifact file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required {label}: {path}")


def _resolve_model_dir(model_dir_raw: str) -> Path:
    """Resolve provenance model_dir to an absolute canonical Path."""
    model_dir = Path(model_dir_raw)
    if not model_dir.is_absolute():
        model_dir = PACKAGE_ROOT / model_dir
    return model_dir.resolve()


def _extract_model_dir_from_payload(payload: dict[str, Any], label: str) -> Path:
    """Extract and validate provenance.model_dir from an artifact payload."""
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{label} missing required provenance object.")
    model_dir_raw = provenance.get("model_dir")
    if not isinstance(model_dir_raw, str) or not model_dir_raw.strip():
        raise ValueError(f"{label} missing required provenance.model_dir.")
    return _resolve_model_dir(model_dir_raw)


def _extract_provenance(payload: dict[str, Any], label: str) -> dict[str, Any]:
    """Extract provenance dict and validate required shape."""
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{label} missing required provenance object.")
    if not isinstance(provenance.get("model_dir"), str):
        raise ValueError(f"{label} missing required provenance.model_dir.")
    return provenance


def _provenance_signature(provenance: dict[str, Any]) -> str:
    """Canonical run signature for strict same-run figure artifact matching."""
    return json.dumps(provenance, sort_keys=True, separators=(",", ":"), default=str)


def _assert_common_model_dir(model_dirs: dict[str, Path]) -> Path:
    """Ensure all required artifacts point to the same model_dir."""
    unique = sorted({str(path.resolve()) for path in model_dirs.values()})
    if len(unique) != 1:
        details = ", ".join(f"{name}={path}" for name, path in model_dirs.items())
        raise ValueError(
            "Artifact provenance mismatch across figure inputs. "
            f"Expected a single model_dir, got: {details}. Re-run make validate/make baselines."
        )
    return Path(unique[0])


def _assert_common_run_signature(signatures: dict[str, str]) -> str:
    """Ensure all required artifacts share identical provenance payloads."""
    unique = sorted(set(signatures.values()))
    if len(unique) != 1:
        details = ", ".join(signatures.keys())
        raise ValueError(
            "Artifact provenance mismatch across figure inputs (same model_dir but different run metadata). "
            f"Inputs: {details}. Re-run make baselines/make validate together."
        )
    return unique[0]


def _apply_base_style() -> None:
    """Apply clean academic plot style."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "0.8",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def load_baseline_results(artifacts_dir: Path) -> tuple[dict, Path, str]:
    """Load baseline comparison results JSON with strict provenance."""
    path = artifacts_dir / "baseline_comparison_results.json"
    require_file(path, "baseline comparison results JSON")
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("baseline_comparison_results.json must contain a JSON object.")
    label = "baseline_comparison_results.json"
    model_dir = _extract_model_dir_from_payload(payload, label)
    provenance = _extract_provenance(payload, label)
    return payload, model_dir, _provenance_signature(provenance)


def load_per_domain_csv(artifacts_dir: Path) -> tuple[pd.DataFrame, Path, str]:
    """Load per-domain comparison CSV with strict provenance sidecar checks."""
    csv_path = artifacts_dir / "baseline_comparison_per_domain.csv"
    meta_path = artifacts_dir / "baseline_comparison_per_domain.meta.json"
    require_file(csv_path, "per-domain comparison CSV")
    require_file(meta_path, "per-domain comparison CSV metadata")

    with open(meta_path) as f:
        meta_payload = json.load(f)
    if not isinstance(meta_payload, dict):
        raise ValueError("baseline_comparison_per_domain.meta.json must be a JSON object.")
    label = "baseline_comparison_per_domain.meta.json"
    model_dir = _extract_model_dir_from_payload(meta_payload, label)
    provenance = _extract_provenance(meta_payload, label)

    artifact_meta = meta_payload.get("artifact")
    if not isinstance(artifact_meta, dict):
        raise ValueError(
            "baseline_comparison_per_domain.meta.json missing required artifact metadata."
        )
    expected_sha = artifact_meta.get("sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError(
            "baseline_comparison_per_domain.meta.json missing valid artifact.sha256."
        )
    actual_sha = file_sha256(csv_path)
    if actual_sha.lower() != expected_sha.lower():
        raise ValueError(
            "baseline_comparison_per_domain.csv SHA-256 mismatch vs metadata sidecar. "
            "Re-run make baselines."
        )

    df = pd.read_csv(csv_path)
    expected_rows = artifact_meta.get("n_rows")
    if isinstance(expected_rows, int) and len(df) != expected_rows:
        raise ValueError(
            "baseline_comparison_per_domain.csv row count mismatch vs metadata sidecar. "
            "Re-run make baselines."
        )
    expected_cols = artifact_meta.get("n_columns")
    if isinstance(expected_cols, int) and len(df.columns) != expected_cols:
        raise ValueError(
            "baseline_comparison_per_domain.csv column count mismatch vs metadata sidecar. "
            "Re-run make baselines."
        )
    return df, model_dir, _provenance_signature(provenance)


def load_item_selection_csv(artifacts_dir: Path) -> pd.DataFrame | None:
    """Load adaptive_topk item selection CSV (if present)."""
    path = artifacts_dir / "adaptive_topk_item_selection.csv"
    if not path.exists():
        log.warning("Item selection CSV not found at %s — skipping related figures.", path)
        return None
    return pd.read_csv(path)


def load_ml_vs_avg(artifacts_dir: Path) -> tuple[dict, Path, str]:
    """Load ML vs averaging comparison JSON with strict provenance."""
    path = artifacts_dir / "ml_vs_averaging_comparison.json"
    require_file(path, "ML vs averaging comparison JSON")
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("ml_vs_averaging_comparison.json must contain a JSON object.")
    label = "ml_vs_averaging_comparison.json"
    model_dir = _extract_model_dir_from_payload(payload, label)
    provenance = _extract_provenance(payload, label)
    return payload, model_dir, _provenance_signature(provenance)


def require_pearson_ci(entry: dict, strategy: str, k: int) -> tuple[float, float]:
    """Require bootstrap CI fields for Figure 1 publication output."""
    ci = entry.get("pearson_r_ci")
    if not (
        isinstance(ci, list)
        and len(ci) == 2
        and all(isinstance(v, (int, float)) for v in ci)
    ):
        raise ValueError(
            "Missing pearson_r_ci for "
            f"strategy={strategy!r}, k={k}. "
            "Re-run baselines with --bootstrap-n 1000 before generating paper figures."
        )
    return float(ci[0]), float(ci[1])


# ---------------------------------------------------------------------------
# Figure 1: Efficiency Curves (r vs K)
# ---------------------------------------------------------------------------


def figure_1_efficiency_curves(data: dict, fig_dir: Path) -> None:
    """Line plot of overall Pearson r vs number of items for each strategy."""
    log.info("  Generating Figure 1: Efficiency Curves...")
    _apply_base_style()
    if "overall" not in data:
        raise KeyError(
            "baseline_comparison_results.json is missing top-level key 'overall'"
        )
    overall = data["overall"]

    k_values = sorted(int(k) for k in overall.keys())
    if k_values != EXPECTED_K_VALUES:
        raise ValueError(
            "baseline_comparison_results.json has unexpected K values: "
            f"{k_values}. Expected {EXPECTED_K_VALUES}."
        )
    available_methods = {
        method
        for k in overall.values()
        for method in k.keys()
    }
    missing_strategies = [s for s in STRATEGY_ORDER if s not in available_methods]
    if missing_strategies:
        raise ValueError(
            "baseline_comparison_results.json missing required strategies for Figure 1: "
            + ", ".join(missing_strategies)
        )
    strategies = STRATEGY_ORDER

    fig, ax = plt.subplots(figsize=(7, 4.8))

    for strat in strategies:
        ks, rs, ci_lo, ci_hi = [], [], [], []
        for k in k_values:
            entry = overall[str(k)].get(strat)
            if entry is None:
                raise ValueError(
                    f"baseline_comparison_results.json missing strategy={strat!r} at K={k}."
                )
            ks.append(k)
            rs.append(entry["pearson_r"])
            ci_lower, ci_upper = require_pearson_ci(entry, strat, k)
            ci_lo.append(ci_lower)
            ci_hi.append(ci_upper)
        ks_arr, rs_arr = np.array(ks), np.array(rs)
        ci_lo_arr, ci_hi_arr = np.array(ci_lo), np.array(ci_hi)

        ax.plot(
            ks_arr,
            rs_arr,
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=COLORS[strat],
            label=STRATEGY_LABELS[strat],
            zorder=3,
        )
        ax.fill_between(
            ks_arr,
            ci_lo_arr,
            ci_hi_arr,
            alpha=0.12,
            color=COLORS[strat],
            zorder=2,
        )

    # Mini-IPIP at K=20
    k20 = overall.get("20")
    if k20 is None:
        raise KeyError(
            "baseline_comparison_results.json is missing key overall['20']"
        )
    mini = k20.get("mini_ipip")
    if not isinstance(mini, dict):
        raise ValueError(
            "baseline_comparison_results.json missing required mini_ipip entry at K=20."
        )
    ax.plot(
        20,
        mini["pearson_r"],
        marker="*",
        markersize=14,
        color=COLORS["mini_ipip"],
        zorder=5,
        label=STRATEGY_LABELS["mini_ipip"],
        markeredgecolor="white",
        markeredgewidth=0.5,
    )

    # North-star target line
    ax.axhline(y=0.90, color="0.45", linestyle="--", linewidth=1.0, zorder=1)
    ax.text(
        5.5, 0.905, "r = 0.90 target", fontsize=8.5, color="0.40", ha="left", va="bottom"
    )

    ax.set_xlabel("Number of items")
    ax.set_ylabel("Pearson r (overall)")
    ax.set_xlim(3, 52)
    ax.set_ylim(0.50, 1.02)
    ax.set_xticks([5, 10, 15, 20, 25, 30, 40, 50])
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.025))

    # Light horizontal grid on major y ticks only
    ax.yaxis.grid(True, which="major", linewidth=0.4, color="0.85", zorder=0)

    ax.legend(loc="lower right", frameon=True)
    ax.set_title("Assessment Efficiency: Accuracy vs. Number of Items", pad=10)

    for fmt in ("png", "pdf"):
        fig.savefig(fig_dir / f"fig1_efficiency_curves.{fmt}")
    plt.close(fig)
    log.info("    Saved fig1_efficiency_curves.png/.pdf")


# ---------------------------------------------------------------------------
# Figure 2: Domain Starvation Heatmap
# ---------------------------------------------------------------------------


def figure_2_domain_starvation(df: pd.DataFrame, fig_dir: Path) -> None:
    """Heatmap showing items per domain for each strategy at K=5,10,15,20."""
    log.info("  Generating Figure 2: Domain Starvation Heatmap...")
    _apply_base_style()

    k_vals = [5, 10, 15, 20]
    available_methods = set(df["method"].dropna().unique())
    missing_strategies = [s for s in STRATEGY_ORDER if s not in available_methods]
    if missing_strategies:
        raise ValueError(
            "Per-domain CSV missing required strategies for Figure 2: "
            + ", ".join(missing_strategies)
        )
    strategies = STRATEGY_ORDER
    domains = DOMAIN_ORDER

    # CSV has columns like items_Extraversion, items_Agreeableness, etc.
    domain_col_map = {
        "ext": "items_Extraversion",
        "agr": "items_Agreeableness",
        "csn": "items_Conscientiousness",
        "est": "items_EmotionalStability",
        "opn": "items_Intellect",
    }
    missing_columns = [col for col in domain_col_map.values() if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "Per-domain CSV missing required columns for Figure 2: "
            + ", ".join(missing_columns)
        )

    # Create a panel: one sub-heatmap per K value
    fig, axes = plt.subplots(1, len(k_vals), figsize=(8.5, 3.2), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    vmin, vmax = 0, 10

    for idx, k in enumerate(k_vals):
        ax = axes[idx]
        sub = df[df["n_items"] == k]
        if sub.empty:
            raise ValueError(
                f"Per-domain CSV missing required rows for n_items={k}. Re-run make baselines."
            )
        matrix = np.zeros((len(strategies), len(domains)))
        for si, strat in enumerate(strategies):
            row = sub[sub["method"] == strat]
            if row.empty:
                raise ValueError(
                    f"Per-domain CSV missing row for n_items={k}, method={strat}. "
                    "Re-run make baselines."
                )
            if len(row) != 1:
                raise ValueError(
                    f"Per-domain CSV must contain exactly one row for n_items={k}, "
                    f"method={strat}; found {len(row)}."
                )
            for di, dom in enumerate(domains):
                col_name = domain_col_map[dom]
                matrix[si, di] = row[col_name].values[0]

        im = ax.imshow(
            matrix, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax
        )

        # Annotate cells
        for si in range(len(strategies)):
            for di in range(len(domains)):
                val = float(matrix[si, di])
                if np.isclose(val, round(val)):
                    label = f"{int(round(val))}"
                else:
                    label = f"{val:.1f}"
                text_color = "white" if val >= 6 else "black"
                ax.text(
                    di,
                    si,
                    label,
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color=text_color,
                )

        ax.set_xticks(range(len(domains)))
        ax.set_xticklabels(
            [DOMAIN_LABELS_SHORT[d][:4] for d in domains], fontsize=8
        )
        ax.set_title(f"K = {k}", fontsize=10, pad=4)

        if idx == 0:
            ax.set_yticks(range(len(strategies)))
            ax.set_yticklabels(
                [STRATEGY_LABELS[s].split(" (")[0] for s in strategies],
                fontsize=8.5,
            )
        else:
            ax.set_yticks([])

        # Remove spines for heatmap
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)

    # Colorbar
    cbar_ax = fig.add_axes([0.93, 0.18, 0.015, 0.65])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Items", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle("Domain Coverage by Item Selection Strategy", fontsize=12, y=1.02)

    for fmt in ("png", "pdf"):
        fig.savefig(fig_dir / f"fig2_domain_starvation.{fmt}")
    plt.close(fig)
    log.info("    Saved fig2_domain_starvation.png/.pdf")


# ---------------------------------------------------------------------------
# Figure 3: ML vs Simple Averaging
# ---------------------------------------------------------------------------


def figure_3_ml_vs_averaging(data: dict, fig_dir: Path) -> None:
    """Grouped bar chart comparing ML scoring vs simple averaging."""
    log.info("  Generating Figure 3: ML vs Simple Averaging...")
    _apply_base_style()
    if "comparisons" not in data:
        raise KeyError(
            "ml_vs_averaging_comparison.json is missing key 'comparisons'"
        )
    comps = data["comparisons"]
    if not isinstance(comps, list):
        raise ValueError(
            "ml_vs_averaging_comparison.json key 'comparisons' must be a list."
        )

    expected_pairs = [
        ("domain_balanced", 10),
        ("domain_balanced", 15),
        ("domain_balanced", 20),
        ("domain_balanced", 25),
        ("mini_ipip", 20),
    ]
    expected_pair_set = set(expected_pairs)

    entries_by_pair: dict[tuple[str, int], dict[str, Any]] = {}
    for idx, c in enumerate(comps):
        if not isinstance(c, dict):
            raise ValueError(
                "ml_vs_averaging_comparison.json comparisons entries must be objects "
                f"(invalid index {idx})."
            )

        method = c.get("method")
        n_items_raw = c.get("n_items")
        if not isinstance(method, str) or not method.strip():
            raise ValueError(
                "ml_vs_averaging_comparison.json has comparison entry with invalid method "
                f"at index {idx}."
            )
        if isinstance(n_items_raw, bool) or not isinstance(n_items_raw, (int, float)):
            raise ValueError(
                "ml_vs_averaging_comparison.json has comparison entry with invalid n_items "
                f"for method={method!r}."
            )
        n_items = int(n_items_raw)
        if float(n_items_raw) != float(n_items):
            raise ValueError(
                "ml_vs_averaging_comparison.json has non-integer n_items for "
                f"method={method!r}: {n_items_raw!r}."
            )

        pair = (method, n_items)
        if pair not in expected_pair_set:
            raise ValueError(
                "ml_vs_averaging_comparison.json contains unexpected comparison "
                f"method={method!r}, n_items={n_items}. "
                f"Expected exactly: {expected_pairs}."
            )
        if pair in entries_by_pair:
            raise ValueError(
                "ml_vs_averaging_comparison.json contains duplicate comparison "
                f"method={method!r}, n_items={n_items}."
            )

        numeric_values: dict[str, float] = {}
        for key in ("ml_r", "avg_r", "delta_r"):
            raw = c.get(key)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(
                    "ml_vs_averaging_comparison.json has invalid numeric field "
                    f"{key!r} for method={method!r}, n_items={n_items}."
                )
            value = float(raw)
            if not np.isfinite(value):
                raise ValueError(
                    "ml_vs_averaging_comparison.json has non-finite numeric field "
                    f"{key!r} for method={method!r}, n_items={n_items}."
                )
            numeric_values[key] = value

        label_base = STRATEGY_LABELS.get(method, method).split(" (")[0]
        entries_by_pair[pair] = {
            "label": f"{label_base}\n({n_items} items)",
            "ml_r": numeric_values["ml_r"],
            "avg_r": numeric_values["avg_r"],
            "delta": numeric_values["delta_r"],
            "n_items": n_items,
            "method": method,
        }

    missing_pairs = [pair for pair in expected_pairs if pair not in entries_by_pair]
    if missing_pairs:
        raise ValueError(
            "ml_vs_averaging_comparison.json is missing required comparisons: "
            + ", ".join(f"{m}:{k}" for m, k in missing_pairs)
        )

    entries_ordered = [entries_by_pair[pair] for pair in expected_pairs]

    labels = [e["label"] for e in entries_ordered]
    ml_rs = [e["ml_r"] for e in entries_ordered]
    avg_rs = [e["avg_r"] for e in entries_ordered]
    deltas = [e["delta"] for e in entries_ordered]

    x = np.arange(len(labels))
    bar_w = 0.32

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.bar(
        x - bar_w / 2,
        ml_rs,
        bar_w,
        label="XGBoost ML Scoring",
        color="#0072B2",
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax.bar(
        x + bar_w / 2,
        avg_rs,
        bar_w,
        label="Simple Scale Averaging",
        color="#E69F00",
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )

    # Delta annotations above each pair
    for i, delta in enumerate(deltas):
        top = max(ml_rs[i], avg_rs[i])
        ax.annotate(
            f"{delta:+.3f}",
            xy=(x[i], top + 0.003),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#333333",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Pearson r (overall)")
    ax.set_ylim(0.82, 0.96)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.02))
    ax.yaxis.grid(True, which="major", linewidth=0.4, color="0.85", zorder=0)

    ax.legend(loc="upper left", frameon=True)
    ax.set_title("ML Scoring Advantage: XGBoost vs. Simple Averaging", pad=10)

    for fmt in ("png", "pdf"):
        fig.savefig(fig_dir / f"fig3_ml_vs_averaging.{fmt}")
    plt.close(fig)
    log.info("    Saved fig3_ml_vs_averaging.png/.pdf")


# ---------------------------------------------------------------------------
# Figure 4: Per-Domain Accuracy at K=20
# ---------------------------------------------------------------------------


def figure_4_per_domain_k20(df: pd.DataFrame, fig_dir: Path) -> None:
    """Grouped bar chart of per-domain r for selected strategies at K=20."""
    log.info("  Generating Figure 4: Per-Domain Accuracy at K=20...")
    _apply_base_style()
    sub = df[df["n_items"] == 20]
    if sub.empty:
        raise ValueError("Per-domain comparison CSV has no rows for n_items == 20.")

    strategies_to_show = ["domain_balanced", "mini_ipip", "adaptive_topk"]
    strategy_colors = {
        "domain_balanced": "#0072B2",
        "mini_ipip": "#56B4E9",
        "adaptive_topk": "#D55E00",
    }
    strategy_short = {
        "domain_balanced": "Domain-Balanced",
        "mini_ipip": "Mini-IPIP",
        "adaptive_topk": "Greedy Top-K",
    }

    domain_r_cols = {
        "ext": "r_Extraversion",
        "agr": "r_Agreeableness",
        "csn": "r_Conscientiousness",
        "est": "r_EmotionalStability",
        "opn": "r_Intellect",
    }
    missing_domain_cols = [col for col in domain_r_cols.values() if col not in sub.columns]
    if missing_domain_cols:
        raise ValueError(
            "Per-domain CSV missing required columns for Figure 4: "
            + ", ".join(missing_domain_cols)
        )

    # Also try to get CI columns for error bars
    domain_r_ci_lower_cols = {
        "ext": "r_Extraversion_ci_lower",
        "agr": "r_Agreeableness_ci_lower",
        "csn": "r_Conscientiousness_ci_lower",
        "est": "r_EmotionalStability_ci_lower",
        "opn": "r_Intellect_ci_lower",
    }
    domain_r_ci_upper_cols = {
        "ext": "r_Extraversion_ci_upper",
        "agr": "r_Agreeableness_ci_upper",
        "csn": "r_Conscientiousness_ci_upper",
        "est": "r_EmotionalStability_ci_upper",
        "opn": "r_Intellect_ci_upper",
    }

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    n_domains = len(DOMAIN_ORDER)
    n_strats = len(strategies_to_show)
    bar_w = 0.22
    x = np.arange(n_domains)

    for si, strat in enumerate(strategies_to_show):
        row = sub[sub["method"] == strat]
        if row.empty:
            raise ValueError(
                f"Per-domain CSV missing required row for n_items=20, method={strat}. "
                "Re-run make baselines."
            )
        if len(row) != 1:
            raise ValueError(
                f"Per-domain CSV must contain exactly one row for n_items=20, "
                f"method={strat}; found {len(row)}."
            )
        vals = [row[domain_r_cols[d]].values[0] for d in DOMAIN_ORDER]
        offset = (si - (n_strats - 1) / 2) * bar_w

        # Try to compute error bars from bootstrap CIs
        yerr_lower = []
        yerr_upper = []
        has_ci = True
        for d in DOMAIN_ORDER:
            ci_lo_col = domain_r_ci_lower_cols[d]
            ci_hi_col = domain_r_ci_upper_cols[d]
            if ci_lo_col in row.columns and ci_hi_col in row.columns:
                lo = row[ci_lo_col].values[0]
                hi = row[ci_hi_col].values[0]
                r_val = row[domain_r_cols[d]].values[0]
                if pd.notna(lo) and pd.notna(hi):
                    yerr_lower.append(r_val - lo)
                    yerr_upper.append(hi - r_val)
                else:
                    has_ci = False
                    break
            else:
                has_ci = False
                break

        if has_ci:
            yerr = [yerr_lower, yerr_upper]
            ax.bar(
                x + offset,
                vals,
                bar_w,
                label=strategy_short[strat],
                color=strategy_colors[strat],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
                yerr=yerr,
                capsize=2,
                error_kw={"linewidth": 0.8, "capthick": 0.8},
            )
        else:
            ax.bar(
                x + offset,
                vals,
                bar_w,
                label=strategy_short[strat],
                color=strategy_colors[strat],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )

        # Value labels on bars
        for xi, v in zip(x + offset, vals):
            ax.text(
                xi,
                v + 0.008,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="0.3",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [DOMAIN_LABELS_SHORT[d] for d in DOMAIN_ORDER], fontsize=10
    )
    ax.set_ylabel("Pearson r (per domain)")
    ax.set_ylim(0.35, 1.02)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.10))
    ax.yaxis.grid(True, which="major", linewidth=0.4, color="0.85", zorder=0)

    # Highlight Intellect collapse for Greedy Top-K
    topk_row = sub[sub["method"] == "adaptive_topk"]
    if not topk_row.empty and "r_Intellect" in topk_row.columns:
        intellect_topk_r = topk_row["r_Intellect"].values[0]
        n_intellect_items = None
        if "items_Intellect" in topk_row.columns:
            n_raw = topk_row["items_Intellect"].values[0]
            if pd.notna(n_raw):
                n_intellect_items = float(n_raw)
        if n_intellect_items is None:
            items_label = "Intellect items unavailable"
        elif np.isclose(n_intellect_items, round(n_intellect_items)):
            items_label = f"{int(round(n_intellect_items))} Intellect items"
        else:
            items_label = f"{n_intellect_items:.1f} Intellect items"
        # Find the x position for the adaptive_topk bar on Intellect
        si_topk = strategies_to_show.index("adaptive_topk")
        offset_topk = (si_topk - (n_strats - 1) / 2) * bar_w
        bar_x = 4 + offset_topk  # index 4 = opn
        ax.annotate(
            f"r = {intellect_topk_r:.3f}\n({items_label})",
            xy=(bar_x, intellect_topk_r),
            xytext=(4.45, 0.55),
            fontsize=8,
            color="#D55E00",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.2),
            ha="left",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.3", fc="white", ec="#D55E00", alpha=0.9
            ),
        )

    ax.legend(loc="lower left", frameon=True)
    ax.set_title("Per-Domain Accuracy at 20 Items", pad=10)

    for fmt in ("png", "pdf"):
        fig.savefig(fig_dir / f"fig4_per_domain_k20.{fmt}")
    plt.close(fig)
    log.info("    Saved fig4_per_domain_k20.png/.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for the IPIP-BFFM paper"
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=PACKAGE_ROOT / "artifacts" / "variants" / "reference",
        help="Path to artifacts directory (default: artifacts/variants/reference/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "figures",
        help="Path to output figures directory (default: figures/)",
    )
    args = parser.parse_args()

    artifacts_dir = args.artifacts_dir
    fig_dir = args.output_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    log.info("Generating publication figures for IPIP-BFFM paper...")
    log.info("  Artifacts directory: %s", artifacts_dir)
    log.info("  Output directory: %s", fig_dir)
    log.info("")

    baseline_data, baseline_model_dir, baseline_run_sig = load_baseline_results(artifacts_dir)
    per_domain_df, per_domain_model_dir, per_domain_run_sig = load_per_domain_csv(artifacts_dir)
    ml_vs_avg_data, ml_vs_avg_model_dir, ml_vs_avg_run_sig = load_ml_vs_avg(artifacts_dir)

    common_model_dir = _assert_common_model_dir(
        {
            "baseline_comparison_results.json": baseline_model_dir,
            "baseline_comparison_per_domain.meta.json": per_domain_model_dir,
            "ml_vs_averaging_comparison.json": ml_vs_avg_model_dir,
        }
    )
    _assert_common_run_signature(
        {
            "baseline_comparison_results.json": baseline_run_sig,
            "baseline_comparison_per_domain.meta.json": per_domain_run_sig,
            "ml_vs_averaging_comparison.json": ml_vs_avg_run_sig,
        }
    )
    log.info("  Provenance verified: all figure inputs target model_dir=%s", common_model_dir)

    figure_1_efficiency_curves(baseline_data, fig_dir)
    figure_2_domain_starvation(per_domain_df, fig_dir)
    figure_3_ml_vs_averaging(ml_vs_avg_data, fig_dir)
    figure_4_per_domain_k20(per_domain_df, fig_dir)

    # Build and write figures/manifest.json
    log.info("")
    log.info("Writing figures/manifest.json...")

    baseline_results_path = artifacts_dir / "baseline_comparison_results.json"
    per_domain_csv_path = artifacts_dir / "baseline_comparison_per_domain.csv"
    per_domain_meta_path = artifacts_dir / "baseline_comparison_per_domain.meta.json"
    ml_vs_avg_path = artifacts_dir / "ml_vs_averaging_comparison.json"

    source_artifacts: dict[str, dict[str, str]] = {}
    for label, path in [
        ("baseline_comparison_results", baseline_results_path),
        ("baseline_comparison_per_domain_csv", per_domain_csv_path),
        ("baseline_comparison_per_domain_meta", per_domain_meta_path),
        ("ml_vs_averaging_comparison", ml_vs_avg_path),
    ]:
        source_artifacts[label] = {
            "path": relative_to_root(path),
            "sha256": file_sha256(path),
        }

    manifest = {
        "schema_version": 1,
        "provenance": build_provenance(Path(__file__).name),
        "model_dir": relative_to_root(common_model_dir),
        "source_artifacts": source_artifacts,
        "figures": [
            {
                "filename": "fig1_efficiency_curves",
                "formats": ["png", "pdf"],
                "source_artifacts": ["baseline_comparison_results"],
            },
            {
                "filename": "fig2_domain_starvation",
                "formats": ["png", "pdf"],
                "source_artifacts": ["baseline_comparison_per_domain_csv"],
            },
            {
                "filename": "fig3_ml_vs_averaging",
                "formats": ["png", "pdf"],
                "source_artifacts": ["ml_vs_averaging_comparison"],
            },
            {
                "filename": "fig4_per_domain_k20",
                "formats": ["png", "pdf"],
                "source_artifacts": ["baseline_comparison_per_domain_csv"],
            },
        ],
    }

    manifest_path = fig_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("  Saved %s", manifest_path)

    log.info("")
    log.info("Done. All figures saved to %s", fig_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
