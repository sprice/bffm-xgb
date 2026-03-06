"""
IPIP-BFFM Sparse Quantile Model — Python inference module.

Loads a single merged ONNX model and provides prediction from either a dict
of item responses or a raw numpy array.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.stats import norm

DOMAINS = ["ext", "agr", "csn", "est", "opn"]
QUANTILES = ["q05", "q50", "q95"]


class IPIPBFFMPredictor:
    """Predict Big Five personality scores from IPIP-BFFM item responses."""

    def __init__(self, model_dir: str | Path | None = None) -> None:
        """Load config.json and the merged ONNX session.

        Args:
            model_dir: Path to the directory containing model.onnx and config.json.
                       Defaults to the parent of this file's directory.
        """
        if model_dir is None:
            model_dir = Path(__file__).resolve().parent.parent / "output" / "reference"
        self.model_dir = Path(model_dir)

        with open(self.model_dir / "config.json") as f:
            self.config = json.load(f)

        self.feature_names: list[str] = self.config["input"]["feature_names"]
        self._feature_index = {
            name: i for i, name in enumerate(self.feature_names)
        }
        self.norms: dict = self.config["norms"]

        # Load single merged ONNX session
        model_path = self.model_dir / self.config["model_file"]
        self._session = ort.InferenceSession(str(model_path))
        self._output_names: list[str] = self.config["outputs"]

    def predict(self, responses: dict[str, float]) -> dict:
        """Predict from a dict mapping item IDs to response values (1-5).

        Missing items are treated as NaN (unanswered).

        Args:
            responses: e.g. {"ext3": 4.0, "agr1": 3.0, ...}

        Returns:
            Per-domain results with raw scores and percentiles for each quantile.

        Raises:
            ValueError: If any keys in *responses* are not valid item IDs.
        """
        unknown = sorted(set(responses) - set(self._feature_index))
        if unknown:
            valid = ", ".join(self.feature_names)
            raise ValueError(
                f"Unrecognized item IDs: {', '.join(unknown)}. "
                f"Valid IDs: {valid}"
            )

        bad = {
            k: v for k, v in responses.items()
            if not (isinstance(v, (int, float)) and np.isfinite(v) and 1 <= v <= 5)
        }
        if bad:
            raise ValueError(
                f"Response values must be finite numbers in [1, 5]. "
                f"Got: {bad}"
            )

        arr = np.full((1, len(self.feature_names)), np.nan, dtype=np.float32)
        for item_id, value in responses.items():
            idx = self._feature_index.get(item_id)
            if idx is not None:
                arr[0, idx] = value
        return self.predict_array(arr)

    def _calibration_regime(self, responses: np.ndarray) -> str:
        n_answered = int(np.sum(~np.isnan(responses)))
        return "full_50" if n_answered >= 50 else "sparse_20_balanced"

    def predict_array(self, responses: np.ndarray) -> dict:
        """Predict from a raw [1, 50] float32 array.

        Args:
            responses: Shape (1, 50) float32 array. NaN for unanswered items.

        Returns:
            Per-domain results with raw scores and percentiles for each quantile.

        Raises:
            ValueError: If *responses* does not have the expected shape.
        """
        expected_shape = (1, len(self.feature_names))
        if responses.shape != expected_shape:
            raise ValueError(
                f"Expected array with shape {expected_shape}, "
                f"got {responses.shape}"
            )

        # Single session.run() for all 15 outputs
        raw_outputs = self._session.run(
            self._output_names, {"input": responses}
        )
        raw_dict = {
            name: float(output.flatten()[0])
            for name, output in zip(self._output_names, raw_outputs)
        }

        calibration = self.config.get("calibration", {})
        regime = self._calibration_regime(responses[0])
        regime_cal = calibration.get(regime, {})

        results = {}
        for domain in DOMAINS:
            raw_scores = {}
            for q in QUANTILES:
                key = f"{domain}_{q}"
                raw_scores[q] = raw_dict[key]

            # Convert raw scores to percentiles
            n = self.norms[domain]
            pct = {}
            for q in QUANTILES:
                z = (raw_scores[q] - n["mean"]) / n["sd"]
                pct[q] = float(norm.cdf(z) * 100)

            # Sort percentiles so q05 <= q50 <= q95
            pct_vals = sorted([pct["q05"], pct["q50"], pct["q95"]])
            pct["q05"], pct["q50"], pct["q95"] = pct_vals

            scale = float(regime_cal.get(domain, {}).get("scale_factor", 1.0))
            if scale != 1.0:
                half_width = 0.5 * (pct["q95"] - pct["q05"]) * scale
                pct["q05"] = float(np.clip(pct["q50"] - half_width, 0.0, 100.0))
                pct["q95"] = float(np.clip(pct["q50"] + half_width, 0.0, 100.0))

            results[domain] = {
                "raw": {q: round(raw_scores[q], 4) for q in QUANTILES},
                "percentile": {q: round(pct[q], 1) for q in QUANTILES},
            }
        return results
