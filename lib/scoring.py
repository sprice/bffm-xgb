"""
Scoring utilities for raw-score to percentile conversion.

Converts raw IPIP-BFFM domain scores (1-5 scale) to percentiles (0-100)
using z-score transformation with committed norms from stage 03.
"""

import numpy as np
import scipy.special

from .norms import load_norms


def raw_score_to_percentile(raw_score, domain, norms=None):
    """Convert raw domain score (1-5 scale) to percentile (0-100).

    Uses z-score transformation with stage-03 norms.

    Args:
        raw_score: Raw score(s) on 1-5 scale (float or ndarray)
        domain: Domain key (ext, agr, csn, est, opn)
        norms: Optional pre-loaded norms dict ({domain: {mean, sd}})

    Returns:
        Percentile(s) on 0-100 scale
    """
    norms_map = norms if norms is not None else load_norms()
    norm = norms_map[domain]
    z = (raw_score - norm["mean"]) / norm["sd"]
    return np.clip(0.5 * (1.0 + scipy.special.erf(z / np.sqrt(2.0))) * 100, 0, 100)
