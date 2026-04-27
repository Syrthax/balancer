from __future__ import annotations
import math
from collections import Counter


def _entropy(values: list) -> float:
    """Shannon entropy H(X)."""
    n = len(values)
    if n == 0:
        return 0.0
    counts = Counter(values)
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)


def _mutual_info(x: list, y: list) -> float:
    """Homemade mutual information I(X;Y) = H(X) + H(Y) - H(X,Y)."""
    joint = list(zip(x, y))
    return max(0.0, _entropy(x) + _entropy(y) - _entropy(joint))


def _discretize(values: list[float], bins: int = 5) -> list[int]:
    """Bin continuous values into equal-width buckets."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if lo == hi:
        return [0] * len(values)
    width = (hi - lo) / bins
    return [min(int((v - lo) / width), bins - 1) for v in values]


MI_THRESHOLD = 0.1


def detect_proxies(
    scores_raw: list[dict],
    pair_map: dict[int, dict],
    threshold: float = MI_THRESHOLD,
) -> list[dict]:
    """
    Compute mutual information between each profile feature and demographic group.
    Features with MI > threshold are flagged as potential proxies.
    In a properly constructed dataset all MIs should be ~0 (profiles are identical).
    """
    if not pair_map:
        return []

    profile_fields = [
        "years_experience",
        "github_commits_per_month",
        "patents",
        "open_source_projects",
    ]

    # Build feature vectors and group labels (0=white, 1=black)
    feature_vecs: dict[str, list] = {f: [] for f in profile_fields}
    feature_vecs["num_skills"] = []
    groups: list[int] = []

    for row in scores_raw:
        pid = row["pair_id"]
        if pid not in pair_map:
            continue
        pair = pair_map[pid]
        # Each pair contributes two rows (one per candidate)
        for cand_key, grp_label in [("candidate_a", 0), ("candidate_b", 1)]:
            cand = pair[cand_key]
            for field in profile_fields:
                feature_vecs[field].append(cand[field])
            feature_vecs["num_skills"].append(len(cand.get("skills", [])))
            groups.append(grp_label)

    proxies = []
    for feat, raw_vals in feature_vecs.items():
        if not raw_vals:
            continue
        discretized = _discretize([float(v) for v in raw_vals])
        mi = _mutual_info(discretized, groups)
        if mi > threshold:
            proxies.append({"feature": feat, "mutual_information": round(mi, 4)})

    return proxies
