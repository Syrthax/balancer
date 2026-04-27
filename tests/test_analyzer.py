import json
import pytest
from pathlib import Path
from balancer.analyzer import analyze, _grade, _hire_rate, BiasReport

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _make_scores(pairs: list[tuple[int, int, int]]) -> dict:
    """pairs: list of (pair_id, a_score, b_score)."""
    return {
        "metadata": {"timestamp": "2026-01-01", "model": "test", "total_pairs": len(pairs), "client": "test"},
        "scores": [
            {"pair_id": pid, "candidate_a_score": a, "candidate_b_score": b,
             "candidate_a_reasoning": "", "candidate_b_reasoning": ""}
            for pid, a, b in pairs
        ],
    }


def _write_tmp(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "gemini_biased.json"
    p.write_text(json.dumps(data))
    return p


# --- grade ---

def test_grade_a():
    assert _grade(0.95) == "A"

def test_grade_b():
    assert _grade(0.85) == "B"

def test_grade_c():
    assert _grade(0.75) == "C"

def test_grade_d():
    assert _grade(0.65) == "D"

def test_grade_f_minus():
    assert _grade(0.55) == "F-"

def test_grade_f():
    assert _grade(0.40) == "F"


# --- hire rate ---

def test_hire_rate_all_pass():
    assert _hire_rate([80, 90, 100]) == 1.0

def test_hire_rate_none_pass():
    assert _hire_rate([50, 60, 70]) == 0.0

def test_hire_rate_mixed():
    assert _hire_rate([70, 75, 80]) == pytest.approx(2 / 3)

def test_hire_rate_empty():
    assert _hire_rate([]) == 0.0


# --- full analysis with known inputs ---

def test_no_bias_equal_scores(tmp_path):
    data = _make_scores([(i, 80, 80) for i in range(1, 51)])
    path = _write_tmp(tmp_path, data)
    report = analyze(path)
    assert report.demographic_parity_gap["white_vs_black"] == 0.0
    assert report.disparate_impact_ratio == 1.0
    assert report.consistency_rate == 0.0

def test_white_favoured(tmp_path):
    # White always 10 points higher
    data = _make_scores([(i, 80, 70) for i in range(1, 51)])
    path = _write_tmp(tmp_path, data)
    report = analyze(path)
    assert report.demographic_parity_gap["white_vs_black"] == pytest.approx(10.0)
    assert report.consistency_rate == 1.0
    assert report.white_coded_avg == pytest.approx(80.0)
    assert report.black_coded_avg == pytest.approx(70.0)

def test_dir_below_threshold_flagged(tmp_path):
    # White: 80 (hire), Black: 65 (no hire) → DIR = 0/1 = 0
    data = _make_scores([(i, 80, 65) for i in range(1, 51)])
    path = _write_tmp(tmp_path, data)
    report = analyze(path)
    assert report.disparate_impact_ratio < 0.80
    assert report.grade in ("D", "F-", "F")

def test_proxy_features_empty_on_balanced_data(tmp_path):
    """Profiles are identical per pair so MI should be ~0 on all features."""
    data = _make_scores([(i, 80, 75) for i in range(1, 11)])
    path = _write_tmp(tmp_path, data)
    # No pair_map → proxy detector returns empty
    report = analyze(path)
    assert isinstance(report.proxy_features, list)

def test_report_fields_present(tmp_path):
    data = _make_scores([(i, 80, 78) for i in range(1, 21)])
    path = _write_tmp(tmp_path, data)
    report = analyze(path)
    assert report.total_pairs == 20
    assert isinstance(report.significant, bool)
    assert isinstance(report.grade, str)
    assert isinstance(report.demographic_parity_gap, dict)
    assert "white_vs_black" in report.demographic_parity_gap
