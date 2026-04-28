from __future__ import annotations
import json
import webbrowser
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from balancer.analyzer import analyze, BiasReport

RESULTS_DIR = Path(__file__).parent.parent / "results"
TEMPLATES_DIR = Path(__file__).parent / "templates"


def _safe_load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _report_dict(r: BiasReport) -> dict:
    return {
        "grade": r.grade,
        "dpg_overall": r.demographic_parity_gap.get("white_vs_black", 0),
        "dpg_female": r.demographic_parity_gap.get("white_female_vs_black_female", 0),
        "dpg_male": r.demographic_parity_gap.get("white_male_vs_black_male", 0),
        "dir": r.disparate_impact_ratio,
        "consistency_rate": r.consistency_rate,
        "p_value": r.p_value,
        "t_stat": r.t_stat,
        "significant": r.significant,
        "white_avg": r.white_coded_avg,
        "black_avg": r.black_coded_avg,
        "white_hire_rate": round(r.white_hire_rate * 100, 1),
        "black_hire_rate": round(r.black_hire_rate * 100, 1),
        "total_pairs": r.total_pairs,
        "proxy_features": [p["feature"] for p in r.proxy_features],
    }


def _score_buckets(scores: list[float]) -> list[int]:
    buckets = [0] * 10
    for s in scores:
        idx = min(int(s) // 10, 9)
        buckets[idx] += 1
    return buckets


def generate_report(output_path: Path = RESULTS_DIR / "report.html") -> Path:
    biased_path = RESULTS_DIR / "gemini_biased.json"
    if not biased_path.exists():
        raise FileNotFoundError("results/gemini_biased.json not found. Run `balancer run` first.")

    biased_data = _safe_load(biased_path)
    math_data = _safe_load(RESULTS_DIR / "math_fixed.json")
    sc_data = _safe_load(RESULTS_DIR / "gemini_self_corrected.json")
    narrative = _safe_load(RESULTS_DIR / "audit_narrative.json") or {}

    biased_report = _report_dict(analyze(biased_path))
    math_report = _report_dict(analyze(RESULTS_DIR / "math_fixed.json")) if math_data else None
    sc_report = _report_dict(analyze(RESULTS_DIR / "gemini_self_corrected.json")) if sc_data else None

    def extract(data: dict | None, key: str) -> list[float]:
        if data is None:
            return []
        return [s[key] for s in data["scores"]]

    bw = extract(biased_data, "candidate_a_score")
    bb = extract(biased_data, "candidate_b_score")
    mw = extract(math_data, "candidate_a_score")
    mb = extract(math_data, "candidate_b_score")
    sw = extract(sc_data, "candidate_a_score")
    sb = extract(sc_data, "candidate_b_score")

    pair_scatter = [
        {"x": s["candidate_a_score"], "y": s["candidate_b_score"]}
        for s in biased_data["scores"]
    ]

    chart_data = {
        "biased": {"white": _score_buckets(bw), "black": _score_buckets(bb)},
        "math": {"white": _score_buckets(mw), "black": _score_buckets(mb)} if mw else None,
        "sc": {"white": _score_buckets(sw), "black": _score_buckets(sb)} if sw else None,
        "scatter": pair_scatter,
    }

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=False)
    template = env.get_template("report.html.j2")
    html = template.render(
        metadata=biased_data.get("metadata", {}),
        biased=biased_report,
        math=math_report,
        sc=sc_report,
        narrative=narrative,
        chart_data_json=json.dumps(chart_data),
        reports_json=json.dumps({"biased": biased_report, "math": math_report, "sc": sc_report}),
    )

    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    return output_path


def open_report(path: Path) -> None:
    webbrowser.open(path.as_uri())
