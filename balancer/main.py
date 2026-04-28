import sys
import os
import importlib
import urllib.request
import warnings
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(help="Balancer — AI Bias Detection & Correction CLI")
console = Console()


@app.callback()
def _root():
    """Balancer v0.1.0 — AI bias detection and correction CLI."""

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"


@app.command()
def check():
    """Health check — verify all dependencies and environment requirements."""
    console.print("\n[bold]Balancer v0.1.0 — Health Check[/bold]")
    console.print("─" * 40)

    all_passed = True

    # Python version
    major, minor = sys.version_info.major, sys.version_info.minor
    py_version = f"Python {major}.{minor}.{sys.version_info.micro}"
    if (major, minor) >= (3, 11):
        console.print(f"[green]✓[/green]  {py_version:<30}")
    else:
        console.print(f"[red]✗[/red]  {py_version:<30}  (requires 3.11+)")
        all_passed = False

    # Required packages
    required = [
        "typer",
        "google.generativeai",
        "ollama",
        "pandas",
        "numpy",
        "scipy",
        "jinja2",
        "aiohttp",
        "rich",
    ]
    deps_ok = True
    missing = []
    for pkg in required:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                importlib.import_module(pkg)
        except ImportError:
            deps_ok = False
            missing.append(pkg)

    if deps_ok:
        console.print(f"[green]✓[/green]  {'Dependencies':<30}")
    else:
        console.print(f"[red]✗[/red]  {'Dependencies':<30}  missing: {', '.join(missing)}")
        all_passed = False

    # GEMINI_API_KEY
    if os.environ.get("GEMINI_API_KEY"):
        console.print(f"[green]✓[/green]  {'GEMINI_API_KEY':<30}")
    else:
        console.print(f"[red]✗[/red]  {'GEMINI_API_KEY':<30}  not set")
        all_passed = False

    # Ollama reachability (warn, don't fail)
    ollama_ok = False
    try:
        req = urllib.request.urlopen("http://localhost:11434", timeout=2)
        ollama_ok = True
    except Exception:
        pass

    if ollama_ok:
        console.print(f"[green]✓[/green]  {'Ollama (localhost)':<30}")
    else:
        console.print(
            f"[yellow]⚠[/yellow]  {'Ollama (localhost)':<30}  not reachable — will use Gemini only"
        )

    # data/names.json
    names_path = DATA_DIR / "names.json"
    if names_path.exists():
        console.print(f"[green]✓[/green]  {'data/names.json':<30}")
    else:
        console.print(f"[red]✗[/red]  {'data/names.json':<30}  not found")
        all_passed = False

    # results/ directory
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]✓[/green]  {'results/ directory':<30}")
    except Exception as e:
        console.print(f"[red]✗[/red]  {'results/ directory':<30}  {e}")
        all_passed = False

    console.print()
    if all_passed:
        console.print("[bold green]All checks passed.[/bold green] Run `balancer run` to start.")
    else:
        console.print("[bold red]Some checks failed.[/bold red] Fix the issues above before running.")
        raise typer.Exit(code=1)


@app.command()
def generate():
    """Generate 100 candidate pairs (seed=42, reproducible)."""
    from balancer.generator import generate_pairs, save_pairs, _verify_identical_profiles, OUTPUT_PATH

    console.print("\n[bold]Generating candidate pairs...[/bold]")
    pairs = generate_pairs(n=100, seed=42)

    # Verify profiles are identical within each pair
    profiles_ok = _verify_identical_profiles(pairs)

    save_pairs(pairs)

    # Stats
    wf_bf = sum(1 for p in pairs if p["candidate_a"]["demographic_group"] == "white_female")
    wm_bm = sum(1 for p in pairs if p["candidate_a"]["demographic_group"] == "white_male")

    console.print(f"Generated [green]{len(pairs)}[/green] candidate pairs")
    console.print(f"Seed: [cyan]42[/cyan] (reproducible)")
    console.print(f"Groups: [cyan]{wf_bf}[/cyan] white_female/black_female, [cyan]{wm_bm}[/cyan] white_male/black_male pairs")
    if profiles_ok:
        console.print("Profile fields identical within pairs: [green]✓[/green]")
    else:
        console.print("Profile fields identical within pairs: [red]✗ MISMATCH DETECTED[/red]")
        raise typer.Exit(code=1)
    console.print(f"Saved to [cyan]{OUTPUT_PATH.relative_to(OUTPUT_PATH.parent.parent)}[/cyan]")


@app.command(name="test-api")
def test_api():
    """Send 2 test pairs to the active LLM client and verify the response."""
    import time
    from balancer.generator import generate_pairs
    from balancer.router import score_with_fallback

    console.print("\nSending 2 test pairs...")
    pairs = generate_pairs(n=2, seed=42)
    start = time.time()
    try:
        scores, name = score_with_fallback(pairs, batch_size=2)
    except Exception as e:
        console.print(f"[red]✗ Request failed:[/red] {e}")
        raise typer.Exit(code=1)
    elapsed = time.time() - start

    console.print(f"Active client: [cyan]{name}[/cyan]")

    console.print(f"Response received in [cyan]{elapsed:.1f}s[/cyan]")

    # Validate JSON structure
    valid = isinstance(scores, list) and len(scores) > 0
    console.print(f"JSON valid: {'[green]✓[/green]' if valid else '[red]✗[/red]'}")
    if not valid:
        console.print("[red]✗ No scores returned[/red]")
        raise typer.Exit(code=1)

    # Check all pairs have scores
    all_have_scores = all(
        "candidate_a_score" in s and "candidate_b_score" in s for s in scores
    )
    console.print(
        f"Scores present for all pairs: {'[green]✓[/green]' if all_have_scores else '[red]✗[/red]'}"
    )
    if not all_have_scores:
        raise typer.Exit(code=1)

    # Show sample
    first = scores[0]
    console.print(
        f"Sample: pair_{first['pair_id']} → "
        f"candidate_a: [cyan]{first['candidate_a_score']}[/cyan], "
        f"candidate_b: [cyan]{first['candidate_b_score']}[/cyan]"
    )


@app.command()
def run(seed: int = typer.Option(42, help="Seed for pair generation")):
    """Full pipeline: load pairs → score → save results/gemini_biased.json."""
    import json
    import time
    from datetime import datetime
    from balancer.generator import generate_pairs, save_pairs, load_pairs, OUTPUT_PATH
    from balancer.router import score_with_fallback

    console.print("\n[bold]Balancer v0.1.0[/bold]")
    console.print("─" * 35)

    # [1/6] Load candidates
    console.print("[bold cyan][1/6][/bold cyan] Loading candidates...", end="          ")
    if OUTPUT_PATH.exists():
        pairs = load_pairs()
    else:
        pairs = generate_pairs(n=100, seed=seed)
        save_pairs(pairs)
    console.print(f"[green]{len(pairs)} pairs ✓[/green]")

    # Check existing results — offer to skip re-scoring
    biased_path = RESULTS_DIR / "gemini_biased.json"
    if biased_path.exists():
        with open(biased_path) as f:
            existing = json.load(f)
        ts = existing.get("metadata", {}).get("timestamp", "unknown")
        answer = typer.prompt(
            f"\nResults exist from {ts}. Rerun scoring?",
            default="N",
        )
        if answer.strip().upper() != "Y":
            console.print(
                "Using existing results. "
                "Run [cyan]balancer analyze[/cyan] to view bias metrics."
            )
            return

    # [2/6] Connect (handled inside score_with_fallback)
    console.print("[bold cyan][2/6][/bold cyan] Connecting to LLM...              ")

    # [3/6] Score all pairs
    console.print("[bold cyan][3/6][/bold cyan] Scoring candidates...")

    def _progress(batch_num: int, total: int) -> None:
        console.print(f"      Batch {batch_num:>2}/{total} [green]✓[/green]")

    start = time.time()
    try:
        scores, name = score_with_fallback(pairs, batch_size=10, progress_cb=_progress)
    except Exception as e:
        console.print(f"[red]✗ Scoring failed:[/red] {e}")
        raise typer.Exit(code=1)
    elapsed = time.time() - start
    console.print(f"      Client: [cyan]{name}[/cyan]  ({elapsed:.1f}s)")

    # [4/6] Persist raw results
    console.print("[bold cyan][4/6][/bold cyan] Saving raw results...", end="          ")
    result_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": name,
            "total_pairs": len(scores),
            "client": "gemini" if "Gemini" in name else "ollama",
            "scoring_time_seconds": round(elapsed, 1),
        },
        "scores": scores,
    }
    with open(biased_path, "w") as f:
        json.dump(result_data, f, indent=2)
    console.print("[green]results/gemini_biased.json ✓[/green]")

    # [5/6] and [6/6] placeholders
    console.print("[bold cyan][5/6][/bold cyan] Running bias analysis...   [dim](phase 5)[/dim]")
    console.print("[bold cyan][6/6][/bold cyan] Generating report...       [dim](phase 7)[/dim]")

    console.print(
        f"\n[green]Done.[/green] {len(scores)} pairs scored in {elapsed:.1f}s. "
        "Run [cyan]balancer analyze[/cyan] next."
    )


@app.command()
def analyze():
    """Compute homemade bias metrics on results/gemini_biased.json."""
    from balancer.analyzer import analyze as run_analysis, RESULTS_DIR as RD

    biased_path = RD / "gemini_biased.json"
    if not biased_path.exists():
        console.print("[red]✗[/red] results/gemini_biased.json not found. Run [cyan]balancer run[/cyan] first.")
        raise typer.Exit(code=1)

    console.print("\n[bold]Bias Analysis Results[/bold]")
    console.print("─" * 45)

    report = run_analysis(biased_path)

    dir_color = "green" if report.disparate_impact_ratio >= 0.80 else "red"
    sig_color = "red" if report.significant else "green"
    cons_color = "red" if report.consistency_rate > 0.65 else "green"

    gap = report.demographic_parity_gap.get("white_vs_black", 0)
    gap_str = f"{gap:+.1f} points"

    console.print(f"Demographic Parity Gap     : [{('red' if gap > 0 else 'green')}]{gap_str}[/]")
    console.print(
        f"Disparate Impact Ratio     : [{dir_color}]{report.disparate_impact_ratio:.3f}[/]"
        f"{'  ← FAIL (threshold: 0.80)' if report.disparate_impact_ratio < 0.80 else '  ← PASS'}"
    )
    console.print(
        f"Consistency Rate           : [{cons_color}]{report.consistency_rate:.0%}[/]"
        f"   ← white-coded scored higher in {int(report.consistency_rate * report.total_pairs)}/{report.total_pairs} pairs"
    )
    console.print(
        f"p-value                    : [{sig_color}]{report.p_value:.4f}[/]"
        f"{'  ← statistically significant' if report.significant else '  ← not significant'}"
    )

    proxy_str = "none  ← bias is NAME-ONLY" if not report.proxy_features else ", ".join(p["feature"] for p in report.proxy_features)
    console.print(f"Proxy Features Detected    : {proxy_str}")
    console.print(f"Bias Grade                 : [bold]{report.grade}[/bold]")
    console.print(f"\nWhite-coded avg score      : {report.white_coded_avg}")
    console.print(f"Black-coded avg score      : {report.black_coded_avg}")
    console.print(f"White hire rate (≥75)      : {report.white_hire_rate:.1%}")
    console.print(f"Black hire rate (≥75)      : {report.black_hire_rate:.1%}")

    console.print()
    if report.significant or report.disparate_impact_ratio < 0.80:
        console.print("[bold yellow]⚠  BIAS DETECTED[/bold yellow]")
        console.print(
            f"   Scored white-coded candidates [bold]{abs(gap):.1f} points[/bold] higher on average\n"
            f"   with identical qualifications."
            + (f" Statistically significant (p={report.p_value:.3f})." if report.significant else "")
        )
        if report.disparate_impact_ratio < 0.80:
            console.print(f"   Violates EEOC 4/5 rule (DIR={report.disparate_impact_ratio:.3f} < 0.80).")
        if not report.proxy_features:
            console.print("   No proxy features detected — bias source is the candidate name alone.")
    else:
        console.print("[bold green]✓  No significant bias detected.[/bold green]")


@app.command()
def fix():
    """Apply both bias fixes: math reweighing + self-correction re-prompt + audit narrative."""
    import json
    from balancer.analyzer import analyze as run_analysis, RESULTS_DIR as RD
    from balancer.generator import load_pairs
    from balancer.mitigator import (
        reweigh, save_math_fixed,
        generate_self_corrected, save_self_corrected,
        generate_audit_narrative, save_audit_narrative,
    )

    biased_path = RD / "gemini_biased.json"
    if not biased_path.exists():
        console.print("[red]✗[/red] results/gemini_biased.json not found. Run [cyan]balancer run[/cyan] first.")
        raise typer.Exit(code=1)

    with open(biased_path) as f:
        biased_data = json.load(f)

    console.print("\n[bold]Fix bias?[/bold]")
    console.print("  [1] Apply mathematical correction (instant, guaranteed)")
    console.print("  [2] Re-prompt LLM for self-correction (may not fully fix)")
    console.print("  [3] Both (recommended for comparison)")
    console.print("  [0] Skip — view report with bias only")
    choice = typer.prompt("Choice", default="3")

    do_math = choice in ("1", "3")
    do_reprompt = choice in ("2", "3")

    biased_report = run_analysis(biased_path)
    math_data = None
    self_corrected_data = None

    # --- Method 1: Math reweighing ---
    if do_math:
        console.print("\n[bold cyan]Method 1:[/bold cyan] Mathematical reweighing...")
        math_data = reweigh(biased_data)
        save_math_fixed(math_data)
        math_report = run_analysis(RD / "math_fixed.json")
        console.print(f"  Before → Grade [bold]{biased_report.grade}[/bold]  DIR {biased_report.disparate_impact_ratio:.3f}  Gap {biased_report.demographic_parity_gap.get('white_vs_black', 0):+.1f}pts")
        console.print(f"  After  → Grade [bold green]{math_report.grade}[/bold green]  DIR {math_report.disparate_impact_ratio:.3f}  Gap {math_report.demographic_parity_gap.get('white_vs_black', 0):+.1f}pts  [green]✓[/green]")
    else:
        # Still need math_fixed.json to exist for report
        if not (RD / "math_fixed.json").exists():
            math_data = reweigh(biased_data)
            save_math_fixed(math_data)
        math_report = run_analysis(RD / "math_fixed.json")

    # --- Method 2: Self-correction re-prompt ---
    if do_reprompt:
        console.print("\n[bold cyan]Method 2:[/bold cyan] LLM self-correction re-prompt...")
        pairs = load_pairs()

        def _prog(n, t):
            console.print(f"  Batch {n:>2}/{t} [green]✓[/green]")

        self_corrected_data = generate_self_corrected(pairs, progress_cb=_prog)
        save_self_corrected(self_corrected_data)
        sc_report = run_analysis(RD / "gemini_self_corrected.json")
        console.print(f"  Before → Grade [bold]{biased_report.grade}[/bold]  DIR {biased_report.disparate_impact_ratio:.3f}")
        console.print(f"  After  → Grade [bold]{sc_report.grade}[/bold]  DIR {sc_report.disparate_impact_ratio:.3f}")
    else:
        if not (RD / "gemini_self_corrected.json").exists():
            # Generate a placeholder so report can load all three
            save_self_corrected({"metadata": {"method": "skipped"}, "scores": biased_data["scores"]})
        sc_report = run_analysis(RD / "gemini_self_corrected.json")

    # --- Audit narrative ---
    console.print("\n[bold cyan]Audit:[/bold cyan] Generating narrative...")
    model_name = biased_data.get("metadata", {}).get("model", "unknown")
    narrative = generate_audit_narrative(biased_report, math_report, sc_report, model_name)
    save_audit_narrative(narrative)

    console.print(f"\n  Severity : [bold]{narrative.get('severity', '?')}[/bold]")
    console.print(f"  Verdict  : {narrative.get('verdict', '?')}")

    console.print("\n[green]✓[/green]  results/math_fixed.json")
    console.print("[green]✓[/green]  results/gemini_self_corrected.json")
    console.print("[green]✓[/green]  results/audit_narrative.json")
    console.print("\nRun [cyan]balancer report[/cyan] to generate the HTML report.")


@app.command()
def report(
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Auto-open in browser"),
    output: str = typer.Option("results/report.html", help="Output path for the HTML report"),
):
    """Generate a self-contained HTML bias audit report (no CDN)."""
    from pathlib import Path
    from balancer.reporter import generate_report, open_report

    out_path = Path(output)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    biased_path = RESULTS_DIR / "gemini_biased.json"
    if not biased_path.exists():
        console.print("[red]✗[/red] results/gemini_biased.json not found. Run [cyan]balancer run[/cyan] first.")
        raise typer.Exit(code=1)

    console.print("\n[bold]Generating bias audit report...[/bold]")

    try:
        path = generate_report(out_path)
    except Exception as e:
        console.print(f"[red]✗ Report generation failed:[/red] {e}")
        raise typer.Exit(code=1)

    size_kb = path.stat().st_size // 1024
    console.print(f"[green]✓[/green]  {path.relative_to(REPO_ROOT)}  ({size_kb} KB, fully self-contained)")

    if open_browser:
        open_report(path)
        console.print("Opened in browser.")
