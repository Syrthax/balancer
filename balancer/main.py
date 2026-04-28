import sys
import os
import importlib
import math
import urllib.request
import warnings
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress, BarColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn,
)
from rich.rule import Rule
from rich.table import Table

app = typer.Typer(help="Balancer — AI Bias Detection & Correction CLI", add_completion=False)
console = Console()


@app.callback()
def _root():
    """Balancer v0.1.0 — AI bias detection and correction CLI."""


REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"


# ── helpers ────────────────────────────────────────────────────────────────

def _make_progress() -> Progress:
    return Progress(
        TextColumn("  "),
        BarColumn(bar_width=30, complete_style="cyan", finished_style="green"),
        MofNCompleteColumn(),
        TextColumn("[dim]batches[/dim]"),
        TimeElapsedColumn(),
        TextColumn("·"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def _score_local(pairs, progress_cb):
    from balancer import ollama_client
    if not ollama_client.is_reachable():
        console.print("\n[red]✗ Ollama not reachable. Run `ollama serve`.[/red]")
        raise typer.Exit(code=1)
    scores = ollama_client.score_all_pairs(pairs, batch_size=5, progress_cb=progress_cb)
    return scores, f"Ollama ({ollama_client.OLLAMA_MODEL})"


def _score_auto(pairs, progress_cb):
    from balancer.router import score_with_fallback
    return score_with_fallback(pairs, batch_size=10, progress_cb=progress_cb)


def _run_scoring(pairs, local: bool) -> tuple[list[dict], str, float]:
    """Run scoring and return (scores, model_name, elapsed_seconds)."""
    import time
    est = math.ceil(len(pairs) / 5)
    start = time.time()

    with _make_progress() as progress:
        task = progress.add_task("", total=est)

        def cb(n, t):
            progress.update(task, completed=n, total=t)

        try:
            scores, name = _score_local(pairs, cb) if local else _score_auto(pairs, cb)
        except typer.Exit:
            raise
        except Exception as e:
            console.print(f"\n[red]✗ Scoring failed:[/red] {e}")
            raise typer.Exit(code=1)

    return scores, name, time.time() - start


def _print_analysis(report) -> None:
    gap = report.demographic_parity_gap.get("white_vs_black", 0)
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim", width=28)
    t.add_column()

    dir_color = "green" if report.disparate_impact_ratio >= 0.80 else "red"
    sig_color = "red" if report.significant else "green"
    cons_hi = int(report.consistency_rate * report.total_pairs)

    t.add_row("Demographic Parity Gap",
              f"[{'red' if gap > 0 else 'green'}]{gap:+.1f} pts[/]  "
              f"({'white-coded scored higher' if gap > 0 else 'equal'})")
    t.add_row("Disparate Impact Ratio",
              f"[{dir_color}]{report.disparate_impact_ratio:.3f}[/]  "
              f"({'PASS' if report.disparate_impact_ratio >= 0.80 else 'FAIL — EEOC threshold 0.80'})")
    t.add_row("Consistency Rate",
              f"[{'red' if report.consistency_rate > 0.55 else 'green'}]"
              f"{report.consistency_rate:.0%}[/]  "
              f"({cons_hi}/{report.total_pairs} pairs white scored higher)")
    t.add_row("p-value (paired t-test)",
              f"[{sig_color}]{report.p_value:.4f}[/]  "
              f"({'significant ✗' if report.significant else 'not significant ✓'})")
    t.add_row("Proxy features",
              "none — bias is name-only" if not report.proxy_features
              else ", ".join(p["feature"] for p in report.proxy_features))
    t.add_row("Bias grade", f"[bold]{report.grade}[/bold]")
    t.add_row("White avg / hire rate",
              f"{report.white_coded_avg}  ·  {report.white_hire_rate:.0%} hired (≥75)")
    t.add_row("Black avg / hire rate",
              f"{report.black_coded_avg}  ·  {report.black_hire_rate:.0%} hired (≥75)")

    console.print(t)
    console.print()

    if report.significant or report.disparate_impact_ratio < 0.80:
        lines = [
            f"[bold red]⚠  BIAS DETECTED[/bold red]  [dim](Grade {report.grade})[/dim]",
            "",
            f"[white]{abs(gap):.1f}-point gap[/white] on [italic]identical[/italic] resumes — "
            f"only the name differs.",
            f"White hire rate [cyan]{report.white_hire_rate:.0%}[/cyan] vs "
            f"black hire rate [magenta]{report.black_hire_rate:.0%}[/magenta]  "
            f"({report.white_hire_rate - report.black_hire_rate:.0%} gap).",
        ]
        if report.significant:
            lines.append(f"Statistically significant at [bold]p={report.p_value:.4f}[/bold].")
        if not report.proxy_features:
            lines.append("[dim]No proxy features — the name alone is driving the bias.[/dim]")
        console.print(Panel("\n".join(lines), border_style="red", padding=(0, 1)))
    else:
        console.print(Panel("[bold green]✓  No significant bias detected.[/bold green]",
                            border_style="green", padding=(0, 1)))


# ── commands ───────────────────────────────────────────────────────────────

@app.command()
def check():
    """Health check — verify all dependencies and environment requirements."""
    console.print()
    console.rule("[bold]Balancer — Health Check[/bold]")

    all_passed = True

    major, minor = sys.version_info.major, sys.version_info.minor
    py_ver = f"Python {major}.{minor}.{sys.version_info.micro}"
    if (major, minor) >= (3, 11):
        console.print(f"[green]✓[/green]  {py_ver}")
    else:
        console.print(f"[red]✗[/red]  {py_ver}  (requires 3.11+)")
        all_passed = False

    required = ["typer", "google.generativeai", "ollama", "pandas",
                "numpy", "scipy", "jinja2", "aiohttp", "rich"]
    missing = []
    for pkg in required:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        console.print(f"[red]✗[/red]  Dependencies  missing: {', '.join(missing)}")
        all_passed = False
    else:
        console.print("[green]✓[/green]  Dependencies")

    if os.environ.get("GEMINI_API_KEY"):
        console.print("[green]✓[/green]  GEMINI_API_KEY")
    else:
        console.print("[yellow]⚠[/yellow]  GEMINI_API_KEY not set  (Ollama will be used)")

    ollama_ok = False
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=2)
        ollama_ok = True
    except Exception:
        pass
    if ollama_ok:
        console.print("[green]✓[/green]  Ollama (localhost:11434)")
    else:
        console.print("[red]✗[/red]  Ollama not reachable — run `ollama serve`")
        all_passed = False

    if (DATA_DIR / "names.json").exists():
        console.print("[green]✓[/green]  data/names.json")
    else:
        console.print("[red]✗[/red]  data/names.json not found")
        all_passed = False

    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        console.print("[green]✓[/green]  results/")
    except Exception as e:
        console.print(f"[red]✗[/red]  results/  {e}")
        all_passed = False

    console.print()
    if all_passed:
        console.print("[bold green]All checks passed.[/bold green]  Run [cyan]balancer run --local[/cyan] to start.")
    else:
        console.print("[bold red]Some checks failed.[/bold red]  Fix the issues above first.")
        raise typer.Exit(code=1)


@app.command()
def generate():
    """Generate 100 candidate pairs (seed=42, reproducible)."""
    from balancer.generator import generate_pairs, save_pairs, _verify_identical_profiles, OUTPUT_PATH

    console.print("\nGenerating candidate pairs...")
    pairs = generate_pairs(n=100, seed=42)
    ok = _verify_identical_profiles(pairs)
    save_pairs(pairs)

    wf = sum(1 for p in pairs if p["candidate_a"]["demographic_group"] == "white_female")
    wm = len(pairs) - wf
    console.print(f"[green]✓[/green]  {len(pairs)} pairs  ·  {wf} WF/BF + {wm} WM/BM  ·  seed=42")
    console.print(f"[green]✓[/green]  Profiles identical within pairs: {'yes' if ok else '[red]MISMATCH[/red]'}")
    console.print(f"[green]✓[/green]  Saved to data/candidates.json")


@app.command(name="test-api")
def test_api():
    """Send 2 test pairs to the active LLM and verify the response."""
    import time
    from balancer.generator import generate_pairs

    console.print("\nSending 2 test pairs...")
    pairs = generate_pairs(n=2, seed=42)
    start = time.time()

    with _make_progress() as progress:
        task = progress.add_task("", total=1)

        def cb(n, t):
            progress.update(task, completed=n, total=t)

        try:
            scores, name = _score_auto(pairs, cb)
        except Exception as e:
            console.print(f"[red]✗ Request failed:[/red] {e}")
            raise typer.Exit(code=1)

    elapsed = time.time() - start
    valid = isinstance(scores, list) and len(scores) > 0
    all_scores = all("candidate_a_score" in s and "candidate_b_score" in s for s in scores)

    console.print(f"  Client   : [cyan]{name}[/cyan]")
    console.print(f"  Time     : {elapsed:.1f}s")
    console.print(f"  JSON     : {'[green]✓[/green]' if valid else '[red]✗[/red]'}")
    console.print(f"  Scores   : {'[green]✓[/green]' if all_scores else '[red]✗[/red]'}")
    if scores:
        s = scores[0]
        console.print(f"  Sample   : pair_{s['pair_id']} → "
                      f"white [cyan]{s['candidate_a_score']}[/cyan]  "
                      f"black [magenta]{s['candidate_b_score']}[/magenta]  "
                      f"(gap {s['candidate_a_score'] - s['candidate_b_score']:+d})")


@app.command()
def run(
    seed: int = typer.Option(42, help="Seed for pair generation"),
    local: bool = typer.Option(False, "--local", help="Skip cloud APIs, use Ollama directly"),
    offline: bool = typer.Option(False, "--offline", help="Alias for --local"),
):
    """Score 100 candidate pairs and save raw results."""
    import json
    from datetime import datetime
    from balancer.generator import generate_pairs, save_pairs, load_pairs, OUTPUT_PATH

    use_local = local or offline

    console.print()
    console.rule("[bold]Balancer — Score[/bold]")

    # Load pairs
    console.print("[bold cyan][1/3][/bold cyan] Loading candidates...", end="  ")
    pairs = load_pairs() if OUTPUT_PATH.exists() else generate_pairs(n=100, seed=seed)
    if not OUTPUT_PATH.exists():
        save_pairs(pairs)
    console.print(f"[green]{len(pairs)} pairs ✓[/green]")

    # Check existing results
    biased_path = RESULTS_DIR / "gemini_biased.json"
    if biased_path.exists():
        with open(biased_path) as f:
            existing = json.load(f)
        ts = existing.get("metadata", {}).get("timestamp", "unknown")
        answer = typer.prompt(f"\nResults exist from {ts}. Rerun?", default="N")
        if answer.strip().upper() != "Y":
            console.print("Keeping existing results.  Next: [cyan]balancer analyze[/cyan]")
            return

    # Score
    console.print("[bold cyan][2/3][/bold cyan] Scoring with llama3.2:1b...")
    scores, name, elapsed = _run_scoring(pairs, use_local)
    console.print(f"  [dim]{name}  ·  {elapsed:.0f}s  ·  {len(scores)} pairs[/dim]")

    # Save
    console.print("[bold cyan][3/3][/bold cyan] Saving...", end="  ")
    result_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": name,
            "total_pairs": len(scores),
            "client": "ollama",
            "scoring_time_seconds": round(elapsed, 1),
        },
        "scores": scores,
    }
    with open(biased_path, "w") as f:
        json.dump(result_data, f, indent=2)
    console.print("[green]results/gemini_biased.json ✓[/green]")

    console.print(f"\n[green]Done.[/green]  Next: [cyan]balancer analyze[/cyan]  →  "
                  "[cyan]balancer fix[/cyan]  →  [cyan]balancer report[/cyan]")


@app.command()
def analyze():
    """Compute homemade bias metrics on results/gemini_biased.json."""
    from balancer.analyzer import analyze as run_analysis, RESULTS_DIR as RD

    biased_path = RD / "gemini_biased.json"
    if not biased_path.exists():
        console.print("[red]✗[/red]  results/gemini_biased.json not found.  "
                      "Run [cyan]balancer run --local[/cyan] first.")
        raise typer.Exit(code=1)

    console.print()
    console.rule("[bold]Bias Analysis[/bold]")
    report = run_analysis(biased_path)
    _print_analysis(report)


@app.command()
def fix(
    local: bool = typer.Option(False, "--local", help="Use Ollama for self-correction"),
    offline: bool = typer.Option(False, "--offline", help="Alias for --local"),
):
    """Apply bias corrections: math reweighing + self-correction re-prompt + audit narrative."""
    import json
    from balancer.analyzer import analyze as run_analysis, RESULTS_DIR as RD
    from balancer.generator import load_pairs
    from balancer.mitigator import (
        reweigh, save_math_fixed,
        generate_self_corrected, save_self_corrected,
        generate_audit_narrative, save_audit_narrative,
    )

    use_local = local or offline
    biased_path = RD / "gemini_biased.json"
    if not biased_path.exists():
        console.print("[red]✗[/red]  results/gemini_biased.json not found.  "
                      "Run [cyan]balancer run --local[/cyan] first.")
        raise typer.Exit(code=1)

    with open(biased_path) as f:
        biased_data = json.load(f)

    console.print()
    console.rule("[bold]Bias Fix[/bold]")
    console.print("  [bold]1[/bold]  Mathematical correction  [dim](instant, guaranteed)[/dim]")
    console.print("  [bold]2[/bold]  LLM self-correction re-prompt  [dim](slower, may not fully fix)[/dim]")
    console.print("  [bold]3[/bold]  Both  [dim](recommended for comparison)[/dim]")
    console.print("  [bold]0[/bold]  Skip")
    choice = typer.prompt("\nChoice", default="3")

    do_math = choice in ("1", "3")
    do_reprompt = choice in ("2", "3")

    biased_report = run_analysis(biased_path)

    # Method 1: Math
    if do_math:
        console.print()
        console.print("[bold cyan]Method 1[/bold cyan]  Mathematical reweighing...", end="  ")
        math_data = reweigh(biased_data)
        save_math_fixed(math_data)
        math_report = run_analysis(RD / "math_fixed.json")
        gap_b = biased_report.demographic_parity_gap.get("white_vs_black", 0)
        gap_a = math_report.demographic_parity_gap.get("white_vs_black", 0)
        console.print(
            f"[dim]before[/dim] Grade [bold]{biased_report.grade}[/bold]  "
            f"DIR {biased_report.disparate_impact_ratio:.3f}  Gap {gap_b:+.1f}pts  →  "
            f"[dim]after[/dim] Grade [bold green]{math_report.grade}[/bold green]  "
            f"DIR {math_report.disparate_impact_ratio:.3f}  Gap {gap_a:+.1f}pts  [green]✓[/green]"
        )
    else:
        if not (RD / "math_fixed.json").exists():
            save_math_fixed(reweigh(biased_data))
        math_report = run_analysis(RD / "math_fixed.json")

    # Method 2: Self-correction
    if do_reprompt:
        console.print()
        console.print("[bold cyan]Method 2[/bold cyan]  LLM self-correction re-prompt...")
        pairs = load_pairs()

        with _make_progress() as progress:
            task = progress.add_task("", total=math.ceil(len(pairs) / 5))

            def _prog(n, t):
                progress.update(task, completed=n, total=t)

            sc_data = generate_self_corrected(pairs, progress_cb=_prog)

        save_self_corrected(sc_data)
        sc_report = run_analysis(RD / "gemini_self_corrected.json")
        console.print(
            f"  [dim]before[/dim] Grade [bold]{biased_report.grade}[/bold]  "
            f"DIR {biased_report.disparate_impact_ratio:.3f}  →  "
            f"[dim]after[/dim] Grade [bold]{sc_report.grade}[/bold]  "
            f"DIR {sc_report.disparate_impact_ratio:.3f}"
        )
    else:
        if not (RD / "gemini_self_corrected.json").exists():
            save_self_corrected({"metadata": {"method": "skipped"}, "scores": biased_data["scores"]})
        sc_report = run_analysis(RD / "gemini_self_corrected.json")

    # Audit narrative
    console.print()
    console.print("[bold cyan]Audit[/bold cyan]  Generating narrative...", end="  ")
    model_name = biased_data.get("metadata", {}).get("model", "unknown")
    narrative = generate_audit_narrative(biased_report, math_report, sc_report, model_name)
    save_audit_narrative(narrative)
    sev = narrative.get("severity", "?")
    sev_color = {"CRITICAL": "red", "HIGH": "orange3", "MEDIUM": "yellow", "LOW": "green"}.get(sev, "white")
    console.print(f"[{sev_color}]{sev}[/{sev_color}]  [dim]{narrative.get('verdict', '')}[/dim]")

    console.print()
    console.print("[green]✓[/green]  results/math_fixed.json")
    console.print("[green]✓[/green]  results/gemini_self_corrected.json")
    console.print("[green]✓[/green]  results/audit_narrative.json")
    console.print("\nNext: [cyan]balancer report[/cyan]")


@app.command()
def report(
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Auto-open in browser"),
    output: str = typer.Option("results/report.html", help="Output path"),
):
    """Generate a self-contained HTML bias audit report (no CDN)."""
    from balancer.reporter import generate_report, open_report

    out_path = Path(output)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    if not (RESULTS_DIR / "gemini_biased.json").exists():
        console.print("[red]✗[/red]  No results found.  Run [cyan]balancer run --local[/cyan] first.")
        raise typer.Exit(code=1)

    console.print("\nGenerating report...", end="  ")
    try:
        path = generate_report(out_path)
    except Exception as e:
        console.print(f"[red]failed:[/red] {e}")
        raise typer.Exit(code=1)

    size_kb = path.stat().st_size // 1024
    console.print(f"[green]{path.relative_to(REPO_ROOT)}[/green]  [dim]({size_kb} KB, self-contained)[/dim]")
    if open_browser:
        open_report(path)
        console.print("[dim]Opened in browser.[/dim]")


@app.command()
def demo(
    rescore: bool = typer.Option(False, "--rescore", help="Force re-scoring even if results exist"),
):
    """End-to-end live demo: score → analyze → fix → report (local Ollama, no cloud APIs)."""
    import json
    from datetime import datetime
    from balancer.generator import generate_pairs, save_pairs, load_pairs, OUTPUT_PATH
    from balancer.analyzer import analyze as run_analysis
    from balancer.mitigator import reweigh, save_math_fixed, generate_audit_narrative, save_audit_narrative
    from balancer.mitigator import save_self_corrected
    from balancer.reporter import generate_report, open_report

    console.print()
    console.print(Panel(
        "[bold white]Balancer — AI Hiring Bias Detection[/bold white]\n"
        "[dim]100 identical resumes · only the name changes · seed=42[/dim]\n\n"
        "[dim]H2S × Google for Developers Hackathon — Unbiased AI Decision Track[/dim]",
        border_style="bright_blue",
        padding=(1, 3),
    ))

    # ── Step 1: Pairs ─────────────────────────────────────────────────────
    console.print(Rule("[bold cyan]Step 1 / 4[/bold cyan]  Candidate Pairs", style="cyan"))
    pairs = load_pairs() if OUTPUT_PATH.exists() else generate_pairs(n=100, seed=42)
    if not OUTPUT_PATH.exists():
        save_pairs(pairs)
    wf = sum(1 for p in pairs if p["candidate_a"]["demographic_group"] == "white_female")
    console.print(f"  [green]✓[/green]  {len(pairs)} pairs  ·  {wf} female pairs + {len(pairs) - wf} male pairs")
    console.print("  [dim]Each pair has identical technical qualifications — only the name differs.[/dim]")
    console.print("  [dim]Names drawn from Bertrand & Mullainathan (2004) audit study.[/dim]")

    # ── Step 2: Scoring ───────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold cyan]Step 2 / 4[/bold cyan]  LLM Scoring (llama3.2:1b)", style="cyan"))

    biased_path = RESULTS_DIR / "gemini_biased.json"
    scores = None

    if biased_path.exists() and not rescore:
        with open(biased_path) as f:
            existing = json.load(f)
        ts = existing.get("metadata", {}).get("timestamp", "unknown")
        console.print(f"  [green]✓[/green]  Using cached results from [cyan]{ts}[/cyan]")
        console.print("  [dim](run with --rescore to force re-scoring)[/dim]")
    else:
        console.print("  Asking llama3.2:1b to score all 100 pairs as a hiring manager...")
        console.print("  [dim]The model sees names but is told to judge on technical merit only.[/dim]")
        scores, name, elapsed = _run_scoring(pairs, local=True)
        console.print(f"  [dim]{name}  ·  {elapsed:.0f}s[/dim]")

        result_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": name,
                "total_pairs": len(scores),
                "client": "ollama",
                "scoring_time_seconds": round(elapsed, 1),
            },
            "scores": scores,
        }
        with open(biased_path, "w") as f:
            json.dump(result_data, f, indent=2)
        console.print(f"  [green]✓[/green]  results/gemini_biased.json saved")

    # ── Step 3: Analyze ───────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold cyan]Step 3 / 4[/bold cyan]  Bias Analysis", style="cyan"))
    biased_report = run_analysis(biased_path)
    _print_analysis(biased_report)

    # ── Step 4: Fix ───────────────────────────────────────────────────────
    console.print(Rule("[bold cyan]Step 4 / 4[/bold cyan]  Mathematical Correction", style="cyan"))

    with open(biased_path) as f:
        biased_data = json.load(f)

    console.print("  Applying score reweighing:  score × (overall_mean / group_mean)", end="  ")
    math_data = reweigh(biased_data)
    save_math_fixed(math_data)
    math_report = run_analysis(RESULTS_DIR / "math_fixed.json")

    gap_b = biased_report.demographic_parity_gap.get("white_vs_black", 0)
    gap_a = math_report.demographic_parity_gap.get("white_vs_black", 0)
    console.print("[green]done[/green]")

    # Comparison panel
    cmp = Table.grid(padding=(0, 3))
    cmp.add_column(style="dim", width=18)
    cmp.add_column(justify="center", width=14)
    cmp.add_column(justify="center", width=14)
    cmp.add_row("", "[bold red]Biased Run[/bold red]", "[bold green]Math Fixed[/bold green]")
    cmp.add_row("Grade",
                f"[bold]{biased_report.grade}[/bold]",
                f"[bold green]{math_report.grade}[/bold green]")
    cmp.add_row("DIR",
                f"[red]{biased_report.disparate_impact_ratio:.3f}[/red]",
                f"[green]{math_report.disparate_impact_ratio:.3f}[/green]")
    cmp.add_row("Gap",
                f"[red]{gap_b:+.1f} pts[/red]",
                f"[green]{gap_a:+.1f} pts[/green]")
    cmp.add_row("White hire rate",
                f"{biased_report.white_hire_rate:.0%}",
                f"{math_report.white_hire_rate:.0%}")
    cmp.add_row("Black hire rate",
                f"[red]{biased_report.black_hire_rate:.0%}[/red]",
                f"[green]{math_report.black_hire_rate:.0%}[/green]")
    console.print(Panel(cmp, border_style="green", title="Before vs After", padding=(0, 2)))

    # Placeholder self-corrected for report
    sc_path = RESULTS_DIR / "gemini_self_corrected.json"
    if not sc_path.exists():
        save_self_corrected({"metadata": {"method": "skipped"}, "scores": biased_data["scores"]})
    sc_report = run_analysis(sc_path)

    # Narrative
    console.print("  Generating audit narrative...", end="  ")
    model_name = biased_data.get("metadata", {}).get("model", "unknown")
    narrative = generate_audit_narrative(biased_report, math_report, sc_report, model_name)
    save_audit_narrative(narrative)
    sev = narrative.get("severity", "?")
    sev_color = {"CRITICAL": "red", "HIGH": "orange3", "MEDIUM": "yellow", "LOW": "green"}.get(sev, "white")
    console.print(f"[{sev_color}]{sev}[/{sev_color}]")

    # Report
    report_path = RESULTS_DIR / "report.html"
    console.print()
    console.print("  Generating HTML report...", end="  ")
    generate_report(report_path)
    console.print(f"[green]{report_path.relative_to(REPO_ROOT)}[/green]")
    open_report(report_path)

    console.print()
    console.print(Panel(
        f"[bold green]Demo complete.[/bold green]\n\n"
        f"  [white]Bias detected:[/white]  [red]{gap_b:+.1f} pt gap[/red] on identical resumes  "
        f"(p={biased_report.p_value:.4f})\n"
        f"  [white]After fix:[/white]      [green]{gap_a:+.1f} pt gap[/green]  "
        f"(DIR {biased_report.disparate_impact_ratio:.3f} → {math_report.disparate_impact_ratio:.3f})\n\n"
        f"  [dim]Report opened in browser.[/dim]",
        border_style="green",
        padding=(1, 3),
    ))
