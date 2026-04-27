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
