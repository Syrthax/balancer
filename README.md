# Balancer — AI Bias Detection CLI

A CLI tool that sends identical candidate profiles (only names differ, demographically coded per Bertrand & Mullainathan 2004) to Gemini, detects hiring bias statistically, fixes it two ways (math reweighing + Gemini self-correction), and generates an HTML report.

---

## 🚨 For the new collaborator — read this first

### What's done (Phases 1–3, partial 4)

| Phase | Status | What it does |
|---|---|---|
| 1 | ✅ Done | `balancer check` — health check, dependency verify |
| 2 | ✅ Done | `balancer generate` — 100 candidate pairs, seed=42 |
| 3 | ✅ Done | Gemini + Ollama clients, router with fallback |
| 4 | 🔴 Blocked | `balancer run` — pipeline exists but Gemini quota exhausted |

### The actual problem blocking Phase 4

The `GEMINI_API_KEY` in `~/.zshenv` is from an **AI Studio free tier project called "Balancer"**. We burned through the daily quota (1500 req/day) while debugging concurrent batching. It resets at midnight Pacific.

**The fix that's already coded** (just needs a working key to verify):
- Switched model: `gemini-2.0-flash` → `gemini-2.0-flash-lite`
- Batches now run **sequentially** with 5s gaps (no more concurrent hammering)
- 429 retry backoff: 15s → 30s → 45s before giving up

### What you need to do

**Option A — Wait for quota reset (simplest)**
Just run this tomorrow morning (quota resets at midnight Pacific):
```bash
balancer test-api    # should return clean JSON from Gemini
balancer run         # scores 100 pairs, ~90s total with 5s gaps
```

**Option B — Get a fresh API key right now**
1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Click **Create API key → Create API key in new project**
3. Update `~/.zshenv`:
   ```bash
   export GEMINI_API_KEY="YOUR_NEW_KEY"
   ```
4. Open a new terminal and run `balancer test-api` to verify it works

> ⚠️ Do NOT use Google Cloud Console keys — those projects have `limit: 0` on the free tier and won't work without billing enabled. Only AI Studio keys work.

### Key things to know about the codebase

- **`balancer/gemini_client.py`** — Gemini API client. Model is `gemini-2.0-flash-lite`. Batching is sequential. Don't change it back to concurrent.
- **`balancer/ollama_client.py`** — Ollama fallback using `gemma3:4b` (must be pulled locally). Also sequential — do NOT make it concurrent or it will overheat the CPU.
- **`balancer/router.py`** — Routes to Gemini if key is set, falls back to Ollama with a warning. Also has `score_with_fallback()` which auto-falls-back on 429.
- **`balancer/config.py`** — Prompts user to enter `GEMINI_API_KEY` if not set in env.
- **`data/names.json`** — B&M 2004 name list. Do not modify.
- **`results/`** — Output files go here (gitignored except `.gitkeep`).

### How to run after fixing the key

```bash
# 1. Verify environment
balancer check

# 2. Generate candidate pairs (already done, but safe to rerun)
balancer generate

# 3. Verify Gemini responds
balancer test-api

# 4. Run full pipeline (saves results/gemini_biased.json)
balancer run

# 5. Run all tests
pytest tests/
```

### Phases still to implement (4 through 9)

See `prompt.md` in the repo root for the full spec. Each phase has a **STOP AND TEST** block — don't skip them.

| Phase | Command | What it adds |
|---|---|---|
| 4 | `balancer run` | Full pipeline, saves `results/gemini_biased.json` ← **start here** |
| 5 | `balancer analyze` | Homemade bias metrics (DPG, DIR, p-value, proxy scan) |
| 6 | — | Math reweighing + Gemini self-correction + audit narrative |
| 7 | `balancer report` | Self-contained HTML report, zero CDN deps |
| 8 | `balancer demo` | Demo mode with offline cache |
| 9 | — | Packaging, README, fresh install test |

### Rules (never break these)

- Bias metrics are **homemade** — no fairlearn, no aif360
- `scipy.stats` allowed for p-value only
- HTML report must have **zero CDN dependencies**
- Candidate profiles must be identical except name (test enforces this)
- Seed 42 is fixed — never change it
- Ollama fallback must **always warn the user** — never silent

### Git protocol

```bash
git add .
git commit -m "phase N: one line description"
git push origin main
```

---

## Install

```bash
pip install -e .
export GEMINI_API_KEY="your_key_from_aistudio"
balancer check
```

## Project structure

```
balancer/
├── balancer/
│   ├── main.py          # CLI entrypoint (typer)
│   ├── generator.py     # 100 candidate pairs, seed=42
│   ├── gemini_client.py # Gemini API, sequential batching, retry
│   ├── ollama_client.py # Ollama fallback (gemma3:4b)
│   ├── router.py        # Client routing + score_with_fallback()
│   └── config.py        # API key prompting
├── data/
│   └── names.json       # B&M 2004 name list
├── results/             # Output files (auto-created)
├── tests/
│   └── test_generator.py
├── prompt.md            # Full phase-by-phase spec ← read this
└── pyproject.toml
```
