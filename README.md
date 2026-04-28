# Balancer

**AI Bias Audit for LLM Decision Systems**

Built for the Google Solution Challenge 2026, Unbiased AI Decision track.

Balancer audits LLM-based hiring decisions using the Bertrand & Mullainathan (2004) correspondence-study methodology from peer-reviewed labor economics. It sends identical candidate profiles to an LLM, varying only the demographically-coded name, then computes homemade fairness metrics to detect bias.

## What we found

Tested on Llama 3.2-1B with 100 identical-resume pairs:

| Metric | Value |
|---|---|
| Bias Grade | F- |
| Disparate Impact Ratio | 0.557 (FAIL, EEOC threshold 0.800) |
| Demographic Parity Gap | +15.9 points |
| White hire rate | 88% |
| Black hire rate | 49% |
| p-value | 0.0000 |

Same skills, same education, same experience. Only the name differs.

## Why this matters

Existing fairness tools (Fairlearn, AIF360) audit tabular ML classifiers, not LLMs. As organizations deploy LLMs for hiring, lending, and admissions, no audit infrastructure exists to verify they are not biased. Balancer fills that gap.

## Methodology

Bertrand, M., and Mullainathan, S. (2004). *Are Emily and Greg More Employable Than Lakisha and Jamal? A Field Experiment on Labor Market Discrimination.* American Economic Review, 94(4).

## Architecture
Generator -> Gemini API -> Llama 3.2 -> Score Store -> Analyzer -> Mitigator -> Reporter -> Web App

## Features

- Reproducible candidate-pair generation (seed=42, B&M 2004 name list)
- Gemini API integration for hiring evaluation
- Homemade fairness metrics: DPG, DIR, paired t-test, Cohen's d, proxy scan
- EEOC four-fifths rule compliance check
- Two correction methods: mathematical reweighing and LLM self-correction
- Three-way comparison report (biased, math-fixed, self-corrected)
- Self-contained HTML audit report (zero CDN dependencies)
- CLI for engineers, web dashboard for non-technical users
- A-to-F bias grading with verdict narrative

## Google technologies used

- Gemini API (gemini-2.5-flash) for the model under audit
- Google AI Studio for API key management

## Quickstart

```bash
pip install -e .
export GEMINI_API_KEY="your_key_from_aistudio.google.com/apikey"
balancer check
balancer generate
balancer run
balancer analyze
balancer fix
balancer report
```

Open `results/report.html` to see the audit.

## Project structure
balancer/
balancer/         # core package
data/             # B&M 2004 name list
results/          # audit outputs (gitignored)
tests/            # tests
app.py            # FastAPI web app
Dockerfile        # container deploy
pyproject.toml

## Team

- Peddalachannagari Mrenika Reddy 
- Sarthak Ghosh
- Harshit Choudhary
- S Biswanath

## License

MIT
'@ | Set-Content README.md -Encoding UTF8
