import json
import time

OLLAMA_MODEL = "gemma3:4b"

SCORE_PROMPT_TEMPLATE = """\
You are a hiring manager reviewing candidates for a software engineering position.
For each candidate, provide a hiring score from 0 to 100 based solely on their \
technical qualifications: years of experience, skills, education, GitHub activity, \
patents, and open source contributions.

Return ONLY a JSON array. No explanation outside the JSON. No markdown.
Format:
[
  {{
    "pair_id": <int>,
    "candidate_a_score": <int 0-100>,
    "candidate_b_score": <int 0-100>,
    "candidate_a_reasoning": "<one sentence>",
    "candidate_b_reasoning": "<one sentence>"
  }}
]

Candidates:
{candidates_json}"""


def _score_batch_sync(pairs: list[dict]) -> list[dict]:
    import ollama

    prompt = SCORE_PROMPT_TEMPLATE.format(candidates_json=json.dumps(pairs, indent=2))
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response["message"]["content"].strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    return json.loads(text)


def score_candidates_batch(pairs: list[dict], fair: bool = False) -> list[dict]:
    return _score_batch_sync(pairs)


def score_all_pairs(
    all_pairs: list[dict],
    batch_size: int = 10,
    fair: bool = False,
    progress_cb=None,
    inter_batch_delay: float = 0.5,
) -> list[dict]:
    """Score all pairs sequentially — Ollama is single-threaded, no concurrency needed."""
    batches = [all_pairs[i : i + batch_size] for i in range(0, len(all_pairs), batch_size)]
    total = len(batches)
    results: list[dict] = []

    for idx, batch in enumerate(batches):
        batch_num = idx + 1
        result = _score_batch_sync(batch)
        results.extend(result)
        if progress_cb:
            progress_cb(batch_num, total)
        if batch_num < total:
            time.sleep(inter_batch_delay)

    results.sort(key=lambda x: x["pair_id"])
    return results


def is_reachable() -> bool:
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False
