import asyncio
import json

OLLAMA_MODEL = "llama3.2"

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

    candidates_json = json.dumps(pairs, indent=2)
    prompt = SCORE_PROMPT_TEMPLATE.format(candidates_json=candidates_json)

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response["message"]["content"].strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    return json.loads(text)


async def _score_batch_async(
    pairs: list[dict],
    batch_num: int,
    total_batches: int,
    progress_cb=None,
) -> list[dict]:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _score_batch_sync, pairs)
    if progress_cb:
        progress_cb(batch_num, total_batches)
    return result


def score_candidates_batch(pairs: list[dict], fair: bool = False) -> list[dict]:
    return _score_batch_sync(pairs)


def score_all_pairs(
    all_pairs: list[dict],
    batch_size: int = 10,
    fair: bool = False,
    progress_cb=None,
) -> list[dict]:
    async def _run():
        batches = [
            all_pairs[i : i + batch_size] for i in range(0, len(all_pairs), batch_size)
        ]
        total = len(batches)
        tasks = [
            _score_batch_async(batch, idx + 1, total, progress_cb=progress_cb)
            for idx, batch in enumerate(batches)
        ]
        results = await asyncio.gather(*tasks)
        flat = [item for batch in results for item in batch]
        flat.sort(key=lambda x: x["pair_id"])
        return flat

    return asyncio.run(_run())


def is_reachable() -> bool:
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False
