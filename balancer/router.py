from balancer import gemini_client, ollama_client
from balancer.config import ensure_gemini_key
from balancer.gemini_client import GeminiQuotaError


def get_client():
    """Return active client module. Prompts for key if needed, falls back to Ollama."""
    ensure_gemini_key()

    if gemini_client.is_reachable():
        return gemini_client

    print(
        "⚠  Gemini unreachable. Falling back to Ollama (gemma3:4b). Results may differ."
    )

    if not ollama_client.is_reachable():
        raise RuntimeError(
            "Both Gemini and Ollama are unreachable. "
            "Check your GEMINI_API_KEY or run `ollama serve`."
        )
    return ollama_client


def get_client_name(client) -> str:
    if client is gemini_client:
        return f"Gemini ({gemini_client.GEMINI_MODEL})"
    return f"Ollama ({ollama_client.OLLAMA_MODEL})"


def score_with_fallback(
    pairs: list[dict],
    batch_size: int = 10,
    fair: bool = False,
    progress_cb=None,
) -> tuple[list[dict], str]:
    """Score pairs using Gemini, falling back to Ollama on quota errors.

    Returns (scores, client_name_used).
    """
    ensure_gemini_key()

    if gemini_client.is_reachable():
        try:
            scores = gemini_client.score_all_pairs(
                pairs, batch_size=batch_size, fair=fair, progress_cb=progress_cb
            )
            return scores, get_client_name(gemini_client)
        except GeminiQuotaError:
            print(
                "\n⚠  Gemini quota exceeded. Falling back to Ollama (gemma3:4b). "
                "Results may differ from a Gemini run."
            )

    if not ollama_client.is_reachable():
        raise RuntimeError(
            "Both Gemini and Ollama are unreachable. "
            "Check your GEMINI_API_KEY or run `ollama serve`."
        )

    scores = ollama_client.score_all_pairs(
        pairs, batch_size=batch_size, progress_cb=progress_cb
    )
    return scores, get_client_name(ollama_client)
