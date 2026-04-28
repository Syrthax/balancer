from balancer import gemini_client, ollama_client, claude_client
from balancer.config import ensure_gemini_key
from balancer.gemini_client import GeminiQuotaError


def get_client():
    """Return active client: Gemini → Claude → Ollama, with warnings on fallback."""
    ensure_gemini_key()

    if gemini_client.is_reachable():
        return gemini_client

    if claude_client.is_reachable():
        print("⚠  Gemini unreachable. Falling back to Claude (claude-haiku-4-5).")
        return claude_client

    print("⚠  Gemini + Claude unreachable. Falling back to Ollama (llama3.2:1b).")
    if not ollama_client.is_reachable():
        raise RuntimeError(
            "No LLM available. Check GEMINI_API_KEY / ANTHROPIC_API_KEY or run `ollama serve`."
        )
    return ollama_client


def get_client_name(client) -> str:
    if client is gemini_client:
        return f"Gemini ({gemini_client.GEMINI_MODEL})"
    if client is claude_client:
        return f"Claude ({claude_client.CLAUDE_MODEL})"
    return f"Ollama ({ollama_client.OLLAMA_MODEL})"


def score_with_fallback(
    pairs: list[dict],
    batch_size: int = 10,
    fair: bool = False,
    progress_cb=None,
) -> tuple[list[dict], str]:
    """Score pairs: Gemini first, Claude on quota error, Ollama as last resort."""
    ensure_gemini_key()

    if gemini_client.is_reachable():
        try:
            scores = gemini_client.score_all_pairs(
                pairs, batch_size=batch_size, fair=fair, progress_cb=progress_cb
            )
            return scores, get_client_name(gemini_client)
        except GeminiQuotaError:
            print("\n⚠  Gemini quota exceeded. Falling back to Claude (claude-haiku-4-5).")

    if claude_client.is_reachable():
        try:
            scores = claude_client.score_all_pairs(
                pairs, batch_size=batch_size, fair=fair, progress_cb=progress_cb
            )
            return scores, get_client_name(claude_client)
        except Exception as e:
            print(f"\n⚠  Claude failed ({e}). Falling back to Ollama (llama3.2:1b).")

    if not ollama_client.is_reachable():
        raise RuntimeError(
            "No LLM available. Check GEMINI_API_KEY / ANTHROPIC_API_KEY or run `ollama serve`."
        )
    scores = ollama_client.score_all_pairs(
        pairs, batch_size=batch_size, progress_cb=progress_cb
    )
    return scores, get_client_name(ollama_client)
