from balancer import gemini_client, ollama_client, claude_client
from balancer.config import ensure_gemini_key
from balancer.gemini_client import GeminiQuotaError

# Track clients that failed this process so we skip them immediately on retry
_session_failures: set = set()


def get_client_name(client) -> str:
    if client is gemini_client:
        return f"Gemini ({gemini_client.GEMINI_MODEL})"
    if client is claude_client:
        return f"Claude ({claude_client.CLAUDE_MODEL})"
    return f"Ollama ({ollama_client.OLLAMA_MODEL})"


def get_client():
    """Return active client: Gemini → Claude → Ollama, with warnings on fallback."""
    ensure_gemini_key()

    if gemini_client not in _session_failures and gemini_client.is_reachable():
        return gemini_client

    if claude_client not in _session_failures and claude_client.is_reachable():
        print("⚠  Gemini unavailable. Falling back to Claude.")
        return claude_client

    print("⚠  Falling back to Ollama (llama3.2:1b).")
    if not ollama_client.is_reachable():
        raise RuntimeError(
            "No LLM available. Check GEMINI_API_KEY / ANTHROPIC_API_KEY or run `ollama serve`."
        )
    return ollama_client


def score_with_fallback(
    pairs: list[dict],
    batch_size: int = 10,
    fair: bool = False,
    progress_cb=None,
) -> tuple[list[dict], str]:
    """Score pairs: Gemini → Claude → Ollama. Failed clients are skipped for the rest of the session."""
    ensure_gemini_key()

    if gemini_client not in _session_failures and gemini_client.is_reachable():
        try:
            scores = gemini_client.score_all_pairs(
                pairs, batch_size=batch_size, fair=fair, progress_cb=progress_cb
            )
            return scores, get_client_name(gemini_client)
        except GeminiQuotaError:
            _session_failures.add(gemini_client)
            print("\n⚠  Gemini quota exceeded — switching to Claude.")

    if claude_client not in _session_failures and claude_client.is_reachable():
        try:
            scores = claude_client.score_all_pairs(
                pairs, batch_size=batch_size, fair=fair, progress_cb=progress_cb
            )
            return scores, get_client_name(claude_client)
        except Exception as e:
            _session_failures.add(claude_client)
            print(f"\n⚠  Claude unavailable ({e}) — switching to Ollama.")

    if not ollama_client.is_reachable():
        raise RuntimeError(
            "No LLM available. Check GEMINI_API_KEY / ANTHROPIC_API_KEY or run `ollama serve`."
        )
    scores = ollama_client.score_all_pairs(
        pairs, batch_size=batch_size, fair=fair, progress_cb=progress_cb
    )
    return scores, get_client_name(ollama_client)
