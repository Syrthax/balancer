from balancer import gemini_client, ollama_client


def get_client():
    """Return active client module. Warns and falls back to Ollama if Gemini unreachable."""
    if gemini_client.is_reachable():
        return gemini_client

    print(
        "⚠  Gemini unreachable. Falling back to Ollama (local). Results may differ."
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
