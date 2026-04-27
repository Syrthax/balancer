import os
from pathlib import Path


def ensure_gemini_key() -> str:
    """Return GEMINI_API_KEY, prompting the user to enter it if not in environment."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key

    print("\n⚠  GEMINI_API_KEY not found in environment.")
    print("   Get one at https://aistudio.google.com/app/apikey")
    try:
        key = input("   Enter GEMINI_API_KEY (or press Enter to use Ollama fallback): ").strip()
    except (EOFError, KeyboardInterrupt):
        key = ""

    if key:
        os.environ["GEMINI_API_KEY"] = key
        _offer_save(key)

    return key


def _offer_save(key: str) -> None:
    """Offer to persist the key in ~/.zshenv."""
    try:
        answer = input("   Save to ~/.zshenv permanently? [y/N]: ").strip().upper()
    except (EOFError, KeyboardInterrupt):
        return

    if answer != "Y":
        return

    zshenv = Path.home() / ".zshenv"
    content = zshenv.read_text() if zshenv.exists() else ""
    if "GEMINI_API_KEY" in content:
        print(f"   ℹ  GEMINI_API_KEY already in {zshenv} — not overwritten")
        return

    with open(zshenv, "a") as f:
        f.write(f'\nexport GEMINI_API_KEY="{key}"\n')
    print(f"   ✓  Saved to {zshenv}")
