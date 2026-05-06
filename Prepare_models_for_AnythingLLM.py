#!/usr/bin/env python3
"""
Prepare AnythingLLM to use the existing local models without duplicating them.

AnythingLLM's desktop app ships a BUNDLED Ollama. Pointing it at the system
Ollama that LM Studio + consolidate_models.py already populate means every
model you've set up is instantly available, with zero extra disk usage.

This script edits the AnythingLLM storage .env (with a timestamped backup)
to switch:

    LLM_PROVIDER='anythingllm_ollama'         →   'ollama'
    + adds OLLAMA_BASE_PATH='http://localhost:11434'
    + sets OLLAMA_MODEL_PREF to a reasonable default

Usage:
  python3 Prepare_models_for_AnythingLLM.py                  # patch + report
  python3 Prepare_models_for_AnythingLLM.py --dry-run        # preview
  python3 Prepare_models_for_AnythingLLM.py --prefer MODEL   # pick default model
  python3 Prepare_models_for_AnythingLLM.py --revert         # restore backup
  python3 Prepare_models_for_AnythingLLM.py --list-only      # show ollama models
"""
import argparse
import json
import os
import re
import shutil
import sys
import time
import urllib.request
from pathlib import Path

HOME = Path.home()
ANYTHINGLLM_ENV = HOME / "Library" / "Application Support" / "anythingllm-desktop" / "storage" / ".env"
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


# ── .env parsing ───────────────────────────────────────────────

ENV_LINE = re.compile(r"^([A-Z_][A-Z0-9_]*)\s*=\s*(.*)$")


def parse_env(text: str) -> dict:
    """Parse dotenv-style content into an ordered dict."""
    out = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = ENV_LINE.match(s)
        if not m:
            continue
        k, v = m.group(1), m.group(2).strip()
        if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
            v = v[1:-1]
        out[k] = v
    return out


def format_env(kv: dict) -> str:
    lines = [f"# Patched by Prepare_models_for_AnythingLLM.py on {time.strftime('%Y-%m-%d %H:%M:%S %Z')}"]
    for k, v in kv.items():
        # Always single-quote values for safety (matches AnythingLLM's own style)
        v_escaped = v.replace("'", r"\'")
        lines.append(f"{k}='{v_escaped}'")
    return "\n".join(lines) + "\n"


# ── Ollama discovery ───────────────────────────────────────────

def fetch_ollama_models(base: str) -> list[dict]:
    """Return list of {name, size, modified_at} from ollama /api/tags."""
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=5) as r:
            data = json.load(r)
        return data.get("models", []) or []
    except Exception as e:
        print(f"  WARN: could not reach Ollama at {base}: {e}")
        return []


def choose_default_model(models: list[dict], prefer: str | None) -> str | None:
    names = [m.get("name", "") for m in models]
    if not names:
        return None
    if prefer:
        for n in names:
            if prefer == n or prefer in n:
                return n
        print(f"  WARN: --prefer {prefer!r} not found, using first available")
    # Heuristic: prefer a small fast model as default
    for needle in ("llama3.2:3b", "phi", "qwen2.5", "llama", "mistral"):
        for n in names:
            if needle in n.lower():
                return n
    return names[0]


# ── Patch ──────────────────────────────────────────────────────

def backup(path: Path, dry_run: bool) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    b = path.with_suffix(path.suffix + f".bak.{ts}")
    if dry_run:
        print(f"  [DRY RUN] would back up {path.name} -> {b.name}")
    else:
        shutil.copy2(path, b)
        print(f"  Backed up {path.name} -> {b.name}")
    return b


def find_latest_backup(path: Path) -> Path | None:
    candidates = sorted(path.parent.glob(path.name + ".bak.*"), reverse=True)
    return candidates[0] if candidates else None


def patch_env(prefer: str | None, dry_run: bool) -> int:
    if not ANYTHINGLLM_ENV.exists():
        print(f"  ERROR: {ANYTHINGLLM_ENV} does not exist. Is AnythingLLM installed?")
        return 2

    original = ANYTHINGLLM_ENV.read_text()
    env = parse_env(original)
    prior_provider = env.get("LLM_PROVIDER", "(unset)")

    # Fetch the model list BEFORE writing so we can pick a default
    models = fetch_ollama_models(OLLAMA_URL)
    default_model = choose_default_model(models, prefer)
    if not models:
        print(f"  ERROR: No models returned by Ollama at {OLLAMA_URL}.")
        print(f"  Start Ollama (`ollama serve`) and ensure models are installed first.")
        return 3

    # Apply the switch
    new_env = dict(env)
    new_env["LLM_PROVIDER"] = "ollama"
    new_env["OLLAMA_BASE_PATH"] = OLLAMA_URL
    if default_model:
        new_env["OLLAMA_MODEL_PREF"] = default_model
        # AnythingLLM needs an explicit token limit for the selected model.
        # Default to 8192; most modern local models handle that or more.
        new_env.setdefault("OLLAMA_MODEL_TOKEN_LIMIT", "8192")

    if new_env == env:
        print("  No changes needed — AnythingLLM already points at the external Ollama.")
        return 0

    print(f"  LLM_PROVIDER:       {prior_provider}  →  {new_env['LLM_PROVIDER']}")
    print(f"  OLLAMA_BASE_PATH:   {env.get('OLLAMA_BASE_PATH', '(unset)')}  →  {new_env['OLLAMA_BASE_PATH']}")
    print(f"  OLLAMA_MODEL_PREF:  {env.get('OLLAMA_MODEL_PREF', '(unset)')}  →  {new_env['OLLAMA_MODEL_PREF']}")
    print()
    print(f"  Ollama has {len(models)} model(s) available — first 10:")
    for m in models[:10]:
        print(f"    {m.get('name')}")
    if len(models) > 10:
        print(f"    … and {len(models) - 10} more")
    print()

    backup(ANYTHINGLLM_ENV, dry_run)
    if dry_run:
        print("  [DRY RUN] would write new .env")
    else:
        ANYTHINGLLM_ENV.write_text(format_env(new_env))
        print(f"  Wrote {ANYTHINGLLM_ENV}")
    return 0


def revert(dry_run: bool) -> int:
    latest = find_latest_backup(ANYTHINGLLM_ENV)
    if not latest:
        print("  No backup found to revert from.")
        return 1
    print(f"  Restoring from: {latest.name}")
    if dry_run:
        print("  [DRY RUN] would copy backup over .env")
    else:
        shutil.copy2(latest, ANYTHINGLLM_ENV)
        print(f"  Restored {ANYTHINGLLM_ENV.name}")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Point AnythingLLM at the existing system Ollama.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--prefer", type=str, default=None,
                    help="Preferred default model (substring match on ollama name)")
    ap.add_argument("--revert", action="store_true",
                    help="Restore the most recent .env backup")
    ap.add_argument("--list-only", action="store_true",
                    help="Just list models visible from the external Ollama and exit")
    args = ap.parse_args()

    mode = "DRY RUN" if args.dry_run else "LIVE"
    print("=" * 60)
    print(f"AnythingLLM → External Ollama — {mode}")
    print("=" * 60)
    print(f"Env file:    {ANYTHINGLLM_ENV}")
    print(f"Ollama URL:  {OLLAMA_URL}")
    print()

    if args.list_only:
        models = fetch_ollama_models(OLLAMA_URL)
        print(f"Found {len(models)} model(s) in Ollama:")
        for m in models:
            size_gb = (m.get("size") or 0) / (1024**3)
            print(f"  {m.get('name'):60s}  {size_gb:5.1f} GB")
        return

    if args.revert:
        sys.exit(revert(args.dry_run))

    rc = patch_env(args.prefer, args.dry_run)
    print()
    if rc == 0:
        print("Done. Restart AnythingLLM so it picks up the new .env.")
    sys.exit(rc)


if __name__ == "__main__":
    main()
