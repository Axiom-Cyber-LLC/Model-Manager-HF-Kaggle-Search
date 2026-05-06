#!/usr/bin/env python3
"""
Reconcile Ollama's registry against GGUFs on disk.

Walks the same scan roots as Prepare_models_for_Ollama.py, lists what
Ollama currently has, computes the diff, and registers ONLY the missing
ones. Does NOT use any local state file — the source of truth is
`ollama list`.

Use this when:
  - You suspect Prepare_models_for_Ollama.py's state file is stale
  - You did `ollama rm`, switched OLLAMA_MODELS, or rebuilt ~/.ollama/
  - You want to verify "47 cached" actually maps to 47 ollama list entries

Usage:
    python reconcile_ollama.py             # show diff + register missing
    python reconcile_ollama.py --dry-run   # show diff only, no writes
    python reconcile_ollama.py --verify    # also probe `ollama show` per entry
"""

import argparse
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

# ── Config (mirrors Prepare_models_for_Ollama.py) ───────────────────

SCAN_ROOTS = [
    Path("/Volumes/ModelStorage/models"),
    Path("/Volumes/ModelStorage/models-flat"),
    Path("/Volumes/ModelStorage/.cache/huggingface"),
    Path.home() / ".cache" / "huggingface",
    Path("/Volumes/ModelStorage/.cache/modelscope"),
]

# Diffusion/image GGUFs that crash llama.cpp at load time
IMAGE_PATTERNS = re.compile(
    r"(flux|z[-_.]?image|z[-_.]?img|sdxl|sd[-_.]xl|pixelwave|"
    r"qwen[-_.]?image|wan[-_.]?\d|stable[-_.]?diffusion)",
    re.IGNORECASE,
)

NON_FIRST_SHARD = re.compile(r"-0*(?!0*1\b)\d+-of-\d+\.gguf$", re.IGNORECASE)

QUANT_RE = re.compile(
    r"[-_](Q\d[_\w]*|IQ\d[_\w]*|F16|F32|BF16|FP16)\.gguf$",
    re.IGNORECASE,
)

SKIP_DIRS = {"blobs", ".locks", "refs", ".studio_links", ".git", "__pycache__"}


# ── GGUF probing ────────────────────────────────────────────────────

def is_valid_gguf(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"GGUF"
    except OSError:
        return False


# ── Discovery (subset of the main script's logic) ──────────────────

def discover_ggufs():
    seen = set()
    seen_names = set()
    for root in SCAN_ROOTS:
        if not root.is_dir():
            continue
        for gguf in sorted(root.rglob("*.gguf")):
            try:
                if not gguf.is_file():
                    continue
            except OSError:
                continue

            try:
                rel = gguf.relative_to(root).parts
            except ValueError:
                continue

            if any(p in SKIP_DIRS for p in rel):
                continue
            if any(p.startswith(".") and p not in (".", "..") for p in rel[:-1]):
                continue
            if "mmproj" in gguf.name.lower():
                continue
            if NON_FIRST_SHARD.search(gguf.name):
                continue
            if IMAGE_PATTERNS.search(gguf.name):
                continue
            if not is_valid_gguf(gguf):
                continue

            try:
                real = gguf.resolve()
            except (OSError, RuntimeError):
                real = gguf
            if real in seen:
                continue
            seen.add(real)

            # Short-name derivation (HF cache → publisher-model, else parent dir)
            short = None
            for anc in gguf.parents:
                if anc.name.startswith("models--"):
                    bits = anc.name[len("models--"):].split("--", 1)
                    if len(bits) == 2 and all(bits):
                        short = f"{bits[0]}-{bits[1]}"
                    break
            if not short:
                short = gguf.parent.name or gguf.stem
            if short in ("hub", "blobs", "models"):
                short = gguf.stem
            short = short.strip(".-_") or gguf.stem

            m = QUANT_RE.search(gguf.name)
            quant = m.group(1).upper() if m else "default"

            try:
                siblings = [s for s in gguf.parent.glob("*.gguf") if s.is_file()]
            except OSError:
                siblings = [gguf]

            ollama_name = f"local/{short}".lower()
            if len(siblings) > 1:
                ollama_name = f"local/{short}:{quant}".lower()

            if ollama_name in seen_names:
                tag = real.name[-8:].lower() if real.name else gguf.stem[-8:]
                ollama_name = f"{ollama_name}-{tag}"
            seen_names.add(ollama_name)

            yield ollama_name, gguf


# ── Ollama ops ──────────────────────────────────────────────────────

def ollama_list_names() -> set[str]:
    out = subprocess.run(["ollama", "list"], capture_output=True,
                         text=True, check=True, timeout=15)
    names = set()
    for line in out.stdout.splitlines()[1:]:
        if not line.strip():
            continue
        n = line.split()[0]
        names.add(n)
        # Also record without explicit ":latest" tag for matching
        if n.endswith(":latest"):
            names.add(n[: -len(":latest")])
    return names


def ollama_show_ok(name: str) -> bool:
    r = subprocess.run(["ollama", "show", name],
                       capture_output=True, timeout=10)
    return r.returncode == 0


def register(name: str, gguf: Path, dry_run: bool) -> tuple[bool, str]:
    if dry_run:
        return True, "dry-run"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".Modelfile",
                                     delete=False) as mf:
        mf.write(f"FROM {gguf}\n")
        mf_path = mf.name
    try:
        r = subprocess.run(["ollama", "create", name, "-f", mf_path],
                           capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            return False, (r.stderr or r.stdout or "unknown").strip()[-300:]
        # Post-create verification
        if not ollama_show_ok(name):
            return False, "create returned 0 but `ollama show` failed"
        return True, "ok"
    finally:
        Path(mf_path).unlink(missing_ok=True)


# ── Main ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="Also run `ollama show` for every existing entry")
    args = ap.parse_args()

    print("=" * 66)
    print(f"Ollama Reconciler — {'DRY RUN' if args.dry_run else 'LIVE'}")
    print("=" * 66)

    try:
        existing = ollama_list_names()
    except subprocess.CalledProcessError as e:
        sys.exit(f"`ollama list` failed: {e.stderr}")

    print(f"Ollama currently has {len(existing) // 2} models registered "
          f"({len(existing)} match-keys with/without :latest)\n")

    print("Scanning for GGUFs...")
    candidates = list(discover_ggufs())
    print(f"Found {len(candidates)} valid GGUFs on disk\n")

    missing = []
    present = []
    for name, gguf in candidates:
        # Check both with and without :latest
        if name in existing or f"{name}:latest" in existing or \
           name.split(":", 1)[0] in existing:
            present.append((name, gguf))
        else:
            missing.append((name, gguf))

    print(f"Already in Ollama:  {len(present)}")
    print(f"Missing from Ollama: {len(missing)}")
    print()

    if args.verify and present:
        print(f"Verifying {len(present)} existing entries with `ollama show`...")
        broken = []
        for name, gguf in present:
            base = name.split(":")[0]
            check = name if name in existing else (
                f"{name}:latest" if f"{name}:latest" in existing else base
            )
            if not ollama_show_ok(check):
                broken.append((check, gguf))
        if broken:
            print(f"  ⚠️  {len(broken)} entries listed but unusable (ghost manifests):")
            for n, _ in broken[:20]:
                print(f"     {n}")
            if not args.dry_run:
                print(f"  Re-registering them...")
                missing.extend([(n.replace(":latest", ""), g) for n, g in broken])
        else:
            print(f"  ✓ all {len(present)} verified")
        print()

    if not missing:
        print("Nothing to register. Ollama is in sync with disk.")
        return

    print(f"Registering {len(missing)} missing model(s):")
    ok_count = 0
    fail_count = 0
    for name, gguf in missing:
        print(f"  {name}  <-  {gguf.name}")
        ok, msg = register(name, gguf, args.dry_run)
        if ok:
            ok_count += 1
        else:
            fail_count += 1
            print(f"      FAILED: {msg}")

    print()
    print(f"Result: registered={ok_count} failed={fail_count} "
          f"{'(dry-run, no writes)' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
