#!/usr/bin/env python3
"""
Prepare existing local models for GPT4All.

GPT4All on macOS auto-discovers any .gguf file sitting in:
    ~/Library/Application Support/nomic.ai/GPT4All/

This script creates SYMLINKS to the canonical LM Studio copies so you don't
duplicate the weights. Filenames are flattened as {author}__{model}__{file}.gguf
to avoid collisions across providers.

Usage:
  python3 Prepare_models_for_GPT4All.py             # link everything
  python3 Prepare_models_for_GPT4All.py --dry-run   # preview
  python3 Prepare_models_for_GPT4All.py --clean     # remove dangling links
"""
import argparse
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_print_lock = threading.Lock()
def _log(msg):
    with _print_lock:
        print(msg)

HOME = Path.home()
GPT4ALL_DIR = HOME / "Library" / "Application Support" / "nomic.ai" / "GPT4All"
# Unified scan roots — searched recursively (rglob, unlimited depth) and
# deduped by resolved physical path so symlinked overlaps don't double up.
# Missing roots are skipped silently. The GPT4All target dir is intentionally
# kept in this shared list for cross-app consistency, then filtered out at
# runtime in this script (see source-root resolution in main()) to avoid
# walking our own destination as a source.
DEFAULT_GGUF_ROOTS = [
    Path("/Volumes/ModelStorage/models"),
    Path("/Volumes/ModelStorage/models-flat"),
    Path("/Volumes/ModelStorage/models/huggingface/model"),
    Path("/Volumes/ModelStorage/models-flat/local"),
    Path("/Volumes/ModelStorage/.cache/huggingface"),
    HOME / ".cache" / "huggingface",   # symlink to the SSDE one
    Path("/Volumes/ModelStorage/.cache/modelscope"),
    Path("/Volumes/ModelStorage/.cache/model_manager"),
    HOME / "model_downloads" / "huggingface" / "model",
    GPT4ALL_DIR,                       # filtered out at runtime — we ARE this dir
    HOME / ".lmstudio" / "models",     # legacy fallback
    Path("~/skill-scanner/scan-results/20260503T074319Z"),
    Path("~/skill-scanner/scan-results/20260503T074334Z"),
]
GGUF_EXT = ".gguf"

# Directory tokens we never DESCEND into below a scan root. These are
# tool-internal layouts (HF cache structure, Ollama blob store, LM Studio
# symlink hashes) that contain symlinks/aliases to blobs we'll already see
# via their canonical location. The user's `.cache/huggingface` and
# `.cache/modelscope` are top-level roots, so `.cache` is intentionally
# NOT in this list — relative-path checks won't include the root prefix.
SKIP_DIR_NAMES = {
    "blobs", ".locks", "refs",
    ".studio_links", ".git", "__pycache__",
}

# Structural filename patterns that are NEVER standalone models. These are
# format-level filters (the file genuinely cannot be loaded as a complete
# model), not content-judgment filters about chat-vs-image.
SKIP_FILENAME_PATTERNS = [
    re.compile(r"^mmproj", re.IGNORECASE),       # multimodal projection helper
    re.compile(r"-mmproj", re.IGNORECASE),       # multimodal projection helper
    re.compile(r"_lora(?:[_-]|\.)", re.IGNORECASE),  # LoRA adapters need a base
    re.compile(r"^lora[_-]", re.IGNORECASE),     # LoRA adapters need a base
]

# Top-level author dirs whose CONTENT is never standalone models
SKIP_AUTHOR_DIRS = {"lora-adapters", "loras", "adapters"}

MIN_GGUF_BYTES = 50 * 1024 * 1024  # 50MB floor — anything smaller is a partial/corrupt download


def sanitize(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]", "-", name)
    name = re.sub(r"-+", "-", name)
    return name.strip("-") or "model"


def flattened_name(author: str, model_dir: str, file_name: str) -> str:
    """Build a collision-safe flat filename for GPT4All's flat dir."""
    stem = Path(file_name).stem
    prefix = f"{sanitize(author)}__{sanitize(model_dir)}__" if author != "local" else f"{sanitize(model_dir)}__"
    # Avoid double-up when the gguf name already contains the model dir
    if sanitize(model_dir) in sanitize(stem):
        prefix = f"{sanitize(author)}__" if author != "local" else ""
    return f"{prefix}{stem}{GGUF_EXT}"


def iter_ggufs(root: Path):
    if not root.exists():
        return
    for p in root.rglob("*" + GGUF_EXT):
        # Walk-internal exclusions (HF/Ollama metadata dirs that aren't models)
        try:
            rel_parts = p.relative_to(root).parts
        except ValueError:
            rel_parts = ()
        if any(part in SKIP_DIR_NAMES for part in rel_parts):
            continue
        # Skip dot-prefixed ancestors BELOW the root (e.g. `.studio_links`,
        # `.aidiff`). The root itself may begin with a dot (`/.cache/...`) —
        # those are intentionally scanned.
        if any(part.startswith(".") and part not in (".", "..")
               for part in rel_parts[:-1]):
            continue
        # Authors that are never standalone models (LoRA / adapter dumps)
        if rel_parts and rel_parts[0] in SKIP_AUTHOR_DIRS:
            continue
        # Structural file-level skips (mmproj, lora — not chat/image filtering)
        fname = p.name
        if any(pat.search(fname) for pat in SKIP_FILENAME_PATTERNS):
            continue
        # Exclude obviously truncated files — GGUF parser crashes on these
        try:
            if p.stat().st_size < MIN_GGUF_BYTES:
                continue
        except OSError:
            continue
        # Best-effort author/model derivation from path layout
        if len(rel_parts) >= 3:
            author, model_dir = rel_parts[0], rel_parts[1]
        elif len(rel_parts) == 2:
            author, model_dir = "local", rel_parts[0]
        else:
            author, model_dir = "local", p.stem
        # HF cache layout: models--{publisher}--{model}/snapshots/{sha}/file.gguf
        for anc in p.parents:
            if anc.name.startswith("models--"):
                bits = anc.name[len("models--"):].split("--", 1)
                if len(bits) == 2 and all(bits):
                    author, model_dir = bits[0], bits[1]
                break
        yield author, model_dir, p


def safe_symlink(src: Path, dst: Path, dry_run: bool) -> str:
    if dst.is_symlink():
        current = Path(os.readlink(dst))
        if current == src.resolve() or current == src:
            return "exists"
        if dry_run:
            return "replace"
        dst.unlink()
    elif dst.exists():
        return "skip-real-file"
    if dry_run:
        return "create"
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst)
    return "created"


def clean_dangling(root: Path, dry_run: bool) -> int:
    if not root.exists():
        return 0
    removed = 0
    for p in list(root.iterdir()):
        if p.suffix != GGUF_EXT:
            continue
        if p.is_symlink() and not p.exists():
            if dry_run:
                print(f"  [DRY RUN] UNLINK (dangling): {p.name}")
            else:
                try:
                    p.unlink()
                    print(f"  UNLINK (dangling): {p.name}")
                except OSError as e:
                    print(f"  ERROR unlinking {p.name}: {e}")
                    continue
            removed += 1
    return removed


def main():
    ap = argparse.ArgumentParser(
        description="Symlink local GGUF files into the GPT4All model directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--clean", action="store_true",
                    help="Also remove symlinks whose target no longer exists")
    ap.add_argument("--gguf-root", type=Path, action="append", default=[],
                    help=f"Source GGUF root (repeat for multiple). "
                         f"Default: {', '.join(str(p) for p in DEFAULT_GGUF_ROOTS)}")
    ap.add_argument("--target", type=Path, default=GPT4ALL_DIR)
    ap.add_argument("--workers", type=int, default=16,
                    help="Parallel symlink workers (default 16; symlinks are cheap)")
    args = ap.parse_args()

    # Resolve source roots: user-supplied (if any) else defaults, keep only
    # existing ones, and exclude the target dir itself (we're writing INTO it,
    # so walking it as a source would cause double-linking under different
    # flattened names).
    def _safe_resolve(p: Path) -> Path:
        try:
            return p.resolve()
        except OSError:
            return p.absolute()
    target_real = _safe_resolve(args.target)
    candidate_roots = args.gguf_root if args.gguf_root else DEFAULT_GGUF_ROOTS
    source_roots = [r for r in candidate_roots
                    if r.exists() and _safe_resolve(r) != target_real]
    if not source_roots:
        sys.exit(f"Error: no GGUF source roots exist. Tried: {candidate_roots}")

    mode = "DRY RUN" if args.dry_run else "LIVE"
    print("=" * 60)
    print(f"GPT4All Model Linker — {mode}")
    print("=" * 60)
    for r in source_roots:
        print(f"GGUF source: {r}")
    print(f"Target:      {args.target}")
    print()

    if not args.target.exists():
        print(f"  Target does not exist. Is GPT4All installed?")
        if not args.dry_run:
            args.target.mkdir(parents=True, exist_ok=True)
            print(f"  Created: {args.target}")

    # Gather all tasks first (serial walk), then parallelize the symlink ops
    stats = {"created": 0, "exists": 0, "replace": 0, "skip-real-file": 0}
    seen_targets = set()
    tasks = []
    for root in source_roots:
        for author, model_dir, src in iter_ggufs(root):
            name = flattened_name(author, model_dir, src.name)
            if name in seen_targets:
                continue
            seen_targets.add(name)
            tasks.append((src, args.target / name, name))

    def _do_one(task):
        src, dst, name = task
        status = safe_symlink(src, dst, args.dry_run)
        tag = "[DRY RUN] " if args.dry_run else ""
        _log(f"  {tag}{status:14s} {name}")
        return status

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for status in pool.map(_do_one, tasks):
            stats[status] = stats.get(status, 0) + 1

    print()
    print(f"Result: {stats}")

    if args.clean:
        print()
        print("Removing dangling symlinks…")
        removed = clean_dangling(args.target, args.dry_run)
        print(f"  Dangling symlinks removed: {removed}")

    print()
    print("Done. Restart GPT4All to refresh its model list.")


if __name__ == "__main__":
    main()
