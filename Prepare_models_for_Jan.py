#!/usr/bin/env python3
"""
Prepare existing local models for Jan (jan.ai).

Canonical models live in LM Studio's directory (~/.lmstudio/models/).
This script creates SYMLINKS inside Jan's models directories so Jan can
load them without duplicating gigabytes of weights.

Jan on macOS uses two backends:
  - llama.cpp (GGUF):  ~/Library/Application Support/Jan/data/llamacpp/models/{id}/model.gguf
                       + model.yml metadata sibling file
  - MLX (directories): ~/Library/Application Support/Jan/data/mlx/models/{id}/
                       whole directory with config.json + tokenizer + *.safetensors

Usage:
  python3 Prepare_models_for_Jan.py                 # link everything found
  python3 Prepare_models_for_Jan.py --dry-run       # preview only
  python3 Prepare_models_for_Jan.py --clean         # remove symlinks whose
                                                    # targets no longer exist
  python3 Prepare_models_for_Jan.py --include-gguf-only   # skip MLX scan
  python3 Prepare_models_for_Jan.py --mlx-root DIR  # extra MLX scan root
"""
import argparse
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_print_lock = threading.Lock()
def _log(msg):
    with _print_lock:
        print(msg)

HOME = Path.home()

# Unified scan roots — searched recursively (rglob, unlimited depth)
# and deduped at link-creation time by safe_symlink. Missing roots skipped.
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
    HOME / "Library" / "Application Support" / "nomic.ai" / "GPT4All",
    HOME / ".lmstudio" / "models",     # legacy fallback
    Path("~/skill-scanner/scan-results/20260503T074319Z"),
    Path("~/skill-scanner/scan-results/20260503T074334Z"),
]
# MLX roots = unified set + MLX-specific extras
DEFAULT_MLX_ROOTS = [
    Path("/Volumes/ModelStorage/models"),
    Path("/Volumes/ModelStorage/models-flat"),
    Path("/Volumes/ModelStorage/models/huggingface/model"),
    Path("/Volumes/ModelStorage/models-flat/local"),
    Path("/Volumes/ModelStorage/.cache/huggingface"),
    HOME / ".cache" / "huggingface",
    Path("/Volumes/ModelStorage/.cache/modelscope"),
    Path("/Volumes/ModelStorage/.cache/model_manager"),
    HOME / "model_downloads" / "huggingface" / "model",
    HOME / "Library" / "Application Support" / "nomic.ai" / "GPT4All",
    HOME / "mlx-lm",
    HOME / "Library" / "Caches" / "huggingface" / "hub",
    HOME / ".lmstudio" / "models",
    Path("~/skill-scanner/scan-results/20260503T074319Z"),
    Path("~/skill-scanner/scan-results/20260503T074334Z"),
]

# Jan's target directories
JAN_ROOT = HOME / "Library" / "Application Support" / "Jan" / "data"
JAN_LLAMACPP = JAN_ROOT / "llamacpp" / "models"
JAN_MLX = JAN_ROOT / "mlx" / "models"

GGUF_EXT = ".gguf"
MLX_MARKER_FILES = {"config.json"}
MLX_WEIGHT_EXTS = {".safetensors", ".npz", ".mlx"}
# Walk-internal exclusions (HF cache structure, Ollama blob store, LM Studio
# symlink hashes). These are tool-internal layouts that contain aliases of
# blobs we'll see via canonical paths — scanning them creates duplicates.
# `.cache` is not here because the user's HF/ModelScope cache roots include
# `.cache` as a path component; relative-path checks won't include the root.
SKIP_DIR_NAMES = {
    "blobs", ".locks", "refs",
    ".studio_links", ".git", "__pycache__",
}

# Structural-only filename skips: mmproj is a multimodal projection helper,
# not a complete model. Chat-vs-image content filtering removed per user
# direction — every standalone GGUF gets a Jan entry.
SKIP_FILENAME_PATTERNS = [
    re.compile(r"^mmproj", re.IGNORECASE),
    re.compile(r"-mmproj", re.IGNORECASE),
]

MIN_GGUF_BYTES = 50 * 1024 * 1024  # 50MB floor — anything smaller = partial download


def sanitize(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]", "-", name)
    name = re.sub(r"-+", "-", name)
    return name.strip("-") or "model"


def safe_symlink(src: Path, dst: Path, dry_run: bool) -> str:
    """Create dst -> src symlink. Returns status tag."""
    if dst.is_symlink():
        current = os.readlink(dst)
        if Path(current) == src.resolve() or Path(current) == src:
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


# ── GGUF discovery ──────────────────────────────────────────────

def find_ggufs(root: Path):
    """Yield (author, model_dir, gguf_path) for every .gguf under root."""
    if not root.exists():
        return
    for p in root.rglob("*" + GGUF_EXT):
        try:
            rel_parts = p.relative_to(root).parts
        except ValueError:
            rel_parts = ()
        if any(part in SKIP_DIR_NAMES for part in rel_parts):
            continue
        # Skip dotted ancestors BELOW the root (the root itself may be `.cache/...`)
        if any(part.startswith(".") and part not in (".", "..")
               for part in rel_parts[:-1]):
            continue
        fname = p.name
        # Structural skips only (mmproj). No chat-vs-image content filtering.
        if any(pat.search(fname) for pat in SKIP_FILENAME_PATTERNS):
            continue
        try:
            if p.stat().st_size < MIN_GGUF_BYTES:
                continue
        except OSError:
            continue
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
        yield sanitize(author), sanitize(model_dir), p


# ── MLX discovery ──────────────────────────────────────────────

def looks_like_mlx_dir(d: Path) -> bool:
    """An MLX model dir has config.json + at least one safetensors/npz weight."""
    if not d.is_dir():
        return False
    if not (d / "config.json").is_file():
        return False
    for f in d.iterdir():
        if f.is_file() and f.suffix.lower() in MLX_WEIGHT_EXTS:
            return True
    return False


def find_mlx_models(roots):
    """Yield (author, model_dir, dir_path) for MLX model directories."""
    seen = set()
    for root in roots:
        if not root or not root.exists():
            continue
        # scan to a sensible depth to avoid walking blob trees
        for d in root.rglob("*"):
            if not d.is_dir():
                continue
            if any(part in SKIP_DIR_NAMES for part in d.parts):
                continue
            if any(part.startswith(".") for part in d.parts):
                continue
            if not looks_like_mlx_dir(d):
                continue
            real = d.resolve()
            if real in seen:
                continue
            seen.add(real)
            # Best-effort author/model split relative to the scan root
            try:
                rel = d.relative_to(root).parts
                if len(rel) >= 2:
                    author, model_dir = rel[-2], rel[-1]
                else:
                    author, model_dir = "local", rel[-1] if rel else d.name
            except ValueError:
                author, model_dir = "local", d.name
            yield sanitize(author), sanitize(model_dir), d


# ── Link placement ─────────────────────────────────────────────

def install_gguf(author: str, model_dir: str, src: Path, dry_run: bool):
    """Create Jan llamacpp entry: {id}/model.gguf + model.yml."""
    model_id = f"{author}__{model_dir}" if author != "local" else model_dir
    dest_dir = JAN_LLAMACPP / model_id
    gguf_link = dest_dir / "model.gguf"
    yml = dest_dir / "model.yml"

    status = safe_symlink(src, gguf_link, dry_run)
    size = src.stat().st_size if src.exists() else 0

    yml_content = (
        f"embedding: false\n"
        f"model_path: llamacpp/models/{model_id}/model.gguf\n"
        f"name: {model_id}\n"
        f"size_bytes: {size}\n"
    )
    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)
        if not yml.exists() or yml.read_text() != yml_content:
            yml.write_text(yml_content)
    return status, model_id


def install_mlx(author: str, model_dir: str, src: Path, dry_run: bool):
    """Symlink the whole MLX model directory into Jan's mlx/models/."""
    model_id = f"{author}__{model_dir}" if author != "local" else model_dir
    dest = JAN_MLX / model_id
    return safe_symlink(src, dest, dry_run), model_id


def clean_dangling(root: Path, dry_run: bool) -> int:
    """Remove symlinks under root whose target no longer exists."""
    removed = 0
    if not root.exists():
        return 0
    for p in list(root.rglob("*")):
        if p.is_symlink() and not p.exists():
            if dry_run:
                print(f"  [DRY RUN] UNLINK (dangling): {p.relative_to(root)}")
            else:
                try:
                    p.unlink()
                    print(f"  UNLINK (dangling): {p.relative_to(root)}")
                except OSError as e:
                    print(f"  ERROR unlinking {p}: {e}")
                    continue
            removed += 1
    return removed


def main():
    ap = argparse.ArgumentParser(
        description="Symlink local GGUF + MLX models into Jan's model directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--clean", action="store_true",
                    help="Also remove symlinks whose targets no longer exist")
    ap.add_argument("--include-gguf-only", action="store_true",
                    help="Skip the MLX scan entirely")
    ap.add_argument("--gguf-root", type=Path, action="append", default=[],
                    help=f"Root to scan for .gguf (repeatable). "
                         f"Default: {', '.join(str(p) for p in DEFAULT_GGUF_ROOTS)}")
    ap.add_argument("--mlx-root", action="append", type=Path, default=[],
                    help="Extra root(s) to scan for MLX directories (repeatable)")
    ap.add_argument("--workers", type=int, default=16,
                    help="Parallel symlink workers (default 16)")
    args = ap.parse_args()

    gguf_roots = [r for r in (args.gguf_root or DEFAULT_GGUF_ROOTS) if r.exists()]

    mode = "DRY RUN — no changes" if args.dry_run else "LIVE"
    print("=" * 60)
    print(f"Jan Model Linker — {mode}")
    print("=" * 60)
    for r in gguf_roots:
        print(f"GGUF source:   {r}")
    print(f"Jan llamacpp:  {JAN_LLAMACPP}")
    print(f"Jan MLX:       {JAN_MLX}")
    print()

    if not args.dry_run:
        JAN_LLAMACPP.mkdir(parents=True, exist_ok=True)
        JAN_MLX.mkdir(parents=True, exist_ok=True)

    # ── GGUF ─────────────────────────────────────────────────
    # Gather first (serial walk), then parallelize symlink creation.
    print("Scanning GGUF files…")
    gguf_stats = {"created": 0, "exists": 0, "replace": 0, "skip-real-file": 0}
    seen_model_ids = set()
    gguf_tasks = []  # (author, model_dir, src, root)
    for root in gguf_roots:
        for author, model_dir, src in find_ggufs(root):
            # install_gguf derives model_id from (author, model_dir); precompute for dedup
            model_id = f"{author}__{model_dir}" if author != "local" else model_dir
            if model_id in seen_model_ids:
                continue
            seen_model_ids.add(model_id)
            gguf_tasks.append((author, model_dir, src, root))

    def _gguf_work(task):
        author, model_dir, src, root = task
        status, model_id = install_gguf(author, model_dir, src, args.dry_run)
        tag = "[DRY RUN] " if args.dry_run else ""
        _log(f"  {tag}{status:14s} {model_id} <- {src.relative_to(root)}")
        return status

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for status in pool.map(_gguf_work, gguf_tasks):
            gguf_stats[status] = gguf_stats.get(status, 0) + 1
    print(f"  GGUF: {gguf_stats}")
    print()

    # ── MLX ──────────────────────────────────────────────────
    if not args.include_gguf_only:
        roots = args.mlx_root + DEFAULT_MLX_ROOTS
        print(f"Scanning MLX directories in: {[str(r) for r in roots if r.exists()]}")
        mlx_stats = {"created": 0, "exists": 0, "replace": 0, "skip-real-file": 0}
        mlx_tasks = list(find_mlx_models(roots))

        def _mlx_work(task):
            author, model_dir, src = task
            status, model_id = install_mlx(author, model_dir, src, args.dry_run)
            tag = "[DRY RUN] " if args.dry_run else ""
            _log(f"  {tag}{status:14s} {model_id} <- {src}")
            return status

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for status in pool.map(_mlx_work, mlx_tasks):
                mlx_stats[status] = mlx_stats.get(status, 0) + 1
        print(f"  MLX:  {mlx_stats}")
        print()

    # ── Cleanup ─────────────────────────────────────────────
    if args.clean:
        print("Removing dangling symlinks…")
        removed = clean_dangling(JAN_LLAMACPP, args.dry_run)
        removed += clean_dangling(JAN_MLX, args.dry_run)
        print(f"  Dangling symlinks removed: {removed}")
        print()

    print("Done. Restart Jan to pick up the new model entries.")


if __name__ == "__main__":
    main()
