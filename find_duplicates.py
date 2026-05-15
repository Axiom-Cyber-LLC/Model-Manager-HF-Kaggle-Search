#!/usr/bin/env python3
"""find_duplicates.py — locate duplicate Ollama-style blob files across roots.

Walks every storage root for files named `sha256-*` (Ollama's blob naming
convention), groups them by basename (= same content), and reports any
blob that exists at more than one filesystem path.

Each duplicate is annotated with:
  - total size, total reclaimable (size × (copies - 1))
  - whether the blob is referenced by any Ollama manifest (= load-bearing)
    or orphaned (= safe-to-delete from any location after manual review)
  - whether the multiple paths are hardlinks (same inode → not a real dup)
    versus separate copies (different inodes → real dup, real reclaim)

Why the previous version did not work:
  - It compared Ollama's `digest` (from /api/tags) to blob filenames on disk.
    But /api/tags returns the MANIFEST digest, not blob digests. Blobs are
    named after their own sha256 (referenced inside the manifest's layers).
    The two hashes never collide, so no output ever appeared.
  - It only walked the SSD, missing <REDACTED_PATH> on the boot drive.

Usage:
  python3 find_duplicates.py                                # default roots
  python3 find_duplicates.py --roots <REDACTED_PATH> <REDACTED_PATH>
  python3 find_duplicates.py --all                          # also list singletons
  python3 find_duplicates.py --json                         # machine-readable
  python3 find_duplicates.py --min-size 100M                # only blobs ≥ 100 MB
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

DEFAULT_ROOTS = [
    Path.home() / ".ollama" / "models",
    Path("<REDACTED_PATH>"),
]
OLLAMA_MANIFESTS = Path.home() / ".ollama" / "models" / "manifests"


# ── helpers ──────────────────────────────────────────────────────────


def parse_size(value: str) -> int:
    """Parse '100M', '1.5G', '500K', '12345' (bytes) → bytes."""
    value = value.strip().upper()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([KMGT]?)B?", value)
    if not m:
        raise argparse.ArgumentTypeError(f"unrecognized size: {value!r}")
    num = float(m.group(1))
    mult = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}[m.group(2)]
    return int(num * mult)


def human_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(n)
    for u in units:
        if size < 1024 or u == units[-1]:
            return f"{size:.1f} {u}" if u != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} PB"


# ── manifest parsing ─────────────────────────────────────────────────


def in_use_blobs() -> dict[str, list[str]]:
    """Walk Ollama manifests; return {blob_filename: [model_label, ...]}.

    Manifest path layout: <REDACTED_PATH><host>/<owner>/<name>/<tag>
    Manifest content (JSON): {"layers": [{"digest": "sha256:abc..."}, ...],
                              "config": {"digest": "sha256:def..."}}
    Blob files on disk are named `sha256-abc...` (colon → dash).
    """
    in_use: dict[str, list[str]] = defaultdict(list)
    if not OLLAMA_MANIFESTS.is_dir():
        return in_use
    for manifest in OLLAMA_MANIFESTS.rglob("*"):
        if not manifest.is_file():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        try:
            tail = manifest.relative_to(OLLAMA_MANIFESTS).parts
            # Drop the host segment (e.g. registry.ollama.ai); join the rest.
            model_label = "/".join(tail[1:]) if len(tail) > 1 else manifest.name
        except ValueError:
            model_label = manifest.name
        digests: list[str] = []
        for layer in data.get("layers") or []:
            if isinstance(layer, dict):
                d = layer.get("digest", "")
                if isinstance(d, str) and d.startswith("sha256:"):
                    digests.append(d)
        cfg = data.get("config")
        if isinstance(cfg, dict):
            d = cfg.get("digest", "")
            if isinstance(d, str) and d.startswith("sha256:"):
                digests.append(d)
        for d in digests:
            blob_filename = "sha256-" + d.split(":", 1)[1]
            in_use[blob_filename].append(model_label)
    return in_use


# ── blob discovery ───────────────────────────────────────────────────


def find_blob_files(roots: Iterable[Path]) -> dict[str, list[Path]]:
    """Walk every root for files matching sha256-*; group by basename."""
    out: dict[str, list[Path]] = defaultdict(list)
    for root in roots:
        if not root.exists():
            print(f"  skip (does not exist): {root}", file=sys.stderr)
            continue
        if not root.is_dir():
            print(f"  skip (not a directory): {root}", file=sys.stderr)
            continue
        for path in root.rglob("sha256-*"):
            try:
                if path.is_file():
                    out[path.name].append(path)
            except OSError:
                continue
    return out


def classify_paths(paths: list[Path]) -> tuple[bool, int]:
    """Return (is_hardlinked, distinct_copy_count). Hardlinks share content."""
    inodes: set[tuple[int, int]] = set()
    for p in paths:
        try:
            s = p.stat()
            inodes.add((s.st_dev, s.st_ino))
        except OSError:
            continue
    return (len(inodes) == 1 and len(paths) > 1, len(inodes))


def blob_size(paths: list[Path]) -> int:
    for p in paths:
        try:
            return p.stat().st_size
        except OSError:
            continue
    return 0


# ── main ─────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Find duplicate Ollama blob files across storage roots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:", 1)[1] if "Usage:" in (__doc__ or "") else "",
    )
    ap.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        default=DEFAULT_ROOTS,
        help=f"Roots to scan (default: {[str(r) for r in DEFAULT_ROOTS]})",
    )
    ap.add_argument("--all", action="store_true",
                    help="Show every blob, not just duplicates")
    ap.add_argument("--json", action="store_true",
                    help="Emit JSON instead of human-readable report")
    ap.add_argument("--min-size", type=parse_size, default=0,
                    help="Skip blobs smaller than this (e.g. 100M, 1G)")
    args = ap.parse_args()

    roots = [Path(r).expanduser().resolve() for r in args.roots]
    print(f"Scanning {len(roots)} root(s) for sha256-* blob files...", file=sys.stderr)
    for r in roots:
        print(f"  - {r}", file=sys.stderr)
    blobs = find_blob_files(roots)
    in_use = in_use_blobs()

    if args.min_size:
        blobs = {sha: ps for sha, ps in blobs.items() if blob_size(ps) >= args.min_size}

    duplicates = {sha: ps for sha, ps in blobs.items() if len(ps) > 1}
    print(
        f"  Found {len(blobs)} unique blob name(s); {len(duplicates)} appear at >1 path.",
        file=sys.stderr,
    )

    target = blobs if args.all else duplicates

    if args.json:
        out = []
        for sha, paths in target.items():
            size = blob_size(paths)
            hardlinked, distinct = classify_paths(paths)
            out.append({
                "blob": sha,
                "size_bytes": size,
                "size_human": human_size(size),
                "paths": [str(p) for p in paths],
                "copy_count": len(paths),
                "distinct_copies": distinct,
                "hardlinked": hardlinked,
                "reclaimable_bytes": 0 if hardlinked else size * (len(paths) - 1),
                "in_use_by": in_use.get(sha, []),
            })
        # Largest reclaim first
        out.sort(key=lambda x: x["reclaimable_bytes"], reverse=True)
        json.dump(out, sys.stdout, indent=2)
        print()
        return 0

    if not target:
        msg = "no blobs found at the given roots." if args.all else "no duplicate blobs found."
        print(f"\n{msg}")
        return 0

    # Sort by reclaim size descending so the biggest wins surface first.
    sorted_target = sorted(target.items(), key=lambda kv: blob_size(kv[1]), reverse=True)
    total_reclaim = 0
    in_use_dup_count = 0
    orphan_dup_count = 0

    print()
    print("=" * 78)
    print("Duplicate Ollama-style blobs" if not args.all else "All Ollama-style blobs")
    print("=" * 78)
    for sha, paths in sorted_target:
        size = blob_size(paths)
        hardlinked, distinct = classify_paths(paths)
        used_by = in_use.get(sha, [])
        reclaim = 0 if hardlinked else size * (len(paths) - 1)
        if not args.all and not hardlinked:
            total_reclaim += reclaim
            if used_by:
                in_use_dup_count += 1
            else:
                orphan_dup_count += 1

        if used_by:
            usage = "IN USE: " + ", ".join(used_by[:3])
            if len(used_by) > 3:
                usage += f"  (+{len(used_by) - 3} more)"
        else:
            usage = "ORPHAN (no Ollama manifest references)"

        link_note = "  [HARDLINKED — same inode, not real reclaim]" if hardlinked else ""
        print()
        print(f"  {sha}")
        print(f"    size={human_size(size)}  copies={len(paths)}  reclaim={human_size(reclaim)}{link_note}")
        print(f"    {usage}")
        for p in paths:
            print(f"      - {p}")

    print()
    print("=" * 78)
    print("Summary")
    print("=" * 78)
    print(f"  Duplicate blob entries:        {len(duplicates)}")
    print(f"    in use by Ollama manifests:  {in_use_dup_count}")
    print(f"    orphan (no manifest ref):    {orphan_dup_count}")
    print(f"  Total reclaimable (non-hardlink dups): {human_size(total_reclaim)}")
    print()
    print("Note: this script never deletes anything. Inspect the path list and")
    print("decide which copy of each duplicate to remove manually.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
