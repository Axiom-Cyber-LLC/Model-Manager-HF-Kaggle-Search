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


def find_manifest_roots(roots: Iterable[Path]) -> list[Path]:
    """Discover every Ollama-style manifests/ root.

    Always includes the canonical <REDACTED_PATH> if it exists.
    Also walks each configured scan root looking for sibling `manifests/`
    directories that contain a `registry.ollama.ai/` (or any registry host)
    subtree — catches secondary stores like <REDACTED_PATH>
    or <REDACTED_PATH>
    """
    found: list[Path] = []
    seen: set[Path] = set()

    def _add(p: Path) -> None:
        try:
            resolved = p.resolve()
        except OSError:
            resolved = p
        if resolved in seen:
            return
        seen.add(resolved)
        found.append(p)

    canonical = Path.home() / ".ollama" / "models" / "manifests"
    if canonical.is_dir():
        _add(canonical)

    for root in roots:
        if not root.is_dir():
            continue
        for manifests_dir in root.rglob("manifests"):
            try:
                if not manifests_dir.is_dir():
                    continue
                # Heuristic: only treat it as a real Ollama manifests root if
                # it contains a registry.* subdir (Ollama's standard layout).
                children = [c for c in manifests_dir.iterdir() if c.is_dir()]
                if not any(c.name.startswith("registry.") for c in children):
                    continue
                _add(manifests_dir)
            except OSError:
                continue
    return found


def in_use_blobs(roots: Iterable[Path] | None = None) -> dict[str, list[str]]:
    """Walk every discovered Ollama manifests/ root; aggregate blob references.

    Returns: {blob_filename: ["[store-prefix] model/tag", ...]}.
    The store-prefix lets you tell which store referenced the blob when
    multiple stores exist on the system.
    """
    in_use: dict[str, list[str]] = defaultdict(list)
    manifest_roots = find_manifest_roots(roots or [])
    for mroot in manifest_roots:
        # A short store identifier for the label — strip a common prefix where possible
        store_label = str(mroot)
        for prefix, short in (
            (str(Path.home() / ".ollama" / "models" / "manifests"), "[<REDACTED_PATH>]"),
        ):
            if store_label == prefix:
                store_label = short
                break
        else:
            # Compact label for SSD-side stores
            store_label = f"[{Path(store_label).parent.name}]"
        for manifest in mroot.rglob("*"):
            if not manifest.is_file():
                continue
            # Skip dotfiles and obvious junk (.DS_Store, ._foo, etc.)
            # macOS Finder leaves binary .DS_Store files in any browsed
            # directory; if treated as JSON they hit UnicodeDecodeError.
            if manifest.name.startswith(".") or manifest.name.startswith("._"):
                continue
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError):
                # ValueError covers UnicodeDecodeError and any other parse weirdness
                continue
            try:
                tail = manifest.relative_to(mroot).parts
                # Drop the host segment (e.g. registry.ollama.ai); join the rest.
                model_label = "/".join(tail[1:]) if len(tail) > 1 else manifest.name
            except ValueError:
                model_label = manifest.name
            label = f"{store_label} {model_label}"
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
                in_use[blob_filename].append(label)
    return in_use


# ── blob discovery ───────────────────────────────────────────────────


def find_orphan_blobs(
    blobs: dict[str, list[Path]],
    in_use: dict[str, list[str]],
) -> list[tuple[str, list[Path], int]]:
    """Find blobs that exist on disk but are NOT referenced by ANY manifest.

    Returns: [(blob_filename, [paths], total_size_one_copy), ...]
    Sorted by size descending so the biggest wins surface first.
    """
    out: list[tuple[str, list[Path], int]] = []
    for sha, paths in blobs.items():
        if sha in in_use:
            continue  # referenced — not an orphan
        # Use the first reachable path's size; all copies should be byte-identical
        size = 0
        for p in paths:
            try:
                size = p.stat().st_size
                break
            except OSError:
                continue
        out.append((sha, paths, size))
    out.sort(key=lambda t: t[2], reverse=True)
    return out


def report_orphan_blobs(orphans: list[tuple[str, list[Path], int]]) -> int:
    """Print orphan-blob report. Returns total reclaimable bytes (counts all
    copies of each orphan since none are referenced)."""
    if not orphans:
        print("\nNo orphan blobs found.")
        return 0
    total = 0
    print()
    print("=" * 78)
    print("Orphan Ollama-style blobs (sha256-* files NOT referenced by ANY manifest)")
    print("=" * 78)
    print("These are safe to delete — no Ollama model registration points at them.")
    print()
    for sha, paths, size in orphans:
        per_blob_total = size * len(paths)
        total += per_blob_total
        copy_label = f"copies={len(paths)}" if len(paths) > 1 else ""
        print(f"  {sha[:32]}...  size={_human_size_inline(size):>10}  {copy_label}")
        for p in paths:
            print(f"    - {p}")
    print()
    print(f"  Total orphans: {len(orphans)} blob(s); reclaimable: {_human_size_inline(total)}")
    return total


def _human_size_inline(n: int) -> str:
    # Local alias for the existing human_size helper — keeps this section
    # self-contained without circular import concerns.
    return human_size(n)


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


# ── multi-quant cluster discovery ────────────────────────────────────
# Catches the "modelmgr/hfdownloader pulled the entire mradermacher kitchen-
# sink repo and now I have 8-10 quants of the same model" scenario.
# Different from blob-dup detection: each quant is a DIFFERENT file with a
# unique sha256, but they all derive from the same base model. Group by base
# name (with quant suffix stripped) so the user can pick which quant(s) to
# keep and delete the rest.

# Recognized quant tokens, in approximate quality-descending order. Matched
# case-insensitively. Ordering is used for the "recommended keep" suggestion.
_QUANT_TOKENS_ORDERED = [
    "F32", "F16", "BF16", "FP16", "FP32",
    "Q8_0", "Q6_K",
    "Q5_K_M", "Q5_K_S", "Q5_0", "Q5_1",
    "Q4_K_M", "Q4_K_S", "Q4_0", "Q4_1",
    "IQ4_XS", "IQ4_NL",
    "Q3_K_L", "Q3_K_M", "Q3_K_S",
    "IQ3_M", "IQ3_S", "IQ3_XS", "IQ3_XXS",
    "Q2_K", "Q2_K_S",
    "IQ2_M", "IQ2_S", "IQ2_XS", "IQ2_XXS",
    "IQ1_M", "IQ1_S",
]
# Recommended keep — first match in this list wins. Q4_K_M is the standard
# size/quality sweet spot for general-purpose use; falls back through other
# common picks if the user only has higher- or lower-precision variants.
_KEEP_PREFERENCE = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q4_K_S", "Q3_K_M", "Q8_0", "F16"]

_QUANT_RE = re.compile(
    r"""
    [._-]                       # separator before quant token (. _ -)
    (?:i\d+[._-])?              # optional i1- / i2- prefix (mradermacher imatrix series)
    (
        I?Q\d+(?:_[A-Z0-9]+)*   # Q-series: Q2_K, Q4_K_M, IQ3_XXS, IQ4_XS, etc.
        | F32 | F16 | BF16 | FP16 | FP32
    )
    (?:[._-]\d+[-_]of[-_]\d+)?  # shard suffix BEFORE .gguf: -00001-of-00003
    (?:\.gguf|\.safetensors)
    (?:\.part\d+of\d+)?         # shard suffix AFTER .gguf (mradermacher: .part1of3)
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)


def model_name_signature(filename: str) -> tuple[str, str | None]:
    """Return (base_model_name, quant_token) by stripping the quant suffix.

    Returns (filename, None) when no recognized quant token is found —
    such files are unique enough to skip from grouping."""
    m = _QUANT_RE.search(filename)
    if not m:
        return (filename, None)
    base = filename[: m.start()]
    quant = m.group(1).upper()
    # Normalize FP16 → F16 for grouping consistency
    if quant == "FP16":
        quant = "F16"
    if quant == "FP32":
        quant = "F32"
    return (base, quant)


def find_quant_clusters(roots: Iterable[Path]) -> dict[str, dict[str, list[Path]]]:
    """Walk every root for *.gguf and *.safetensors; group by base model name.

    Returns: {base_name: {quant_token: [path1, path2, ...]}}
    Multi-shard quants legitimately have multiple paths under one quant key.
    """
    out: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for ext in (".gguf", ".safetensors"):
            for path in root.rglob(f"*{ext}"):
                try:
                    if not path.is_file():
                        continue
                except OSError:
                    continue
                base, quant = model_name_signature(path.name)
                if quant is None:
                    continue  # not a recognized quant variant; skip
                out[base][quant].append(path)
    return out


def quant_total_size(paths: list[Path]) -> int:
    total = 0
    for p in paths:
        try:
            total += p.stat().st_size
        except OSError:
            continue
    return total


def recommend_keep_quant(quants_present: list[str]) -> str | None:
    """Pick the most useful quant to keep from the available set."""
    upper = [q.upper() for q in quants_present]
    for pref in _KEEP_PREFERENCE:
        if pref in upper:
            return pref
    # Otherwise: highest in the quality-descending list that's present
    for q in _QUANT_TOKENS_ORDERED:
        if q in upper:
            return q
    return quants_present[0] if quants_present else None


def report_quant_clusters(
    clusters: dict[str, dict[str, list[Path]]],
    min_size_bytes: int = 0,
    min_variants: int = 2,
) -> tuple[int, int]:
    """Print the multi-quant report. Returns (cluster_count, reclaimable_bytes)."""
    # Drop clusters below the variant-count threshold and the size floor
    interesting: list[tuple[str, dict[str, list[Path]], int, int]] = []
    for base, by_quant in clusters.items():
        if len(by_quant) < min_variants:
            continue
        cluster_total = sum(quant_total_size(paths) for paths in by_quant.values())
        if cluster_total < min_size_bytes:
            continue
        keep = recommend_keep_quant(list(by_quant.keys()))
        keep_size = quant_total_size(by_quant.get(keep, [])) if keep else 0
        reclaim = cluster_total - keep_size
        interesting.append((base, by_quant, cluster_total, reclaim))
    if not interesting:
        return (0, 0)
    interesting.sort(key=lambda t: t[3], reverse=True)
    total_reclaim = 0
    print()
    print("=" * 78)
    print("Multi-quant clusters (same base model, different quants)")
    print("=" * 78)
    for base, by_quant, cluster_total, reclaim in interesting:
        keep = recommend_keep_quant(list(by_quant.keys()))
        total_reclaim += reclaim
        print()
        print(f"  Base: {base.rstrip('._-')}")
        print(f"    {len(by_quant)} quants on disk, total {human_size(cluster_total)}; "
              f"keep {keep} → reclaim {human_size(reclaim)}")
        # Sort quants by quality (highest first) for display
        order = {q: i for i, q in enumerate(_QUANT_TOKENS_ORDERED)}
        sorted_quants = sorted(by_quant.items(), key=lambda kv: order.get(kv[0], 999))
        for quant, paths in sorted_quants:
            sz = quant_total_size(paths)
            keep_marker = "  ← KEEP" if quant == keep else ""
            print(f"      {quant:<8} {human_size(sz):>10}{keep_marker}")
            for p in paths:
                print(f"        - {p}")
    return (len(interesting), total_reclaim)


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
    ap.add_argument("--mode", choices=["both", "blobs", "quants", "orphans", "all"], default="both",
                    help=("Which analysis to run.\n"
                          "  blobs   — sha256-* file dedup (byte-identical, manifest-aware)\n"
                          "  quants  — same-base-model multi-quant clusters in *.gguf/*.safetensors\n"
                          "  orphans — sha256-* files on disk NOT referenced by any Ollama manifest\n"
                          "  both    — blobs + quants (default)\n"
                          "  all     — blobs + quants + orphans"))
    ap.add_argument("--min-variants", type=int, default=2,
                    help="In quants mode: only show clusters with at least N quant variants. Default 2.")
    args = ap.parse_args()

    roots = [Path(r).expanduser().resolve() for r in args.roots]
    print(f"Scanning {len(roots)} root(s)...", file=sys.stderr)
    for r in roots:
        print(f"  - {r}", file=sys.stderr)

    # ── Multi-quant cluster analysis (early — cheap stat()-only walk) ──
    if args.mode in ("both", "quants", "all"):
        print(f"  Walking *.gguf and *.safetensors for multi-quant clusters...",
              file=sys.stderr)
        clusters = find_quant_clusters(roots)
        cluster_count, quant_reclaim = report_quant_clusters(
            clusters,
            min_size_bytes=args.min_size,
            min_variants=args.min_variants,
        )
        if cluster_count == 0:
            print("\n(no multi-quant clusters meeting the variant threshold)")
        else:
            print()
            print("=" * 78)
            print("Multi-quant summary")
            print("=" * 78)
            print(f"  Clusters with >= {args.min_variants} quant(s): {cluster_count}")
            print(f"  Total reclaimable if you keep one quant per cluster: {human_size(quant_reclaim)}")

    if args.mode == "quants":
        # Skip the sha256-blob analysis entirely
        print()
        print("Note: this script never deletes anything. Inspect the path list and")
        print("decide which file(s) to remove manually.")
        return 0

    print(f"  Walking sha256-* files for byte-identical dups...", file=sys.stderr)
    blobs = find_blob_files(roots)

    # ── Orphan-blob analysis (sha256-* on disk, NOT referenced by any manifest) ──
    # Runs in `orphans` and `all` modes. Also computed and shown opportunistically
    # in `both` mode as a small summary line (full report needs --mode orphans/all).
    if args.mode in ("orphans", "all"):
        print(f"  Aggregating manifests from all stores to compute orphans...",
              file=sys.stderr)
        in_use_for_orphans = in_use_blobs(roots)
        orphans = find_orphan_blobs(blobs, in_use_for_orphans)
        # Apply min-size filter to orphan section too
        if args.min_size:
            orphans = [(sha, paths, sz) for sha, paths, sz in orphans if sz >= args.min_size]
        orphan_total = report_orphan_blobs(orphans)
        if args.mode == "orphans":
            print()
            print("Note: this script never deletes anything. Inspect the path list and")
            print("decide which file(s) to remove manually.")
            return 0
    # Aggregate manifests from every Ollama-style store under any root,
    # not just <REDACTED_PATH> Catches secondary stores like models-flat/manifests/.
    in_use = in_use_blobs(roots)

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
