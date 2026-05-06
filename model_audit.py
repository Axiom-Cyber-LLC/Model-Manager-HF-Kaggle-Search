#!/usr/bin/env python3
"""
model_audit.py
--------------
Shared audit module + standalone CLI for the local-model tree.

Detects:
  - Duplicate models (bytewise-identical files via SHA-256)
  - Corrupt files (bad GGUF magic, LFS pointers, impossibly small)
  - Orphan HF blobs (blobs/ entries no snapshot symlinks point at)
  - Orphan multimodal projectors (mmproj-* without a base GGUF)
  - Orphan shards (shard 2..N without shard 00001)
  - Dangling symlinks (under app dirs; target missing)

For each finding, prints a recommendation + reason and follows the
user-defined deletion workflow (per-variant sizes, all associated locations,
explicit confirm, 3-second wait between confirm and rm).

Standalone:
    python model_audit.py [--workers N] [--dry-run] [--non-interactive]

Library:
    import model_audit
    report = model_audit.run_audit(roots, app_dirs=[...], workers=8)
    bytes_freed = model_audit.interactive_prompt(report, dry_run=False)
"""
from __future__ import annotations
import argparse
import hashlib
import os
import re
import shutil
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Iterable

# ── Defaults ─────────────────────────────────────────────────────────

HOME = Path.home()
DEFAULT_SCAN_ROOTS = [
    Path("/Volumes/ModelStorage/models"),
    Path("/Volumes/ModelStorage/models-flat"),
    Path("/Volumes/ModelStorage/.cache/huggingface"),
    HOME / ".cache" / "huggingface",
    Path("/Volumes/ModelStorage/.cache/modelscope"),
]
DEFAULT_APP_DIRS = [
    HOME / "Library" / "Application Support" / "nomic.ai" / "GPT4All",
    HOME / "Library" / "Application Support" / "Jan" / "data" / "llamacpp" / "models",
    HOME / "Library" / "Application Support" / "Jan" / "data" / "mlx" / "models",
    HOME / ".lmstudio" / "models",
]
CANONICAL_KEEP_DIR = Path("/Volumes/ModelStorage/models-flat/local")

GGUF_MAGIC = b"GGUF"
LFS_SIGNATURE = b"version https://git-lfs.github.com/spec/v1"
MIN_GGUF_BYTES = 50 * 1024 * 1024  # below this is a partial / corrupt file
MIN_SAFETENSORS_BYTES = 10 * 1024 * 1024
PROJECTOR_PREFIXES = ("mmproj-", "mmproj_", "mm-projector", "mm_projector")
NON_FIRST_SHARD = re.compile(r"-0*(?!0*1\b)\d+-of-\d+\.gguf$", re.IGNORECASE)
FIRST_SHARD = re.compile(r"-0*1-of-(\d+)\.gguf$", re.IGNORECASE)
ANY_SHARD = re.compile(r"-(0*\d+)-of-(\d+)\.gguf$", re.IGNORECASE)
PROJECTOR_QUANT_RE = re.compile(
    r"[-_](?:F16|F32|BF16|Q[0-9]+(?:_[A-Z0-9]+)*|FP16|FP32)$",
    re.IGNORECASE,
)
SKIP_DIR_NAMES = {"blobs", ".locks", "refs", ".studio_links", ".git", "__pycache__"}

_print_lock = Lock()


def _log(msg: str) -> None:
    with _print_lock:
        print(msg)


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} PB"


# ── Findings ─────────────────────────────────────────────────────────


@dataclass
class Finding:
    """Base type. Each subclass represents one kind of audit hit."""
    paths: list[Path] = field(default_factory=list)
    total_bytes: int = 0
    detail: str = ""

    @property
    def kind(self) -> str:
        return type(self).__name__


@dataclass
class DuplicateGroup(Finding):
    sha256: str = ""


@dataclass
class CorruptFile(Finding):
    reason: str = ""


@dataclass
class OrphanBlob(Finding):
    """An entry under an HF blobs/ dir with no snapshot symlink pointing at it."""


@dataclass
class OrphanProjector(Finding):
    likely_base: str = ""


@dataclass
class OrphanShard(Finding):
    expected_first_shard: str = ""


@dataclass
class DanglingSymlink(Finding):
    """A symlink under an app dir whose target no longer exists."""
    missing_target: str = ""
    repair_candidates: list[Path] = field(default_factory=list)


@dataclass
class AuditReport:
    findings: list[Finding] = field(default_factory=list)

    @property
    def total_recoverable_bytes(self) -> int:
        return sum(f.total_bytes for f in self.findings)

    def grouped(self) -> dict[type, list[Finding]]:
        out: dict[type, list[Finding]] = {}
        for f in self.findings:
            out.setdefault(type(f), []).append(f)
        return out

    def __len__(self) -> int:
        return len(self.findings)


# ── Low-level helpers ────────────────────────────────────────────────


def is_lfs_pointer(path: Path) -> bool:
    """A git-lfs pointer file: small text starting with the LFS signature."""
    try:
        if path.stat().st_size > 4096:
            return False
        with open(path, "rb") as f:
            return f.read(len(LFS_SIGNATURE)) == LFS_SIGNATURE
    except OSError:
        return False


def gguf_magic_ok(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == GGUF_MAGIC
    except OSError:
        return False


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def is_projector(path: Path) -> bool:
    name = path.name.lower()
    return any(name.startswith(p) for p in PROJECTOR_PREFIXES)


def projector_base_name(path: Path) -> str:
    stem = path.stem
    for p in PROJECTOR_PREFIXES:
        if stem.lower().startswith(p):
            stem = stem[len(p):]
            break
    return PROJECTOR_QUANT_RE.sub("", stem)


def _within_skip_dir(rel_parts: tuple) -> bool:
    return any(part in SKIP_DIR_NAMES for part in rel_parts)


def _below_dotted(rel_parts: tuple) -> bool:
    # the root itself may be `.cache/...`; we only flag dotted parts BELOW the root
    return any(part.startswith(".") and part not in (".", "..") for part in rel_parts[:-1])


def _index_model_files_by_name(roots: Iterable[Path]) -> dict[str, list[Path]]:
    """Index model-ish files by basename so dangling app symlinks can be repaired."""
    wanted_suffixes = {
        ".gguf", ".safetensors", ".bin", ".pt", ".pth", ".ckpt",
        ".onnx", ".mlmodel", ".mlpackage", ".h5", ".hdf5", ".keras",
    }
    by_name: dict[str, list[Path]] = {}
    seen: set[Path] = set()
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            try:
                if not p.is_file() or p.suffix.lower() not in wanted_suffixes:
                    continue
                real = p.resolve()
            except OSError:
                continue
            if real in seen:
                continue
            seen.add(real)
            by_name.setdefault(p.name.lower(), []).append(p)
    for paths in by_name.values():
        paths.sort(key=lambda x: (0 if CANONICAL_KEEP_DIR in x.parents else 1, str(x)))
    return by_name


def _find_symlink_repair_candidates(target_text: str, by_name: dict[str, list[Path]]) -> list[Path]:
    if not target_text:
        return []
    name = Path(target_text).name.lower()
    if not name:
        return []
    return list(by_name.get(name, []))


# ── Scanners ─────────────────────────────────────────────────────────


def _iter_model_files(roots: Iterable[Path]) -> Iterable[Path]:
    """Yield every .gguf and .safetensors file under any root, deduped by realpath."""
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in ("*.gguf", "*.safetensors"):
            for p in root.rglob(pattern):
                try:
                    if not p.is_file():
                        continue
                    rel = p.relative_to(root).parts
                except (OSError, ValueError):
                    continue
                if _within_skip_dir(rel) or _below_dotted(rel):
                    continue
                try:
                    real = p.resolve()
                except OSError:
                    real = p
                if real in seen:
                    continue
                seen.add(real)
                yield p


def find_corrupt_files(roots: Iterable[Path]) -> list[CorruptFile]:
    out: list[CorruptFile] = []
    for p in _iter_model_files(roots):
        try:
            size = p.stat().st_size
        except OSError as e:
            out.append(CorruptFile(paths=[p], total_bytes=0,
                                   reason=f"stat failed: {e}"))
            continue
        if is_lfs_pointer(p):
            out.append(CorruptFile(paths=[p], total_bytes=size,
                                   reason="git-lfs pointer (file body never downloaded)"))
            continue
        if p.suffix.lower() == ".gguf":
            if size < MIN_GGUF_BYTES:
                out.append(CorruptFile(paths=[p], total_bytes=size,
                                       reason=f"gguf too small ({human_bytes(size)} < "
                                              f"{human_bytes(MIN_GGUF_BYTES)})"))
                continue
            if not gguf_magic_ok(p):
                out.append(CorruptFile(paths=[p], total_bytes=size,
                                       reason="GGUF magic bytes missing"))
                continue
        elif p.suffix.lower() == ".safetensors":
            if size < MIN_SAFETENSORS_BYTES:
                out.append(CorruptFile(paths=[p], total_bytes=size,
                                       reason=f"safetensors too small "
                                              f"({human_bytes(size)})"))
    return out


def find_duplicates(roots: Iterable[Path], workers: int = 8) -> list[DuplicateGroup]:
    """SHA-256 groups of bytewise-identical files. Only hashes when sizes match."""
    by_size: dict[int, list[Path]] = {}
    for p in _iter_model_files(roots):
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if size < MIN_SAFETENSORS_BYTES:
            continue  # corruption check covers tiny files
        by_size.setdefault(size, []).append(p)

    candidates = [(size, paths) for size, paths in by_size.items() if len(paths) > 1]
    if not candidates:
        return []

    by_hash: dict[str, list[Path]] = {}

    def _hash_one(p: Path) -> tuple[Path, str | None]:
        try:
            return p, sha256_of(p)
        except OSError:
            return p, None

    todo: list[Path] = [p for _, paths in candidates for p in paths]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for path, digest in pool.map(_hash_one, todo):
            if digest is None:
                continue
            by_hash.setdefault(digest, []).append(path)

    out: list[DuplicateGroup] = []
    for digest, paths in by_hash.items():
        if len(paths) < 2:
            continue
        try:
            size = paths[0].stat().st_size
        except OSError:
            size = 0
        wasted = size * (len(paths) - 1)
        out.append(DuplicateGroup(paths=sorted(paths), total_bytes=wasted,
                                  sha256=digest, detail=f"{len(paths)} copies"))
    return out


def find_orphan_blobs(roots: Iterable[Path]) -> list[OrphanBlob]:
    """For every HF cache root, find blobs/ files no snapshots/* symlink points at."""
    out: list[OrphanBlob] = []
    for root in roots:
        if not root.is_dir():
            continue
        # HF layout: <root>/hub/models--{pub}--{model}/{blobs,refs,snapshots}/...
        for repo_dir in list(root.rglob("models--*--*")):
            blobs_dir = repo_dir / "blobs"
            snapshots_dir = repo_dir / "snapshots"
            if not blobs_dir.is_dir():
                continue
            referenced: set[Path] = set()
            if snapshots_dir.is_dir():
                for snap in snapshots_dir.rglob("*"):
                    if snap.is_symlink():
                        try:
                            target = (snap.parent / os.readlink(snap)).resolve()
                            referenced.add(target)
                        except OSError:
                            continue
            for blob in blobs_dir.iterdir():
                if not blob.is_file():
                    continue
                try:
                    real = blob.resolve()
                except OSError:
                    real = blob
                if real not in referenced:
                    try:
                        size = blob.stat().st_size
                    except OSError:
                        size = 0
                    out.append(OrphanBlob(paths=[blob], total_bytes=size,
                                          detail=f"under {repo_dir.name}"))
    return out


def find_orphan_projectors(roots: Iterable[Path]) -> list[OrphanProjector]:
    """mmproj-* GGUFs whose presumed base GGUF doesn't exist on disk."""
    all_ggufs: list[Path] = []
    projectors: list[Path] = []
    for p in _iter_model_files(roots):
        if p.suffix.lower() != ".gguf":
            continue
        all_ggufs.append(p)
        if is_projector(p):
            projectors.append(p)

    base_stems = [g.stem.lower() for g in all_ggufs if not is_projector(g)]
    out: list[OrphanProjector] = []
    for proj in projectors:
        target = projector_base_name(proj).lower()
        if target and any(target in stem for stem in base_stems):
            continue
        try:
            size = proj.stat().st_size
        except OSError:
            size = 0
        out.append(OrphanProjector(paths=[proj], total_bytes=size,
                                   likely_base=target,
                                   detail=f"no base matching '{target}'"))
    return out


def find_orphan_shards(roots: Iterable[Path]) -> list[OrphanShard]:
    """Shard 2..N where shard 00001 is missing in the same dir."""
    out: list[OrphanShard] = []
    for p in _iter_model_files(roots):
        if p.suffix.lower() != ".gguf":
            continue
        m = NON_FIRST_SHARD.search(p.name)
        if not m:
            continue
        # build the expected first-shard name in the same dir
        prefix = ANY_SHARD.sub("", p.name)
        first = next(p.parent.glob(f"{prefix}-*1-of-*.gguf"), None)
        if first is not None:
            continue
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        out.append(OrphanShard(paths=[p], total_bytes=size,
                               expected_first_shard=f"{prefix}-00001-of-*.gguf",
                               detail="first shard missing"))
    return out


def find_dangling_symlinks(app_dirs: Iterable[Path],
                           roots: Iterable[Path] | None = None) -> list[DanglingSymlink]:
    out: list[DanglingSymlink] = []
    by_name = _index_model_files_by_name(roots or [])
    for d in app_dirs:
        if not d.is_dir():
            continue
        for p in d.rglob("*"):
            if p.is_symlink() and not p.exists():
                try:
                    target = os.readlink(p)
                except OSError:
                    target = ""
                candidates = _find_symlink_repair_candidates(target, by_name)
                detail = f"target gone: {target}"
                if candidates:
                    detail += f"; repair candidate: {candidates[0]}"
                out.append(DanglingSymlink(paths=[p], total_bytes=0, detail=detail,
                                           missing_target=target,
                                           repair_candidates=candidates))
    return out


# ── Top-level audit ──────────────────────────────────────────────────


def run_audit(roots: Iterable[Path] | None = None,
              app_dirs: Iterable[Path] | None = None,
              workers: int = 8,
              skip_duplicates: bool = False) -> AuditReport:
    """Run every detector and bundle results into one AuditReport."""
    if roots is None:
        roots = DEFAULT_SCAN_ROOTS
    if app_dirs is None:
        app_dirs = DEFAULT_APP_DIRS
    roots = [Path(r) for r in roots]
    app_dirs = [Path(d) for d in app_dirs]

    report = AuditReport()

    _log("  → scanning for corrupt files / LFS pointers / size violations…")
    report.findings.extend(find_corrupt_files(roots))

    _log("  → scanning for orphan multimodal projectors…")
    report.findings.extend(find_orphan_projectors(roots))

    _log("  → scanning for orphan multi-part shards…")
    report.findings.extend(find_orphan_shards(roots))

    _log("  → scanning for orphan HF blobs (this can be slow)…")
    report.findings.extend(find_orphan_blobs(roots))

    _log("  → scanning for dangling symlinks under app dirs…")
    report.findings.extend(find_dangling_symlinks(app_dirs, roots=roots))

    if not skip_duplicates:
        _log(f"  → hashing for byte-identical duplicates ({workers} workers)…")
        report.findings.extend(find_duplicates(roots, workers=workers))

    return report


# ── Recommendations ──────────────────────────────────────────────────


def recommend(finding: Finding) -> tuple[bool, str]:
    """
    Return (delete_recommended, reason). Conservative: anything ambiguous
    returns False so the user makes the call, with the reason explaining.
    """
    if isinstance(finding, CorruptFile):
        return True, f"file is unloadable: {finding.reason}"

    if isinstance(finding, DuplicateGroup):
        keep = _pick_canonical(finding.paths)
        others = [p for p in finding.paths if p != keep]
        return True, (
            f"{len(finding.paths)} byte-identical copies (sha256 "
            f"{finding.sha256[:12]}…). Recommend keep {keep}, "
            f"delete {len(others)} other(s) to reclaim "
            f"{human_bytes(finding.total_bytes)}."
        )

    if isinstance(finding, OrphanBlob):
        return True, ("HF blob not referenced by any snapshot; the cache has "
                      "lost its symlink — the blob is unreachable from the model API.")

    if isinstance(finding, OrphanProjector):
        return False, ("multimodal projector with no base GGUF found locally. "
                       "Confirm whether you can re-download the matching base "
                       "before deleting; flagging only.")

    if isinstance(finding, OrphanShard):
        return False, ("non-first shard of a multi-part GGUF; the first shard "
                       "is missing locally. Confirm the model is unrecoverable "
                       "before deleting.")

    if isinstance(finding, DanglingSymlink):
        if finding.repair_candidates:
            return False, ("symlink target no longer exists, but a same-named model file "
                           "was found locally. Recommend rebuilding the symlink, not deleting data.")
        return False, ("symlink target no longer exists and no same-named replacement was "
                       "found. Recommend deleting only the broken symlink, not model data.")

    return False, "unknown finding type."


def _pick_canonical(paths: list[Path]) -> Path:
    """Prefer paths under CANONICAL_KEEP_DIR; tie-break on newest mtime."""
    canon_root = CANONICAL_KEEP_DIR.resolve()
    canon = []
    other = []
    for p in paths:
        try:
            real = p.resolve()
        except OSError:
            real = p
        if canon_root in real.parents or real == canon_root:
            canon.append(p)
        else:
            other.append(p)
    pool = canon or other

    def mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0
    return max(pool, key=mtime)


# ── Interactive deletion workflow ────────────────────────────────────


def _print_finding_block(idx: int, total: int, finding: Finding) -> None:
    print()
    print("=" * 78)
    print(f"[{idx}/{total}] {finding.kind} — {human_bytes(finding.total_bytes)}")
    print("=" * 78)
    if isinstance(finding, DuplicateGroup):
        print(f"sha256: {finding.sha256}")
        print(f"copies ({len(finding.paths)}):")
        for p in finding.paths:
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
            print(f"  - [{human_bytes(size):>10}] {p}")
    elif isinstance(finding, CorruptFile):
        print(f"reason: {finding.reason}")
        for p in finding.paths:
            print(f"  - {p}")
    elif isinstance(finding, OrphanProjector):
        print(f"likely base name: '{finding.likely_base}'")
        for p in finding.paths:
            print(f"  - [{human_bytes(finding.total_bytes):>10}] {p}")
    elif isinstance(finding, OrphanShard):
        print(f"expected first shard: {finding.expected_first_shard}")
        for p in finding.paths:
            print(f"  - [{human_bytes(finding.total_bytes):>10}] {p}")
    elif isinstance(finding, OrphanBlob):
        print(f"detail: {finding.detail}")
        for p in finding.paths:
            print(f"  - [{human_bytes(finding.total_bytes):>10}] {p}")
    elif isinstance(finding, DanglingSymlink):
        for p in finding.paths:
            print(f"  - {p}  →  {finding.detail}")
        if finding.repair_candidates:
            print("repair candidates:")
            for n, c in enumerate(finding.repair_candidates[:10], 1):
                print(f"  {n}. {c}")
            if len(finding.repair_candidates) > 10:
                print(f"  … {len(finding.repair_candidates) - 10} more")

    rec_delete, reason = recommend(finding)
    if isinstance(finding, DanglingSymlink) and finding.repair_candidates:
        rec = "REBUILD SYMLINK"
    elif isinstance(finding, DanglingSymlink):
        rec = "DELETE BROKEN SYMLINK ONLY / FLAG"
    else:
        rec = "DELETE" if rec_delete else "KEEP / FLAG"
    print()
    print(f"Recommendation: {rec}")
    print(f"  Why: {reason}")


def _resolve_repair_target(finding: DanglingSymlink) -> Path | None:
    if not finding.repair_candidates:
        return None
    return finding.repair_candidates[0]


def _repair_dangling_symlink(finding: DanglingSymlink, dry_run: bool) -> bool:
    if not finding.paths:
        return False
    link = finding.paths[0]
    target = _resolve_repair_target(finding)
    if target is None:
        print("  No repair candidate found.")
        return False
    print("  Will rebuild symlink:")
    print(f"    {link}")
    print(f"    -> {target}")
    if dry_run:
        print("  [DRY] would unlink the broken symlink and recreate it.")
        return True
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)
        print("  repaired symlink.")
        return True
    except OSError as e:
        print(f"  ERROR repairing symlink: {e}")
        return False


def _delete_broken_symlink_only(finding: DanglingSymlink, dry_run: bool) -> int:
    targets = [p for p in finding.paths if p.is_symlink() and not p.exists()]
    if not targets:
        print("  No broken symlink target remains.")
        return 0
    print(f"  Will delete {len(targets)} broken symlink(s) only in 3 seconds — Ctrl-C to abort…")
    try:
        time.sleep(3)
    except KeyboardInterrupt:
        print("\n  Aborted by user.")
        return 0
    for p in targets:
        if dry_run:
            print(f"  [DRY] would unlink broken symlink: {p}")
            continue
        try:
            p.unlink()
            print(f"  removed broken symlink: {p}")
        except OSError as e:
            print(f"  ERROR removing broken symlink {p}: {e}")
    return 0


def _resolve_deletion_targets(finding: Finding) -> list[Path]:
    """Which files would actually be removed if the user says yes."""
    if isinstance(finding, DanglingSymlink):
        return []
    if isinstance(finding, DuplicateGroup):
        keep = _pick_canonical(finding.paths)
        return [p for p in finding.paths if p != keep]
    return list(finding.paths)


def _delete_with_wait(targets: list[Path], dry_run: bool) -> int:
    """Per CLAUDE.md: wait 3s between confirm and rm; report bytes freed."""
    print(f"  Will delete {len(targets)} file(s) in 3 seconds — Ctrl-C to abort…")
    try:
        time.sleep(3)
    except KeyboardInterrupt:
        print("\n  Aborted by user.")
        return 0
    freed = 0
    for p in targets:
        try:
            size = p.stat().st_size if p.exists() else 0
        except OSError:
            size = 0
        if dry_run:
            print(f"  [DRY] would unlink: {p}")
            freed += size
            continue
        try:
            if p.is_symlink() or p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
            print(f"  removed: {p}")
            freed += size
        except OSError as e:
            print(f"  ERROR removing {p}: {e}")
    print(f"  Reclaimed: {human_bytes(freed)}")
    return freed


def interactive_prompt(report: AuditReport, dry_run: bool = False,
                       non_interactive: bool = False) -> int:
    """
    Walk every finding. Per the user's CLAUDE.md model-deletion workflow:
    list variants, list associated paths, recommend, ask explicitly, wait 3s.
    Returns total bytes freed.
    """
    if not report.findings:
        print()
        print("Audit clean — no duplicates, corrupt files, or orphans found.")
        return 0

    print()
    print(f"Audit found {len(report.findings)} finding(s), "
          f"{human_bytes(report.total_recoverable_bytes)} potentially recoverable.")
    grouped = report.grouped()
    for cls, group in grouped.items():
        print(f"  {cls.__name__}: {len(group)}")

    if non_interactive:
        print()
        print("--non-interactive set; skipping deletion prompts.")
        for i, f in enumerate(report.findings, 1):
            _print_finding_block(i, len(report.findings), f)
        return 0

    freed_total = 0
    for i, finding in enumerate(report.findings, 1):
        _print_finding_block(i, len(report.findings), finding)
        targets = _resolve_deletion_targets(finding)
        if not targets:
            continue
        try:
            answer = input(
                "\nAre you sure you want to delete this model with all "
                "associated data?\n  [yes / no / skip-all] > "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Aborted.")
            break
        if answer in ("skip-all", "stop", "quit", "q"):
            print("  Skipping the rest.")
            break
        if answer in ("yes", "y"):
            freed_total += _delete_with_wait(targets, dry_run)
        else:
            print("  Skipped.")

    print()
    print(f"Total reclaimed: {human_bytes(freed_total)}")
    return freed_total


# ── CLI ──────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel hashing workers for duplicate detection (default 8)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be deleted without unlinking anything")
    ap.add_argument("--non-interactive", action="store_true",
                    help="Print findings + recommendations, never prompt or delete")
    ap.add_argument("--root", type=Path, action="append", default=[],
                    help=f"Scan root (repeatable). Default: "
                         f"{', '.join(str(r) for r in DEFAULT_SCAN_ROOTS)}")
    ap.add_argument("--app-dir", type=Path, action="append", default=[],
                    help="App symlink dir to check for dangling links (repeatable)")
    ap.add_argument("--skip-duplicates", action="store_true",
                    help="Skip the SHA-256 duplicate-detection pass (fastest)")
    args = ap.parse_args()

    roots = args.root or DEFAULT_SCAN_ROOTS
    app_dirs = args.app_dir or DEFAULT_APP_DIRS

    print("=" * 78)
    print(f"Model audit — {'DRY RUN' if args.dry_run else 'LIVE'}")
    print("=" * 78)
    for r in roots:
        print(f"  scan root: {r}{'' if Path(r).is_dir() else '  (missing)'}")
    for d in app_dirs:
        print(f"  app dir  : {d}{'' if Path(d).is_dir() else '  (missing)'}")
    print()

    report = run_audit(roots=roots, app_dirs=app_dirs, workers=args.workers,
                       skip_duplicates=args.skip_duplicates)
    interactive_prompt(report, dry_run=args.dry_run,
                       non_interactive=args.non_interactive)
    return 0


if __name__ == "__main__":
    sys.exit(main())
