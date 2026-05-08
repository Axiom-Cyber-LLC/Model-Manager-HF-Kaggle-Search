#!/usr/bin/env python3
"""
Register the flat-dir models with Ollama (multi-threaded).

For every GGUF in <Your Model Directory>/local/{short}/ that is
actually a text-generation model (not an image/video diffusion GGUF), this
script creates a Modelfile and runs `ollama create` under a threaded pool.

Image/video diffusion GGUFs (Flux, SDXL, Z-Image, Pixelwave, Wan, Qwen-
Image-Edit, etc.) are detected and skipped — they crash llama.cpp's loader
and throw 500 errors in Ollama.

State is cached at <REDACTED_PATH> (mtime-based);
unchanged blobs won't be re-registered unless --force is passed.

Usage:
    python Prepare_models_for_Ollama.py                  # register all
    python Prepare_models_for_Ollama.py --dry-run        # preview
    python Prepare_models_for_Ollama.py --force          # re-register all
    python Prepare_models_for_Ollama.py --workers 16     # tune parallelism
    python Prepare_models_for_Ollama.py --clean-orphans  # remove broken entries
"""

import argparse
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from prepare_models_env import extend_scan_roots


# ── Defaults ──────────────────────────────────────────────────────────

HOME = Path.home()
FLAT_MODELS = Path("<Your Model Directory>/local")
STATE_FILE = HOME / ".ollama" / "models" / "ai_model_state.json"

# Unified scan roots — scanned recursively (rglob, unlimited depth).
# Dedup is by resolved physical path so symlinks / overlapping paths don't
# double-register the same blob. Missing roots are skipped silently.
SCAN_ROOTS = extend_scan_roots([
    Path("<Your Model Directory>"),
    Path("<Your Model Directory>"),
    Path("<Your Model Directory>/huggingface/model"),
    Path("<Your Model Directory>/local"),
    Path("<REDACTED_PATH>"),
    Path.home() / ".cache" / "huggingface",   # symlink to the SSDE one
    Path("<Your Model Directory>"),
    Path("<REDACTED_PATH>"),
    Path.home() / "model_downloads" / "huggingface" / "model",
    Path.home() / "Library" / "Application Support" / "nomic.ai" / "GPT4All",
    Path.home() / ".lmstudio" / "models",
    Path("<REDACTED_PATH>"),
    Path("<REDACTED_PATH>"),
])

# Kept for the --clean-orphans path; NOT applied during registration anymore
# (user instruction: register every GGUF without name-based filtering).
IMAGE_MODEL_PATTERNS = re.compile(r"(?!x)x")  # never matches(flux|z.?image|z.?img|sdxl|sd[-_.]xl|pixelwave|"

# Filename pattern for shards 2+ in a multi-part GGUF (first shard = 00001).
# This is structural (Ollama only knows shard 00001 — given any other shard it
# can't stitch the model). Kept as a structural exclusion.
NON_FIRST_SHARD = re.compile(r"-0*(?!0*1\b)\d+-of-\d+\.gguf$", re.IGNORECASE)

print_lock = threading.Lock()
def _log(msg):
    with print_lock:
        print(msg)


# ── GGUF parsing ─────────────────────────────────────────────────────

GGUF_MAGIC = b"GGUF"


def probe_gguf(path: Path) -> dict:
    """Read GGUF header: magic, version, tensor_count, kv_count."""
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != GGUF_MAGIC:
                return {"ok": False, "reason": "not GGUF"}
            version = struct.unpack("<I", f.read(4))[0]
            tensor_count = struct.unpack("<Q", f.read(8))[0]
            kv_count = struct.unpack("<Q", f.read(8))[0]
            return {
                "ok": True, "version": version,
                "tensors": tensor_count, "kv_pairs": kv_count,
            }
    except (OSError, struct.error) as e:
        return {"ok": False, "reason": str(e)}


def is_chat_gguf(path: Path, short_name: str) -> tuple[bool, str]:
    """
    Decide whether to register a GGUF with Ollama. Per user direction we
    register every GGUF that is structurally a complete, standalone file —
    we no longer second-guess "is this a chat model" by name or metadata.

    The two remaining exclusions are STRUCTURAL, not content judgments:
      - mmproj companions are multimodal-projection helpers, not standalone
        models; registering one creates a broken Ollama entry by definition.
      - Shards 2+ of a split GGUF cannot be loaded on their own — Ollama
        stitches a multi-part GGUF when given shard 00001; pointing it at
        shard 2 is meaningless.
    """
    if "mmproj" in path.name.lower():
        return False, "mmproj companion (multimodal projection — not standalone)"
    if NON_FIRST_SHARD.search(path.name):
        return False, "non-first shard of a split GGUF (Ollama loads via shard 00001)"

    probe = probe_gguf(path)
    if not probe["ok"]:
        return False, f"GGUF probe failed: {probe['reason']}"
    return True, f"{probe['tensors']} tensors / {probe['kv_pairs']} kv (registered as-is)"


# ── State ────────────────────────────────────────────────────────────

def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text()) if STATE_FILE.is_file() else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_state(state: dict):
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, indent=2))
    except OSError as e:
        _log(f"  WARN: couldn't save state: {e}")


# ── Candidate discovery ─────────────────────────────────────────────

def find_gguf_candidates(scan_roots):
    """
    Yield (short_name, gguf_path, ollama_name) for every .gguf under any of
    the scan roots, recursively. Dedup is by resolved physical path AND by
    inode, so the same blob reached via symlinks OR hardlinks is registered
    exactly once.

    `scan_roots` may be a single Path (for backward-compat with the old
    signature) or any iterable of Paths.
    """
    # Backward-compat: accept a single Path
    if isinstance(scan_roots, (str, Path)):
        scan_roots = [Path(scan_roots)]

    seen_real: set[Path] = set()
    seen_inodes: set[tuple] = set()
    seen_ollama_names: set[str] = set()

    # Subdirectory NAMES we never descend into during a scan. These are
    # tool-internal layouts that contain symlinks/aliases to blobs we'll
    # already see via their canonical location.
    #   blobs/snapshots/.locks/refs  → HF cache + Ollama internals
    #   .studio_links                → LM Studio's per-blob hash dirs
    #   .git/__pycache__             → developer junk
    # NB: `.cache` is intentionally NOT here — the user's HF/ModelScope cache
    # roots include `.cache` as their own path component.
    SKIP_DIR_NAMES = {
        "blobs", ".locks", "refs",
        ".studio_links", ".git", "__pycache__",
    }

    quant_re = re.compile(
        r"[-_](Q\d[_\w]*|IQ\d[_\w]*|F16|F32|BF16|FP16)\.gguf$",
        re.IGNORECASE,
    )

    for root in scan_roots:
        if not root or not root.is_dir():
            continue
        try:
            root_resolved = root.resolve()
        except (OSError, RuntimeError):
            root_resolved = root

        for gguf in sorted(root.rglob("*.gguf")):
            try:
                if not gguf.is_file():
                    continue
            except OSError:
                continue

            # Determine the path relative to root for ancestor checks.
            try:
                rel_parts = gguf.relative_to(root).parts
            except ValueError:
                rel_parts = gguf.parts

            # Drop SKIP_DIR_NAMES anywhere in the relative path (the root
            # itself may legitimately start with a `.`).
            if any(p in SKIP_DIR_NAMES for p in rel_parts):
                continue

            # Drop any other dotted ancestor BELOW the root (e.g. random
            # `.aidiff`, `.tmp`). The root's own dotted prefix is fine.
            if any(p.startswith(".") and p not in (".", "..") for p in rel_parts[:-1]):
                continue

            # Dedup by resolved real path AND inode (catches hardlinks too)
            try:
                real = gguf.resolve()
            except (OSError, RuntimeError):
                real = gguf
            if real in seen_real:
                continue
            try:
                st = gguf.stat()
                inode_key = (st.st_dev, st.st_ino)
                if inode_key in seen_inodes:
                    continue
                seen_inodes.add(inode_key)
            except OSError:
                pass
            seen_real.add(real)

            # Short-name derivation. Best signal in priority order:
            #   1. HF-cache layout: models--{pub}--{model}/snapshots/{sha}/file.gguf
            #   2. flat layout:     {root}/{model_dir}/file.gguf  (parent.name)
            #   3. random GGUF:     file stem
            short = None
            for anc in gguf.parents:
                if anc.name.startswith("models--"):
                    bits = anc.name[len("models--"):].split("--", 1)
                    if len(bits) == 2 and all(bits):
                        short = f"{bits[0]}-{bits[1]}"
                    break
            if not short:
                short = gguf.parent.name or gguf.stem
            # Reject useless short-names (hash-only dirs, internal markers)
            if not short or short in ("hub", "blobs", "models"):
                short = gguf.stem
            short = short.strip(".-_") or gguf.stem

            # Quant tag detection
            m = quant_re.search(gguf.name)
            quant = m.group(1).upper() if m else "default"

            # If this dir holds multiple GGUFs, suffix with quant
            try:
                siblings = [s for s in gguf.parent.glob("*.gguf") if s.is_file()]
            except OSError:
                siblings = [gguf]
            ollama_name = f"local/{short}".lower()
            if len(siblings) > 1:
                ollama_name = f"local/{short}:{quant}".lower()

            # Avoid name collisions across different physical blobs
            if ollama_name in seen_ollama_names:
                tag = (real.name[-8:] if real.name else gguf.stem[-8:]).lower()
                ollama_name = f"{ollama_name}-{tag}"
            seen_ollama_names.add(ollama_name)

            yield short, gguf, ollama_name


# ── Ollama ops ──────────────────────────────────────────────────────

def ollama_available() -> bool:
    try:
        subprocess.run(["ollama", "--version"], capture_output=True,
                       check=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired):
        return False


def detect_ollama_storage_mismatch() -> dict:
    """
    Check whether OLLAMA_MODELS points somewhere that doesn't have the blob
    store + manifests Ollama actually needs. Returns {ok, target, blobs_dir,
    manifests_dir, hint} — if not ok, `hint` suggests a symlink fix.
    """
    target = os.environ.get("OLLAMA_MODELS")
    if not target:
        # Daemon default is <REDACTED_PATH> when env var unset; read daemon env directly
        try:
            # Find daemon PID and read its environment
            out = subprocess.run(["pgrep", "-f", "ollama serve"],
                                 capture_output=True, text=True, timeout=3)
            pid = out.stdout.split()[0] if out.stdout.strip() else None
            if pid:
                env_out = subprocess.run(["ps", "-E", "-p", pid],
                                         capture_output=True, text=True, timeout=3)
                for tok in env_out.stdout.split():
                    if tok.startswith("OLLAMA_MODELS="):
                        target = tok.split("=", 1)[1]
                        break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                FileNotFoundError):
            pass
    if not target:
        target = str(Path.home() / ".ollama" / "models")

    target_path = Path(target)
    blobs = target_path / "blobs"
    manifests = target_path / "manifests"
    blobs_ok = blobs.is_dir() and any(blobs.iterdir()) if blobs.exists() else False

    hint = None
    if not blobs_ok:
        # Look for a real blobs dir nearby (common pattern: <REDACTED_PATH> when
        # daemon points at <REDACTED_PATH>). Probe common siblings.
        for candidate in [
            Path("<Your Model Directory>"),
            Path.home() / ".ollama" / "models",
        ]:
            if (candidate / "blobs").is_dir() and any((candidate / "blobs").iterdir()):
                hint = (
                    f"Ollama daemon has OLLAMA_MODELS={target} but the real\n"
                    f"  blob store appears to live at {candidate}.\n"
                    f"  Fix with:\n"
                    f"    rmdir {target_path / 'blobs'} 2>/dev/null\n"
                    f"    rmdir {target_path / 'manifests'} 2>/dev/null\n"
                    f"    ln -s {candidate / 'blobs'} {target_path / 'blobs'}\n"
                    f"    ln -s {candidate / 'manifests'} {target_path / 'manifests'}"
                )
                break

    return {
        "ok": blobs_ok,
        "target": target,
        "blobs_dir": str(blobs),
        "manifests_dir": str(manifests),
        "hint": hint,
    }


def ollama_list() -> list[str]:
    try:
        out = subprocess.run(["ollama", "list"], capture_output=True,
                             check=True, text=True, timeout=10)
        return [line.split()[0] for line in out.stdout.splitlines()[1:]
                if line.strip()]
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []


def register_one(short: str, gguf: Path, ollama_name: str,
                 temp_dir: Path, dry_run: bool) -> tuple[str, str]:
    """
    Run `ollama create`. Returns (status, message) where status is one of:
      created | skipped-not-chat | skipped-cached | failed | dry-run
    """
    ok, reason = is_chat_gguf(gguf, short)
    if not ok:
        _log(f"  SKIP   {short:<50} ({reason})")
        return "skipped-not-chat", reason

    if dry_run:
        _log(f"  [DRY] REGISTER {ollama_name} <- {short}/{gguf.name} ({reason})")
        return "dry-run", reason

    modelfile = temp_dir / f"Modelfile-{ollama_name.replace('/', '-').replace(':', '_')}"
    modelfile.write_text(f"FROM {gguf}\n")

    try:
        subprocess.run(
            ["ollama", "create", ollama_name, "-f", str(modelfile)],
            capture_output=True, check=True, timeout=300,
        )
        _log(f"  OK     {ollama_name}")
        return "created", reason
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(errors="replace").strip() if e.stderr else "unknown"
        _log(f"  FAIL   {ollama_name}: {err.splitlines()[-1] if err else 'unknown'}")
        return "failed", err
    except subprocess.TimeoutExpired:
        _log(f"  FAIL   {ollama_name}: timeout")
        return "failed", "timeout"


def find_missing_blob_orphans(manifests_dir: Path, blobs_dir: Path) -> list[tuple[str, Path, list[str]]]:
    """
    Walk Ollama manifests and return (ollama_name, manifest_path, missing_digests)
    for every entry whose config or layer blobs no longer exist on disk.
    """
    out: list[tuple[str, Path, list[str]]] = []
    base = manifests_dir / "registry.ollama.ai"
    if not base.is_dir():
        return out

    for manifest in base.rglob("*"):
        if not manifest.is_file():
            continue
        try:
            content = json.loads(manifest.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        try:
            rel = manifest.relative_to(base)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) < 3:
            continue
        owner = parts[0]
        repo = "/".join(parts[1:-1])
        tag = parts[-1]
        ollama_name = f"{repo}:{tag}" if owner == "library" else f"{owner}/{repo}:{tag}"

        missing: list[str] = []
        cfg_digest = content.get("config", {}).get("digest", "") or ""
        if cfg_digest.startswith("sha256:"):
            if not (blobs_dir / cfg_digest.replace(":", "-")).exists():
                missing.append(cfg_digest)
        for layer in content.get("layers", []) or []:
            digest = layer.get("digest", "") or ""
            if digest.startswith("sha256:") and not (blobs_dir / digest.replace(":", "-")).exists():
                missing.append(digest)

        if missing:
            out.append((ollama_name, manifest, missing))
    return out


def clean_orphans(state: dict, dry_run: bool):
    """
    Remove Ollama registrations that no longer have working backing.

    Two passes:
      1. Image/diffusion entries by name pattern (legacy mistakes).
      2. Entries whose manifest references a config or layer blob that does
         not exist on disk (the "deleted the GGUF, registration sticks
         around" case).
    """
    current = ollama_list()
    removed = 0

    # Pass 1: image/diffusion name-pattern removals (preserved from old behavior)
    image_orphans: set[str] = set()
    for name in current:
        base = name.split(":", 1)[0]
        if IMAGE_MODEL_PATTERNS.search(base):
            image_orphans.add(name)
            _log(f"  RM (image)   {name}")
            if not dry_run:
                subprocess.run(["ollama", "rm", name], capture_output=True)
            removed += 1

    # Pass 2: missing-blob orphans (the actual "stale registration" cleanup)
    pre = detect_ollama_storage_mismatch()
    if pre.get("ok"):
        manifests_dir = Path(pre["manifests_dir"])
        blobs_dir = Path(pre["blobs_dir"])
        for ollama_name, _manifest, missing in find_missing_blob_orphans(manifests_dir, blobs_dir):
            if ollama_name in image_orphans:
                continue
            sample = ", ".join(d.split(":", 1)[1][:8] + "…" for d in missing[:2])
            extra = f" +{len(missing) - 2} more" if len(missing) > 2 else ""
            _log(f"  RM (orphan)  {ollama_name:<50}  missing blob: {sample}{extra}")
            if not dry_run:
                subprocess.run(["ollama", "rm", ollama_name], capture_output=True)
            removed += 1

    return removed


# ── Main ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Re-register even if mtime matches the cache")
    ap.add_argument("--root", type=Path, action="append", default=[],
                    help=f"Scan root (repeatable). If omitted, scans all default "
                         f"roots: {', '.join(str(r) for r in SCAN_ROOTS)}")
    ap.add_argument("--flat", type=Path, default=None,
                    help="Legacy single-flat-dir mode. Equivalent to --root.")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel ollama create workers (default 4; Ollama "
                         "serializes internally so more than ~8 rarely helps)")
    ap.add_argument("--clean-orphans", action="store_true",
                    help="Remove Ollama entries matching image/diffusion patterns")
    args = ap.parse_args()

    # Resolve the scan-root list. CLI flags override defaults.
    if args.root:
        scan_roots = args.root
    elif args.flat:
        scan_roots = [args.flat]
    else:
        scan_roots = SCAN_ROOTS

    mode = "DRY RUN" if args.dry_run else "LIVE"
    print("=" * 66)
    print(f"Ollama Registration — {mode}")
    print("=" * 66)
    print(f"Scan roots ({len(scan_roots)}):")
    for r in scan_roots:
        marker = "✓" if r.is_dir() else "✗ (missing)"
        print(f"  {marker} {r}")
    print(f"Workers:  {args.workers}")
    print()

    if not ollama_available():
        sys.exit("Error: `ollama` command not found or not responding.")

    # Detect OLLAMA_MODELS vs actual blob-store mismatch (the "CACHED but
    # ollama list empty" failure mode).
    diag = detect_ollama_storage_mismatch()
    if not diag["ok"]:
        print(f"  WARNING: Ollama's storage dir appears empty:")
        print(f"    OLLAMA_MODELS = {diag['target']}")
        print(f"    blobs dir     = {diag['blobs_dir']} (empty or missing)")
        if diag["hint"]:
            print()
            print(f"  {diag['hint']}")
            print()
            print(f"  Without this fix, `ollama create` calls will return HTTP 500 and")
            print(f"  our state file will falsely report 'CACHED'. Aborting.")
            sys.exit(1)
        else:
            print(f"  Could not find a backup blob store. Continuing anyway.")
            print()

    if args.clean_orphans:
        print("Cleaning image/diffusion orphans from Ollama…")
        removed = clean_orphans({}, args.dry_run)
        print(f"  Removed: {removed}\n")

    state = load_state() if not args.force else {}
    candidates = list(find_gguf_candidates(scan_roots))
    print(f"Found {len(candidates)} GGUF files across "
          f"{len(set(c[0] for c in candidates))} unique short-names "
          f"(deduped by resolved blob path)\n")

    if not candidates:
        print("Nothing to register.")
        return

    temp_dir = Path(tempfile.mkdtemp(prefix="ollama_modelfiles_"))
    stats = {"created": 0, "skipped-not-chat": 0,
             "skipped-cached": 0, "failed": 0, "dry-run": 0}
    errors = []

    try:
        def work(item):
            short, gguf, ollama_name = item
            try:
                mtime = str(int(gguf.stat().st_mtime))
            except OSError:
                return "failed", gguf, "stat failed"

            if not args.force and state.get(ollama_name) == mtime:
                _log(f"  CACHED  {ollama_name}")
                return "skipped-cached", gguf, "cached"

            status, msg = register_one(short, gguf, ollama_name,
                                       temp_dir, args.dry_run)
            if status == "created":
                state[ollama_name] = mtime
            elif status == "failed":
                errors.append((ollama_name, msg))
            return status, gguf, msg

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(work, c) for c in candidates]
            for f in as_completed(futures):
                status, _, _ = f.result()
                stats[status] = stats.get(status, 0) + 1
        elapsed = time.time() - t0

        if not args.dry_run:
            save_state(state)

    finally:
        try:
            # Cleanup modelfiles
            for p in temp_dir.iterdir():
                p.unlink(missing_ok=True)
            temp_dir.rmdir()
        except OSError:
            pass

    print()
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Result:  {stats}")
    if errors:
        print()
        print(f"Errors ({len(errors)}):")
        for name, msg in errors[:15]:
            print(f"  {name}: {msg[:200]}")
        if len(errors) > 15:
            print(f"  … +{len(errors) - 15} more")

    print()
    print("Done. Try `ollama list` to see registered models.")


if __name__ == "__main__":
    main()
