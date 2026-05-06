#!/usr/bin/env python3
"""
Prepare local models for Anaconda AI Navigator (a.k.a. "AI Studio").

AI Navigator only displays models that are registered in its LokiJS database
(`ailauncher.db`). It does NOT scan the filesystem for local GGUFs on its own.

This script registers our flat-dir models by injecting `LocalModel` entries
into the DB, matching each local file against AI Nav's catalog entries where
possible. Our file's actual SHA256/size/path replaces the catalog's — AI Nav
will show them as "Downloaded" and ready to run.

Also creates HF-style symlinks at the downloadLocation so any future catalog
scanner can find the files by path too.

Safety:
  - AI Navigator MUST be quit before running (it holds the DB open on some
    configurations and won't see our changes otherwise).
  - A timestamped backup of ailauncher.db is written before any mutation.

Usage:
    python Prepare_models_for_AIStudio.py                  # inject + symlink
    python Prepare_models_for_AIStudio.py --dry-run        # preview only
    python Prepare_models_for_AIStudio.py --workers 16     # parallel hashing
    python Prepare_models_for_AIStudio.py --no-hash        # skip sha256 (faster)
    python Prepare_models_for_AIStudio.py --revert         # restore latest backup
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


# ── Defaults ──────────────────────────────────────────────────────────

HOME = Path.home()
FLAT_ROOT = Path("/Volumes/ModelStorage/models-flat")
FLAT_MODELS = FLAT_ROOT / "local"
RENAME_PLAN = FLAT_ROOT / "rename-plan.json"

AI_NAV_DIR = HOME / "Library" / "Application Support" / "ai-navigator"
AI_NAV_CONFIG = AI_NAV_DIR / "config.json"
AI_NAV_DB = AI_NAV_DIR / "ailauncher.db"
AI_NAV_CATALOG = AI_NAV_DIR / "catalog.json"

_print_lock = threading.Lock()
def _log(msg):
    with _print_lock:
        print(msg)


# ── Helpers ──────────────────────────────────────────────────────────

def read_ai_nav_download_location() -> Path:
    if AI_NAV_CONFIG.is_file():
        try:
            cfg = json.loads(AI_NAV_CONFIG.read_text())
            if loc := cfg.get("downloadLocation"):
                return Path(loc)
        except (json.JSONDecodeError, OSError):
            pass
    return AI_NAV_DIR / "models"


def extract_publisher_and_model(original: str):
    """
    `models--{publisher}--{model}` → (publisher, model) or None.

    For `local`-publisher entries, we still return ("local", model) — these
    are local-only models that AI Nav should still know about, just without
    a real HF source.
    """
    if not original.startswith("models--"):
        return None
    rest = original[len("models--"):]
    parts = rest.split("--", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    publisher, model = parts
    # Filter only obvious non-models. `local` is allowed through — those are
    # legit user-converted GGUFs without HF attribution.
    if publisher == "_image-models":
        return None
    return publisher, model


def sha256_of(path: Path, chunk=1 << 20) -> str:
    """Stream-hash a file with SHA-256."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
    except OSError as e:
        raise RuntimeError(f"could not read {path}: {e}")
    return h.hexdigest()


def load_catalog():
    if not AI_NAV_CATALOG.is_file():
        sys.exit(f"Error: AI Nav catalog not found at {AI_NAV_CATALOG}. "
                 f"Launch AI Navigator once to download it.")
    cat = json.loads(AI_NAV_CATALOG.read_text())
    return {m["id"]: m for m in cat}


# ── LokiJS DB helpers ────────────────────────────────────────────────

def load_db(path: Path) -> dict:
    return json.loads(path.read_text())


def save_db(db: dict, path: Path):
    # LokiJS uses compact JSON (no indent) in its serialization.
    path.write_text(json.dumps(db, separators=(",", ":")))


def get_models_collection(db: dict) -> dict:
    for c in db.get("collections", []):
        if c.get("name") == "models":
            return c
    # Shouldn't happen in a real AI Nav DB, but create if missing.
    coll = {
        "name": "models", "data": [], "idIndex": [], "binaryIndices": {},
        "constraints": None, "uniqueNames": ["name"], "transforms": {},
        "objType": "models", "dirty": True, "cachedIndex": None,
        "cachedBinaryIndex": None, "cachedData": None,
        "adaptiveBinaryIndices": True, "transactional": False,
        "cloneObjects": False, "cloneMethod": "parse-stringify",
        "asyncListeners": False, "disableMeta": False,
        "disableChangesApi": True, "disableDeltaChangesApi": True,
        "autoupdate": False, "ttl": None, "maxId": 0,
        "DynamicViews": [], "events": {}, "changes": [],
    }
    db.setdefault("collections", []).append(coll)
    return coll


def next_loki_id(coll: dict) -> int:
    max_id = coll.get("maxId", 0)
    return (max_id or 0) + 1


def now_meta(rev: int = 0) -> dict:
    return {
        "revision": rev,
        "created": int(time.time() * 1000),
        "version": 0,
    }


QUANT_RE = re.compile(r"[-_.](Q\d[_\w]*|IQ\d[_\w]*|F16|F32|BF16|FP16)\.gguf$",
                      re.IGNORECASE)

# Architectures AI Navigator's bundled llama.cpp CANNOT LOAD. AI Nav 1.x ships
# a frozen llama.cpp from pre-mid-2025; archs added upstream after that throw
# `unknown model architecture` on load. This list is a RUNTIME-CAPABILITY
# filter, not a content judgment — per user policy: "if it won't run in the
# app, exclude it; otherwise every model it's capable of running needs to show".
# For models with these archs use Ollama (latest) or LM Studio (with updated
# runtime backend) instead.
UNSUPPORTED_ARCHS = {
    # Text LLMs with new architectures llama.cpp < mid-2025 didn't know
    "mistral3", "qwen35", "qwen3_5", "qwen3next", "qwen3_next",
    "gemma3", "gemma4",
    "nemotron_h", "nemotron_h_moe", "phi4moe",
    # Vision models — AI Nav is text-only
    "qwen2vl", "qwen25vl", "qwen2_vl", "qwen25_vl", "llama4",
    # Diffusion — not chat at all, llama.cpp can't load image-gen GGUFs
    "flux", "sdxl", "stable_diffusion", "qwen_image", "lumina2", "wan",
    "z_image", "zimage",
}


def gguf_architecture(path: Path):
    """Read first string KV to extract general.architecture. Returns None if unknown."""
    import struct
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return None
            f.read(4)        # version
            f.read(8)        # tensor count
            kv_count = struct.unpack("<Q", f.read(8))[0]
            for _ in range(min(kv_count, 50)):
                klen = struct.unpack("<Q", f.read(8))[0]
                key = f.read(klen).decode("utf-8", errors="replace")
                vtype = struct.unpack("<I", f.read(4))[0]
                if vtype == 8:
                    slen = struct.unpack("<Q", f.read(8))[0]
                    val = f.read(slen).decode("utf-8", errors="replace")
                    if key == "general.architecture":
                        return val
                else:
                    return None
    except (OSError, struct.error):
        return None
    return None


def parse_quant(filename: str) -> str:
    m = QUANT_RE.search(filename)
    if m:
        return m.group(1).upper()
    low = filename.lower()
    if ".safetensors" in low:
        return "FP16"
    if "mlx" in low and "4bit" in low:
        return "4bit"
    if "mlx" in low and "8bit" in low:
        return "8bit"
    return "unknown"


def estimate_file_format(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".gguf":
        return "GGUF"
    if ext == ".safetensors":
        return "safetensors"
    return "unknown"


def estimate_max_ram(size_bytes: int, quant: str) -> int:
    """
    Estimate RAM needed to run the model. File size is a good lower bound;
    add ~15% for KV cache, CUDA/Metal overhead, activations.
    """
    return int(size_bytes * 1.15)


def estimate_n_cpus(size_bytes: int) -> int:
    """Rough CPU recommendation scaling with model size."""
    gb = size_bytes / (1024 ** 3)
    if gb < 3:
        return 2
    if gb < 10:
        return 4
    if gb < 30:
        return 6
    if gb < 80:
        return 8
    return 10


def build_local_model_file(catalog_file: dict | None, real_path: Path,
                           real_sha: str, real_size: int) -> dict:
    """Build a LocalModelFile from a catalog template (optional) + our real file."""
    quant = parse_quant(real_path.name)
    if catalog_file:
        out = dict(catalog_file)
    else:
        out = {
            "name": real_path.name,
            "downloadUrl": "",
            "quantization": quant,
            "format": estimate_file_format(real_path),
            "id": real_sha,
            "benchmarks": [],
            "evaluations": [],
            "generatedOn": "",
            "quantEngine": "",
        }
    # Override with reality
    out["name"] = real_path.name
    out["sizeBytes"] = real_size
    out["sha256checksum"] = real_sha
    out["id"] = real_sha
    out["localPath"] = str(real_path)
    out["isDownloaded"] = True
    out["downloadStatus"] = {
        "status": "completed",
        "localPath": str(real_path),
    }
    # Fill any missing numeric fields that AI Nav's UI formats (avoids "NaN undefined")
    if not out.get("maxRamUsage"):
        out["maxRamUsage"] = estimate_max_ram(real_size, quant)
    if out.get("estimatedNCpusReq") in (None, 0):
        out["estimatedNCpusReq"] = estimate_n_cpus(real_size)
    if not out.get("quantization"):
        out["quantization"] = quant
    if not out.get("format"):
        out["format"] = estimate_file_format(real_path)
    return out


PARAM_NAME_RE = re.compile(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*[Bb](?![A-Za-z])")


def estimate_n_parameters(model_name: str, largest_file_bytes: int,
                          quant: str) -> int:
    """
    Infer parameter count. Prefer parsing "7B", "32B", "120B" from the name;
    fall back to size / bits-per-weight estimation.
    """
    m = PARAM_NAME_RE.search(model_name)
    if m:
        return int(float(m.group(1)) * 1_000_000_000)
    # Bits per weight by quant (roughly)
    bits_per_weight = {
        "Q2_K": 3.0, "Q3_K_S": 3.5, "Q3_K_M": 3.9, "Q3_K_L": 4.3,
        "Q4_0": 4.5, "Q4_1": 5.0, "Q4_K_S": 4.6, "Q4_K_M": 4.8, "Q4_K_L": 5.2,
        "Q5_K_S": 5.5, "Q5_K_M": 5.7, "Q6_K": 6.6, "Q8_0": 8.5,
        "F16": 16, "FP16": 16, "BF16": 16, "F32": 32,
        "4BIT": 4.5, "8BIT": 8.5,
    }.get(quant.upper(), 5.0)
    bits_per_byte = 8
    approx = int(largest_file_bytes * bits_per_byte / bits_per_weight)
    # Round to nearest 10M; floor at 1M so tiny models don't come out 0
    rounded = round(approx / 10_000_000) * 10_000_000
    return max(1_000_000, rounded)


def build_local_model(catalog_entry: dict | None,
                      model_id: str,
                      publisher: str,
                      model_name: str,
                      local_files: list[dict]) -> dict:
    """Build a LocalModel suitable for insertion into the models collection."""
    if catalog_entry:
        out = {k: v for k, v in catalog_entry.items() if k != "files"}
    else:
        largest = max((f.get("sizeBytes", 0) for f in local_files), default=0)
        quant = local_files[0].get("quantization", "unknown") if local_files else "unknown"
        n_params = estimate_n_parameters(model_name, largest, quant)
        out = {
            "publisher": publisher,
            "name": model_name,
            "id": model_id,
            "resources": {"canonicalUrl": f"https://huggingface.co/{publisher}"},
            "license": "other",
            "languages": ["en"],
            "library_name": "transformers",
            "trainedFor": "text-generation",
            "numParameters": n_params,
            "description": f"Locally registered model — {publisher}/{model_name}",
            "model_type": "unknown",
            "contextWindowSize": 8192,
            "baseModel": "",
            "baseModels": [],
            "datasets": [],
            "groups": [{"name": "Local", "order": 0}],
            "tags": ["local"],
            "knowledgeCutOff": "",
            "sourceUrl": f"https://huggingface.co/{publisher}/{model_name}",
            "modelId": model_id,
            "infoUrl": "",
            "datePublished": "",
        }
    # Even catalog-backed entries might be missing some UI-used fields — fill defensively.
    # `is None` rather than falsy — numParameters can legitimately be small.
    if out.get("numParameters") in (None, 0):
        largest = max((f.get("sizeBytes", 0) for f in local_files), default=0)
        quant = local_files[0].get("quantization", "unknown") if local_files else "unknown"
        out["numParameters"] = estimate_n_parameters(model_name, largest, quant)
    if out.get("contextWindowSize") in (None, 0):
        out["contextWindowSize"] = 8192
    out["files"] = local_files
    return out


def safe_symlink_dir(target: Path, link: Path, dry_run: bool) -> str:
    try:
        if link.is_symlink():
            current = Path(os.readlink(link))
            if current == target.resolve() or current == target:
                return "exists"
            if dry_run:
                return "would-replace"
            link.unlink()
        elif link.exists():
            return "skipped-real-dir"
        if dry_run:
            return "would-create"
        link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(target, link)
        return "created"
    except OSError as e:
        return f"error:{e}"


# ── Main injection logic ────────────────────────────────────────────

def discover_local_models(plan_path: Path, flat_models: Path):
    """
    Yield (hf_publisher, hf_model, short_name, src_dir) for every model in the
    rename-plan that maps to a real HF publisher (not `local`/`_image-models`).
    """
    if not plan_path.is_file():
        return
    plan = json.loads(plan_path.read_text())
    for entry in plan.get("entries", []):
        short = entry["short_name"]
        src_dir = flat_models / short
        if not src_dir.is_dir():
            continue
        pm = extract_publisher_and_model(entry.get("original", ""))
        if not pm:
            continue
        publisher, model = pm
        yield publisher, model, short, src_dir


def pick_primary_files(src_dir: Path, gguf_only: bool = False) -> list[Path]:
    """
    Return the weight files AI Nav can actually load.

    AI Navigator's bundled llama.cpp only handles GGUF — safetensors won't
    load. By default we return GGUFs only; pass gguf_only=False to fall back
    to safetensors (for tools that can load them).
    """
    try:
        files = sorted(f for f in src_dir.iterdir()
                       if f.is_file() and not f.name.startswith("."))
    except OSError:
        return []
    ggufs = [f for f in files if f.suffix.lower() == ".gguf"
             and "mmproj" not in f.name.lower()]
    if ggufs:
        return ggufs
    if gguf_only:
        return []
    for f in files:
        if f.suffix.lower() == ".safetensors":
            return [f]
    return []


def build_entries(plan_path: Path, flat_models: Path, catalog: dict,
                  hash_files: bool, workers: int) -> list[dict]:
    """
    Walk the plan, gather candidate LocalModel entries. Each entry is a dict
    ready to insert into the `models` collection.
    """
    discovered = list(discover_local_models(plan_path, flat_models))
    _log(f"  Discovered {len(discovered)} local models with HF-style ids")

    # Precompute file hashes in parallel (expensive: 10-100s per 5GB file)
    hash_cache: dict[Path, tuple[str, int]] = {}
    file_tasks = []
    for _, _, _, src_dir in discovered:
        for f in pick_primary_files(src_dir):
            file_tasks.append(f)

    if hash_files and file_tasks:
        _log(f"  Hashing {len(file_tasks)} files (workers={workers})...")
        def _hash(f):
            try:
                sz = f.stat().st_size
                h = sha256_of(f)
                return f, (h, sz)
            except (OSError, RuntimeError) as e:
                _log(f"    skip {f.name}: {e}")
                return f, None
        t0 = time.time()
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for path, result in pool.map(_hash, file_tasks):
                done += 1
                if done % 10 == 0:
                    _log(f"    hashed {done}/{len(file_tasks)} "
                         f"({time.time() - t0:.0f}s)")
                if result:
                    hash_cache[path] = result
        _log(f"  Hashing done in {time.time() - t0:.0f}s.")
    else:
        # Fill with zeroed hashes + real sizes (AI Nav may accept)
        for f in file_tasks:
            try:
                sz = f.stat().st_size
            except OSError:
                continue
            hash_cache[f] = ("0" * 64, sz)

    entries = []
    skipped_by_arch = []     # (model_id, arch)  → unsupported llama.cpp arch
    skipped_no_gguf = []     # (model_id, src_dir, formats_seen) → safetensors-only
    for publisher, model, short, src_dir in discovered:
        model_id = f"{publisher}/{model}"
        catalog_entry = catalog.get(model_id)

        # AI Nav's bundled llama.cpp only loads GGUF — skip safetensors-only
        # models. Report them so the user knows what to find a GGUF for.
        files = pick_primary_files(src_dir, gguf_only=False)
        if not files:
            # Identify the formats that ARE present, for the report
            try:
                fmts = sorted({f.suffix.lower() for f in src_dir.iterdir()
                              if f.is_file()} - {".json", ".md", ".txt", "",
                                                  ".jinja", ".py", ".yaml"})
            except OSError:
                fmts = []
            skipped_no_gguf.append((model_id, src_dir, fmts))
            continue

        # Architecture gate
        ggufs = [f for f in files if f.suffix.lower() == ".gguf"]
        if ggufs:
            arch = gguf_architecture(ggufs[0])
            if arch and arch.lower() in UNSUPPORTED_ARCHS:
                skipped_by_arch.append((model_id, arch))
                continue

        local_files = []
        for f in files:
            if f not in hash_cache:
                continue
            sha, size = hash_cache[f]
            cat_file = None
            if catalog_entry:
                for cf in catalog_entry.get("files", []):
                    if cf.get("name") == f.name:
                        cat_file = cf
                        break
            local_files.append(
                build_local_model_file(cat_file, f, sha, size)
            )

        if not local_files:
            continue

        entries.append(build_local_model(
            catalog_entry, model_id, publisher, model, local_files
        ))

    if skipped_no_gguf:
        _log(f"\n  Skipped {len(skipped_no_gguf)} model(s) with no GGUF "
             f"(AI Navigator only loads GGUF):")
        for mid, src, fmts in skipped_no_gguf:
            fmts_str = ",".join(fmts) if fmts else "no model files"
            _log(f"    - {mid:<60} [{fmts_str}]")
        _log("    -> use Ollama or LM Studio for these, or find a GGUF "
             "quant from bartowski/unsloth/mradermacher")

    if skipped_by_arch:
        _log(f"\n  Skipped {len(skipped_by_arch)} model(s) with unsupported "
             f"architectures (use Ollama / LM Studio for these):")
        for mid, arch in skipped_by_arch:
            _log(f"    - {mid}  (arch={arch!r})")
    return entries


def _dedupe_input_by_id(entries: list[dict]) -> list[dict]:
    """
    If multiple input entries have the same model id (happens when
    collision-disambiguated flat-dir entries map back to the same HF repo),
    keep the one with the largest primary file — that's likely the fuller version.
    """
    by_id: dict[str, dict] = {}
    for e in entries:
        mid = e.get("id")
        if not mid:
            continue
        if mid not in by_id:
            by_id[mid] = e
            continue
        # Pick the entry whose first file is largest
        cur_sz = (by_id[mid].get("files") or [{}])[0].get("sizeBytes", 0)
        new_sz = (e.get("files") or [{}])[0].get("sizeBytes", 0)
        if new_sz > cur_sz:
            by_id[mid] = e
    return list(by_id.values())


def inject(db: dict, entries: list[dict], dry_run: bool) -> tuple[int, int, int, int, int]:
    """
    Insert entries into the `models` collection.
    Returns (added, updated, removed_dupes, removed_incompatible, skipped).

    Also purges any pre-existing DB entry whose first GGUF has an architecture
    in UNSUPPORTED_ARCHS — those will throw "unknown model architecture" in
    AI Nav and should never be listed.
    """
    coll = get_models_collection(db)
    entries = _dedupe_input_by_id(entries)
    valid_ids = {e.get("id") for e in entries if e.get("id")}

    # Collapse pre-existing duplicates AND drop incompatible entries.
    seen: dict[str, int] = {}
    new_data: list[dict] = []
    new_id_index: list[int] = []
    removed_dupes = 0
    removed_incompat = 0
    for m in coll.get("data", []):
        mid = m.get("id")
        # Drop incompatible entries: we excluded them from this run's input,
        # so if the DB still has them they're stale (from before arch filtering).
        if mid and mid not in valid_ids:
            # Only remove if we previously INJECTED it (has a file with localPath
            # pointing into /Volumes/ModelStorage/models-flat). Don't touch
            # models AI Nav downloaded itself.
            files = m.get("files") or []
            is_ours = any(
                str((f.get("localPath") or "")).startswith("/Volumes/ModelStorage/models-flat/")
                for f in files
            )
            if is_ours:
                removed_incompat += 1
                continue
        if mid and mid in seen:
            removed_dupes += 1
            continue
        if mid:
            seen[mid] = len(new_data)
        new_data.append(m)
        if "$loki" in m:
            new_id_index.append(m["$loki"])
    coll["data"] = new_data
    coll["idIndex"] = new_id_index

    existing = {m.get("id"): i for i, m in enumerate(coll["data"]) if m.get("id")}
    added = updated = skipped = 0

    for entry in entries:
        mid = entry.get("id")
        if not mid:
            skipped += 1
            continue
        if mid in existing:
            if dry_run:
                updated += 1
                continue
            idx = existing[mid]
            old = coll["data"][idx]
            entry["$loki"] = old.get("$loki")
            entry["meta"] = now_meta(rev=(old.get("meta", {}).get("revision", 0) + 1))
            coll["data"][idx] = entry
            updated += 1
        else:
            if dry_run:
                added += 1
                continue
            loki_id = next_loki_id(coll)
            entry["$loki"] = loki_id
            entry["meta"] = now_meta()
            coll["data"].append(entry)
            coll.setdefault("idIndex", []).append(loki_id)
            existing[mid] = len(coll["data"]) - 1
            coll["maxId"] = loki_id
            added += 1

    if not dry_run:
        coll["dirty"] = True
    return added, updated, removed_dupes, removed_incompat, skipped


def check_ai_nav_running() -> bool:
    try:
        import subprocess
        out = subprocess.run(["pgrep", "-f", "AI-Navigator"],
                             capture_output=True, text=True, timeout=3)
        return bool(out.stdout.strip())
    except Exception:
        return False


def backup_db(path: Path) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    backup = path.with_suffix(path.suffix + f".bak.{ts}")
    shutil.copy2(path, backup)
    return backup


def find_latest_backup(path: Path) -> Path | None:
    backups = sorted(path.parent.glob(path.name + ".bak.*"))
    return backups[-1] if backups else None


# ── Symlink farm (HF-style) ──────────────────────────────────────────

def make_symlinks(plan_path: Path, flat_models: Path, target: Path,
                  dry_run: bool, workers: int) -> dict:
    discovered = list(discover_local_models(plan_path, flat_models))
    tasks = []
    for publisher, model, short, src_dir in discovered:
        tasks.append((src_dir, target / publisher / model, f"{publisher}/{model}"))
    stats = {}
    def _do(task):
        src, link, disp = task
        status = safe_symlink_dir(src, link, dry_run).split(":", 1)[0]
        tag = "[DRY] " if dry_run else ""
        _log(f"  {tag}{status:<17} {disp}")
        return status
    if not dry_run:
        target.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for s in pool.map(_do, tasks):
            stats[s] = stats.get(s, 0) + 1
    return stats


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--plan", type=Path, default=RENAME_PLAN)
    ap.add_argument("--target", type=Path, default=None,
                    help="Override AI Nav downloadLocation")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-hash", action="store_true",
                    help="Skip SHA-256 computation (faster; uses zeroed hashes)")
    ap.add_argument("--no-symlinks", action="store_true",
                    help="Skip the HF-style symlink farm (DB-only)")
    ap.add_argument("--revert", action="store_true",
                    help="Restore the latest ailauncher.db backup and exit")
    args = ap.parse_args()

    mode = "DRY RUN" if args.dry_run else "LIVE"
    print("=" * 68)
    print(f"AI Navigator (AI Studio) Model Registration — {mode}")
    print("=" * 68)

    if args.revert:
        backup = find_latest_backup(AI_NAV_DB)
        if not backup:
            sys.exit("  No backup found.")
        print(f"  Restoring {backup.name} → {AI_NAV_DB.name}")
        if not args.dry_run:
            shutil.copy2(backup, AI_NAV_DB)
        print("  Done.")
        return

    if not AI_NAV_DB.is_file():
        sys.exit(f"Error: AI Nav DB not found at {AI_NAV_DB}. "
                 f"Launch AI Navigator once to create it.")

    if check_ai_nav_running():
        print("  WARNING: AI Navigator is running. Quit it before running this\n"
              "  script — otherwise AI Nav won't see the new entries until\n"
              "  next launch. Continuing anyway.")
        print()

    target = (args.target or read_ai_nav_download_location()).expanduser().resolve()
    print(f"DB:       {AI_NAV_DB}")
    print(f"Plan:     {args.plan}")
    print(f"Target:   {target}")
    print(f"Hashing:  {'yes' if not args.no_hash else 'no (zeroed)'}")
    print(f"Symlinks: {'yes' if not args.no_symlinks else 'no'}")
    print(f"Workers:  {args.workers}")
    print()

    catalog = load_catalog()
    print(f"Catalog has {len(catalog)} models.")
    print()

    # Build entries
    print("Building LocalModel entries…")
    entries = build_entries(args.plan, FLAT_MODELS, catalog,
                            hash_files=not args.no_hash, workers=args.workers)
    in_cat = sum(1 for e in entries if e["id"] in catalog)
    print(f"  Built {len(entries)} entries ({in_cat} matched to catalog, "
          f"{len(entries) - in_cat} custom).")
    print()

    # Back up + inject
    if not args.dry_run:
        backup = backup_db(AI_NAV_DB)
        print(f"  Backed up DB → {backup.name}")
    db = load_db(AI_NAV_DB)
    added, updated, removed_dupes, removed_incompat, skipped = inject(
        db, entries, args.dry_run)
    if not args.dry_run:
        save_db(db, AI_NAV_DB)
    print(f"  DB: {added} added, {updated} updated, "
          f"{removed_dupes} dupes removed, "
          f"{removed_incompat} incompatible-arch entries removed, "
          f"{skipped} skipped")
    print()

    # HF-style symlink farm (belt + suspenders)
    if not args.no_symlinks:
        print("Creating HF-style symlink farm…")
        stats = make_symlinks(args.plan, FLAT_MODELS, target,
                              args.dry_run, args.workers)
        print(f"  Symlinks: {stats}")
        print()

    print("Done. Launch AI Navigator — the models should appear under 'Downloaded'.")
    if not args.dry_run:
        print(f"(Revert with: {sys.argv[0]} --revert)")


if __name__ == "__main__":
    main()
