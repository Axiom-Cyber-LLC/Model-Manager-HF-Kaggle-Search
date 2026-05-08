#!/usr/bin/env python3
"""
Prepare_models_for_Lmstudio.py

Scans the LM Studio hub models directory and prepares all models for both
LM Studio and Ollama. Runs without arguments — auto-detects the directory.

What it does:
  1. Reads LM Studio manifests + scans for loose model files
  2. Scans HuggingFace cache for models that need downloading
  3. Fixes folder structure (moves misplaced models into publisher/model/)
  4. Registers GGUF models with Ollama via `ollama create` (reference blobs)
  5. Detects and deletes duplicate models
  6. Cleans up empty directories
  7. Reports status for every model found

Usage:
    python Prepare_models_for_Lmstudio.py              # scan default dir
    python Prepare_models_for_Lmstudio.py --dry-run    # preview only
    python Prepare_models_for_Lmstudio.py --no-ollama  # skip Ollama registration
    python Prepare_models_for_Lmstudio.py --force      # re-register all with Ollama
    python Prepare_models_for_Lmstudio.py --workers 16
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from prepare_models_env import default_manager_models_dir, extend_scan_roots

_print_lock = threading.Lock()
def _tlog(msg):
    """Thread-safe print."""
    with _print_lock:
        print(msg)


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

GGUF_EXTENSIONS = {".gguf"}
SAFETENSORS_EXTENSIONS = {".safetensors"}
LEGACY_EXTENSIONS = {".bin", ".pt", ".pth"}
ALL_MODEL_EXTENSIONS = GGUF_EXTENSIONS | SAFETENSORS_EXTENSIONS | LEGACY_EXTENSIONS

COMPANION_FILENAMES = {
    "config.json", "tokenizer.json", "tokenizer_config.json",
    "tokenizer.model", "special_tokens_map.json", "generation_config.json",
    "preprocessor_config.json", "added_tokens.json", "vocab.json",
    "vocab.txt", "merges.txt", "model.safetensors.index.json",
    "pytorch_model.bin.index.json", "chat_template.json",
    "quantize_config.json", "weights.json",
}

COMPANION_SUFFIXES = {".tiktoken"}

LFS_SIGNATURE = "version https://git-lfs.github.com/spec/v1"

KNOWN_PUBLISHERS = {
    "lmstudio-community", "mlx-community", "featherless-ai-quants",
    "mradermacher", "bartowski", "thebloke", "unsloth",
    "google", "meta", "meta-llama", "deepseek", "deepseek-ai",
    "microsoft", "mistralai", "qwen", "huggingface", "nvidia",
    "01-ai", "tiiuae", "essentialai", "silveroxides", "jayn7",
    "mikeyandfriends", "fig", "yur968869", "library",
    "_image-models", "mistral", "lm_studio", "starcoder,"
}

# Directories that are LM Studio internals, not model folders
# `local/` is intentionally NOT here — in the flat-dir layout it's a legitimate
# pseudo-publisher holding all short-named models.
INTERNAL_DIRS = {"blobs", "manifests", "src", "Reference", "paper", ".git"}

DEFAULT_MODELS_DIR = default_manager_models_dir(Path("<Your Model Directory>"))

# All HF/ModelScope cache locations the user has on disk. scan_hf_cache iterates
# this list; previously it scanned only the first (most other tools' caches
# were silently invisible). Deduped in scan_hf_cache by resolved real path so
# the symlink between <REDACTED_PATH> and <REDACTED_PATH>
# doesn't cause double-scanning.
def _hf_cache_dirs_from_env() -> list[Path]:
    """Read HF_HUB_CACHE and HF_HOME from env and return any cache hub paths
    they imply. Empty list if the env vars are unset."""
    out: list[Path] = []
    hub = os.environ.get("HF_HUB_CACHE", "").strip()
    if hub:
        out.append(Path(hub).expanduser())
    home = os.environ.get("HF_HOME", "").strip()
    if home:
        out.append(Path(home).expanduser() / "hub")
    return out


HF_CACHE_DIRS = extend_scan_roots(_hf_cache_dirs_from_env() + [
    Path("<Your Model Directory>"),  # legacy primary
    Path("<Your Model Directory>/huggingface/hub"),  # legacy alt
    Path("<REDACTED_PATH>"),         # post-move location
    Path("<Your Model Directory>/huggingface/model"),
    Path.home() / ".cache" / "huggingface" / "hub",       # symlink fallback
    Path("<Your Model Directory>"),       # ModelScope cache
    Path("<REDACTED_PATH>"),
    Path.home() / "model_downloads" / "huggingface" / "model",
    Path.home() / "Library" / "Application Support" / "nomic.ai" / "GPT4All",
    Path("<REDACTED_PATH>"),
    Path("<REDACTED_PATH>"),
])
# Backward-compat alias — kept for code paths that referenced the singular name.
HF_CACHE_DIR = HF_CACHE_DIRS[0]
OLLAMA_STATE_FILE = Path.home() / ".ollama" / "models" / "model_state.json"


def _detect_lmstudio_downloads_folder() -> Path:
    """Read LM Studio's downloadsFolder setting; fall back to <Your Model Directory>."""
    settings_path = Path.home() / ".lmstudio" / "settings.json"
    fallback = Path("<Your Model Directory>")
    try:
        data = json.loads(settings_path.read_text())
        folder = data.get("downloadsFolder")
        if folder:
            return Path(folder).expanduser().resolve()
    except (OSError, json.JSONDecodeError):
        pass
    return fallback


DEFAULT_LMSTUDIO_DIR = _detect_lmstudio_downloads_folder()


# ──────────────────────────────────────────────────────────────────────
# Short-name key (user-defined vocabulary)
# ──────────────────────────────────────────────────────────────────────
#
# Rules:
#   • Single-letter codes concatenate (iCM = intent + classifier + merged).
#   • Multi-letter codes and body tokens are dash-separated.
#   • Case distinguishes: I=Instruct, i=Intent / M=Merged, m=Mini / T=Tokenizer, t=Testing.
#   • Filler tokens (LLM, GGUF, Model, HF) are dropped — they're implicit.
#   • Quant tags are compressed per the user's map (Q4_K_M → 4KM, etc.).

FILLER_TOKENS = {
    "llm", "gguf", "model", "models", "hf", "huggingface",
    "release", "final", "main",
}

# Multi-token compound patterns. Lowercase. Longest first → greedy match wins.
# Quants appear here because after splitting on `-` and `_`, they present as
# multi-token sequences (e.g., Q4_K_M → ['q4','k','m']).
COMPOUND_VOCAB = [
    # Compressed quants per user key
    (["iq4", "xs"], "I4XS"),
    (["iq4", "nl"], "I4NL"),
    (["q4", "k", "m"], "4KM"),
    (["q4", "k", "s"], "4KS"),
    (["q4", "k", "l"], "4KL"),
    (["q4", "0"], "4"),
    (["q4", "1"], "41"),
    # Preserved quants (not in user's map but keep recognizable)
    (["q3", "k", "s"], "Q3_K_S"),
    (["q3", "k", "m"], "Q3_K_M"),
    (["q3", "k", "l"], "Q3_K_L"),
    (["q5", "k", "s"], "Q5_K_S"),
    (["q5", "k", "m"], "Q5_K_M"),
    (["q2", "k"], "Q2_K"),
    (["q6", "k"], "Q6_K"),
    (["q8", "0"], "Q8_0"),
    # Word compounds
    (["semantic", "router"], "SR"),
    (["distilled", "instruct"], "DI"),
    (["distilled", "instruction"], "DI"),
    (["distill", "instruct"], "DI"),
]

# Single-token vocab (lowercase key → output code).
SINGLE_VOCAB = {
    "distilled": "D", "distill": "D",
    "instruct": "I", "instruction": "I", "instructed": "I",
    "reasoning": "RE",
    "intent": "i",
    "classifier": "C",
    "merged": "M",
    "large": "LG",
    "lite": "L",
    "turbo": "TU",
    "base": "B",
    "mini": "m",
    "tokenizer": "T",
    "testing": "t",
    "coder": "Co", "coding": "Co", "code": "Co",
    # Preserved single-token quants
    "f16": "F16", "f32": "F32", "bf16": "BF16", "fp16": "FP16",
}

# Publishers to encode rather than drop. Everything else is dropped.
KNOWN_PUBLISHER_CODES = {
    "llm-semantic-router": "SR",
}


def _tokenize_name(s: str):
    """Split on any run of hyphens OR underscores, strip file extensions + padded-digit noise."""
    tokens = [t for t in re.split(r"[-_]+", s) if t]
    cleaned = []
    for t in tokens:
        # Strip stray .gguf/.safetensors/.bin inside a token (e.g., "1.5B.gguf")
        t = re.sub(r"\.(gguf|safetensors|bin|onnx|h5|pt|pth)$", "", t, flags=re.IGNORECASE)
        # Strip trailing runs of 6+ identical digits (padded usernames like `sridhar1111111111`)
        t = re.sub(r"(\d)\1{5,}$", "", t)
        if t:
            cleaned.append(t)
    return cleaned


def _classify_tokens(tokens):
    """Match tokens against vocab. Returns list of (out_str, kind) with kind ∈ {single, multi, body}."""
    out = []
    i = 0
    while i < len(tokens):
        matched = False
        # Try compound (multi-token) patterns first
        for pattern, code in COMPOUND_VOCAB:
            n = len(pattern)
            window = [t.lower() for t in tokens[i:i + n]]
            if window == pattern:
                out.append((code, "single" if len(code) == 1 else "multi"))
                i += n
                matched = True
                break
        if matched:
            continue
        tok = tokens[i]
        if tok.lower() in SINGLE_VOCAB:
            code = SINGLE_VOCAB[tok.lower()]
            out.append((code, "single" if len(code) == 1 else "multi"))
        else:
            out.append((tok, "body"))
        i += 1
    return out


def _assemble(classified):
    """Concatenate single-letter runs; dash-separate everything else."""
    chunks = []
    run = []
    for tok, kind in classified:
        if kind == "single":
            run.append(tok)
        else:
            if run:
                chunks.append("".join(run))
                run = []
            chunks.append(tok)
    if run:
        chunks.append("".join(run))
    return "-".join(chunks)


def shorten_name(full: str) -> str:
    """
    Convert an HF-style `models--{publisher}--{repo}` name (or any hyphen-delimited
    model identifier) into the short form dictated by the user's key.
    """
    if full.startswith("models--"):
        full = full[len("models--"):]

    if "--" in full:
        pub_part, repo_part = full.split("--", 1)
    else:
        pub_part, repo_part = "", full

    repo_part = repo_part.replace("--", "-")

    # Resolve publisher → explicit code OR drop
    publisher_classified = []
    if pub_part:
        pub_code = KNOWN_PUBLISHER_CODES.get(pub_part.lower())
        if pub_code is None:
            # Some publisher names are themselves encodable compounds; try vocab on them
            pub_tokens = _tokenize_name(pub_part)
            pub_tokens = [t for t in pub_tokens if t.lower() not in FILLER_TOKENS]
            tentative = _classify_tokens(pub_tokens)
            # If every token became a code, keep it; otherwise drop.
            if tentative and all(kind != "body" for _, kind in tentative):
                publisher_classified = tentative
        else:
            publisher_classified = [(pub_code, "single" if len(pub_code) == 1 else "multi")]

    # Process repo body
    repo_tokens = _tokenize_name(repo_part)
    repo_tokens = [t for t in repo_tokens if t.lower() not in FILLER_TOKENS]
    repo_classified = _classify_tokens(repo_tokens)

    full_classified = publisher_classified + repo_classified
    if not full_classified:
        return ""
    return _assemble(full_classified)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def is_lfs_pointer(filepath: Path) -> bool:
    try:
        if filepath.stat().st_size > 1024:
            return False
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return f.readline().strip() == LFS_SIGNATURE
    except Exception:
        return False


def format_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def infer_publisher_and_model(source: Path) -> tuple:
    """Infer (publisher, model_name) for a source path."""
    model_dir = source.parent if source.is_file() else source
    parent = model_dir.parent

    if parent.name.lower() in {p.lower() for p in KNOWN_PUBLISHERS}:
        return parent.name, model_dir.name

    readme = model_dir / "README.md"
    if readme.exists():
        try:
            content = readme.read_text(errors="replace")
            qb = re.search(r"^quantized_by:\s*(\S+)", content, re.MULTILINE)
            if qb:
                return qb.group(1), model_dir.name
        except Exception:
            pass

    git_config = model_dir / ".git" / "config"
    if git_config.exists():
        try:
            content = git_config.read_text(errors="replace")
            m = re.search(
                r'url\s*=\s*https://huggingface\.co/([^/\s]+)/([^/\s]+?)(?:\.git)?\s*$',
                content, re.MULTILINE,
            )
            if m:
                return m.group(1), m.group(2)
        except Exception:
            pass

    name = model_dir.name if model_dir.is_dir() else source.stem
    publisher = "local"
    name_lower = name.lower()
    if "mlx" in name_lower:
        publisher = "mlx-community"
    elif "deepseek" in name_lower:
        publisher = "deepseek-ai"
    elif "llama" in name_lower:
        publisher = "meta-llama"
    elif "gemma" in name_lower:
        publisher = "google"
    elif "qwen" in name_lower:
        publisher = "qwen"
    elif "mistral" in name_lower or "ministral" in name_lower:
        publisher = "mistralai"
    elif "phi" in name_lower:
        publisher = "microsoft"
    return publisher, name


def is_model_directory(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_config = (path / "config.json").is_file()
    # Note: glob() returns dirs as well as files — is_file() filter required,
    # otherwise a subdir named `foo.gguf/` would falsely match.
    has_safetensors = any(f.is_file() for f in path.glob("*.safetensors"))
    has_gguf = any(f.is_file() for f in path.glob("*.gguf"))
    has_bin = any(f.is_file() for f in path.glob("pytorch_model*.bin"))
    has_weights_json = (path / "weights.json").is_file()
    return (has_config and (has_safetensors or has_bin or has_weights_json)) or has_gguf


def should_move_file(f: Path) -> bool:
    if f.name.startswith("."):
        return False
    if f.is_dir():
        return False
    if f.suffix in ALL_MODEL_EXTENSIONS:
        return True
    if f.name in COMPANION_FILENAMES:
        return True
    if f.suffix in COMPANION_SUFFIXES:
        return True
    return False


def cleanup_empty_dirs(directory: Path, stop_at: Path):
    try:
        current = directory.resolve()
        boundary = stop_at.resolve()
        while current != boundary and str(current).startswith(str(boundary)):
            if current.is_dir() and not any(current.iterdir()):
                current.rmdir()
                current = current.parent
            else:
                break
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────────────────────────────

class Results:
    def __init__(self):
        self.ready = []           # (name, size_str, model_type)
        self.fixed = []           # (name, source, dest)
        self.registered = []      # (name, ollama_name)
        self.broken = []          # (name, path, reason)
        self.duplicates = []      # (name, path, kept_path)
        self.errors = []          # (name, reason)
        self.skipped_ollama = []  # (name, ollama_name)
        self.needs_download = []  # (name, model_type, hf_repo)


# ──────────────────────────────────────────────────────────────────────
# Manifest scanning — reads LM Studio's registry manifests
# ──────────────────────────────────────────────────────────────────────

def scan_manifests(models_dir: Path, results: Results):
    """
    Read all LM Studio registry manifests to discover models stored in blobs/.
    These are the majority of models — they have Ollama-style manifests that
    reference SHA256 blobs.
    """
    manifest_root = models_dir / "manifests" / "registry.ollama.ai"
    if not manifest_root.is_dir():
        return

    blobs_dir = models_dir / "blobs"

    for manifest_file in sorted(manifest_root.rglob("*")):
        if not manifest_file.is_file():
            continue

        try:
            rel = manifest_file.relative_to(manifest_root)
            parts = rel.parts  # e.g. ("qwen", "qwen3.5-vl-10b-q4_k_m", "latest")
            if len(parts) < 3:
                continue
            publisher = parts[0]
            model_name = parts[1]
            tag = parts[2]
            display = f"{publisher}/{model_name}:{tag}"
        except ValueError:
            continue

        try:
            manifest = json.loads(manifest_file.read_text())
        except (json.JSONDecodeError, OSError):
            results.errors.append((display, "Could not read manifest"))
            continue

        if not isinstance(manifest, dict):
            results.errors.append((display, "Invalid manifest format"))
            continue

        # Find the model layer
        model_layer = None
        for layer in manifest.get("layers") or []:
            if layer.get("mediaType") == "application/vnd.ollama.image.model":
                model_layer = layer
                break

        if not model_layer:
            # Image models or other non-standard types
            for layer in manifest.get("layers") or []:
                if "model" in layer.get("mediaType", ""):
                    model_layer = layer
                    break

        if not model_layer:
            continue

        size = model_layer.get("size", 0)
        digest = model_layer.get("digest", "")
        from_path = model_layer.get("from", "")

        # Check if the blob exists
        if from_path:
            blob = Path(from_path)
        elif digest:
            blob = blobs_dir / digest.replace(":", "-")
        else:
            results.broken.append((display, manifest_file, "No blob reference"))
            continue

        if blob.exists():
            results.ready.append((display, format_size(size), "manifest/blob"))
        else:
            results.broken.append((display, manifest_file,
                                   f"Missing blob: {blob.name} ({format_size(size)})"))


# ──────────────────────────────────────────────────────────────────────
# HuggingFace cache scanning
# ──────────────────────────────────────────────────────────────────────

def scan_hf_cache(results: Results):
    """
    Scan every HF/ModelScope cache root in HF_CACHE_DIRS for model repos.
    Reports models that are LFS pointers (need downloading) vs fully
    downloaded. Roots are deduped by resolved physical path so the
    `<REDACTED_PATH>` symlink doesn't cause double-scanning.
    """
    seen_repo_real: set[Path] = set()
    seen_root_real: set[Path] = set()

    for cache_root in HF_CACHE_DIRS:
        if not cache_root.is_dir():
            continue
        try:
            real_root = cache_root.resolve()
        except (OSError, RuntimeError):
            real_root = cache_root
        if real_root in seen_root_real:
            continue
        seen_root_real.add(real_root)

        for repo_dir in sorted(cache_root.iterdir()):
            if not repo_dir.is_dir() or not repo_dir.name.startswith("models--"):
                continue

            # Dedup repos across roots (symlinks point at the same physical dir)
            try:
                real_repo = repo_dir.resolve()
            except (OSError, RuntimeError):
                real_repo = repo_dir
            if real_repo in seen_repo_real:
                continue
            seen_repo_real.add(real_repo)

            # Parse repo name: models--owner--repo -> owner/repo
            parts = repo_dir.name[len("models--"):].split("--", 1)
            if len(parts) != 2:
                continue
            hf_repo = f"{parts[0]}/{parts[1]}"

            snap_dir = repo_dir / "snapshots"
            if not snap_dir.is_dir():
                continue

            # Get the latest snapshot
            try:
                snapshots = sorted(snap_dir.iterdir(),
                                   key=lambda p: p.stat().st_mtime, reverse=True)
            except OSError:
                continue
            if not snapshots:
                continue
            latest = snapshots[0]

            # Determine model type from files present
            has_gguf = any(latest.glob("*.gguf"))
            has_st = any(latest.glob("*.safetensors"))

            if not has_gguf and not has_st:
                continue  # Not a model repo (dataset, tokenizer-only, etc.)

            model_type = "GGUF" if has_gguf else "safetensors"

            # Check if files are real or LFS pointers/symlinks to tiny stubs
            ext = "*.gguf" if has_gguf else "*.safetensors"
            sample = next(latest.glob(ext), None)
            if sample is None:
                continue

            # Resolve through symlinks and check real size
            try:
                real_path = sample.resolve()
                real_size = real_path.stat().st_size
            except OSError:
                real_size = 0

            if real_size < 1024:
                # LFS pointer — needs download
                results.needs_download.append((hf_repo, model_type, hf_repo))
            # If real_size >= 1024 it's actually downloaded — could be moved,
            # but we leave that for a future enhancement

    return


# ──────────────────────────────────────────────────────────────────────
# Flatten-to-short-names mode — builds a rename plan and hardlinks into
# `{output}/local/{short-name}/{files}` so LM Studio shows one flat list.
# ──────────────────────────────────────────────────────────────────────

# Directory names that indicate an HF cache root or other non-model container.
# When encountered at the publisher OR model level, skip them.
CACHE_ROOT_NAMES = {"hub", "huggingface", "xet", "blobs", "manifests", ".cache"}

# Dirs that look like models (have config.json + safetensors) but are actually
# Stable Diffusion pipeline components, not standalone loadable models.
NON_MODEL_NAMES = {
    "text_encoder", "text_encoder_2", "text_encoder_3",
    "unet", "vae", "vae_1_0", "vae_decoder", "vae_encoder",
    "tokenizer", "tokenizer_2", "tokenizer_3",
    "scheduler", "transformer", "prior",
    "safety_checker", "feature_extractor",
    "image_encoder", "image_processor",
}


def _is_lora_only(path: Path) -> bool:
    """Return True if the dir contains LoRA adapter weights but no base model weights."""
    try:
        files = list(path.iterdir())
    except OSError:
        return False
    has_adapter = any(f.name.startswith("adapter_") and f.suffix in {".safetensors", ".bin"}
                      for f in files)
    has_base_weights = any(
        (f.name in {"model.safetensors", "pytorch_model.bin"}
         or re.match(r"model-\d+-of-\d+\.safetensors$", f.name)
         or f.suffix == ".gguf")
        for f in files
    )
    return has_adapter and not has_base_weights


def _looks_like_hf_cache_root(path: Path) -> bool:
    """A directory is an HF cache root if most of its children are models--*/datasets--*."""
    try:
        children = [c.name for c in path.iterdir() if c.is_dir()]
    except OSError:
        return False
    if not children:
        return False
    cache_like = sum(1 for n in children if n.startswith(("models--", "datasets--")))
    return cache_like >= max(1, len(children) // 2)


def _detect_format(src_dir: Path) -> str:
    """Return 'gguf', 'safetensors', 'pytorch', or 'unknown' based on weight file suffixes."""
    try:
        files = list(src_dir.iterdir())
    except OSError:
        return "unknown"
    has_gguf = any(f.suffix == ".gguf" for f in files)
    has_st = any(f.suffix == ".safetensors" for f in files)
    has_bin = any(f.suffix == ".bin" for f in files)
    if has_gguf and not has_st:
        return "gguf"
    if has_st and not has_gguf:
        return "safetensors"
    if has_gguf and has_st:
        return "mixed"
    if has_bin:
        return "pytorch"
    return "unknown"


def _extract_author(original: str) -> str:
    """Extract the author portion of `models--{author}--{repo}`."""
    if original.startswith("models--"):
        rest = original[len("models--"):]
        if "--" in rest:
            return rest.split("--", 1)[0]
    return ""


def _find_model_sources(input_roots):
    """
    Yield (original_full_name, source_dir, kind) for every model dir found
    across the given input roots. `original_full_name` is the best
    reconstruction of `models--{publisher}--{repo}` for short-name derivation.
    `kind` is one of: 'hf_cache', 'lmstudio', 'standalone'.
    """
    seen_source_dirs = set()

    for root in input_roots:
        root = root.expanduser().resolve() if hasattr(root, "expanduser") else Path(root).resolve()
        if not root.is_dir():
            continue

        # 1. HF cache style: models--owner--repo/snapshots/<hash>/
        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or entry.is_symlink():
                continue
            if not entry.name.startswith("models--"):
                continue
            snap = _find_hf_latest_snapshot(entry)
            if snap is None:
                continue
            # Must have weights (ignore tokenizer-only / dataset-like)
            has_weights = any(f.suffix in ALL_MODEL_EXTENSIONS for f in snap.iterdir())
            if not has_weights:
                continue
            if snap.resolve() in seen_source_dirs:
                continue
            seen_source_dirs.add(snap.resolve())
            yield (entry.name, snap, "hf_cache")

        # 2. LM Studio canonical: publisher/repo/*.gguf or *.safetensors
        # 3. Standalone: direct model dirs at the root (no publisher layer) — e.g. `local/Foo/`
        for pub_dir in sorted(root.iterdir()):
            if not pub_dir.is_dir() or pub_dir.name.startswith("."):
                continue
            if pub_dir.name in CACHE_ROOT_NAMES or pub_dir.name in NON_MODEL_NAMES:
                continue
            if pub_dir.name.startswith(("models--", "datasets--")):
                continue  # handled above
            # Skip pure LM Studio/Ollama internal dirs at top level.
            if pub_dir.name in {"blobs", "manifests", ".git"}:
                continue

            # Case A: `pub_dir` is itself a model dir (no publisher wrapper).
            if is_model_directory(pub_dir) and not _is_lora_only(pub_dir):
                if pub_dir.resolve() not in seen_source_dirs:
                    seen_source_dirs.add(pub_dir.resolve())
                    full_name = f"models--local--{pub_dir.name}"
                    yield (full_name, pub_dir, "standalone")
                continue

            # Case B: publisher dir full of cache-style subdirs — skip.
            if _looks_like_hf_cache_root(pub_dir):
                continue

            # Case C: publisher/model/ layout.
            for model_dir in sorted(pub_dir.iterdir()):
                if not model_dir.is_dir() or model_dir.name.startswith("."):
                    continue
                if model_dir.name in CACHE_ROOT_NAMES or model_dir.name in NON_MODEL_NAMES:
                    continue
                if not is_model_directory(model_dir):
                    continue
                if _is_lora_only(model_dir):
                    continue  # LoRA adapters need a base model — don't flatten standalone
                if model_dir.resolve() in seen_source_dirs:
                    continue
                seen_source_dirs.add(model_dir.resolve())
                full_name = f"models--{pub_dir.name}--{model_dir.name}"
                yield (full_name, model_dir, "lmstudio")


def build_flatten_plan(input_roots, output_dir: Path) -> dict:
    """
    Walk every input root, compute a short name for each model, detect collisions,
    and return a plan dict ready to serialize or apply.
    """
    plan = {
        "output_root": str(output_dir),
        "entries": [],
        "collisions": [],
    }

    # short_name → list of (original_name, source_dir)
    short_map = {}
    for original, src_dir, kind in _find_model_sources(input_roots):
        short = shorten_name(original)
        if not short:
            continue
        short_map.setdefault(short, []).append((original, src_dir, kind))

    def _content_fingerprint(src_dir: Path):
        """Sorted list of (filename, resolved-size) for the model dir — stable across paths."""
        fp = []
        try:
            for f in sorted(src_dir.iterdir()):
                if not (f.is_file() or f.is_symlink()) or f.name.startswith("."):
                    continue
                try:
                    size = f.resolve().stat().st_size
                except OSError:
                    continue
                fp.append((f.name, size))
        except OSError:
            pass
        return tuple(fp)

    def _smart_suffix(original: str, src_dir: Path) -> str:
        """
        Derive a meaningful suffix for disambiguating a collision entry.
        Preference: format tag > author code > hash fallback.
        """
        fmt = _detect_format(src_dir)
        if fmt in ("gguf", "safetensors"):
            fmt_tag = "GGUF" if fmt == "gguf" else "ST"
        else:
            fmt_tag = None
        author = _extract_author(original)
        # Use the author unless it's redundant (e.g., official qwen/Qwen namespace) or empty.
        author_tag = author if author else None
        return fmt_tag, author_tag

    def _choose_suffixes(candidates):
        """
        Given a list of (original, src_dir, kind) entries that collide, return
        a dict mapping index → disambiguating suffix. Tries format first, then
        author, then falls back to a 4-char hash.
        """
        import hashlib
        # Collect formats and authors
        infos = [_smart_suffix(orig, src) for orig, src, _ in candidates]
        formats = [f for f, _ in infos]
        authors = [a for _, a in infos]

        # Case A: formats differ uniquely → use format tags
        if len(set(formats)) == len(candidates) and all(formats):
            return {i: f for i, f in enumerate(formats)}
        # Case B: exactly two entries, one gguf one safetensors → tag only the GGUF
        if len(candidates) == 2 and set(formats) == {"GGUF", "ST"}:
            return {
                i: ("GGUF" if formats[i] == "GGUF" else None)
                for i in range(len(candidates))
            }
        # Case C: authors are all distinct → use authors
        if len(set(authors)) == len(candidates) and all(authors):
            return {i: authors[i] for i in range(len(candidates))}
        # Case D: format + author mix
        mixed = [f"{a or ''}{'-' if a and f else ''}{f or ''}" for a, f in zip(authors, formats)]
        if len(set(mixed)) == len(candidates):
            return {i: mixed[i] for i in range(len(candidates))}
        # Fallback: hash
        out = {}
        for i, (_, src, _) in enumerate(candidates):
            out[i] = hashlib.sha256(str(src).encode()).hexdigest()[:4]
        return out

    for short, sources in short_map.items():
        if len(sources) == 1:
            original, src_dir, kind = sources[0]
            plan["entries"].append({
                "short_name": short,
                "original": original,
                "source": str(src_dir),
                "kind": kind,
                "dest": str(output_dir / "local" / short),
            })
            continue

        # Dedupe by content fingerprint: same filenames + sizes = same model
        # (hardlinks/symlinks in HF cache collapse to the same content).
        by_fingerprint = {}
        for original, src_dir, kind in sources:
            fp = _content_fingerprint(src_dir)
            by_fingerprint.setdefault(fp, []).append((original, src_dir, kind))

        if len(by_fingerprint) == 1:
            # All copies are identical — pick the first, note duplicates, emit one entry.
            first_original, first_src, first_kind = sources[0]
            plan["entries"].append({
                "short_name": short,
                "original": first_original,
                "source": str(first_src),
                "kind": first_kind,
                "dest": str(output_dir / "local" / short),
                "duplicates_dropped": [str(s[1]) for s in sources[1:]],
            })
        else:
            # Real content-level collision. Pick one representative per fingerprint,
            # then disambiguate those by format/author.
            reps = [group[0] for group in by_fingerprint.values()]
            suffixes = _choose_suffixes(reps)
            for i, ((original, src_dir, kind), group) in enumerate(zip(reps, by_fingerprint.values())):
                suffix = suffixes[i]
                disambiguated = f"{short}-{suffix}" if suffix else short
                plan["entries"].append({
                    "short_name": disambiguated,
                    "original": original,
                    "source": str(src_dir),
                    "kind": kind,
                    "dest": str(output_dir / "local" / disambiguated),
                    "collision_note": f"Disambiguated from '{short}' by {suffix or 'base'}",
                    "duplicates_dropped": [str(g[1]) for g in group[1:]],
                })
            plan["collisions"].append({
                "short_name": short,
                "distinct_contents": len(by_fingerprint),
                "total_copies": len(sources),
                "sources": [s[0] for s in sources],
            })

    plan["entries"].sort(key=lambda e: e["short_name"].lower())
    return plan


def apply_flatten_plan(plan: dict, dry_run: bool, results: Results,
                       workers: int = 8):
    """Execute the plan in parallel: hardlink every real source file into `{dest}/`."""
    output_root = Path(plan["output_root"])
    try:
        dest_dev = output_root.parent.stat().st_dev
    except OSError:
        dest_dev = None

    def _process_entry(entry):
        src = Path(entry["source"])
        dst = Path(entry["dest"])
        short = entry["short_name"]

        if not src.is_dir():
            return ("error", short, f"Source missing: {src}")

        files_to_link = []
        total_size = 0
        for f in sorted(src.iterdir()):
            if not (f.is_file() or f.is_symlink()) or f.name.startswith("."):
                continue
            try:
                real = f.resolve()
                if not real.is_file():
                    continue
                total_size += real.stat().st_size
            except OSError:
                continue
            files_to_link.append((f.name, real))

        if not files_to_link:
            return ("skip", short, "no files")

        try:
            src_dev = src.stat().st_dev
            can_hardlink = dest_dev is not None and src_dev == dest_dev
        except OSError:
            can_hardlink = False

        method = "HARDLINK" if can_hardlink else "COPY"
        note = f" [{entry['collision_note']}]" if "collision_note" in entry else ""
        _tlog(f"  FLATTEN  {entry['original']}{note}")
        _tlog(f"        -> local/{short}/  ({len(files_to_link)} files, "
              f"{format_size(total_size)}, {method})")

        if dry_run:
            return ("ok", short, (src, dst))

        try:
            dst.mkdir(parents=True, exist_ok=True)
            for name, real_src in files_to_link:
                target = dst / name
                if target.exists():
                    continue
                if can_hardlink:
                    try:
                        os.link(real_src, target)
                    except OSError:
                        shutil.copy2(real_src, target)
                else:
                    shutil.copy2(real_src, target)
            return ("ok", short, (src, dst))
        except OSError as e:
            _tlog(f"           ERROR: {e}")
            return ("error", short, f"Flatten failed: {e}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for status, short, payload in pool.map(_process_entry, plan["entries"]):
            if status == "ok":
                src, dst = payload
                results.fixed.append((short, src, dst))
            elif status == "error":
                results.errors.append((short, payload))


# ──────────────────────────────────────────────────────────────────────
# Partial-download cleanup — HuggingFace .incomplete / .lock residue
# ──────────────────────────────────────────────────────────────────────

PARTIAL_SUFFIXES = (".incomplete", ".partial", ".downloading", ".tmp")


def clean_partial_downloads(scan_roots, dry_run: bool, results: Results):
    """
    Find and optionally remove HuggingFace partial-download residue
    (`<blob>.incomplete`, `.lock`, etc.) across the given roots.

    Only touches files inside `blobs/` trees, so this can never delete
    actual model weights — those live in `snapshots/` or at the repo root.
    """
    total_bytes = 0
    by_repo = {}  # (root, repo_dir) -> (count, bytes)
    home = str(Path.home())

    # Dedupe by resolved path — <REDACTED_PATH> is often a symlink to an SSD
    seen_roots = set()
    unique_roots = []
    for root in scan_roots:
        if not root.is_dir():
            continue
        resolved = root.resolve()
        if resolved in seen_roots:
            continue
        seen_roots.add(resolved)
        unique_roots.append(resolved)

    seen_inodes = set()  # dedupe across recursive/firmlinked paths
    for root in unique_roots:
        for f in root.rglob("*"):
            try:
                if not f.is_file():
                    continue
            except OSError:
                continue

            is_partial = any(f.name.endswith(suf) for suf in PARTIAL_SUFFIXES)
            is_lock = f.suffix == ".lock" and "blobs" in f.parts
            if not (is_partial or is_lock):
                continue

            # Safety: only act on files under a blobs/ dir (HF cache or Ollama)
            if "blobs" not in f.parts:
                continue

            try:
                st = f.stat()
            except OSError:
                continue

            # Dedupe by (device, inode) — paths reachable via multiple symlinks
            # would otherwise double-count the same blob.
            inode_key = (st.st_dev, st.st_ino)
            if inode_key in seen_inodes:
                continue
            seen_inodes.add(inode_key)

            # Attribute to the enclosing repo dir (parent of blobs/), using
            # the resolved path so symlinked roots collapse to one entry.
            resolved_f = f.resolve()
            repo_dir = None
            for i, part in enumerate(resolved_f.parts):
                if part == "blobs" and i > 0:
                    repo_dir = Path(*resolved_f.parts[:i])
                    break
            key = repo_dir if repo_dir else resolved_f.parent

            count, bytes_ = by_repo.get(key, (0, 0))
            by_repo[key] = (count + 1, bytes_ + st.st_size)
            total_bytes += st.st_size

            if not dry_run:
                try:
                    f.unlink()
                except OSError as e:
                    results.errors.append((str(f), f"Could not delete partial: {e}"))

    if not by_repo:
        return

    action = "Would free" if dry_run else "Freed"
    print(f"\n  {action} {format_size(total_bytes)} across {len(by_repo)} repos:")
    for repo_dir, (count, bytes_) in sorted(by_repo.items(), key=lambda x: -x[1][1]):
        # Pretty-print full path with ~ substitution
        full = str(repo_dir)
        if full.startswith(home):
            full = "~" + full[len(home):]
        print(f"    {count:>3} files, {format_size(bytes_):>10}  {full}")


# ──────────────────────────────────────────────────────────────────────
# Model validation — does this model have all files needed to load?
# ──────────────────────────────────────────────────────────────────────

GGUF_MAGIC = b"GGUF"
SAFETENSORS_MIN_SIZE = 4096  # anything smaller is likely a stub

# Non-weight .pt/.bin files that can legitimately be tiny — don't flag them as broken.
TRAINING_ARTIFACT_FILES = {
    "scheduler.pt", "optimizer.pt", "rng_state.pt",
    "training_args.bin", "trainer_state.pt",
}


def _is_primary_weight_file(name: str) -> bool:
    """
    Return True only for files that should contain actual model weights.
    Training-state artifacts (scheduler.pt, optimizer.pt, etc.) are excluded.
    """
    low = name.lower()
    if low in TRAINING_ARTIFACT_FILES or low.startswith("trainer_state"):
        return False
    # Everything else with a weight-file extension counts as primary.
    return True


def _gguf_header_ok(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == GGUF_MAGIC
    except OSError:
        return False


def _resolve_if_symlink(p: Path) -> Path:
    try:
        return p.resolve() if p.is_symlink() else p
    except OSError:
        return p


def validate_model_dir(model_dir: Path, display: str, results: Results):
    """
    Check that a model dir is 'mount-ready':
      - no broken symlinks
      - no LFS-pointer stubs among weight files
      - GGUF files have valid header magic
      - if a safetensors.index.json is present, all referenced shards exist at real size
    Appends to results.broken when something is off.
    """
    broken_reasons = []

    for f in model_dir.iterdir():
        if f.is_symlink():
            try:
                target = f.resolve(strict=False)
                if not target.exists():
                    broken_reasons.append(f"broken symlink: {f.name}")
                    continue
            except OSError:
                broken_reasons.append(f"unresolvable symlink: {f.name}")
                continue

        if not (f.is_file() or f.is_symlink()):
            continue

        # Weight-file checks
        if f.suffix in ALL_MODEL_EXTENSIONS:
            real = _resolve_if_symlink(f)
            try:
                size = real.stat().st_size
            except OSError:
                broken_reasons.append(f"missing: {f.name}")
                continue

            if is_lfs_pointer(f):
                broken_reasons.append(f"LFS pointer: {f.name}")
                continue
            # Training artifacts (scheduler.pt etc.) are allowed to be small.
            if _is_primary_weight_file(f.name) and size < SAFETENSORS_MIN_SIZE:
                broken_reasons.append(f"too small ({size}B): {f.name}")
                continue
            if f.suffix == ".gguf" and not _gguf_header_ok(real):
                broken_reasons.append(f"bad GGUF magic: {f.name}")
                continue

    # Sharded safetensors: verify index references all shards
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.is_file():
        try:
            idx = json.loads(index_file.read_text())
            shards = set((idx.get("weight_map") or {}).values())
            for shard in shards:
                shard_path = model_dir / shard
                if not shard_path.exists():
                    broken_reasons.append(f"missing shard: {shard}")
                else:
                    real = _resolve_if_symlink(shard_path)
                    try:
                        if real.stat().st_size < SAFETENSORS_MIN_SIZE:
                            broken_reasons.append(f"shard too small: {shard}")
                    except OSError:
                        broken_reasons.append(f"unreadable shard: {shard}")
        except (json.JSONDecodeError, OSError):
            broken_reasons.append("corrupt model.safetensors.index.json")

    if broken_reasons:
        reason = "; ".join(broken_reasons[:3])
        if len(broken_reasons) > 3:
            reason += f" +{len(broken_reasons) - 3} more"
        results.broken.append((display, model_dir, reason))


def validate_all_models(models_dir: Path, results: Results):
    """Walk publisher/model/ dirs and flag any that won't load cleanly."""
    if not models_dir.is_dir():
        return
    for publisher_dir in sorted(models_dir.iterdir()):
        if not publisher_dir.is_dir() or publisher_dir.name.startswith("."):
            continue
        if publisher_dir.name in INTERNAL_DIRS:
            continue
        if publisher_dir.name.startswith("models--") or publisher_dir.name.startswith("datasets--"):
            continue  # HF cache, not yet migrated

        for model_dir in sorted(publisher_dir.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith("."):
                continue
            if not is_model_directory(model_dir):
                continue
            display = f"{publisher_dir.name}/{model_dir.name}"
            validate_model_dir(model_dir, display, results)


# ──────────────────────────────────────────────────────────────────────
# HuggingFace cache migration — convert models--owner--repo/ → owner/repo/
# ──────────────────────────────────────────────────────────────────────

def _parse_hf_cache_name(name: str):
    """Parse 'models--owner--repo' -> ('owner', 'repo'). None if not HF cache."""
    if not name.startswith("models--"):
        return None
    rest = name[len("models--"):]
    parts = rest.split("--", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return parts[0], parts[1]


def _find_hf_latest_snapshot(hf_dir: Path) -> Path:
    """Resolve HF cache's latest snapshot dir via refs/main, else newest mtime."""
    snap_dir = hf_dir / "snapshots"
    if not snap_dir.is_dir():
        return None

    main_ref = hf_dir / "refs" / "main"
    if main_ref.is_file():
        try:
            commit = main_ref.read_text(errors="replace").strip()
            candidate = snap_dir / commit
            if candidate.is_dir():
                return candidate
        except OSError:
            pass

    snapshots = [d for d in snap_dir.iterdir() if d.is_dir()]
    if not snapshots:
        return None
    return max(snapshots, key=lambda d: d.stat().st_mtime)


def migrate_hf_cache_dirs(lmstudio_dir: Path, dry_run: bool, cleanup: bool, results: Results):
    """
    Convert HuggingFace cache-style dirs (`models--owner--repo/snapshots/<hash>/...`)
    found anywhere under HF_CACHE_DIRS into LM Studio's expected publisher/repo/
    layout under `lmstudio_dir`.

    Uses hardlinks when on the same filesystem (instant, zero extra disk) and
    falls back to copy2 otherwise. Leaves the `models--*--*` source intact unless
    cleanup=True so an aborted/partial migration can't destroy data.

    Silently skips `datasets--*--*` and any non-HF-cache directories. Roots and
    repos are deduped by resolved physical path so a symlinked cache root
    (e.g. <REDACTED_PATH> -> <REDACTED_PATH>) doesn't cause
    double-migration.
    """
    if not lmstudio_dir.exists():
        try:
            lmstudio_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return

    try:
        dest_dev = lmstudio_dir.stat().st_dev
    except OSError:
        dest_dev = None

    seen_root_real: set[Path] = set()
    seen_repo_real: set[Path] = set()

    for cache_root in HF_CACHE_DIRS:
        if not cache_root.is_dir():
            continue
        try:
            real_root = cache_root.resolve()
        except (OSError, RuntimeError):
            real_root = cache_root
        if real_root in seen_root_real:
            continue
        seen_root_real.add(real_root)

        try:
            entries = sorted(cache_root.iterdir())
        except OSError:
            continue

        for entry in entries:
            if not entry.is_dir() or entry.is_symlink():
                continue
            name = entry.name

            parsed = _parse_hf_cache_name(name)
            if parsed is None:
                continue  # skip datasets--*, non-HF dirs, etc.

            try:
                real_repo = entry.resolve()
            except (OSError, RuntimeError):
                real_repo = entry
            if real_repo in seen_repo_real:
                continue
            seen_repo_real.add(real_repo)

            publisher, repo = parsed
            display = f"{publisher}/{repo}"

            snap = _find_hf_latest_snapshot(entry)
            if snap is None:
                results.broken.append((display, entry, "No usable snapshot in HF cache dir"))
                continue

            # Is this actually a model (not tokenizer-only / config-only)?
            has_weights = any(
                f.suffix in ALL_MODEL_EXTENSIONS for f in snap.iterdir()
            )
            if not has_weights:
                continue

            # Flag LFS pointers — migrating them would produce useless stubs
            lfs_pointers = []
            weight_files = []
            for f in sorted(snap.iterdir()):
                if not (f.is_file() or f.is_symlink()):
                    continue
                if f.name.startswith("."):
                    continue
                if f.suffix in ALL_MODEL_EXTENSIONS:
                    weight_files.append(f)
                    if is_lfs_pointer(f):
                        lfs_pointers.append(f.name)

            if lfs_pointers:
                sample = ", ".join(lfs_pointers[:3])
                if len(lfs_pointers) > 3:
                    sample += f" +{len(lfs_pointers) - 3} more"
                results.broken.append(
                    (display, entry, f"LFS pointers — needs download: {sample}")
                )
                continue

            dest_dir = lmstudio_dir / publisher / repo

            # Idempotency: if destination already has every weight file at same size, skip
            if dest_dir.is_dir():
                existing = {f.name: f.stat().st_size
                            for f in dest_dir.iterdir() if f.is_file()}
                try:
                    src_sizes = {f.name: f.resolve().stat().st_size for f in weight_files}
                except OSError:
                    src_sizes = {}
                if src_sizes and all(existing.get(n) == sz for n, sz in src_sizes.items()):
                    if cleanup and not dry_run:
                        try:
                            shutil.rmtree(entry)
                            print(f"  CLEANUP {name}/  (already migrated)")
                        except OSError as e:
                            print(f"  ERROR cleaning {name}: {e}")
                    continue

            # Gather every file to migrate (weights + companions)
            files_to_migrate = []
            total_size = 0
            for f in sorted(snap.iterdir()):
                if not (f.is_file() or f.is_symlink()) or f.name.startswith("."):
                    continue
                try:
                    real = f.resolve()
                    if not real.is_file():
                        continue
                    total_size += real.stat().st_size
                except OSError:
                    continue
                files_to_migrate.append((f, real))

            if not files_to_migrate:
                continue

            # Decide link strategy
            try:
                src_dev = snap.stat().st_dev
                can_hardlink = dest_dev is not None and src_dev == dest_dev
            except OSError:
                can_hardlink = False

            method = "HARDLINK" if can_hardlink else "COPY"
            print(f"  HF-MIGRATE  {name}/ -> {display}/ "
                  f"({len(files_to_migrate)} files, {format_size(total_size)}, {method})")

            if dry_run:
                results.fixed.append((display, entry, dest_dir))
                continue

            try:
                dest_dir.mkdir(parents=True, exist_ok=True)
                for link, real_src in files_to_migrate:
                    dst = dest_dir / link.name
                    if dst.exists():
                        continue
                    if can_hardlink:
                        try:
                            os.link(real_src, dst)
                        except OSError:
                            # Fall back to copy if hardlink fails (e.g., cross-device, perms)
                            shutil.copy2(real_src, dst)
                    else:
                        shutil.copy2(real_src, dst)
                results.fixed.append((display, entry, dest_dir))

                if cleanup:
                    shutil.rmtree(entry)
                    print(f"              source removed")
            except OSError as e:
                print(f"              ERROR: {e}")
                results.errors.append((display, f"HF migration failed: {e}"))


# ──────────────────────────────────────────────────────────────────────
# Models-flat → LM Studio mirror
#
# `models_dir` (default: models-flat) is the manager's view, but LM Studio
# reads from its own `downloadsFolder` (lmstudio_dir). When the two paths
# differ, publisher/repo/ trees in models_dir are invisible to LM Studio.
# This mirror walks models_dir, finds publisher/repo/<files> trees that
# aren't already present in lmstudio_dir, and hardlinks the files in.
# Same-volume → instant, zero extra disk.
# ──────────────────────────────────────────────────────────────────────

# Top-level names under models_dir that are NOT publisher dirs and should
# be skipped during mirror (flat short-name bucket, staging, internal stores).
_MIRROR_SKIP_TOPLEVEL = {
    "local", "hf_downloads", ".incoming", ".cache",
    "blobs", "manifests", "refs", "snapshots", ".studio_links",
    "huggingface", "xet", "dataset",
}


def mirror_models_flat_to_lmstudio(
    models_dir: Path,
    lmstudio_dir: Path,
    dry_run: bool,
    results: Results,
) -> None:
    """
    Mirror publisher/repo/ trees from `models_dir` into `lmstudio_dir` so LM
    Studio's downloadsFolder actually sees them. Hardlinks when same-volume,
    falls back to copy2 otherwise. Idempotent: skips repos whose destination
    already has matching file sizes.

    Skips:
      - The flat-short-name bucket `local/<short>/` (LM Studio doesn't read
        that layout — that's for the manager + Ollama).
      - Anything starting with `.` or `models--` (HF cache style — handled
        by migrate_hf_cache_dirs).
      - Names in _MIRROR_SKIP_TOPLEVEL.
    """
    if not models_dir.is_dir() or not lmstudio_dir.exists():
        return

    try:
        if models_dir.resolve() == lmstudio_dir.resolve():
            return  # nothing to mirror, paths are the same
    except (OSError, RuntimeError):
        pass

    try:
        dest_dev = lmstudio_dir.stat().st_dev
    except OSError:
        dest_dev = None

    for publisher_dir in sorted(models_dir.iterdir()):
        if not publisher_dir.is_dir() or publisher_dir.is_symlink():
            continue
        name = publisher_dir.name
        if name in _MIRROR_SKIP_TOPLEVEL:
            continue
        if name.startswith(".") or name.startswith("models--") or name.startswith("datasets--"):
            continue

        try:
            repos = sorted(publisher_dir.iterdir())
        except OSError:
            continue

        for repo_dir in repos:
            if not repo_dir.is_dir() or repo_dir.is_symlink():
                continue

            try:
                files = [f for f in repo_dir.iterdir()
                         if (f.is_file() or f.is_symlink()) and not f.name.startswith(".")]
            except OSError:
                continue
            if not files:
                continue

            display = f"{publisher_dir.name}/{repo_dir.name}"
            dest_repo = lmstudio_dir / publisher_dir.name / repo_dir.name

            # Idempotency: same files at same sizes already in destination.
            if dest_repo.is_dir():
                try:
                    existing = {f.name: f.stat().st_size
                                for f in dest_repo.iterdir() if f.is_file()}
                except OSError:
                    existing = {}
                src_sizes: dict[str, int] = {}
                for f in files:
                    try:
                        src_sizes[f.name] = f.resolve().stat().st_size
                    except OSError:
                        pass
                if src_sizes and all(existing.get(n) == sz for n, sz in src_sizes.items()):
                    continue

            # Decide link strategy
            try:
                src_dev = repo_dir.stat().st_dev
                can_hardlink = dest_dev is not None and src_dev == dest_dev
            except OSError:
                can_hardlink = False

            method = "HARDLINK" if can_hardlink else "COPY"
            total_size = 0
            for f in files:
                try:
                    total_size += f.resolve().stat().st_size
                except OSError:
                    pass

            print(
                f"  MIRROR  {display}/  -> {dest_repo}  "
                f"({len(files)} files, {format_size(total_size)}, {method})"
            )

            if dry_run:
                results.fixed.append((display, repo_dir, dest_repo))
                continue

            try:
                dest_repo.mkdir(parents=True, exist_ok=True)
                for src in files:
                    dst = dest_repo / src.name
                    if dst.exists():
                        continue
                    try:
                        real_src = src.resolve()
                        if not real_src.is_file():
                            continue
                    except OSError:
                        continue
                    if can_hardlink:
                        try:
                            os.link(real_src, dst)
                        except OSError:
                            shutil.copy2(real_src, dst)
                    else:
                        shutil.copy2(real_src, dst)
                results.fixed.append((display, repo_dir, dest_repo))
            except OSError as e:
                print(f"          ERROR: {e}")
                results.errors.append((display, f"models-flat mirror failed: {e}"))


# ──────────────────────────────────────────────────────────────────────
# HuggingFace cache resume — re-download interrupted models--owner--repo/
# dirs that have only a refs/ entry (no blobs/, no snapshots/).
# ──────────────────────────────────────────────────────────────────────

def find_broken_hf_cache_dirs():
    """
    Walk HF_CACHE_DIRS and yield (cache_root, entry, owner, repo) for every
    `models--owner--repo` dir whose snapshots/ is missing or empty. Roots and
    repos are deduped by resolved physical path.
    """
    seen_root_real: set[Path] = set()
    seen_repo_real: set[Path] = set()

    for cache_root in HF_CACHE_DIRS:
        if not cache_root.is_dir():
            continue
        try:
            real_root = cache_root.resolve()
        except (OSError, RuntimeError):
            real_root = cache_root
        if real_root in seen_root_real:
            continue
        seen_root_real.add(real_root)

        try:
            entries = sorted(cache_root.iterdir())
        except OSError:
            continue

        for entry in entries:
            if not entry.is_dir() or entry.is_symlink():
                continue
            parsed = _parse_hf_cache_name(entry.name)
            if parsed is None:
                continue
            try:
                real_repo = entry.resolve()
            except (OSError, RuntimeError):
                real_repo = entry
            if real_repo in seen_repo_real:
                continue
            seen_repo_real.add(real_repo)

            owner, repo = parsed
            if _find_hf_latest_snapshot(entry) is None:
                yield (cache_root, entry, owner, repo)


def resume_broken_hf_downloads(dry_run: bool, workers: int = 1) -> None:
    """
    Resume broken HF hub cache downloads via huggingface_hub.snapshot_download.
    snapshot_download is idempotent — it hashes existing blobs and only fetches
    missing ones. Honors HF_TOKEN env var (or <REDACTED_PATH>) for
    gated repos. Skips on auth/404/network errors with a clear reason.
    """
    try:
        from huggingface_hub import snapshot_download
        try:
            from huggingface_hub.errors import (
                GatedRepoError, RepositoryNotFoundError, HfHubHTTPError,
            )
        except ImportError:  # huggingface_hub < 0.20
            from huggingface_hub.utils import (
                GatedRepoError, RepositoryNotFoundError, HfHubHTTPError,
            )
    except ImportError:
        print("  [!] huggingface_hub not installed; pip install huggingface_hub")
        return

    candidates = list(find_broken_hf_cache_dirs())
    if not candidates:
        print("  No broken HF cache dirs to resume.")
        return

    print(f"  Found {len(candidates)} broken HF cache dir(s).")

    if dry_run:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
        except ImportError:
            api = None
        total = 0
        for cache_root, entry, owner, repo in candidates:
            repo_id = f"{owner}/{repo}"
            size_str = "size unknown"
            if api is not None:
                try:
                    info = api.model_info(repo_id, files_metadata=True)
                    size = sum((s.size or 0) for s in info.siblings if s.size)
                    if size:
                        total += size
                        size_str = format_size(size)
                except (GatedRepoError, RepositoryNotFoundError, HfHubHTTPError) as e:
                    size_str = f"size unavailable ({type(e).__name__})"
                except Exception as e:
                    size_str = f"size unavailable ({type(e).__name__})"
            print(f"    DRY  {repo_id:<60} cache={cache_root}  {size_str}")
        if total:
            print(f"  Estimated total to download: {format_size(total)}")
        return

    success: list[str] = []
    failed: list[tuple[str, str]] = []

    def _resume_one(args_tuple):
        cache_root, entry, owner, repo = args_tuple
        repo_id = f"{owner}/{repo}"
        try:
            snapshot_download(repo_id=repo_id, cache_dir=str(cache_root))
            return (repo_id, True, None)
        except GatedRepoError:
            return (repo_id, False, "GATED — set HF_TOKEN")
        except RepositoryNotFoundError:
            return (repo_id, False, "NOT_FOUND on HF")
        except HfHubHTTPError as e:
            status = getattr(e.response, "status_code", "?") if getattr(e, "response", None) else "?"
            return (repo_id, False, f"HTTP {status}")
        except OSError as e:
            return (repo_id, False, f"OSError: {e}")
        except Exception as e:
            return (repo_id, False, f"{type(e).__name__}: {e}")

    if workers <= 1:
        for tup in candidates:
            repo_id, ok, err = _resume_one(tup)
            if ok:
                _tlog(f"  ✓ RESUMED  {repo_id}")
                success.append(repo_id)
            else:
                _tlog(f"  ✗ FAILED   {repo_id}  ({err})")
                failed.append((repo_id, err))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_resume_one, tup): tup for tup in candidates}
            for future in as_completed(futures):
                repo_id, ok, err = future.result()
                if ok:
                    _tlog(f"  ✓ RESUMED  {repo_id}")
                    success.append(repo_id)
                else:
                    _tlog(f"  ✗ FAILED   {repo_id}  ({err})")
                    failed.append((repo_id, err))

    print(f"  Resume summary: {len(success)} resumed, {len(failed)} failed")


def _read_input(prompt: str) -> str:
    """input() wrapper that handles EOF/Ctrl-C cleanly during interactive prompts."""
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        print()
        return "q"


def interactive_broken_actions(broken: list[tuple[str, Path, str]]) -> None:
    """
    For each broken entry, prompt the user for an action: resume, delete, skip, quit.
    Resume: calls huggingface_hub.snapshot_download for repos with parseable owner/repo.
    Delete: requires explicit "Are you sure?" confirmation, then queues an rm command
            (does NOT execute — printed at the end for user to run, per local convention).
    """
    try:
        from huggingface_hub import snapshot_download
        try:
            from huggingface_hub.errors import (
                GatedRepoError, RepositoryNotFoundError, HfHubHTTPError,
            )
        except ImportError:
            from huggingface_hub.utils import (
                GatedRepoError, RepositoryNotFoundError, HfHubHTTPError,
            )
        hf_available = True
    except ImportError:
        hf_available = False

    print()
    print("─" * 78)
    print("Interactive broken-entry handler")
    print(f"  {len(broken)} broken entries — actions: [r]esume  [d]elete  [s]kip  [q]uit")
    print("─" * 78)

    delete_queue: list[Path] = []

    for idx, (name, path, reason) in enumerate(broken, 1):
        print()
        print(f"[{idx}/{len(broken)}]  {name}")
        print(f"    path:   {path}")
        print(f"    reason: {reason}")

        while True:
            choice = _read_input("    Action [r/d/s/q]? ").strip().lower()
            if choice in {"r", "resume"}:
                if not hf_available:
                    print("    huggingface_hub not installed; cannot resume here. Skipping.")
                    break
                if "/" not in name:
                    print("    Cannot derive owner/repo from name; skipping.")
                    break
                cache_root = path.parent if path.parent.is_dir() else path
                print(f"    Resuming via huggingface_hub.snapshot_download(repo_id={name!r}, cache_dir={str(cache_root)!r})…")
                try:
                    snapshot_download(repo_id=name, cache_dir=str(cache_root))
                    print(f"    ✓ Resumed: {name}")
                except GatedRepoError:
                    print("    ✗ Gated repo — set HF_TOKEN to access this model.")
                except RepositoryNotFoundError:
                    print("    ✗ Repo not found on HF.")
                except HfHubHTTPError as e:
                    status = getattr(getattr(e, "response", None), "status_code", "?")
                    print(f"    ✗ HTTP error {status}")
                except Exception as e:
                    print(f"    ✗ {type(e).__name__}: {e}")
                break
            if choice in {"d", "delete"}:
                confirm = _read_input(f"    Are you sure you want to delete {path}? [y/N] ").strip().lower()
                if confirm in {"y", "yes"}:
                    delete_queue.append(path)
                    print(f"    Queued for deletion (you'll run the rm at the end): {path}")
                else:
                    print("    Cancelled.")
                break
            if choice in {"s", "skip", ""}:
                break
            if choice in {"q", "quit", "exit"}:
                print("    Quitting interactive handler.")
                if delete_queue:
                    _print_delete_queue(delete_queue)
                return
            print("    Invalid choice. Use r, d, s, or q.")

    if delete_queue:
        _print_delete_queue(delete_queue)
    else:
        print()
        print("No entries queued for deletion.")


def _print_delete_queue(paths: list[Path]) -> None:
    print()
    print("─" * 78)
    print(f"You queued {len(paths)} path(s) for deletion. Copy and run:")
    print("─" * 78)
    print()
    print("rm -rf \\")
    for i, p in enumerate(paths):
        sep = " \\" if i < len(paths) - 1 else ""
        print(f"  {str(p)!r}{sep}")
    print()


# Extra location where model_manager.py steers huggingface_hub.snapshot_download
# refs/<revision> stubs to keep them out of the user's real HF_HUB_CACHE.
_MODEL_MANAGER_STUB_CACHE = Path.home() / ".cache" / "model_manager" / "hf_refs_stubs"


def report_orphan_hf_stubs() -> None:
    """
    Walk every HF cache root (the user's real caches plus the model_manager
    throwaway stub cache) and report every `models--owner--repo/` directory
    that has only `refs/` and no blobs/ or snapshots/ — the residue left by
    huggingface_hub.snapshot_download(local_dir=...) when the actual files go
    to local_dir but the library still writes a refs/<revision> stub.

    READ-ONLY. Prints exact `rm -rf` commands grouped by parent dir for the
    user to copy and run; never deletes anything itself.
    """
    locations = list(HF_CACHE_DIRS) + [_MODEL_MANAGER_STUB_CACHE]

    seen_root_real: set[Path] = set()
    seen_repo_real: set[Path] = set()
    orphans: list[Path] = []

    for cache_root in locations:
        if not cache_root.is_dir():
            continue
        try:
            real_root = cache_root.resolve()
        except (OSError, RuntimeError):
            real_root = cache_root
        if real_root in seen_root_real:
            continue
        seen_root_real.add(real_root)

        try:
            entries = sorted(cache_root.iterdir())
        except OSError:
            continue

        for entry in entries:
            if not entry.is_dir() or entry.is_symlink():
                continue
            if not entry.name.startswith(("models--", "datasets--")):
                continue
            try:
                real_repo = entry.resolve()
            except (OSError, RuntimeError):
                real_repo = entry
            if real_repo in seen_repo_real:
                continue
            seen_repo_real.add(real_repo)

            try:
                children = {p.name for p in entry.iterdir()}
            except OSError:
                continue
            if "refs" not in children:
                continue
            blobs_dir = entry / "blobs"
            snapshots_dir = entry / "snapshots"
            try:
                has_blobs = blobs_dir.is_dir() and any(blobs_dir.iterdir())
            except OSError:
                has_blobs = False
            try:
                has_snapshots = snapshots_dir.is_dir() and any(snapshots_dir.iterdir())
            except OSError:
                has_snapshots = False
            if has_blobs or has_snapshots:
                continue
            orphans.append(entry)

    if not orphans:
        print("  No orphan HF cache stubs found.")
        return

    print(f"  Found {len(orphans)} orphan HF cache stub(s) (refs/ only, no blobs/snapshots).")
    print(f"  These are bookkeeping residue from huggingface_hub.snapshot_download —")
    print(f"  the actual model data is elsewhere. Each stub is ~96 bytes.")
    print()
    print(f"  To delete them, copy and run these commands:")
    print()

    by_parent: dict[Path, list[str]] = {}
    for o in orphans:
        by_parent.setdefault(o.parent, []).append(o.name)
    for parent in sorted(by_parent):
        names = sorted(by_parent[parent])
        print(f"  cd {parent} && rm -rf \\")
        for i, name in enumerate(names):
            sep = " \\" if i < len(names) - 1 else ""
            print(f"    {name}{sep}")
        print()


# Model-file extensions we treat as "weights" when looking for dangling symlinks.
_DANGLING_LINK_EXTS = (".gguf", ".safetensors", ".bin", ".pt", ".pth", ".mlmodel", ".mlpackage")

_FIRST_SHARD_RE = re.compile(r"-0*1-of-0*\d+\.gguf$", re.IGNORECASE)


def _find_primary_gguf_for_ready_entry(name: str, models_dir: Path, lmstudio_dir: Path) -> Path | None:
    """For a READY entry like 'CohereForAI/c4ai-command-r-plus' or 'local/foo',
    locate the primary GGUF (first shard if split, else largest single-file).
    Returns None if not findable."""
    candidates = [lmstudio_dir / name, models_dir / name]
    for target in candidates:
        try:
            if not target.is_dir():
                continue
        except OSError:
            continue
        ggufs: list[Path] = []
        try:
            ggufs = list(target.rglob("*.gguf"))
        except OSError:
            continue
        if not ggufs:
            continue
        # Prefer first shard of a split GGUF, else largest single file.
        for g in ggufs:
            if _FIRST_SHARD_RE.search(g.name):
                return g
        try:
            return max(ggufs, key=lambda p: p.stat().st_size)
        except OSError:
            return ggufs[0]
    return None


def _sum_split_gguf_size(gguf_path: Path) -> int:
    """If gguf_path is the first shard of a split set, sum sizes of all shards
    in the same parent dir. Otherwise return just gguf_path's size."""
    try:
        if _FIRST_SHARD_RE.search(gguf_path.name):
            parent = gguf_path.parent
            shards = list(parent.glob("*.gguf"))
            return sum(p.stat().st_size for p in shards if p.is_file())
        return gguf_path.stat().st_size
    except OSError:
        return 0


def report_dangling_symlinks(extra_roots: list[Path] | None = None) -> None:
    """
    Walk LM Studio model dirs (and any extra_roots) for *.gguf / *.safetensors /
    weight-file symlinks that are broken (target no longer exists). Prints
    `rm` commands grouped by parent dir. Read-only — does not delete.
    """
    roots = [
        Path.home() / ".lmstudio" / "models",
        DEFAULT_LMSTUDIO_DIR,
        DEFAULT_MODELS_DIR,
    ]
    if extra_roots:
        roots.extend(extra_roots)

    seen_root_real: set[Path] = set()
    broken: list[Path] = []

    for root in roots:
        if not root.is_dir():
            continue
        try:
            real_root = root.resolve()
        except (OSError, RuntimeError):
            real_root = root
        if real_root in seen_root_real:
            continue
        seen_root_real.add(real_root)

        for ext in _DANGLING_LINK_EXTS:
            for path in root.rglob(f"*{ext}"):
                try:
                    if not path.is_symlink():
                        continue
                    if path.exists():
                        continue  # symlink target resolves
                except OSError:
                    continue
                broken.append(path)

    if not broken:
        print("  No dangling weight-file symlinks found.")
        return

    print(f"  Found {len(broken)} dangling symlink(s) — target no longer exists.")
    print(f"  These are residue from previous prepare runs (source models deleted/moved).")
    print()
    print(f"  To delete them, copy and run these commands:")
    print()

    by_parent: dict[Path, list[str]] = {}
    for p in broken:
        by_parent.setdefault(p.parent, []).append(p.name)
    for parent in sorted(by_parent):
        names = sorted(by_parent[parent])
        print(f"  cd {parent} && rm \\")
        for i, name in enumerate(names):
            sep = " \\" if i < len(names) - 1 else ""
            print(f"    {name!r}{sep}")
        print()

    # Also report empty parent dirs that would be left behind, so the user can rmdir if they want
    empty_parent_candidates = sorted({p.parent for p in broken})
    print("  After removing the symlinks above, these parent dirs may become empty:")
    for parent in empty_parent_candidates:
        print(f"    {parent}")
    print()
    print("  To remove now-empty dirs after the rm above, you can run:")
    print(f"    find {' '.join(repr(str(p)) for p in seen_root_real)} \\")
    print(f"      -type d -empty -depth -delete")


# ──────────────────────────────────────────────────────────────────────
# Loose file scanning — models with actual weight files in directories
# ──────────────────────────────────────────────────────────────────────

def scan_loose_models(models_dir: Path, dry_run: bool, results: Results):
    """
    Scan for models with actual weight files in publisher/model/ directories
    (not referenced through manifests). Fix structure if needed.
    """
    seen_dirs = set()

    for child in sorted(models_dir.iterdir()):
        if not child.is_dir() or child.name.startswith(".") or child.name in INTERNAL_DIRS:
            continue
        # HF cache dirs are owned by migrate_hf_cache_dirs; don't touch them here.
        if child.name.startswith(("models--", "datasets--")):
            continue

        # Direct model directory at top level (needs fixing into publisher/model/)
        if is_model_directory(child):
            seen_dirs.add(child)
            _fix_model_dir(child, models_dir, dry_run, results)
            continue

        # Publisher directory — check model subdirectories
        for grandchild in sorted(child.iterdir()):
            if not grandchild.is_dir() or grandchild.name.startswith("."):
                continue
            if is_model_directory(grandchild):
                seen_dirs.add(grandchild)
                _check_or_fix_model_dir(grandchild, models_dir, results)

    # Standalone GGUF files. Filter out directory matches (HF repos sometimes
    # have names ending in `.gguf`, e.g. `katanemo/Arch-Router-1.5B.gguf/`).
    for gguf in sorted(models_dir.glob("**/*.gguf")):
        if not gguf.is_file():
            continue
        if gguf.parent in seen_dirs:
            continue
        try:
            rel = gguf.relative_to(models_dir)
            if rel.parts[0] in INTERNAL_DIRS:
                continue
            # Never "fix" GGUFs that live inside HF cache — the migration handles them.
            if rel.parts[0].startswith(("models--", "datasets--")):
                continue
        except ValueError:
            continue
        _fix_gguf(gguf, models_dir, dry_run, results)


def _check_or_fix_model_dir(model_dir: Path, models_dir: Path, results: Results):
    """Check a model dir that's already in publisher/model/ structure."""
    try:
        rel = model_dir.relative_to(models_dir)
        if len(rel.parts) == 2:
            # Already correct structure
            # Check for LFS pointers
            for f in model_dir.iterdir():
                if f.suffix in ALL_MODEL_EXTENSIONS and is_lfs_pointer(f):
                    results.broken.append(
                        (f"{rel.parts[0]}/{rel.parts[1]}", model_dir,
                         "Contains LFS pointer files"))
                    return

            model_type = _get_model_type_str(model_dir)
            total_size = sum(
                f.stat().st_size for f in model_dir.iterdir()
                if f.is_file() and f.suffix in ALL_MODEL_EXTENSIONS
            )
            display = f"{rel.parts[0]}/{rel.parts[1]}"
            results.ready.append((display, format_size(total_size), model_type))
            return
    except ValueError:
        pass


def _get_model_type_str(path: Path) -> str:
    if any(path.glob("*.safetensors")):
        if (path / "weights.json").exists() or "mlx" in path.name.lower():
            return "MLX"
        return "safetensors"
    if any(path.glob("*.gguf")):
        return "GGUF"
    if any(path.glob("pytorch_model*.bin")):
        return "pytorch"
    return "unknown"


def _fix_gguf(gguf_path: Path, models_dir: Path, dry_run: bool, results: Results):
    # Path.glob("**/*.gguf") matches DIRECTORIES whose name ends in ".gguf"
    # too (e.g. HF repos like `katanemo/Arch-Router-1.5B.gguf/`). Bail early
    # — those are model dirs handled by _fix_model_dir / _check_or_fix_model_dir.
    if not gguf_path.is_file():
        return

    name = gguf_path.name

    if is_lfs_pointer(gguf_path):
        results.broken.append((name, gguf_path, "Git LFS pointer"))
        return

    try:
        rel = gguf_path.relative_to(models_dir)
        if len(rel.parts) == 3:
            results.ready.append(("/".join(rel.parts[:2]), format_size(gguf_path.stat().st_size), "GGUF"))
            return
    except ValueError:
        pass

    publisher, model_name = infer_publisher_and_model(gguf_path)
    target_dir = models_dir / publisher / model_name
    dest = target_dir / name
    display = f"{publisher}/{model_name}/{name}"

    if dest.resolve() == gguf_path.resolve():
        results.ready.append((display, format_size(gguf_path.stat().st_size), "GGUF"))
        return

    print(f"  FIXED   {name} -> {display}")
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(gguf_path), str(dest))
        cleanup_empty_dirs(gguf_path.parent, models_dir)
    results.fixed.append((display, gguf_path, dest))


def _fix_model_dir(model_dir: Path, models_dir: Path, dry_run: bool, results: Results):
    name = model_dir.name

    # Check for LFS pointers
    for f in model_dir.iterdir():
        if f.suffix in ALL_MODEL_EXTENSIONS and is_lfs_pointer(f):
            results.broken.append((name, model_dir, "Contains LFS pointer files"))
            return

    publisher, model_name = infer_publisher_and_model(model_dir)
    target_dir = models_dir / publisher / model_name
    display = f"{publisher}/{model_name}"

    if target_dir.resolve() == model_dir.resolve():
        total_size = sum(f.stat().st_size for f in model_dir.iterdir()
                         if f.is_file() and f.suffix in ALL_MODEL_EXTENSIONS)
        results.ready.append((display, format_size(total_size), _get_model_type_str(model_dir)))
        return

    files_to_move = [f for f in sorted(model_dir.iterdir()) if should_move_file(f)]
    if not files_to_move:
        return

    total_size = sum(f.stat().st_size for f in files_to_move)
    model_type = _get_model_type_str(model_dir)
    print(f"  FIXED   {name}/ -> {display}/ ({model_type}, {format_size(total_size)})")

    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
        for f in files_to_move:
            shutil.move(str(f), str(target_dir / f.name))
        if model_dir.is_dir() and not any(
            f for f in model_dir.iterdir() if not f.name.startswith(".")
        ):
            shutil.rmtree(str(model_dir), ignore_errors=True)
        cleanup_empty_dirs(model_dir.parent, models_dir)

    results.fixed.append((display, model_dir, target_dir))


# ──────────────────────────────────────────────────────────────────────
# Duplicate detection
# ──────────────────────────────────────────────────────────────────────

def find_and_delete_duplicates(models_dir: Path, dry_run: bool, results: Results):
    """Find duplicate loose models (same weight files at multiple paths) and delete extras."""
    gguf_index = {}
    dir_index = {}

    for publisher_dir in sorted(models_dir.iterdir()):
        if not publisher_dir.is_dir() or publisher_dir.name.startswith("."):
            continue
        if publisher_dir.name in INTERNAL_DIRS:
            continue
        for model_dir in sorted(publisher_dir.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith("."):
                continue

            display = f"{publisher_dir.name}/{model_dir.name}"

            gguf_files = list(model_dir.glob("*.gguf"))
            if gguf_files:
                for gf in gguf_files:
                    if is_lfs_pointer(gf):
                        continue
                    key = (gf.name, gf.stat().st_size)
                    gguf_index.setdefault(key, []).append((f"{display}/{gf.name}", gf))
                continue

            weight_files = []
            for f in model_dir.iterdir():
                if f.suffix in ALL_MODEL_EXTENSIONS and not is_lfs_pointer(f):
                    weight_files.append((f.name, f.stat().st_size))
            if weight_files:
                key = frozenset(weight_files)
                dir_index.setdefault(key, []).append((display, model_dir))

    def sort_key(item):
        display, _ = item
        publisher = display.split("/")[0] if "/" in display else ""
        is_known = 0 if publisher.lower() in {p.lower() for p in KNOWN_PUBLISHERS} else 1
        return (is_known, display.lower())

    for key, copies in gguf_index.items():
        if len(copies) <= 1:
            continue
        copies.sort(key=sort_key)
        keep_display, keep_path = copies[0]
        for dup_display, dup_path in copies[1:]:
            # Hardlink-aware: if the "duplicate" and the "keeper" point at the
            # same inode, they're the SAME file via two paths — leave alone.
            try:
                ks = keep_path.stat()
                ds = dup_path.stat()
                if (ks.st_dev, ks.st_ino) == (ds.st_dev, ds.st_ino):
                    continue
            except OSError:
                pass
            print(f"  DUPLICATE  {dup_display} (keeping {keep_display})")
            if not dry_run:
                try:
                    dup_path.unlink()
                except OSError as e:
                    print(f"    ERROR unlinking: {e}")
                    continue
                parent = dup_path.parent
                # Skip parent cleanup for symlinks (they're not real dirs).
                if parent.is_symlink():
                    continue
                try:
                    if parent.is_dir() and not any(parent.iterdir()):
                        parent.rmdir()
                except (OSError, NotADirectoryError):
                    pass
            results.duplicates.append((dup_display, dup_path, keep_path))

    for key, copies in dir_index.items():
        if len(copies) <= 1:
            continue
        copies.sort(key=sort_key)
        keep_display, keep_path = copies[0]
        for dup_display, dup_path in copies[1:]:
            print(f"  DUPLICATE  {dup_display}/ (keeping {keep_display}/)")
            if not dry_run:
                shutil.rmtree(str(dup_path))
            results.duplicates.append((dup_display, dup_path, keep_path))


# ──────────────────────────────────────────────────────────────────────
# Ollama registration
# ──────────────────────────────────────────────────────────────────────

def load_ollama_state() -> dict:
    try:
        if OLLAMA_STATE_FILE.exists():
            return json.loads(OLLAMA_STATE_FILE.read_text())
    except Exception:
        pass
    return {}


def save_ollama_state(state: dict):
    try:
        OLLAMA_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        OLLAMA_STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception as e:
        print(f"  WARNING: Could not save Ollama state: {e}")


def ollama_name_from_manifest(publisher: str, model_name: str) -> str:
    """Derive Ollama model name from manifest path."""
    if publisher == "library":
        return model_name.lower()
    return f"{publisher}/{model_name}".lower()


def register_with_ollama(models_dir: Path, dry_run: bool, force: bool,
                         results: Results, workers: int = 4):
    """
    Register GGUF models with Ollama. Threaded: each `ollama create` call
    runs in its own worker (subprocess is I/O-bound from our side). Ollama's
    daemon serializes internally, so more than ~8 workers rarely helps.
    """
    try:
        subprocess.run(["ollama", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("\n  Ollama not found — skipping registration")
        return

    state = load_ollama_state()
    state_lock = threading.Lock()
    temp_dir = Path(tempfile.mkdtemp(prefix="ollama_modelfiles_"))
    blobs_dir = models_dir / "blobs"
    manifest_root = models_dir / "manifests" / "registry.ollama.ai"

    # Gather all registration tasks first (serial walk), then parallelize the
    # subprocess calls — the walk is cheap but the `ollama create` calls are 1-10s each.
    tasks = []  # (name, blob/gguf_path, display, mtime, is_mmproj_skippable)

    def _queue(name, blob, display):
        try:
            mtime = str(int(blob.stat().st_mtime))
        except OSError:
            return
        if not force and state.get(name) == mtime:
            results.skipped_ollama.append((display, name))
            return
        tasks.append((name, blob, display, mtime))

    # Pass 1: from Ollama manifests (blob-based)
    if manifest_root.is_dir():
        for manifest_file in sorted(manifest_root.rglob("*")):
            if not manifest_file.is_file():
                continue
            try:
                rel = manifest_file.relative_to(manifest_root)
                parts = rel.parts
                if len(parts) < 3:
                    continue
                publisher, model_name = parts[0], parts[1]
            except ValueError:
                continue
            if publisher == "_image-models":
                continue
            if "mmproj" in model_name.lower():
                continue
            try:
                manifest = json.loads(manifest_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(manifest, dict):
                continue
            model_layer = None
            for layer in manifest.get("layers") or []:
                if layer.get("mediaType") == "application/vnd.ollama.image.model":
                    model_layer = layer
                    break
            if not model_layer:
                continue
            from_path = model_layer.get("from", "")
            digest = model_layer.get("digest", "")
            if from_path:
                blob = Path(from_path)
            elif digest:
                blob = blobs_dir / digest.replace(":", "-")
            else:
                continue
            if not blob.exists():
                continue
            name = ollama_name_from_manifest(publisher, model_name)
            _queue(name, blob, f"{publisher}/{model_name}")

    # Pass 2: loose GGUFs in publisher/model/*.gguf
    for gguf in sorted(models_dir.glob("*/*/*.gguf")):
        if not gguf.is_file():
            continue
        try:
            rel = gguf.relative_to(models_dir)
            if rel.parts[0] in INTERNAL_DIRS:
                continue
        except ValueError:
            continue
        if "mmproj" in gguf.name.lower():
            continue
        if is_lfs_pointer(gguf):
            continue
        publisher, model_name = rel.parts[0], rel.parts[1]
        name = f"{publisher}/{model_name}".lower()
        _queue(name, gguf, gguf.name)

    if not tasks:
        shutil.rmtree(str(temp_dir), ignore_errors=True)
        return

    def _register_one(task):
        name, blob, display, mtime = task
        _tlog(f"  REGISTER  {display} -> ollama:{name}")
        if dry_run:
            results.registered.append((display, name))
            return
        modelfile = temp_dir / f"Modelfile-{name.replace('/', '-').replace(':', '_')}"
        try:
            modelfile.write_text(f"FROM {blob}\n")
            subprocess.run(
                ["ollama", "create", name, "-f", str(modelfile)],
                capture_output=True, check=True, timeout=300,
            )
            with state_lock:
                state[name] = mtime
            results.registered.append((display, name))
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode(errors="replace").strip() if e.stderr else "unknown"
            last_line = err.splitlines()[-1] if err else "unknown"
            _tlog(f"           FAILED {name}: {last_line}")
            results.errors.append((display, f"Ollama: {err[:200]}"))
        except subprocess.TimeoutExpired:
            _tlog(f"           FAILED {name}: timeout")
            results.errors.append((display, "Ollama: timeout"))

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Use list() to force all futures to run even though return is None
            list(pool.map(_register_one, tasks))
        if not dry_run:
            save_ollama_state(state)
    finally:
        shutil.rmtree(str(temp_dir), ignore_errors=True)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare models for LM Studio and register with Ollama.",
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help=f"Models directory to scan (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview only.")
    parser.add_argument("--no-ollama", action="store_true", help="Skip Ollama registration.")
    parser.add_argument("--force", action="store_true", help="Re-register all with Ollama.")
    parser.add_argument(
        "--migrate-hf-cache", action=argparse.BooleanOptionalAction, default=True,
        help="Convert HuggingFace cache dirs (models--owner--repo/) under HF_CACHE_DIRS "
             "into LM Studio's publisher/repo/ layout under --lmstudio-dir. "
             "Default: enabled. Use --no-migrate-hf-cache to skip.",
    )
    parser.add_argument(
        "--mirror-models-flat", action=argparse.BooleanOptionalAction, default=True,
        help="Hardlink publisher/repo/ trees from --input (models-flat) into --lmstudio-dir "
             "so LM Studio's downloadsFolder sees them. Default: enabled. "
             "Use --no-mirror-models-flat to skip.",
    )
    parser.add_argument(
        "--lmstudio-dir", type=Path, default=DEFAULT_LMSTUDIO_DIR,
        help=f"Destination root for HF-cache migration (LM Studio's downloadsFolder). "
             f"Default: {DEFAULT_LMSTUDIO_DIR}",
    )
    parser.add_argument(
        "--cleanup-hf-source", action="store_true",
        help="After a successful HF cache migration, delete the original models--*--* dir.",
    )
    parser.add_argument(
        "--resume-broken", action="store_true",
        help="Resume HF hub-cache downloads that have refs/ but no usable snapshot (failed "
             "or interrupted downloads). Uses huggingface_hub.snapshot_download — idempotent, "
             "honors HF_TOKEN. Combined with --dry-run reports per-repo size estimate only.",
    )
    parser.add_argument(
        "--resume-broken-workers", type=int, default=1, metavar="N",
        help="Parallel repos when resuming (default 1; HF rate limits make >2 risky).",
    )
    parser.add_argument(
        "--clean-orphan-stubs", action="store_true",
        help="Scan HF cache roots for `models--*` dirs that contain only refs/ and no "
             "blobs/snapshots/ — bookkeeping residue from huggingface_hub.snapshot_download. "
             "Prints copy-pasteable rm commands; never deletes anything itself.",
    )
    parser.add_argument(
        "--clean-symlinks", action="store_true",
        help="Walk LM Studio model dirs (<REDACTED_PATH>, --lmstudio-dir, --input) for "
             "broken weight-file symlinks (target no longer exists). Prints copy-pasteable "
             "rm commands grouped by parent; never deletes anything itself.",
    )
    parser.add_argument(
        "--interactive-broken", action="store_true",
        help="After the BROKEN list prints, prompt per-entry for an action: "
             "resume (re-download via huggingface_hub), delete (with 'Are you sure?' confirm; "
             "prints rm commands, never auto-deletes), skip, or quit.",
    )
    parser.add_argument(
        "--clean-partial-downloads", action="store_true",
        help="Remove abandoned *.incomplete / *.partial / *.lock files from HF cache blobs/.",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Verify each model is mount-ready (no broken symlinks, LFS stubs, bad GGUF, missing shards).",
    )
    parser.add_argument(
        "--also-scan", type=Path, action="append", default=[],
        help="Additional directory to scan for partial downloads (can repeat).",
    )
    parser.add_argument(
        "--flatten", action="store_true",
        help="Build a short-name rename plan across all input roots and write rename-plan.json.",
    )
    parser.add_argument(
        "--apply-flatten", type=Path, default=None, metavar="PLAN.json",
        help="Read a rename-plan.json produced by --flatten and apply it (hardlinks).",
    )
    parser.add_argument(
        "--flatten-output", type=Path,
        default=Path("<Your Model Directory>"),
        help="Target directory for flattened models (default: <Your Model Directory>).",
    )
    parser.add_argument(
        "--flatten-input", type=Path, action="append", default=[],
        help="Additional model root to include in --flatten scan (can repeat).",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Parallel workers for apply-flatten (default 8).",
    )
    parser.add_argument(
        "--ollama-workers", type=int, default=4,
        help="Parallel workers for Ollama registration (default 4; "
             "ollama daemon serializes internally so more rarely helps).",
    )
    parser.add_argument(
        "--dedupe", action="store_true",
        help="Run the duplicate-detection pass. DESTRUCTIVE — deletes files "
             "across publisher/model/ paths. Don't use when you have the "
             "flat-dir + AI-Studio symlink farm layout.",
    )
    args = parser.parse_args()

    models_dir = (args.input or DEFAULT_MODELS_DIR).expanduser().resolve()

    if not models_dir.exists():
        sys.exit(f"Error: models directory does not exist: {models_dir}")

    if args.dry_run:
        print("=== DRY RUN ===\n")

    print(f"Models: {models_dir}\n")

    results = Results()

    # Step 1: Scan manifests (blob-referenced models)
    print("Scanning manifests...")
    scan_manifests(models_dir, results)

    # Step 1.25: Remove partial-download residue (before anything else touches files).
    if args.clean_partial_downloads:
        print("Cleaning partial downloads...")
        roots = [models_dir, HF_CACHE_DIR] + [p.expanduser().resolve() for p in args.also_scan]
        clean_partial_downloads(roots, args.dry_run, results)

    # Step 1.5: Migrate HF cache dirs if requested (must happen BEFORE scan_loose_models
    # so the newly-created publisher/repo/ dirs get picked up normally)
    if args.clean_orphan_stubs:
        print("Scanning for orphan HF cache stubs...")
        report_orphan_hf_stubs()

    if args.clean_symlinks:
        print("Scanning for dangling weight-file symlinks...")
        report_dangling_symlinks(extra_roots=[args.lmstudio_dir.expanduser().resolve()])

    if args.resume_broken:
        print("Resuming broken HF cache downloads...")
        resume_broken_hf_downloads(args.dry_run, workers=args.resume_broken_workers)

    if args.migrate_hf_cache:
        lmstudio_dir = args.lmstudio_dir.expanduser().resolve()
        print(f"Migrating HuggingFace cache dirs into {lmstudio_dir}...")
        migrate_hf_cache_dirs(lmstudio_dir, args.dry_run, args.cleanup_hf_source, results)

    if args.mirror_models_flat:
        lmstudio_dir = args.lmstudio_dir.expanduser().resolve()
        if models_dir.resolve() != lmstudio_dir.resolve():
            print(f"Mirroring publisher/repo/ trees from {models_dir} into {lmstudio_dir}...")
            mirror_models_flat_to_lmstudio(models_dir, lmstudio_dir, args.dry_run, results)

    # Step 2: Scan loose model files and fix structure
    print("Scanning model files...")
    scan_loose_models(models_dir, args.dry_run, results)

    # Step 2.25: Validate mount-readiness of every model in publisher/repo/ layout
    if args.validate:
        print("Validating model integrity...")
        validate_all_models(models_dir, results)

    # Step 2.5: Flatten mode — build rename plan
    if args.flatten:
        print("\nBuilding flatten rename plan...")
        flatten_roots = [models_dir] + [p.expanduser().resolve() for p in args.flatten_input]
        plan = build_flatten_plan(flatten_roots, args.flatten_output.expanduser().resolve())
        plan_path = args.flatten_output.expanduser().resolve() / "rename-plan.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan, indent=2))
        print(f"  Plan: {len(plan['entries'])} entries, "
              f"{len(plan['collisions'])} collision groups")
        print(f"  Written to: {plan_path}")
        # Preview the plan inline
        print()
        for e in plan["entries"][:40]:
            print(f"    {e['short_name']:<50} <- {e['original']}")
        if len(plan["entries"]) > 40:
            print(f"    ... and {len(plan['entries']) - 40} more (see {plan_path.name})")

    # Step 2.6: Apply a previously-built plan
    if args.apply_flatten:
        plan_path = args.apply_flatten.expanduser().resolve()
        if not plan_path.is_file():
            sys.exit(f"Error: rename plan not found at {plan_path}")
        print(f"\nApplying flatten plan from {plan_path}...")
        plan = json.loads(plan_path.read_text())
        apply_flatten_plan(plan, args.dry_run, results, workers=args.workers)

    # Step 2.5: Scan HuggingFace cache
    print("Scanning HuggingFace cache...")
    scan_hf_cache(results)

    # Step 3: Optional duplicate deletion. NOT default-on anymore — it's
    # destructive and unsafe with the flat-dir + symlink-farm layout (it would
    # delete the canonical `local/{short}/` hardlinks because they look like
    # duplicates of `publisher/{model}/` paths that point at them).
    if args.dedupe:
        print("Checking for duplicates (--dedupe)...")
        find_and_delete_duplicates(models_dir, args.dry_run, results)

    # Step 4: Register with Ollama
    if not args.no_ollama:
        print("Registering with Ollama...")
        register_with_ollama(models_dir, args.dry_run, args.force, results,
                             workers=args.ollama_workers)

    # ── Report ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")

    if results.fixed:
        print(f"\nFIXED ({len(results.fixed)}):")
        for name, *_ in results.fixed:
            print(f"  + {name}")

    if results.registered:
        print(f"\nREGISTERED with Ollama ({len(results.registered)}):")
        for fname, oname in results.registered:
            print(f"  > {oname}")

    if results.ready:
        print(f"\nREADY ({len(results.ready)}):")
        # Lazy import — only needed when there's something to print, and silent on failure.
        _gguf_helpers = None
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import gguf_inspect as _gguf_helpers  # type: ignore
        except ImportError:
            pass

        lmstudio_dir = args.lmstudio_dir.expanduser().resolve()
        for name, size, mtype in results.ready:
            line = f"  . {name}  [{mtype}, {size}]"
            if _gguf_helpers and "GGUF" in mtype.upper():
                gguf_path = _find_primary_gguf_for_ready_entry(
                    name, models_dir, lmstudio_dir,
                )
                if gguf_path is not None:
                    try:
                        meta = _gguf_helpers.parse_gguf_metadata(gguf_path)
                        summary = _gguf_helpers.architecture_summary(meta)
                        total_weight = _sum_split_gguf_size(gguf_path)
                        est = _gguf_helpers.estimate_load_ram(total_weight, summary, [8192])
                        if est:
                            q8_gb = est[8192]["q8"] / (1024 ** 3)
                            line += f"   ≈ {q8_gb:.0f} GB to load @ 8K (Q8 KV)"
                    except Exception:
                        pass
            print(line)

    if results.duplicates:
        print(f"\nDUPLICATES removed ({len(results.duplicates)}):")
        for name, *_ in results.duplicates:
            print(f"  D {name}")

    if results.needs_download:
        print(f"\nHUGGINGFACE CACHE — needs download ({len(results.needs_download)}):")
        for name, mtype, hf_repo in results.needs_download:
            print(f"  ~ {name}  [{mtype}]")
        print(f"\n  To download: huggingface-cli download <repo> --local-dir <dest>")

    if results.broken:
        print(f"\nBROKEN ({len(results.broken)}):")
        for name, path, reason in results.broken:
            print(f"  X {name}: {reason}")
            print(f"      path: {path}")

        if args.interactive_broken:
            interactive_broken_actions(results.broken)

    if results.errors:
        print(f"\nERRORS ({len(results.errors)}):")
        for name, reason in results.errors:
            print(f"  ! {name}: {reason}")

    # Summary
    print(f"\n{'='*60}")
    parts = []
    if results.ready:
        parts.append(f"{len(results.ready)} ready")
    if results.fixed:
        parts.append(f"{len(results.fixed)} fixed")
    if results.registered:
        parts.append(f"{len(results.registered)} registered")
    if results.skipped_ollama:
        parts.append(f"{len(results.skipped_ollama)} already in Ollama")
    if results.duplicates:
        parts.append(f"{len(results.duplicates)} duplicates removed")
    if results.needs_download:
        parts.append(f"{len(results.needs_download)} in HF cache (need download)")
    if results.broken:
        parts.append(f"{len(results.broken)} broken")
    if results.errors:
        parts.append(f"{len(results.errors)} errors")
    print(" | ".join(parts) if parts else "No models found")

    if args.dry_run and (results.fixed or results.registered or results.duplicates):
        print("\nRe-run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
