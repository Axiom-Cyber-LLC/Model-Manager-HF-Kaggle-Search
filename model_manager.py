#!/usr/bin/env python3
"""
model_manager_v1.8_family_filter.py

Interactive model/dataset finder and downloader for Hugging Face and Kaggle.

After download / audit prompts, this program may subprocess optional helpers that live beside
model_manager.py (same directory) or under MODEL_MANAGER_TOOLS_DIR: Prepare_models_for_All.py,
model_conversion.py, model_audit.py, external scanners (modelaudit, modelscan, ModelGuard, …),
the Go hfdownloader binary, and model_manager_gui.swift (native GUI). These are never required
for search or Hub download; they run only when you opt in or pass flags that trigger them.

Core design:
  - Repos are search results.
  - Downloadable model artifacts inside a repo are discovered and selected separately.
  - Downloads are read-only/audit-only by default.
  - Destructive action is never automatic; deletion is offered only after DANGER/BLOCKER findings.

Optional dependencies:
  python3 -m pip install --upgrade huggingface_hub datasets kaggle kagglehub pandas pyarrow safetensors

Auth:
  HF_TOKEN or HUGGINGFACEHUB_API_TOKEN for gated/private Hugging Face repos.
  <REDACTED_PATH> or KAGGLE_USERNAME/KAGGLE_KEY for Kaggle.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

# Directory containing this file; used first when resolving Prepare_*, model_conversion, audit, scanners.
MODEL_MANAGER_SCRIPT_DIR = Path(__file__).expanduser().resolve().parent

MODEL_TOOLS_DIR = Path(os.getenv("MODEL_MANAGER_TOOLS_DIR", str(Path.home() / "model_tools"))).expanduser().resolve()

_DEFAULT_SCANNER_DIR_NAMES = (
    "modelaudit",
    "modelscan",
    "model-scan",
    "ModelGuard",
    "palisade-scan",
    "skillcheck",
    "skill-scanner",
)
_SECURITY_TOOL_ROOTS: list[Path] = []
for _root in (MODEL_MANAGER_SCRIPT_DIR, MODEL_TOOLS_DIR):
    _rp = _root.expanduser().resolve()
    if _rp not in _SECURITY_TOOL_ROOTS:
        _SECURITY_TOOL_ROOTS.append(_rp)

DEFAULT_LOCAL_MODEL_DIRS = [
    Path("<Your Model Directory>"),
    Path("<Your Model Directory>"),
    Path("<REDACTED_PATH>"),
    Path.home() / ".cache" / "huggingface",
    Path("<Your Model Directory>"),
]

DEFAULT_SECURITY_TOOLS = [
    *[root / name for root in _SECURITY_TOOL_ROOTS for name in _DEFAULT_SCANNER_DIR_NAMES],
    Path("<REDACTED_PATH>").expanduser(),
    Path("<REDACTED_PATH>").expanduser(),
    Path("<REDACTED_PATH>").expanduser(),
    Path("<REDACTED_PATH>").expanduser(),
    Path("<REDACTED_PATH>").expanduser(),
    Path("<REDACTED_PATH>").expanduser(),
]

PREP_SCRIPT_CANDIDATES = [
    MODEL_MANAGER_SCRIPT_DIR / "Prepare_models_for_All.py",
    Path.home() / "model_tools" / "Prepare_models_for_All.py",
    Path("<Your Model Directory>/Prepare_models_for_All.py"),
    Path("./Prepare_models_for_All.py"),
    Path.home() / "Prepare_models_for_All.py",
]

CONVERSION_SCRIPT_CANDIDATES = [
    MODEL_MANAGER_SCRIPT_DIR / "model_conversion.py",
    Path.home() / "model_tools" / "model_conversion.py",
    Path("<Your Model Directory>/model_conversion.py"),
    Path("./model_conversion.py"),
    Path.home() / "model_conversion.py",
]

PREP_APP_ORDER = [
    "Lmstudio",
    "Ollama",
    "AnythingLLM",
    "GPT4All",
    "Jan",
    "AIStudio",
    "LocallyAI",
    "LocalAI",
    "Apollo",
    "OffGrid",
]

# Root directory for all managed downloads. Override with MODEL_MANAGER_DOWNLOAD_DIR if needed.
# Expected layout under this root:
#   huggingface/model/<owner>__<repo>/      (Python fallback)
#   huggingface/model/<owner>/<repo>/       (hfdownloader local-dir mode)
#   huggingface/dataset/<owner>__<dataset>/
#   kaggle/model/<owner>__<model>/
#   kaggle/dataset/<owner>__<dataset>/
DEFAULT_DOWNLOAD_DIR = Path(os.getenv("MODEL_MANAGER_DOWNLOAD_DIR", "<Your Model Directory>")).expanduser().resolve()
# Default staging area for in-flight downloads. **Important:** keep this
# OUTSIDE DEFAULT_DOWNLOAD_DIR. When DEFAULT_DOWNLOAD_DIR is also LM Studio's
# downloadsFolder, an `.incoming/` child directory rapidly accumulates many
# small files (RLHF datasets are 18k+ JSON shards, etc.) and pushes the
# downloadsFolder past LM Studio's 7,000-file scanner cap, causing My Models
# to silently show 0 models. The default below puts staging as a sibling of
# the download root. Override with MODEL_MANAGER_INCOMING_DIR.
def _default_incoming_dir(download_dir: Path) -> Path:
    parent = download_dir.parent if download_dir.parent != download_dir else download_dir
    return parent / ".cache" / "model_manager_incoming"

INCOMING_DOWNLOAD_DIR = Path(
    os.getenv("MODEL_MANAGER_INCOMING_DIR", str(_default_incoming_dir(DEFAULT_DOWNLOAD_DIR)))
).expanduser().resolve()
CACHE_DIR = Path(os.getenv("MODEL_MANAGER_CACHE_DIR", str(Path.home() / ".cache" / "model_manager"))).expanduser().resolve()
LEADERBOARD_CACHE_PATH = CACHE_DIR / "leaderboard_cache.json"
REPUTATION_CACHE_PATH = CACHE_DIR / "publisher_reputation.json"
# Persisted in-flight download queue. Records get added before each download
# and cleared on successful completion. Survives crashes/restarts so the
# next modelmgr invocation can resume them without the user having to
# remember and re-search for the repos.
ACTIVE_DOWNLOADS_PATH = CACHE_DIR / "active_downloads.json"
# huggingface_hub.snapshot_download writes a refs/<revision> stub to cache_dir even when
# local_dir is set (the docstring claims cache_dir is unused but the code disagrees). Steer
# those stubs into a throwaway dir so they never pollute HF_HUB_CACHE.
HF_STUB_CACHE_DIR = CACHE_DIR / "hf_refs_stubs"
DEFAULT_HF_DOWNLOAD_MAX_WORKERS = 2
DEFAULT_HF_XET_RANGE_GETS = 16
DEFAULT_HFDOWNLOADER_CONNECTIONS = 4
DEFAULT_HFDOWNLOADER_MAX_ACTIVE = 1
DEFAULT_DOWNLOAD_RETRY_ATTEMPTS = 4
DEFAULT_DOWNLOAD_RETRY_BASE_DELAY_SECONDS = 3
HFDOWNLOADER_SAFE_FILTER_ARTIFACT_TYPES = {"gguf", "gguf-split", "safetensors", "safetensors-sharded"}
MULTIPART_ARTIFACT_TYPES = {"gguf-split", "safetensors-sharded"}
HFDOWNLOADER_MODEL_EXCLUDE_TERMS = [
    ".ggml",
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".mlmodel",
    ".mlpackage",
    ".h5",
    ".hdf5",
    ".keras",
]
DEFAULT_DEBUG_LOG_PATH = MODEL_TOOLS_DIR / "model_manager.debug.log"
DEBUG_ENABLED = os.getenv("MODEL_MANAGER_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
DEBUG_LOG_PATH = Path(os.getenv("MODEL_MANAGER_DEBUG_LOG", "")).expanduser().resolve() if os.getenv("MODEL_MANAGER_DEBUG_LOG") else DEFAULT_DEBUG_LOG_PATH
_OWNER_REPUTATION_CACHE: dict[str, set[str]] | None = None

# Local pre-download risk intelligence. The script searches these files first, then any
# likely risk workbook found in <REDACTED_PATH> Supported formats: .json, .csv, .tsv, .xlsx
# (.xlsx requires openpyxl).
RISK_INTEL_CANDIDATES = [
    MODEL_MANAGER_SCRIPT_DIR / "model_risk_intel.xlsx",
    MODEL_MANAGER_SCRIPT_DIR / "model_risk_intel.csv",
    MODEL_MANAGER_SCRIPT_DIR / "model_risk_intel.json",
    MODEL_TOOLS_DIR / "model_risk_intel.xlsx",
    MODEL_TOOLS_DIR / "model_risk_intel.csv",
    MODEL_TOOLS_DIR / "model_risk_intel.json",
    MODEL_MANAGER_SCRIPT_DIR / "malicious_ai_models.xlsx",
    MODEL_MANAGER_SCRIPT_DIR / "malicious_ai_models.csv",
    MODEL_TOOLS_DIR / "malicious_ai_models.xlsx",
    MODEL_TOOLS_DIR / "malicious_ai_models.csv",
    MODEL_MANAGER_SCRIPT_DIR / "ai_model_risk_workbook.xlsx",
    MODEL_TOOLS_DIR / "ai_model_risk_workbook.xlsx",
]

# Conservative defaults. Edit <REDACTED_PATH> for your own allow/warn/block lists.
DEFAULT_KNOWN_GOOD_OWNERS = {
    "qwen", "deepseek-ai", "microsoft", "codellama", "meta-llama", "mistralai",
    "google", "google-bert", "nomic-ai", "jinaai", "sentence-transformers",
    "unsloth", "bartowski", "lmstudio-community", "ggml-org", "mlx-community",
}
DEFAULT_WARN_NAME_TERMS = {
    "uncensored", "unsensored", "abliterated", "obliterated", "jailbreak",
    "backdoor", "backdoored", "poisoned", "eval-bypass", "bypass",
}
DEFAULT_EXCLUDED_AUTHORS: list[str] = []
DEFAULT_EXCLUDED_TERMS = ["uncensored", "nsfw", "roleplay", "erotic", "jailbreak"]
MODEL_NAME_QUERY_TOKEN_RE = re.compile(r"[A-Za-z]{2,}[A-Za-z0-9._-]*\d[A-Za-z0-9._-]*")
MAX_MODEL_NAME_TOKEN_VARIANTS = 12
MODEL_MANAGER_VERSION = "v1.8.1-staged-cart-scan-fix"
MODEL_FILE_EXTENSIONS = {".gguf", ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx", ".mlmodel", ".mlpackage"}
DATA_SAMPLE_EXTENSIONS = {".csv", ".json", ".jsonl", ".parquet", ".txt", ".tsv"}
LFS_SIGNATURE = b"version https://git-lfs.github.com/spec/v1"
GGUF_MAGIC = b"GGUF"

HF_DATASET_IGNORE_MODEL_WEIGHTS = [
    "*.safetensors", "*.bin", "*.gguf", "*.ggml", "*.ggjt",
    "*.pt", "*.pth", "*.ckpt", "*.h5", "*.hdf5",
    "*.onnx", "*.tflite", "*.mlmodel", "*.mlpackage",
    "*.msgpack", "*.npz", "*.pkl", "*.joblib",
    "pytorch_model*", "tf_model*", "flax_model*",
    "model.safetensors*", "consolidated.*", "adapter_model.*", "*.lora",
]

COMMON_MODEL_COMPANIONS = [
    "README*", "LICENSE*", "NOTICE*", "*.md",
    "config.json", "generation_config.json", "tokenizer*", "vocab.*", "merges.txt",
    "special_tokens_map.json", "chat_template.jinja", "preprocessor_config.json",
    "processor_config.json", "added_tokens.json", "tokenization*.py", "modeling*.py",
]


ARTIFACT_TYPE_OPTIONS = {
    "gguf": {
        "label": "GGUF",
        "patterns": ["*.gguf"],
        "search_terms": [".gguf", "GGUF"],
        "runtime": "LM Studio, Ollama, llama.cpp, Jan, AnythingLLM, GPT4All, LocalAI, KoboldCpp, Open WebUI",
    },
    "coreml": {
        "label": "Core ML (.mlmodel/.mlpackage)",
        "patterns": ["*.mlmodel", "*.mlpackage", "*.mlpackage/*"],
        "search_terms": [".mlmodel", ".mlpackage", "Core ML"],
        "runtime": "Apple Core ML, macOS/iOS apps, Xcode, coremltools",
    },
    "mlx": {
        "label": "MLX / Apple Silicon",
        "patterns": ["*.safetensors", "*.npz", "*.mlx", "config.json"],
        "search_terms": ["MLX"],
        "runtime": "MLX, mlx-lm, Apple Silicon workflows",
    },
    "onnx": {
        "label": "ONNX",
        "patterns": ["*.onnx"],
        "search_terms": [".onnx", "ONNX"],
        "runtime": "ONNX Runtime, OpenVINO, TensorRT conversion workflows",
    },
    "safetensors": {
        "label": "Safetensors / HF Transformers",
        "patterns": ["*.safetensors", "*.safetensors.index.json"],
        "search_terms": [".safetensors", "safetensors"],
        "runtime": "Transformers, vLLM, TGI, Axolotl, PEFT/LoRA tools",
    },
    "keras": {
        "label": "Keras/TensorFlow (.h5/.hdf5/.keras)",
        "patterns": ["*.h5", "*.hdf5", "*.keras", "*.tflite"],
        "search_terms": [".h5", ".keras", "TensorFlow"],
        "runtime": "Keras, TensorFlow, TFLite",
    },
    "pytorch": {
        "label": "Raw PyTorch / pickle-risk formats",
        "patterns": ["*.bin", "*.pt", "*.pth", "*.ckpt"],
        "search_terms": ["PyTorch"],
        "runtime": "PyTorch research workflows; higher deserialization risk",
    },
}
DEFAULT_ARTIFACT_TYPE_KEYS = ["gguf", "coreml"]

LEADERBOARD_SOURCES = {
    "general": [
        ("Open LLM Leaderboard", "https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard"),
        ("OpenEvals every-leaderboards", "https://huggingface.co/spaces/OpenEvals/every-leaderboards"),
        ("LMArena Leaderboard", "https://huggingface.co/spaces/lmarena-ai/arena-leaderboard"),
    ],
    "coding": [
        ("OpenEvals every-leaderboards", "https://huggingface.co/spaces/OpenEvals/every-leaderboards"),
        ("BigCodeBench", "https://huggingface.co/spaces/bigcode/bigcodebench-leaderboard"),
        ("LMArena Leaderboard", "https://huggingface.co/spaces/lmarena-ai/arena-leaderboard"),
    ],
    "visual": [
        ("Open VLM Leaderboard", "https://huggingface.co/spaces/opencompass/open_vlm_leaderboard"),
        ("MMEB Leaderboard", "https://huggingface.co/spaces/TIGER-Lab/MMEB-Leaderboard"),
        ("OpenEvals every-leaderboards", "https://huggingface.co/spaces/OpenEvals/every-leaderboards"),
    ],
    "security": [
        ("OpenEvals every-leaderboards", "https://huggingface.co/spaces/OpenEvals/every-leaderboards"),
        ("HF Spaces search: security leaderboard", "https://huggingface.co/spaces?search=security%20leaderboard"),
        ("HF Datasets search: cybersecurity benchmark", "https://huggingface.co/datasets?search=cybersecurity%20benchmark"),
    ],
    "embedding": [
        ("MTEB Leaderboard", "https://huggingface.co/spaces/mteb/leaderboard"),
        ("MMEB Leaderboard", "https://huggingface.co/spaces/TIGER-Lab/MMEB-Leaderboard"),
    ],
}

# -----------------------------------------------------------------------------
# Data objects
# -----------------------------------------------------------------------------

@dataclass
class SearchResult:
    index: int
    source: str              # hf or kaggle
    kind: str                # model or dataset
    repo_id: str
    title: str = ""
    size_bytes: int | None = None
    downloads: int | None = None
    likes: int | None = None
    pipeline: str | None = None
    updated: str | None = None
    license: str | None = None
    url: str | None = None
    files: list[tuple[str, int | None]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    raw: Any = None
    leaderboard_rank: int | None = None
    leaderboard_label: str | None = None
    selected_artifacts: list[Artifact] = field(default_factory=list)
    visible_artifacts: list[Artifact] = field(default_factory=list)
    recommendation: str = "-"
    duplicate_family: str | None = None
    duplicate_group_size: int = 1
    whole_repo_selected: bool = False
    # True when this result came from an exact repo-ID lookup (e.g. the user
    # typed "owner/repo" as a search term). When set, artifact-type / size /
    # multipart filters skip the result — the user asked for THIS repo, so
    # don't silently drop it because it doesn't have a .gguf or because it's
    # smaller/larger than the chosen size range.
    direct_lookup: bool = False


@dataclass
class Artifact:
    index: int
    label: str
    artifact_type: str       # gguf, gguf-split, safetensors, safetensors-sharded, adapter, raw-weight, whole-repo
    quant: str | None
    files: list[tuple[str, int | None]]
    companion_patterns: list[str] = field(default_factory=lambda: list(COMMON_MODEL_COMPANIONS))
    notes: list[str] = field(default_factory=list)

    @property
    def total_size(self) -> int:
        return sum(size or 0 for _, size in self.files)

    def allow_patterns(self) -> list[str]:
        names = [name for name, _ in self.files]
        return names + self.companion_patterns


@dataclass
class ScannerCommand:
    label: str
    cmd: list[str]
    cwd: Path | None = None

# -----------------------------------------------------------------------------
# Prompting/formatting
# -----------------------------------------------------------------------------

def human_size(num_bytes: int | None) -> str:
    if not num_bytes:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.2f} PB"


def size_range_label(size_range: tuple[int | None, int | None]) -> str:
    lo, hi = size_range
    if lo is None and hi is None:
        return "any"
    if lo is None:
        return f"≤ {human_size(hi)}"
    if hi is None:
        return f"≥ {human_size(lo)}"
    return f"{human_size(lo)} - {human_size(hi)}"


SIZE_UNITS = {
    "b": 1,
    "byte": 1,
    "bytes": 1,
    "kb": 1024,
    "kib": 1024,
    "mb": 1024**2,
    "mib": 1024**2,
    "gb": 1024**3,
    "gib": 1024**3,
    "tb": 1024**4,
    "tib": 1024**4,
    "pb": 1024**5,
    "pib": 1024**5,
}


def parse_size_bytes(value: str) -> int | None:
    raw = value.strip().lower()
    if not raw:
        return None
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([a-z]+)", raw)
    if not m:
        return None
    unit = m.group(2)
    multiplier = SIZE_UNITS.get(unit)
    if multiplier is None:
        return None
    return int(float(m.group(1)) * multiplier)


def parse_size_range(value: str) -> tuple[int | None, int | None] | None:
    raw = value.strip().lower()
    if raw in {"", "any", "all", "none", "no filter", "nofilter", "-"}:
        return (None, None)

    # Single-sided upper bound: <X, <=X, under X, up to X, at most X
    upper = re.fullmatch(r"(?:<=?|under\s+|up\s+to\s+|at\s+most\s+)\s*(.+)", raw)
    if upper:
        hi = parse_size_bytes(upper.group(1))
        return (None, hi) if hi is not None else None

    # Single-sided lower bound: >X, >=X, over X, above X, at least X, min X
    lower = re.fullmatch(r"(?:>=?|over\s+|above\s+|at\s+least\s+|min(?:imum)?\s+)\s*(.+)", raw)
    if lower:
        lo = parse_size_bytes(lower.group(1))
        return (lo, None) if lo is not None else None

    # Two-sided range
    parts = [p.strip() for p in re.split(r"\s+(?:to|through)\s+|\s*-\s*", raw) if p.strip()]
    if len(parts) != 2:
        return None
    lo = parse_size_bytes(parts[0])
    hi = parse_size_bytes(parts[1])
    if lo is None or hi is None:
        return None
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def prompt_model_size_range() -> tuple[int | None, int | None]:
    print()
    print("What size models do you want to see?")
    print("Examples: 200 MB - 2 TB | 4 GB - 80 GB | <8 GB | >50 GB | any")
    default = "200 MB - 2 TB"
    while True:
        value = prompt("Model size range", default)
        parsed = parse_size_range(value)
        if parsed is not None:
            print(f"Model size filter: {size_range_label(parsed)}")
            return parsed
        print("Invalid size range. Include units, for example: 200 MB - 2 TB or <8 GB")


def parse_model_size_range_or_default(value: str | None) -> tuple[int | None, int | None] | None:
    if value is None:
        return None
    parsed = parse_size_range(value)
    if parsed is None:
        print(f"Invalid model size range from arguments: {value!r}. Using prompt default.")
        return None
    lo, hi = parsed
    if lo is None and hi is None:
        print("Model size filter: any")
    else:
        print(f"Model size filter: {human_size(lo)} - {human_size(hi)}")
    return parsed


def multipart_artifact_filter_relevant(selected_types: set[str]) -> bool:
    return not selected_types or bool(selected_types & {"gguf", "safetensors"})


def prompt_exclude_multipart_models(selected_types: set[str]) -> bool:
    print()
    families: list[str] = []
    if not selected_types or "gguf" in selected_types:
        families.append("GGUF split")
    if not selected_types or "safetensors" in selected_types:
        families.append("safetensors shard groups")
    if families:
        print("This hides split/sharded artifacts such as " + " and ".join(families) + ".")
    return prompt_bool("Do you want to exclude multi-part models?", False)


def size_in_range(size_bytes: int | None, size_range: tuple[int | None, int | None]) -> bool:
    lo, hi = size_range
    if lo is None and hi is None:
        return True
    if size_bytes is None:
        return False
    if lo is not None and size_bytes < lo:
        return False
    if hi is not None and size_bytes > hi:
        return False
    return True


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int, minimum: int = 0) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return max(minimum, int(value.strip()))
    except (TypeError, ValueError):
        return default


def env_int_if_set(name: str, minimum: int = 0) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    try:
        return max(minimum, int(value.strip()))
    except (TypeError, ValueError):
        return None


def download_retry_attempts() -> int:
    return env_int("MODEL_MANAGER_DOWNLOAD_RETRIES", DEFAULT_DOWNLOAD_RETRY_ATTEMPTS, minimum=1)


def download_retry_base_delay_seconds() -> int:
    return env_int("MODEL_MANAGER_DOWNLOAD_RETRY_DELAY", DEFAULT_DOWNLOAD_RETRY_BASE_DELAY_SECONDS, minimum=1)


def hf_xet_available() -> bool:
    return importlib.util.find_spec("hf_xet") is not None or shutil.which("hf-xet") is not None


def selected_download_metrics(
    result: SearchResult,
    artifacts: list[Artifact] | None = None,
) -> dict[str, int | None]:
    unique_files: dict[str, int | None] = {}
    source_files = result.files
    if artifacts:
        source_files = [item for art in artifacts for item in art.files]
    for name, size in source_files or []:
        unique_files[name] = size
    total_size = sum(size or 0 for size in unique_files.values()) or result.size_bytes or 0
    return {
        "file_count": len(unique_files) or None,
        "total_size": total_size or None,
    }


def recommend_hf_worker_count(
    result: SearchResult,
    artifacts: list[Artifact] | None = None,
    safe_mode: bool = False,
) -> int:
    explicit = env_int_if_set("MODEL_MANAGER_HF_MAX_WORKERS", minimum=1)
    if explicit is not None:
        return explicit
    if safe_mode:
        return 1

    metrics = selected_download_metrics(result, artifacts)
    total_size = metrics["total_size"] or 0
    file_count = metrics["file_count"] or 0
    recommended = DEFAULT_HF_DOWNLOAD_MAX_WORKERS
    if total_size >= 120 * 1024**3 or file_count >= 48:
        recommended = 16
    elif total_size >= 60 * 1024**3 or file_count >= 24:
        recommended = 12
    elif total_size >= 20 * 1024**3 or file_count >= 8:
        recommended = 8
    elif total_size >= 8 * 1024**3 or file_count >= 3:
        recommended = 4
    return min(max(1, recommended), 32)


def recommend_hf_xet_range_gets(worker_count: int, safe_mode: bool = False) -> int:
    explicit = env_int_if_set("MODEL_MANAGER_HF_XET_RANGE_GETS", minimum=1)
    if explicit is not None:
        return explicit
    if safe_mode:
        return 2
    return min(max(DEFAULT_HF_XET_RANGE_GETS, worker_count * 2), 64)


def recommend_hfdownloader_concurrency(
    result: SearchResult,
    artifacts: list[Artifact] | None = None,
) -> tuple[int, int]:
    explicit_connections = env_int_if_set("MODEL_MANAGER_HFD_CONNECTIONS", minimum=1)
    explicit_active = env_int_if_set("MODEL_MANAGER_HFD_MAX_ACTIVE", minimum=1)
    if explicit_connections is not None and explicit_active is not None:
        return explicit_connections, explicit_active

    metrics = selected_download_metrics(result, artifacts)
    total_size = metrics["total_size"] or 0
    file_count = metrics["file_count"] or 0

    if total_size >= 120 * 1024**3:
        recommended_connections = 16
    elif total_size >= 60 * 1024**3:
        recommended_connections = 12
    elif total_size >= 20 * 1024**3:
        recommended_connections = 8
    else:
        recommended_connections = DEFAULT_HFDOWNLOADER_CONNECTIONS

    if file_count >= 24:
        recommended_active = 6
    elif file_count >= 8:
        recommended_active = 4
    elif file_count >= 3:
        recommended_active = 2
    else:
        recommended_active = DEFAULT_HFDOWNLOADER_MAX_ACTIVE
    if file_count <= 1:
        recommended_active = 1

    connections = explicit_connections if explicit_connections is not None else recommended_connections
    max_active = explicit_active if explicit_active is not None else recommended_active
    return min(max(1, connections), 32), min(max(1, max_active), 16)


def configure_hf_download_environment(
    worker_count: int | None = None,
    range_gets: int | None = None,
) -> dict[str, int | bool | str]:
    """Configure Hugging Face transfers before `huggingface_hub` is imported.

    Default mode uses Hugging Face's fast Xet transfer path. Set
    MODEL_MANAGER_HF_TRANSFER_MODE=safe to lower concurrency for fragile runs.
    """
    mode = os.getenv("MODEL_MANAGER_HF_TRANSFER_MODE", "fast").strip().lower()
    safe_mode = mode in {"safe", "low-memory", "low_memory", "conservative"}
    default_workers = 1 if safe_mode else DEFAULT_HF_DOWNLOAD_MAX_WORKERS
    default_range_gets = 2 if safe_mode else DEFAULT_HF_XET_RANGE_GETS
    max_workers = worker_count or env_int("MODEL_MANAGER_HF_MAX_WORKERS", default_workers, minimum=1)
    range_gets = range_gets or env_int("MODEL_MANAGER_HF_XET_RANGE_GETS", default_range_gets, minimum=1)

    os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
    os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = str(range_gets)
    if safe_mode:
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "0"
        os.environ["HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY"] = "1"
    else:
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
        os.environ["HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY"] = "0"

    return {
        "mode": "safe" if safe_mode else "fast",
        "max_workers": max_workers,
        "high_performance": not safe_mode,
        "range_gets": range_gets,
        "hf_xet_available": hf_xet_available(),
    }


def hfdownloader_enabled() -> bool:
    # Default-on: route model downloads through the Go hfdownloader binary
    # for multipart/chunked-parallel speed. Datasets, artifact selections that
    # hfdownloader can't preserve cleanly, and any download error all fall
    # back to huggingface_hub.snapshot_download automatically. To force the
    # Python path for one run: MODEL_MANAGER_HFDOWNLOADER=0 modelmgr ...
    value = os.getenv("MODEL_MANAGER_HFDOWNLOADER", "1").strip().lower()
    return value not in {"0", "false", "no", "off", "python"}


def find_hfdownloader_binary() -> Path | None:
    configured = os.getenv("MODEL_MANAGER_HFDOWNLOADER_BIN", "").strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            Path(__file__).expanduser().resolve().parent / "bin" / "hfdownloader",
            MODEL_TOOLS_DIR / "bin" / "hfdownloader",
        ]
    )
    found = shutil.which("hfdownloader")
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        try:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
        except OSError:
            continue
    return None


def hfdownloader_filter_terms_from_artifacts(artifacts: list[Artifact] | None) -> list[str]:
    """Build the `--filters` value for hfdownloader from selected artifacts.

    hfdownloader's `--filters` flag expects short *quant tag tokens* that
    appear inside LFS filenames (e.g. `q4_0`, `q5_K_M`, `iq2_xxs`,
    `bf16`). Its matcher fails to recognize a full filename like
    `Foo-Bar.i1-Q2_K.gguf` and silently falls back to downloading the
    whole repo — that's how a 77 GB artifact selection turned into a
    1 TB pull.

    Strategy:
      1. If the artifact has a recognized quant tag, use that (preferred).
      2. Otherwise, fall back to file BASENAME WITHOUT EXTENSION — which
         in practice still contains the quant tag and matches hfdownloader's
         filename-substring logic, while avoiding the `.gguf` suffix and
         path separators that confuse it.
    Returns deduped, lowercase tokens.
    """
    if not artifacts:
        return []
    terms: list[str] = []
    seen: set[str] = set()

    def _add(token: str) -> None:
        tok = token.strip().lower()
        if tok and tok not in seen:
            seen.add(tok)
            terms.append(tok)

    for art in artifacts:
        if art.quant and art.quant != "-":
            _add(art.quant)
            continue
        for name, _ in art.files:
            base = Path(name).name.strip()
            if not base:
                continue
            # Strip the extension so the matcher sees the meaningful tail
            # (which usually contains the quant code).
            stem = base.rsplit(".", 1)[0] if "." in base else base
            _add(stem)
    return terms


def hfdownloader_can_preserve_artifact_selection(artifacts: list[Artifact] | None) -> bool:
    if not artifacts:
        return True
    return all(art.artifact_type in HFDOWNLOADER_SAFE_FILTER_ARTIFACT_TYPES for art in artifacts)


def hfdownloader_exclude_terms_for_artifacts(artifacts: list[Artifact] | None) -> list[str]:
    if not artifacts:
        return []
    selected_exts = {Path(name).suffix.lower() for art in artifacts for name, _ in art.files}
    return [term for term in HFDOWNLOADER_MODEL_EXCLUDE_TERMS if term not in selected_exts]


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _debug_serialize(value: Any) -> str:
    try:
        return json.dumps(value, default=str, ensure_ascii=True)
    except Exception:
        return repr(value)


def configure_debug_mode(enabled: bool, log_path: str | None = None) -> None:
    global DEBUG_ENABLED, DEBUG_LOG_PATH
    DEBUG_ENABLED = bool(enabled)
    DEBUG_LOG_PATH = Path(log_path).expanduser().resolve() if log_path else DEFAULT_DEBUG_LOG_PATH


def debug_log(event: str, **fields: Any) -> None:
    payload = " ".join(f"{key}={_debug_serialize(value)}" for key, value in fields.items())
    line = f"[DEBUG] {event}" + (f" {payload}" if payload else "")
    if DEBUG_ENABLED:
        print(line, file=sys.stderr)
    if DEBUG_LOG_PATH is not None:
        try:
            DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with DEBUG_LOG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass


class InputAborted(RuntimeError):
    """Raised when interactive input is unavailable or intentionally cancelled."""


def read_input(label: str) -> str:
    try:
        value = input(label)
        debug_log("input", prompt=label, value=value)
        return value
    except (EOFError, KeyboardInterrupt) as exc:
        debug_log("input-aborted", prompt=label, exc_type=type(exc).__name__)
        raise InputAborted from exc


def prompt(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    value = read_input(f"{label}{suffix}: ").strip()
    return value if value else (default or "")


def prompt_int(label: str, default: int) -> int:
    value = prompt(label, str(default))
    try:
        return max(1, int(value))
    except ValueError:
        print(f"Invalid number. Using {default}.")
        return default


def prompt_int_range(label: str, default: int, minimum: int, maximum: int) -> int:
    value = prompt(label, str(default))
    try:
        parsed = int(value)
    except ValueError:
        print(f"Invalid number. Using {default}.")
        return default
    if parsed < minimum:
        print(f"Using minimum {minimum}.")
        return minimum
    if parsed > maximum:
        print(f"Using maximum {maximum}.")
        return maximum
    return parsed


def prompt_bool(label: str, default: bool = False) -> bool:
    d = "y" if default else "n"
    value = prompt(f"{label} [y/n]", d).lower()
    return value in {"y", "yes", "true", "1"}


def prompt_choice(label: str, choices: list[str], default: str) -> str:
    print(label)
    for i, c in enumerate(choices, 1):
        tag = " default" if c == default else ""
        print(f"  {i}. {c}{tag}")
    value = prompt(f"Choose 1-{len(choices)}", default).strip()
    if value in choices:
        return value
    try:
        idx = int(value)
        if 1 <= idx <= len(choices):
            return choices[idx - 1]
    except ValueError:
        pass
    print(f"Invalid choice. Using {default}.")
    return default


def parse_selection(selection: str, max_number: int) -> list[int]:
    selection = selection.strip().lower()
    if not selection:
        return []
    if selection == "all":
        return list(range(1, max_number + 1))
    out: set[int] = set()
    for part in selection.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                start, end = int(a), int(b)
            except ValueError:
                continue
            if start > end:
                start, end = end, start
            for n in range(start, end + 1):
                if 1 <= n <= max_number:
                    out.add(n)
        else:
            try:
                n = int(part)
            except ValueError:
                continue
            if 1 <= n <= max_number:
                out.add(n)
    return sorted(out)


def safe_folder_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", name).strip("_") or "download"


def alpha_label(index: int) -> str:
    """1 -> A, 26 -> Z, 27 -> AA."""
    if index < 1:
        return "?"
    chars: list[str] = []
    n = index
    while n:
        n, rem = divmod(n - 1, 26)
        chars.append(chr(ord("A") + rem))
    return "".join(reversed(chars))


def alpha_to_number(label: str) -> int | None:
    label = label.strip().upper()
    if not label or not label.isalpha():
        return None
    n = 0
    for ch in label:
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n


def parse_repo_artifact_tokens(selection: str, results_count: int) -> list[tuple[int, int]]:
    """Parse tokens like 1A, 2C, 10AA into (repo_number, artifact_number)."""
    out: list[tuple[int, int]] = []
    for token in re.split(r"[ ,]+", selection.strip().upper()):
        if not token:
            continue
        m = re.fullmatch(r"(\d+)([A-Z]+)", token)
        if not m:
            continue
        repo_num = int(m.group(1))
        art_num = alpha_to_number(m.group(2))
        if art_num is None:
            continue
        if 1 <= repo_num <= results_count:
            out.append((repo_num, art_num))
    return out


def parse_artifact_page_command(selection: str) -> tuple[str, int] | None:
    """Parse commands like n2, p4, next 2, prev 4 for per-repo artifact paging."""
    s = selection.strip().lower()
    if not s:
        return None
    compact = re.fullmatch(r"(n|p)(\d+)", s)
    if compact:
        direction = "next" if compact.group(1) == "n" else "prev"
        return direction, int(compact.group(2))
    spaced = re.fullmatch(r"(n|next|p|prev|previous)\s+(\d+)", s)
    if spaced:
        token = spaced.group(1)
        direction = "next" if token in {"n", "next"} else "prev"
        return direction, int(spaced.group(2))
    return None


# -----------------------------------------------------------------------------
# Query parsing and result merging
# -----------------------------------------------------------------------------

def split_boolean_query(query: str) -> list[str]:
    """Split simple Boolean OR searches into separate site-compatible queries.

    Supports:
      code OR cybersecurity OR "Threat Detection"
      code or cybersecurity or “Threat Detection”
      code, cybersecurity, Threat Detection
      code | cybersecurity | Threat Detection
      code ; cybersecurity ; Threat Detection

    This intentionally treats only OR / comma / | / ; as split operators. AND/NOT are
    not applied because HF/Kaggle search APIs do not consistently implement
    Boolean semantics.
    """
    q = query.strip().replace("“", '"').replace("”", '"').replace("’", "'")
    if not q:
        return []

    terms: list[str] = []
    current: list[str] = []
    in_quote = False
    i = 0

    while i < len(q):
        ch = q[i]
        if ch == '"':
            in_quote = not in_quote
            i += 1
            continue

        if not in_quote:
            # Split on standalone OR, case-insensitive. This avoids matching
            # words like "forensics" or "organization".
            if q[i:i + 2].lower() == "or":
                before = q[i - 1] if i > 0 else " "
                after = q[i + 2] if i + 2 < len(q) else " "
                if not before.isalnum() and not after.isalnum():
                    term = "".join(current).strip()
                    if term:
                        terms.append(term)
                    current = []
                    i += 2
                    continue

            if ch in {"|", ";", ","}:
                term = "".join(current).strip()
                if term:
                    terms.append(term)
                current = []
                i += 1
                continue

        current.append(ch)
        i += 1

    term = "".join(current).strip()
    if term:
        terms.append(term)

    seen: set[str] = set()
    out: list[str] = []
    for term in terms or [q]:
        cleaned = term.strip().strip('"').strip()
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            out.append(cleaned)
    return out


def query_variant_dedupe_key(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def split_model_name_query_token(token: str) -> tuple[list[str], list[str], list[str]] | None:
    raw = token.strip().strip('"\'')
    if not raw or not MODEL_NAME_QUERY_TOKEN_RE.fullmatch(raw):
        return None
    first_digit = next((i for i, ch in enumerate(raw) if ch.isdigit()), -1)
    if first_digit <= 0:
        return None

    prefix_words = re.findall(r"[A-Za-z]+", raw[:first_digit])
    tail_parts = re.findall(r"[A-Za-z]+|\d+", raw[first_digit:])
    if not prefix_words or not tail_parts:
        return None

    version_parts: list[str] = []
    suffix_parts: list[str] = []
    in_suffix = False
    for part in tail_parts:
        if not in_suffix and part.isdigit():
            version_parts.append(part)
            continue
        in_suffix = True
        suffix_parts.append(part)
    if not version_parts:
        return None
    return prefix_words, version_parts, suffix_parts


def model_name_token_variants(token: str) -> list[str]:
    parsed = split_model_name_query_token(token)
    if not parsed:
        return [token]

    prefix_words, version_parts, suffix_parts = parsed
    if len(prefix_words) == 1:
        prefix_variants = ["".join(prefix_words)]
    else:
        prefix_variants = ["-".join(prefix_words), "".join(prefix_words), " ".join(prefix_words)]

    if len(version_parts) == 1:
        version_variants = [version_parts[0]]
    else:
        version_variants = [".".join(version_parts), "-".join(version_parts), " ".join(version_parts)]

    tail_variants: list[str] = []
    if suffix_parts:
        suffix_hyphen = "-".join(suffix_parts)
        suffix_space = " ".join(suffix_parts)
        suffix_compact = "".join(suffix_parts)
        for version in version_variants:
            tail_variants.append(f"{version}-{suffix_hyphen}")
            tail_variants.append(f"{version} {suffix_space}")
            if len(suffix_parts) == 1:
                tail_variants.append(f"{version}{suffix_compact}")
    else:
        tail_variants.extend(version_variants)

    candidates: list[str] = [token]
    for prefix in prefix_variants:
        for tail in tail_variants:
            for joiner in ("", "-", " "):
                candidates.append(f"{prefix}{joiner}{tail}".strip())

    seen: set[str] = set()
    out: list[str] = []
    for item in candidates:
        key = query_variant_dedupe_key(item)
        if key and key not in seen:
            seen.add(key)
            out.append(item)
        if len(out) >= MAX_MODEL_NAME_TOKEN_VARIANTS:
            break
    return out or [token]


def expand_search_terms_for_model_name_variants(search_terms: list[str]) -> list[str]:
    """Expand mixed letter/number model names across common separator variants.

    Examples:
      GLM5.1 -> GLM-5.1, GLM5-1, GLM-5-1
      Qwen3.6 -> Qwen-3.6, Qwen3-6, Qwen-3-6
    """
    expanded: list[str] = []
    for term in search_terms:
        base = term.strip()
        if not base:
            continue
        expanded.append(base)
        tokens = base.split()
        for i, token in enumerate(tokens):
            for variant in model_name_token_variants(token)[1:]:
                replaced = tokens.copy()
                replaced[i] = variant
                expanded.append(" ".join(replaced))

    seen: set[str] = set()
    out: list[str] = []
    for term in expanded:
        key = query_variant_dedupe_key(term)
        if key not in seen:
            seen.add(key)
            out.append(term)
    return out


def parse_not_publishers_from_query(query: str) -> tuple[str, set[str]]:
    """Extract publisher/owner exclusions from a user search query.

    Supported examples:
      code NOT bartowski
      code NOT owner:bartowski
      code NOT --owner:bartowski
      code -owner:bartowski
      code --owner:bartowski
      code -publisher:qwen -bartowski

    The removed publisher terms are applied locally after provider search because
    Hugging Face/Kaggle search APIs do not consistently implement Boolean NOT.
    """
    q = query.strip().replace("“", '"').replace("”", '"').replace("’", "'")
    excluded: set[str] = set()

    def clean_owner(v: str) -> str:
        v = v.strip().strip('\"\'').lower()
        v = re.sub(r"^[-\s]+", "", v)
        v = re.sub(r"^@", "", v)
        v = re.sub(r"^(publisher:|owner:)", "", v)
        v = re.sub(r"^[-]+", "", v)
        return v.strip()

    def add_owner(raw: str) -> None:
        val = clean_owner(raw)
        if val and not re.fullmatch(r"\d+(\.\d+)?", val):
            excluded.add(val.split("/", 1)[0])

    def repl_not(m: re.Match) -> str:
        add_owner(m.group(1))
        return " "

    # NOT owner filters, including NOT --owner:foo and NOT owner:foo.
    q = re.sub(
        r"(?i)\bNOT\s+(?:--?)?(?:publisher:|owner:)?([A-Za-z][A-Za-z0-9_.-]*(?:/[A-Za-z0-9_.-]+)?)",
        repl_not,
        q,
    )

    def repl_dash_keyed(m: re.Match) -> str:
        add_owner(m.group(1))
        return " "

    # Explicit negative filters: -owner:foo, --owner:foo, -publisher:foo.
    q = re.sub(
        r"(?<!\S)--?(?:publisher:|owner:)([A-Za-z][A-Za-z0-9_.-]*(?:/[A-Za-z0-9_.-]+)?)",
        repl_dash_keyed,
        q,
        flags=re.IGNORECASE,
    )

    def repl_dash_bare(m: re.Match) -> str:
        add_owner(m.group(1))
        return " "

    # Shorthand negative owner filters: -qwen, -bartowski.
    q = re.sub(
        r"(?<!\S)-([A-Za-z][A-Za-z0-9_.-]*(?:/[A-Za-z0-9_.-]+)?)",
        repl_dash_bare,
        q,
    )

    q = re.sub(r"\s+", " ", q).strip()
    return q, excluded

def prompt_excluded_publishers(initial: set[str] | None = None) -> set[str]:
    default_items = list(DEFAULT_EXCLUDED_AUTHORS)
    for item in sorted(initial or set()):
        if item.lower() not in {x.lower() for x in default_items}:
            default_items.append(item)
    default = ", ".join(default_items)
    print()
    print("Author/publisher NOT filter")
    print(f"Default: {default}")
    value = prompt("Authors/publishers to exclude from results", default).strip()
    if not value or value.lower() in {"none", "no", "n", "-"}:
        return set()
    out: set[str] = set()
    for part in re.split(r"[,\s]+", value):
        part = part.strip().strip('"\'').lower()
        if not part or part in {"none", "no", "n"}:
            continue
        part = re.sub(r"^(-|@)", "", part)
        part = re.sub(r"^(publisher:|owner:)", "", part)
        out.add(part.split("/", 1)[0])
    return out


def parse_excluded_publishers_value(value: str) -> set[str]:
    if not value or value.strip().lower() in {"none", "no", "n", "-"}:
        return set()
    out: set[str] = set()
    for part in re.split(r"[,\s]+", value):
        part = part.strip().strip('"\'').lower()
        if not part or part in {"none", "no", "n"}:
            continue
        part = re.sub(r"^(-|@)", "", part)
        part = re.sub(r"^(publisher:|owner:)", "", part)
        out.add(part.split("/", 1)[0])
    return out


def prompt_excluded_terms() -> set[str]:
    default = ", ".join(DEFAULT_EXCLUDED_TERMS)
    print()
    print("Terms/tags NOT filter")
    print(f"Default: {default}")
    value = prompt("Terms/tags to exclude from results", default).strip()
    if not value or value.lower() in {"none", "no", "n", "-"}:
        return set()
    terms: set[str] = set()
    for part in re.split(r"[,;]+", value):
        term = normalize_match_text(part)
        if term and term not in {"none", "no", "n"}:
            terms.add(term)
    return terms


def parse_excluded_terms_value(value: str) -> set[str]:
    if not value or value.strip().lower() in {"none", "no", "n", "-"}:
        return set()
    terms: set[str] = set()
    for part in re.split(r"[,;]+", value):
        term = normalize_match_text(part)
        if term and term not in {"none", "no", "n"}:
            terms.add(term)
    return terms


def apply_publisher_exclusions(results: list[SearchResult], excluded: set[str]) -> list[SearchResult]:
    if not excluded:
        return results
    def is_excluded(r: SearchResult) -> bool:
        owner = owner_of(r.repo_id)
        repo_l = r.repo_id.lower()
        return any(owner == x or repo_l.startswith(x + "/") or x in owner for x in excluded)
    filtered = [r for r in results if not is_excluded(r)]
    removed = len(results) - len(filtered)
    if removed:
        print(f"Applied NOT publisher/owner filter: {', '.join(sorted(excluded))} ({removed} result(s) hidden)")
    return filtered


def result_term_text(result: SearchResult) -> str:
    parts: list[str] = [
        result.repo_id,
        result.title or "",
        result.pipeline or "",
        result.license or "",
        " ".join(result.notes),
    ]
    raw = result.raw
    for attr in ("tags", "cardData", "card_data"):
        value = getattr(raw, attr, None)
        if isinstance(value, dict):
            parts.extend(str(v) for v in value.values())
        elif isinstance(value, (list, tuple, set)):
            parts.extend(str(v) for v in value)
        elif value:
            parts.append(str(value))
    return normalize_match_text(" ".join(parts))


def apply_term_exclusions(results: list[SearchResult], excluded: set[str]) -> list[SearchResult]:
    if not excluded:
        return results
    filtered = [r for r in results if not any(term in result_term_text(r) for term in excluded)]
    removed = len(results) - len(filtered)
    if removed:
        print(f"Applied NOT term/tag filter: {', '.join(sorted(excluded))} ({removed} result(s) hidden)")
    return filtered


# ---------------------------------------------------------------------------
# Existing-content exclusion: skip repos already present in user-specified dirs.
# ---------------------------------------------------------------------------

# Directory names we never want to descend into when scanning for existing repos.
_EXISTING_DIR_SKIP_NAMES = {
    ".git", ".cache", ".huggingface", ".locks", "__pycache__",
    "node_modules", ".DS_Store",
}


def _normalize_repo_id_for_match(repo_id: str) -> str:
    return repo_id.strip().lower()


def _scan_dir_for_existing_repo_ids(directory: Path, max_depth: int = 4) -> set[str]:
    """Walk a directory looking for already-downloaded HF/Kaggle repos.

    Recognized layouts:
      - <publisher>/<repo>/                           (LM Studio + flat layout)
      - <publisher>__<repo>/                          (underscore-flat)
      - models--<owner>--<repo>/                       (HF hub cache)
      - datasets--<owner>--<repo>/                     (HF hub cache)
      - <owner>--<repo>/  (without the models-/datasets- prefix)
      - any directory containing a `config.json` whose grandparent yields owner/

    We stop at max_depth from the requested directory to keep the walk bounded.
    Returns a set of normalized "<owner>/<repo>" strings (lowercased).
    """
    if not directory or not directory.exists() or not directory.is_dir():
        return set()
    found: set[str] = set()
    base = directory.resolve()
    base_depth = len(base.parts)

    def _add(owner: str | None, repo: str | None) -> None:
        if not owner or not repo:
            return
        owner = owner.strip().strip(".").strip()
        repo = repo.strip().strip(".").strip()
        if not owner or not repo:
            return
        # Skip obvious file-extension noise that can sneak in if a path got
        # mistakenly parsed as a repo segment.
        if owner.startswith(".") or repo.startswith("."):
            return
        if "." in owner and len(owner) <= 5:
            return
        found.add(f"{owner.lower()}/{repo.lower()}")

    def _walk(current: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = list(current.iterdir())
        except (PermissionError, FileNotFoundError, OSError):
            return
        for entry in entries:
            try:
                name = entry.name
                if name in _EXISTING_DIR_SKIP_NAMES or name.startswith("."):
                    continue
                if not entry.is_dir():
                    continue
            except OSError:
                continue
            # Layout 1: HF hub cache  models--<owner>--<repo> / datasets--<owner>--<repo>
            m = re.match(r"^(?:models|datasets)--([^-][^/]*?)--(.+)$", name)
            if m:
                _add(m.group(1), m.group(2))
                continue
            # Layout 2: bare hub form  <owner>--<repo>
            m = re.match(r"^([^-][^/]*?)--(.+)$", name)
            if m and "--" not in m.group(2):
                _add(m.group(1), m.group(2))
                # don't descend further: we already have what we need
                continue
            # Layout 3: underscore-flat  <owner>__<repo>
            m = re.match(r"^([^_][^/]*?)__(.+)$", name)
            if m and "__" not in m.group(2):
                _add(m.group(1), m.group(2))
                continue
            # Layout 4: <publisher>/<repo>/ — recognized when this dir contains
            # a child dir that itself looks like a repo (has model/data files
            # or a config.json), so this dir is the publisher.
            try:
                children = [c for c in entry.iterdir() if c.is_dir() and not c.name.startswith(".")]
            except (PermissionError, FileNotFoundError, OSError):
                children = []
            if children:
                publisher_like = False
                for child in children[:50]:
                    try:
                        names = {p.name for p in child.iterdir()}
                    except (PermissionError, FileNotFoundError, OSError):
                        continue
                    if any(
                        n == "config.json"
                        or n.endswith(".gguf")
                        or n.endswith(".safetensors")
                        or n == "tokenizer.json"
                        or n == "model.safetensors.index.json"
                        for n in names
                    ):
                        _add(name, child.name)
                        publisher_like = True
                if publisher_like:
                    continue
            # Recurse if we still have depth budget; this is what catches deeply
            # nested user trees like `<REDACTED_PATH><publisher>/<repo>`.
            depth_now = len(entry.resolve().parts) - base_depth
            if depth_now < max_depth:
                _walk(entry, depth + 1)

    _walk(base, 0)
    return found


def collect_existing_repo_ids(dirs: list[Path]) -> set[str]:
    """Aggregate existing repo IDs across multiple user-supplied directories."""
    out: set[str] = set()
    for d in dirs:
        try:
            ids = _scan_dir_for_existing_repo_ids(d)
        except Exception as e:  # noqa: BLE001 — never block search on a scan error
            print(f"  Warning: could not scan {d}: {e}")
            continue
        if ids:
            print(f"  Found {len(ids)} existing repo dir(s) in {d}")
        out.update(ids)
    return out


def parse_existing_dirs_value(value: str | None) -> list[Path]:
    """Parse a CLI/env list of paths separated by `:` or `,`. Empty → []."""
    if not value:
        return []
    if value.strip().lower() in {"none", "no", "n", "-", "skip"}:
        return []
    parts: list[str] = []
    # Allow both colon (PATH-style) and comma separators.
    for chunk in re.split(r"[:,]+", value):
        chunk = chunk.strip().strip('"\'')
        if chunk and chunk.lower() not in {"none", "no", "n", "-", "skip"}:
            parts.append(chunk)
    out: list[Path] = []
    for part in parts:
        try:
            p = Path(part).expanduser()
        except (RuntimeError, ValueError):
            print(f"  Skipping invalid path: {part!r}")
            continue
        if not p.exists():
            print(f"  Skipping (does not exist): {p}")
            continue
        if not p.is_dir():
            print(f"  Skipping (not a directory): {p}")
            continue
        out.append(p)
    return out


def prompt_existing_dirs() -> list[Path]:
    """Ask the user once per search for directories whose contents should be
    excluded from results. Returns [] if the user skips. Default order of
    precedence:
      1. MODEL_MANAGER_EXISTING_DIRS env var (explicit user override)
      2. DEFAULT_DOWNLOAD_DIR (almost everyone wants to exclude things
         already sitting in their main downloads root)
    """
    default = os.environ.get("MODEL_MANAGER_EXISTING_DIRS", "").strip()
    default_source = "env" if default else None
    if not default and DEFAULT_DOWNLOAD_DIR.exists() and DEFAULT_DOWNLOAD_DIR.is_dir():
        default = str(DEFAULT_DOWNLOAD_DIR)
        default_source = "download-root"
    print()
    print("Already-downloaded filter")
    print("  Provide one or more directories — any owner/repo found inside")
    print("  will be hidden from results. Separate with `:` or `,`. Press")
    print("  Enter to accept the default, or type `none` to skip.")
    print("  Set MODEL_MANAGER_EXISTING_DIRS to override the default.")
    if default and default_source == "env":
        print(f"  Default (from env): {default}")
    elif default and default_source == "download-root":
        print(f"  Default (download root): {default}")
    raw = prompt("Directories with already-downloaded models/datasets", default).strip()
    if not raw or raw.lower() in {"none", "no", "n", "-", "skip"}:
        return []
    return parse_existing_dirs_value(raw)


def apply_existing_repo_exclusions(
    results: list[SearchResult], existing_repo_ids: set[str]
) -> list[SearchResult]:
    if not existing_repo_ids:
        return results
    filtered: list[SearchResult] = []
    hidden = 0
    for r in results:
        rid = _normalize_repo_id_for_match(r.repo_id)
        if rid in existing_repo_ids:
            hidden += 1
            continue
        # Owner/repo may differ in case from disk; also try normalized parts.
        if "/" in rid:
            owner, repo = rid.split("/", 1)
            if f"{owner}/{repo}" in existing_repo_ids:
                hidden += 1
                continue
        filtered.append(r)
    if hidden:
        print(f"Filtered {hidden} already-present repo(s) from existing-dirs scan.")
    return filtered


def choose_artifact_type_filters() -> set[str]:
    print()
    print("What model artifact types do you want to search for?")
    print("Choose the types to include. Anything not selected is excluded automatically.")
    print("Default is GGUF + Core ML only. Safetensors/PyTorch are not included unless selected.")
    print()
    keys = list(ARTIFACT_TYPE_OPTIONS.keys())
    for i, key in enumerate(keys, 1):
        meta = ARTIFACT_TYPE_OPTIONS[key]
        default_mark = "*" if key in DEFAULT_ARTIFACT_TYPE_KEYS else " "
        print(f"  {i}. [{default_mark}] {meta['label']}")
        print(f"      Runs on: {meta['runtime']}")
    print("  0. Any / do not filter")
    print()
    default_nums = ",".join(str(keys.index(k) + 1) for k in DEFAULT_ARTIFACT_TYPE_KEYS)
    raw = prompt("Model types to include, numbers/names, comma-separated", default_nums).strip().lower()
    selected = parse_artifact_type_filter_value(raw)
    if selected is None:
        selected = set(DEFAULT_ARTIFACT_TYPE_KEYS)
    if selected:
        print("Selected artifact types: " + ", ".join(ARTIFACT_TYPE_OPTIONS[k]["label"] for k in keys if k in selected))
    return selected


def parse_artifact_type_filter_value(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    value = raw.strip().lower()
    keys = list(ARTIFACT_TYPE_OPTIONS.keys())
    if value in {"0", "any", "all", "none", "no filter", "nofilter"}:
        return set()
    selected: set[str] = set()
    for part in re.split(r"[,\s]+", value):
        if not part:
            continue
        if part.isdigit() and 1 <= int(part) <= len(keys):
            selected.add(keys[int(part) - 1])
            continue
        for key, meta in ARTIFACT_TYPE_OPTIONS.items():
            label = meta["label"].lower()
            if part == key or part in label or part.lstrip(".") == key:
                selected.add(key)
    return selected or None


def artifact_matches_selected_type(artifact: Artifact, selected_types: set[str]) -> bool:
    if not selected_types:
        return True
    at = artifact.artifact_type.lower()
    if "gguf" in selected_types and at.startswith("gguf"):
        return True
    if "coreml" in selected_types and at in {"coreml", "mlmodel", "mlpackage"}:
        return True
    if "safetensors" in selected_types and at.startswith("safetensors"):
        return True
    if "pytorch" in selected_types and at == "raw-weight":
        return True
    return at in selected_types


def artifacts_for_display(result: SearchResult) -> list[Artifact]:
    if result.visible_artifacts:
        return result.visible_artifacts
    return discover_artifacts(result)


def artifact_is_multipart(artifact: Artifact) -> bool:
    return artifact.artifact_type.lower() in MULTIPART_ARTIFACT_TYPES


def apply_visible_artifacts(result: SearchResult, artifacts: list[Artifact]) -> None:
    for i, art in enumerate(artifacts, 1):
        art.index = i
    result.visible_artifacts = artifacts
    allowed_names = {name for art in artifacts for name, _ in art.files}
    companion_names = {name for name, _ in result.files if any(fnmatch.fnmatch(name, pat) for pat in COMMON_MODEL_COMPANIONS)}
    keep_names = allowed_names | companion_names
    result.files = [(name, size) for name, size in result.files if name in keep_names]


def refresh_hf_file_metadata(result: SearchResult) -> None:
    if result.source != "hf" or result.kind != "model":
        return
    if result.files:
        return
    try:
        info = hf_api().model_info(repo_id=result.repo_id, files_metadata=True)
        total, files = _repo_files_from_info(info)
        result.size_bytes = total or result.size_bytes
        result.files = files
        lic = _card_license(info)
        if lic and not result.license:
            result.license = lic
    except Exception as e:
        note = f"metadata refresh failed: {type(e).__name__}: {e}"
        if note not in result.notes:
            result.notes.append(note)


def filter_results_by_artifact_types(results: list[SearchResult], selected_types: set[str]) -> list[SearchResult]:
    if not selected_types:
        return results
    kept: list[SearchResult] = []
    removed = 0
    for r in results:
        if r.kind != "model":
            kept.append(r)
            continue
        if r.direct_lookup:
            # User typed this exact repo-id; refresh metadata so artifacts
            # show up correctly, but don't drop it on artifact-type mismatch.
            refresh_hf_file_metadata(r)
            artifacts = discover_artifacts(r)
            matching = [a for a in artifacts if artifact_matches_selected_type(a, selected_types)]
            if matching:
                apply_visible_artifacts(r, matching)
                r.notes.append("artifact type match: " + ", ".join(sorted({a.artifact_type for a in matching})) + f"; {len(matching)} direct artifact option(s)")
            else:
                r.notes.append("direct repo lookup kept despite artifact-type filter mismatch")
            kept.append(r)
            continue
        if r.source != "hf":
            r.notes.append("artifact type not confirmed; provider does not expose reliable file metadata")
            kept.append(r)
            continue
        refresh_hf_file_metadata(r)
        artifacts = discover_artifacts(r)
        matching = [a for a in artifacts if artifact_matches_selected_type(a, selected_types)]
        if matching:
            apply_visible_artifacts(r, matching)
            r.notes.append("artifact type match: " + ", ".join(sorted({a.artifact_type for a in matching})) + f"; {len(matching)} direct artifact option(s)")
            kept.append(r)
        else:
            removed += 1
    if removed:
        labels = ", ".join(ARTIFACT_TYPE_OPTIONS[k]["label"] for k in ARTIFACT_TYPE_OPTIONS if k in selected_types)
        print(f"Filtered out {removed} HF model repo(s) that did not expose selected artifact type(s): {labels}")
    return kept


def filter_results_by_model_size(
    results: list[SearchResult],
    size_range: tuple[int | None, int | None],
) -> list[SearchResult]:
    lo, hi = size_range
    if lo is None and hi is None:
        return results
    # Direct repo-ID lookups bypass the size filter — user asked for this
    # exact repo, don't drop it because its size is outside the range.
    direct = [r for r in results if r.direct_lookup]
    if direct:
        results = [r for r in results if not r.direct_lookup]
        for r in direct:
            r.notes.append("direct repo lookup kept despite size-range filter")
    kept: list[SearchResult] = []
    removed = 0
    for r in results:
        if r.kind != "model":
            kept.append(r)
            continue
        if r.source == "hf":
            refresh_hf_file_metadata(r)
            artifacts = artifacts_for_display(r)
            if not artifacts:
                removed += 1
                continue
            matching = [a for a in artifacts if size_in_range(a.total_size, size_range)]
            if matching:
                apply_visible_artifacts(r, matching)
                r.notes.append(
                    f"model size match: {human_size(lo)} - {human_size(hi)}; {len(matching)} artifact option(s)"
                )
                kept.append(r)
            else:
                removed += 1
            continue
        if size_in_range(r.size_bytes, size_range):
            kept.append(r)
        else:
            removed += 1
    if removed:
        print(f"Filtered out {removed} model repo(s) outside size range: {human_size(lo)} - {human_size(hi)}")
    # Direct lookups were set aside above; re-attach them at the end so they
    # appear alongside the search-driven results.
    if direct:
        kept.extend(direct)
    return kept


def filter_results_by_multipart_models(
    results: list[SearchResult],
    exclude_multipart_models: bool,
) -> list[SearchResult]:
    if not exclude_multipart_models:
        return results
    kept: list[SearchResult] = []
    removed = 0
    for r in results:
        if r.kind != "model":
            kept.append(r)
            continue
        if r.direct_lookup:
            r.notes.append("direct repo lookup kept despite multipart filter")
            kept.append(r)
            continue
        if r.source != "hf":
            r.notes.append("multi-part artifact layout not confirmed; provider does not expose reliable file metadata")
            kept.append(r)
            continue
        refresh_hf_file_metadata(r)
        artifacts = artifacts_for_display(r)
        if not artifacts:
            r.notes.append("multi-part artifact layout not confirmed; no recognized artifacts found")
            kept.append(r)
            continue
        matching = [a for a in artifacts if not artifact_is_multipart(a)]
        if matching:
            apply_visible_artifacts(r, matching)
            removed_types = sorted({a.artifact_type for a in artifacts if artifact_is_multipart(a)})
            if removed_types:
                r.notes.append("excluded multi-part artifact type(s): " + ", ".join(removed_types))
            kept.append(r)
            continue
        removed += 1
    if removed:
        print(
            f"Filtered out {removed} HF model repo(s) that only exposed multi-part artifacts "
            "(for example gguf-split or safetensors-sharded)."
        )
    return kept


def expand_search_terms_for_artifacts(search_terms: list[str], selected_types: set[str]) -> list[str]:
    """Add filename-pattern searches such as `.gguf` and `.mlmodel`.

    HF's API does not provide perfect file-extension search semantics, but adding
    extension terms finds many repos that host direct artifacts and improves the
    chance that the picker can show 1A/1B artifact choices.
    """
    if not selected_types:
        return search_terms
    expanded: list[str] = []
    for term in search_terms:
        base = term.strip()
        if base:
            expanded.append(base)
        for key in selected_types:
            for suffix in ARTIFACT_TYPE_OPTIONS.get(key, {}).get("search_terms", []):
                q = f"{base} {suffix}".strip() if base else suffix
                expanded.append(q)
    seen: set[str] = set()
    out: list[str] = []
    for term in expanded:
        canonical = re.sub(r"[^a-z0-9]+", " ", term.lower()).strip()
        if canonical not in seen:
            seen.add(canonical)
            out.append(term)
    return out


def artifact_focused_search_terms(provider_search_terms: list[str], base_search_terms: list[str]) -> list[str]:
    base = {term.strip().lower() for term in base_search_terms}
    focused = [term for term in provider_search_terms if term.strip().lower() not in base]
    return focused or provider_search_terms

def merge_search_results(results: list[SearchResult]) -> list[SearchResult]:
    """De-duplicate search results across split queries and sources."""
    merged: dict[tuple[str, str, str], SearchResult] = {}
    for r in results:
        key = (r.source, r.kind, r.repo_id)
        if key not in merged:
            merged[key] = r
            continue
        existing = merged[key]
        # Keep richer metadata if one result was partial.
        if (existing.size_bytes or 0) == 0 and r.size_bytes:
            existing.size_bytes = r.size_bytes
        if not existing.files and r.files:
            existing.files = r.files
        if not existing.license and r.license:
            existing.license = r.license
        # Sticky: once a repo is marked as a direct user lookup, keep that
        # flag even if a name-search hit duplicates it.
        if r.direct_lookup:
            existing.direct_lookup = True
        if r.notes:
            for note in r.notes:
                if note not in existing.notes:
                    existing.notes.append(note)
    return list(merged.values())



# -----------------------------------------------------------------------------
# Leaderboard cache and exact model helpers
# -----------------------------------------------------------------------------

def load_leaderboard_cache() -> dict[str, Any]:
    if not LEADERBOARD_CACHE_PATH.is_file():
        return {"version": 1, "entries": {}}
    try:
        data = json.loads(LEADERBOARD_CACHE_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"version": 1, "entries": {}}
        data.setdefault("version", 1)
        data.setdefault("entries", {})
        return data
    except Exception:
        return {"version": 1, "entries": {}}


def save_leaderboard_cache(data: dict[str, Any]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data.setdefault("version", 1)
    data.setdefault("entries", {})
    data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    LEADERBOARD_CACHE_PATH.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def add_manual_leaderboard_cache(category: str) -> None:
    print()
    print("Paste ranked Hugging Face model repo IDs or URLs, one per line.")
    print("Example: Qwen/Qwen2.5-Coder-32B-Instruct")
    print("Blank line finishes.")
    cache = load_leaderboard_cache()
    entries: dict[str, Any] = cache.setdefault("entries", {})
    rank = 1
    while True:
        value = read_input(f"Leaderboard #{rank}: ").strip()
        if not value:
            break
        repo_id = parse_hf_repo_id(value)
        if not repo_id:
            print("Could not parse a Hugging Face repo ID from that value.")
            continue
        entries[repo_id.lower()] = {
            "repo_id": repo_id,
            "rank": rank,
            "category": category,
            "label": f"Leaderboard #{rank}",
            "source": "manual",
            "added_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        rank += 1
    save_leaderboard_cache(cache)
    print(f"Saved leaderboard cache: {LEADERBOARD_CACHE_PATH}")


def refresh_leaderboard_cache_best_effort(categories: list[str], limit: int = 20) -> None:
    """Best-effort cache builder using HF model searches.

    This is not a canonical benchmark scrape. It exists so search results can be
    ranked/labeled when the same repo IDs appear later. For authoritative ranks,
    use manual cache entry from leaderboard pages.
    """
    search_terms = {
        "coding": ["coder", "code", "CodeLlama", "Qwen Coder"],
        "security": ["cybersecurity", "malware", "threat detection", "security"],
        "visual": ["vision language", "vlm", "visual"],
        "embedding": ["embedding", "sentence similarity", "retriever"],
        "general": ["instruct", "chat", "llm"],
    }
    cache = load_leaderboard_cache()
    entries: dict[str, Any] = cache.setdefault("entries", {})
    for category in categories:
        print(f"Refreshing best-effort leaderboard cache for {category}...")
        merged: list[SearchResult] = []
        for term in search_terms.get(category, [category]):
            try:
                merged.extend(search_hf_models(term, limit=limit))
            except Exception as e:
                print(f"  search failed for {term}: {type(e).__name__}: {e}")
        ranked = merge_search_results(merged)
        ranked.sort(key=lambda r: (r.downloads or 0, r.likes or 0), reverse=True)
        for rank, r in enumerate(ranked[:limit], 1):
            entries[r.repo_id.lower()] = {
                "repo_id": r.repo_id,
                "rank": rank,
                "category": category,
                "label": f"Leaderboard #{rank}",
                "source": "best-effort-hf-search",
                "added_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
    save_leaderboard_cache(cache)
    print(f"Saved leaderboard cache: {LEADERBOARD_CACHE_PATH}")


def annotate_results_with_leaderboard_cache(results: list[SearchResult]) -> None:
    cache = load_leaderboard_cache()
    entries = cache.get("entries", {}) if isinstance(cache, dict) else {}
    if not isinstance(entries, dict):
        return
    for r in results:
        entry = entries.get(r.repo_id.lower())
        if not isinstance(entry, dict):
            continue
        try:
            r.leaderboard_rank = int(entry.get("rank"))
        except Exception:
            r.leaderboard_rank = None
        r.leaderboard_label = str(entry.get("label") or (f"Leaderboard #{r.leaderboard_rank}" if r.leaderboard_rank else "Leaderboard"))


def parse_hf_repo_id(value: str) -> str | None:
    value = value.strip()
    if not value:
        return None
    value = value.split("?", 1)[0].rstrip("/")
    for prefix in ("https://huggingface.co/datasets/", "https://www.huggingface.co/datasets/"):
        if value.startswith(prefix):
            return value[len(prefix):].strip("/") or None
    for prefix in ("https://huggingface.co/", "https://www.huggingface.co/"):
        if value.startswith(prefix):
            remainder = value[len(prefix):].strip("/")
            parts = remainder.split("/")
            if len(parts) >= 2:
                return "/".join(parts[:2])
            return None
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value):
        return value
    return None


def build_exact_hf_result(repo_id: str, kind: str = "model") -> SearchResult | None:
    api = hf_api()
    try:
        if kind == "dataset":
            info = api.dataset_info(repo_id=repo_id, files_metadata=True)
        else:
            info = api.model_info(repo_id=repo_id, files_metadata=True)
        total, files = _repo_files_from_info(info)
        lic = _card_license(info)
        return SearchResult(
            index=1,
            source="hf",
            kind=kind,
            repo_id=repo_id,
            size_bytes=total,
            downloads=getattr(info, "downloads", None),
            likes=getattr(info, "likes", None),
            pipeline=getattr(info, "pipeline_tag", None),
            updated=str(getattr(info, "lastModified", None) or getattr(info, "last_modified", None) or ""),
            license=lic,
            url=f"https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{repo_id}",
            files=files,
            raw=info,
        )
    except Exception as e:
        print(f"Could not fetch Hugging Face {kind} {repo_id}: {type(e).__name__}: {e}")
        return None

def prompt_multi_choice(label: str, choices: list[str], default: str) -> list[str]:
    """Prompt for one or more numbered choices, e.g. 2,4."""
    print(label)
    for i, c in enumerate(choices, 1):
        tag = " default" if c == default else ""
        print(f"  {i}. {c}{tag}")
    value = prompt(f"Choose 1-{len(choices)}, comma-separated allowed", default).strip()
    if value in choices:
        return [value]
    nums = parse_selection(value, len(choices))
    if nums:
        return [choices[n - 1] for n in nums]
    print(f"Invalid choice. Using {default}.")
    return [default]

# -----------------------------------------------------------------------------
# Hugging Face
# -----------------------------------------------------------------------------

def hf_api():
    configure_hf_download_environment()
    from huggingface_hub import HfApi
    return HfApi(token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"))


# Throttle between HF API metadata calls during search. Default 0 (off).
#
# Throttle is only useful for ANONYMOUS callers (no HF account). With an HF
# account — free OR Pro — `HF_TOKEN` raises the per-user rate limit far
# beyond anything this script's parallel search would hit. The 429 retry
# helper below absorbs the rare ceiling case automatically.
#
# Set MODEL_MANAGER_HF_SEARCH_DELAY_MS=150 (or higher) only if you're
# running anonymous (no HF_TOKEN) AND you start seeing many 429 retries on
# multi-term searches. How to set it:
#
#   One-off (this command only):
#       MODEL_MANAGER_HF_SEARCH_DELAY_MS=150 modelmgr
#
#   Current shell only:
#       export MODEL_MANAGER_HF_SEARCH_DELAY_MS=150
#
#   Persistent (every new shell): add the line above to <REDACTED_PATH>, then
#       source <REDACTED_PATH>
#
# Value is milliseconds between metadata fetches; 0 disables.
_HF_SEARCH_DELAY_S = env_int("MODEL_MANAGER_HF_SEARCH_DELAY_MS", 0, minimum=0) / 1000.0

# Cap the total number of HF search-term variants expanded per query. The two
# expanders (name variants × artifact suffixes) can produce 25+ terms per input,
# each costing a list_models + N×model_info round-trip. A small cap keeps the API
# load proportional to the actual base query. Set to a large number to disable.
_HF_SEARCH_MAX_TERMS = env_int("MODEL_MANAGER_HF_SEARCH_MAX_TERMS", 5, minimum=1)


def _hf_search_throttle() -> None:
    if _HF_SEARCH_DELAY_S > 0:
        time.sleep(_HF_SEARCH_DELAY_S)


def _hf_call_with_retry(fn, *args, max_retries: int = 3, **kwargs):
    """Call an HF API function with 429-aware retry. Honors Retry-After header."""
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            response = getattr(e, "response", None)
            status = getattr(response, "status_code", None) if response is not None else None
            if status != 429 or attempt >= max_retries:
                raise
            retry_after = 30
            try:
                header = response.headers.get("Retry-After") if response is not None else None
                if header:
                    retry_after = max(1, int(float(header)))
            except (TypeError, ValueError, AttributeError):
                pass
            authenticated = bool(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"))
            tip = "" if authenticated else " (set HF_TOKEN to raise your limit; create one at https://huggingface.co/settings/tokens)"
            print(f"  HF API 429 — sleeping {retry_after}s before retry {attempt + 1}/{max_retries}{tip}")
            time.sleep(retry_after)


def _repo_files_from_info(info: Any) -> tuple[int, list[tuple[str, int | None]]]:
    files: list[tuple[str, int | None]] = []
    total = 0
    for s in info.siblings or []:
        name = getattr(s, "rfilename", None) or getattr(s, "filename", None) or "<unknown>"
        size = getattr(s, "size", None)
        files.append((name, int(size) if size is not None else None))
        total += int(size or 0)
    files.sort(key=lambda x: x[1] or 0, reverse=True)
    return total, files


def _card_license(info: Any) -> str | None:
    card = getattr(info, "cardData", None) or getattr(info, "card_data", None)
    if isinstance(card, dict):
        lic = card.get("license")
        if isinstance(lic, list):
            return ",".join(str(x) for x in lic)
        if lic:
            return str(lic)
    return None


def candidate_excluded_before_metadata(
    repo_id: str,
    raw: Any,
    excluded_publishers: set[str] | None = None,
    excluded_terms: set[str] | None = None,
    excluded_family_terms: list[str] | None = None,
) -> bool:
    owner = owner_of(repo_id)
    if excluded_publishers and any(owner == x or repo_id.lower().startswith(x + "/") or x in owner for x in excluded_publishers):
        return True

    text_parts = [repo_id, model_family_key(repo_id)]
    for attr in ("tags", "pipeline_tag", "library_name"):
        value = getattr(raw, attr, None)
        if isinstance(value, (list, tuple, set)):
            text_parts.extend(str(v) for v in value)
        elif value:
            text_parts.append(str(value))
    text = normalize_match_text(" ".join(text_parts))
    if excluded_terms and any(term and term in text for term in excluded_terms):
        return True

    if excluded_family_terms:
        family = normalize_match_text(model_family_key(repo_id))
        repo_text = normalize_match_text(repo_id)
        for term in excluded_family_terms:
            query = normalize_match_text(term)
            if query and (query in repo_text or query in family):
                return True
    return False


def search_hf_models(
    query: str,
    limit: int,
    candidate_limit: int | None = None,
    pre_excluded_publishers: set[str] | None = None,
    pre_excluded_terms: set[str] | None = None,
    pre_excluded_family_terms: list[str] | None = None,
) -> list[SearchResult]:
    api = hf_api()
    fetch_limit = candidate_limit or limit
    try:
        models = list(_hf_call_with_retry(api.list_models, search=query, limit=fetch_limit, sort="downloads", full=True))
    except TypeError:
        models = list(_hf_call_with_retry(api.list_models, search=query, limit=fetch_limit, full=True))

    # First pass: pre-filter (cheap, no network) so we don't waste model_info on excluded repos.
    candidates: list[tuple[str, Any]] = []
    for m in models:
        repo_id = getattr(m, "modelId", None) or getattr(m, "id", None)
        if not repo_id:
            continue
        if candidate_excluded_before_metadata(
            repo_id, m,
            excluded_publishers=pre_excluded_publishers,
            excluded_terms=pre_excluded_terms,
            excluded_family_terms=pre_excluded_family_terms,
        ):
            continue
        candidates.append((repo_id, m))
        # In normal-scan mode, stop after `limit` candidates so we don't fetch
        # 500 metadatas worth in a small search.
        if candidate_limit is None and len(candidates) >= limit:
            break

    # Second pass: parallel metadata fetches. With HF_TOKEN we have plenty of
    # rate-limit headroom for 8-16 concurrent requests; the 429 retry helper
    # absorbs anything that does come back rate-limited.
    workers = env_int("MODEL_MANAGER_HF_SEARCH_WORKERS", 12, minimum=1)

    def _fetch(item: tuple[str, Any]):
        repo_id, raw = item
        try:
            _hf_search_throttle()
            info = _hf_call_with_retry(api.model_info, repo_id=repo_id, files_metadata=True)
            total, files = _repo_files_from_info(info)
            lic = _card_license(info)
            return repo_id, raw, total, files, lic, None
        except Exception as e:
            return repo_id, raw, None, [], None, f"metadata error: {type(e).__name__}: {e}"

    results: list[SearchResult] = []
    if not candidates:
        return results

    if workers <= 1 or len(candidates) <= 2:
        rows = [_fetch(c) for c in candidates]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(workers, len(candidates))) as ex:
            # Preserve input order so search results stay deterministic.
            rows = list(ex.map(_fetch, candidates))

    for repo_id, raw, total, files, lic, err in rows:
        notes: list[str] = []
        if err:
            notes.append(err)
        results.append(SearchResult(
            index=0,
            source="hf",
            kind="model",
            repo_id=repo_id,
            size_bytes=total,
            downloads=getattr(raw, "downloads", None),
            likes=getattr(raw, "likes", None),
            pipeline=getattr(raw, "pipeline_tag", None),
            updated=str(getattr(raw, "lastModified", None) or ""),
            license=lic,
            url=f"https://huggingface.co/{repo_id}",
            files=files,
            notes=notes,
            raw=raw,
        ))
    return results


def search_hf_datasets(query: str, limit: int) -> list[SearchResult]:
    api = hf_api()
    try:
        datasets = list(_hf_call_with_retry(api.list_datasets, search=query, limit=limit, sort="downloads", full=True))
    except Exception:
        from huggingface_hub import list_datasets
        datasets = list(list_datasets(search=query, limit=limit))

    candidates: list[tuple[str, Any]] = []
    for ds in datasets:
        repo_id = getattr(ds, "id", None) or getattr(ds, "repo_id", None)
        if repo_id:
            candidates.append((repo_id, ds))

    workers = env_int("MODEL_MANAGER_HF_SEARCH_WORKERS", 12, minimum=1)

    def _fetch(item: tuple[str, Any]):
        repo_id, raw = item
        try:
            _hf_search_throttle()
            info = _hf_call_with_retry(api.dataset_info, repo_id=repo_id, files_metadata=True)
            total, files = _repo_files_from_info(info)
            lic = _card_license(info)
            return repo_id, raw, total, files, lic, None
        except Exception as e:
            return repo_id, raw, None, [], None, f"metadata error: {type(e).__name__}: {e}"

    results: list[SearchResult] = []
    if not candidates:
        return results

    if workers <= 1 or len(candidates) <= 2:
        rows = [_fetch(c) for c in candidates]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(workers, len(candidates))) as ex:
            rows = list(ex.map(_fetch, candidates))

    for repo_id, raw, total, files, lic, err in rows:
        notes: list[str] = []
        if err:
            notes.append(err)
        results.append(SearchResult(
            index=0,
            source="hf",
            kind="dataset",
            repo_id=repo_id,
            size_bytes=total,
            downloads=getattr(raw, "downloads", None),
            likes=getattr(raw, "likes", None),
            updated=str(getattr(raw, "last_modified", None) or getattr(raw, "lastModified", None) or ""),
            license=lic,
            url=f"https://huggingface.co/datasets/{repo_id}",
            files=files,
            notes=notes,
            raw=raw,
        ))
    return results

# -----------------------------------------------------------------------------
# Kaggle
# -----------------------------------------------------------------------------

def kaggle_api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def search_kaggle_datasets(query: str, limit: int) -> list[SearchResult]:
    api = kaggle_api()
    results: list[SearchResult] = []
    page = 1
    seen: set[str] = set()
    while len(results) < limit:
        chunk = api.dataset_list(search=query, sort_by="updated", page=page)
        if not chunk:
            break
        for ds in chunk:
            ref = getattr(ds, "ref", None)
            if not ref or ref in seen:
                continue
            seen.add(ref)
            size = getattr(ds, "totalBytes", None)
            results.append(SearchResult(
                index=0,
                source="kaggle",
                kind="dataset",
                repo_id=ref,
                title=getattr(ds, "title", "") or "",
                size_bytes=int(size) if size else None,
                downloads=getattr(ds, "downloadCount", None),
                updated=str(getattr(ds, "lastUpdated", None) or getattr(ds, "last_updated", None) or ""),
                url=f"https://www.kaggle.com/datasets/{ref}",
                raw=ds,
            ))
            if len(results) >= limit:
                break
        page += 1
        if page > 10:
            break
        time.sleep(0.5)
    return results


def _kaggle_model_result_from_handle(handle: str, title: str = "", raw: Any = None) -> SearchResult | None:
    handle = (handle or "").strip().strip('"')
    if not handle or "/" not in handle:
        return None
    # Kaggle model handles are usually owner/model or owner/model/framework/variation.
    # Keep longer handles intact for downloads, but reject obvious table noise.
    if handle.lower() in {"ref", "handle", "owner/model"}:
        return None
    notes: list[str] = []
    text = f"{handle} {title}".lower()
    if "huggingface" in text or "hugging-face" in text:
        notes.append("possible Hugging Face mirror/integration")
    return SearchResult(
        index=0,
        source="kaggle",
        kind="model",
        repo_id=handle,
        title=title or handle,
        url=f"https://www.kaggle.com/models/{handle}",
        notes=notes,
        raw=raw,
    )


def search_kaggle_models(query: str, limit: int) -> list[SearchResult]:
    """Best effort Kaggle model search.

    Kaggle model discovery is most reliable through the Kaggle CLI's
    `kaggle models list -s <term>` path. kagglehub is kept as a fallback for
    package versions that expose list/search helpers, but many kagglehub
    releases focus on download/upload rather than search.
    """
    results: list[SearchResult] = []
    seen: set[str] = set()

    def add(result: SearchResult | None) -> None:
        if not result or result.repo_id in seen or len(results) >= limit:
            return
        seen.add(result.repo_id)
        results.append(result)

    cli = shutil.which("kaggle")
    if cli:
        cmd = [cli, "models", "list", "-s", query, "--page-size", str(min(max(limit, 1), 100)), "-v"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "").strip()
                print(f"Kaggle model CLI search returned {proc.returncode}: {err[:300]}")
            else:
                text = proc.stdout or ""
                # New Kaggle CLI supports -v CSV. Parse CSV first, then fall back to whitespace table parsing.
                try:
                    rows = list(csv.DictReader(text.splitlines()))
                except Exception:
                    rows = []
                for row in rows:
                    handle = row.get("ref") or row.get("Ref") or row.get("handle") or row.get("Handle") or row.get("model") or row.get("Model")
                    title = row.get("title") or row.get("Title") or row.get("subtitle") or row.get("Subtitle") or ""
                    add(_kaggle_model_result_from_handle(str(handle or ""), str(title or ""), row))
                if not results:
                    for line in text.splitlines():
                        ln = line.strip()
                        if not ln or ln.lower().startswith(("ref", "warning", "usage")):
                            continue
                        first = ln.split()[0]
                        add(_kaggle_model_result_from_handle(first, ln))
                        if len(results) >= limit:
                            break
        except Exception as e:
            print(f"Kaggle model CLI search failed: {type(e).__name__}: {e}")
    else:
        print("Kaggle CLI not found on PATH; install with: python3 -m pip install --upgrade kaggle")

    if results:
        return results[:limit]

    try:
        import kagglehub
        for fn_name in ("model_list", "models_list", "list_models", "search_models"):
            fn = getattr(kagglehub, fn_name, None)
            if not callable(fn):
                continue
            for kwargs in ({"search": query}, {"search_term": query}, {"query": query}, {}):
                try:
                    items = fn(**kwargs) if kwargs else fn(query)
                    for item in list(items)[:limit]:
                        handle = str(getattr(item, "handle", None) or getattr(item, "ref", None) or getattr(item, "id", None) or item)
                        add(_kaggle_model_result_from_handle(handle, str(item), item))
                    if results:
                        return results[:limit]
                except Exception:
                    continue
        print("kagglehub is installed but did not expose a working model search helper in this environment.")
    except ImportError:
        print("kagglehub not installed; optional fallback install: python3 -m pip install --upgrade kagglehub")
    except Exception as e:
        print(f"Kagglehub model search fallback failed: {type(e).__name__}: {e}")

    return results[:limit]


def kaggle_dataset_files(ref: str) -> list[tuple[str, int | None]]:
    api = kaggle_api()
    out: list[tuple[str, int | None]] = []
    try:
        files = api.dataset_list_files(ref).files
        for f in files or []:
            name = getattr(f, "name", None)
            size = getattr(f, "totalBytes", None) or getattr(f, "size", None)
            if name:
                out.append((name, int(size) if size else None))
    except Exception as e:
        print(f"Could not list Kaggle files for {ref}: {type(e).__name__}: {e}")
    out.sort(key=lambda x: x[1] or 0, reverse=True)
    return out

# -----------------------------------------------------------------------------
# Artifact discovery
# -----------------------------------------------------------------------------

GGUF_SPLIT_RE = re.compile(r"^(.+?)(?:-|\.)(0*\d+)-of-(0*\d+)\.gguf$", re.I)
QUANT_RE = re.compile(r"(?:^|[.\-_])(IQ[0-9][A-Z0-9_]*|Q[0-9](?:_[A-Z0-9]+)*|F16|F32|BF16|FP16|FP32)(?:$|[.\-_])", re.I)
SAFE_SHARD_RE = re.compile(r"^(.+?)-0*(\d+)-of-0*(\d+)\.safetensors$", re.I)


def detect_quant(name: str) -> str | None:
    m = QUANT_RE.search(name)
    return m.group(1).upper() if m else None


def discover_artifacts(result: SearchResult) -> list[Artifact]:
    if result.kind != "model":
        return []
    files = result.files
    artifacts: list[Artifact] = []

    # GGUF split groups.
    split_groups: dict[tuple[str, str], list[tuple[str, int | None]]] = {}
    used: set[str] = set()
    for name, size in files:
        if not name.lower().endswith(".gguf"):
            continue
        m = GGUF_SPLIT_RE.match(Path(name).name)
        if not m:
            continue
        base = m.group(1)
        total = m.group(3)
        key = (str(Path(name).parent), f"{base}-of-{total}")
        split_groups.setdefault(key, []).append((name, size))
        used.add(name)
    for (_, group_name), group_files in split_groups.items():
        group_files.sort(key=lambda x: x[0])
        quant = detect_quant(group_name)
        artifacts.append(Artifact(0, f"GGUF split {quant or group_name}", "gguf-split", quant, group_files))

    # Single GGUFs.
    for name, size in files:
        if name in used or not name.lower().endswith(".gguf"):
            continue
        quant = detect_quant(name)
        label = f"GGUF {quant}" if quant else f"GGUF {Path(name).name}"
        artifacts.append(Artifact(0, label, "gguf", quant, [(name, size)]))
        used.add(name)

    # Core ML artifacts.
    for name, size in files:
        lower = name.lower()
        if name in used:
            continue
        if lower.endswith(".mlmodel"):
            artifacts.append(Artifact(0, f"Core ML {Path(name).name}", "coreml", None, [(name, size)]))
            used.add(name)
        elif ".mlpackage/" in lower or lower.endswith(".mlpackage"):
            package_root = name[:lower.find(".mlpackage") + len(".mlpackage")]
            package_files = [(n, s) for n, s in files if n == package_root or n.startswith(package_root + "/")]
            if package_files:
                artifacts.append(Artifact(0, f"Core ML package {Path(package_root).name}", "coreml", None, package_files))
                used.update(n for n, _ in package_files)

    # ONNX artifacts.
    for name, size in files:
        if name in used or not name.lower().endswith(".onnx"):
            continue
        artifacts.append(Artifact(0, f"ONNX {Path(name).name}", "onnx", None, [(name, size)]))
        used.add(name)

    # Safetensors index group: whole sharded model.
    index_files = [(n, s) for n, s in files if n.endswith(".safetensors.index.json")]
    if index_files:
        shard_files = [(n, s) for n, s in files if n.endswith(".safetensors") and "adapter_model" not in Path(n).name.lower()]
        if shard_files:
            artifacts.append(Artifact(0, "safetensors sharded model", "safetensors-sharded", None, shard_files + index_files))
            used.update(n for n, _ in shard_files + index_files)

    # Safetensors shard filename groups where no index was available.
    shard_groups: dict[tuple[str, str], list[tuple[str, int | None]]] = {}
    for name, size in files:
        if name in used or not name.endswith(".safetensors"):
            continue
        m = SAFE_SHARD_RE.match(Path(name).name)
        if not m:
            continue
        key = (str(Path(name).parent), f"{m.group(1)}-of-{m.group(3)}")
        shard_groups.setdefault(key, []).append((name, size))
        used.add(name)
    for (_, group_name), group_files in shard_groups.items():
        group_files.sort(key=lambda x: x[0])
        artifacts.append(Artifact(0, f"safetensors shard group {group_name}", "safetensors-sharded", None, group_files))

    # Adapter artifacts.
    adapter_files = [(n, s) for n, s in files if Path(n).name.lower() in {"adapter_model.safetensors", "adapter_config.json"} or "/adapter_model.safetensors" in n.lower()]
    if adapter_files:
        artifacts.append(Artifact(0, "adapter / LoRA", "adapter", None, adapter_files, companion_patterns=["README*", "LICENSE*", "tokenizer*", "adapter_config.json", "*.md"]))
        used.update(n for n, _ in adapter_files)

    # Single safetensors models.
    for name, size in files:
        if name in used or not name.endswith(".safetensors"):
            continue
        if Path(name).name == "model.safetensors" or "consolidated" in Path(name).name.lower():
            artifacts.append(Artifact(0, f"safetensors {Path(name).name}", "safetensors", None, [(name, size)]))
            used.add(name)

    # Raw legacy weight files.
    for name, size in files:
        if name in used:
            continue
        if Path(name).suffix.lower() in {".bin", ".pt", ".pth", ".ckpt"}:
            artifacts.append(Artifact(0, f"raw weight {Path(name).name}", "raw-weight", None, [(name, size)], notes=["pickle/deserialization risk; do not load unless trusted"]))
            used.add(name)

    artifacts.sort(key=lambda a: a.total_size, reverse=True)
    for i, art in enumerate(artifacts, 1):
        art.index = i
    return artifacts


def choose_artifacts(result: SearchResult) -> tuple[str, list[str] | None, list[Artifact]]:
    if result.kind != "model" or result.source != "hf":
        return choose_file_patterns_legacy(result) + ([],)

    refresh_hf_file_metadata(result)
    artifacts = artifacts_for_display(result)
    if not artifacts:
        print("No direct model artifacts were discovered from Hugging Face file metadata.")
        if not prompt_bool("Allow whole-repository download fallback?", False):
            return "skip", None, []
        mode, patterns = choose_file_patterns_legacy(result)
        return mode, patterns, []

    print()
    print(f"Discovered downloadable artifacts inside repo: {result.repo_id}")
    print(f"Repo total size: {human_size(result.size_bytes)}")
    print()
    print(f"{'#':>4}  {'SIZE':>12}  {'TYPE':<22}  {'QUANT':<10}  ARTIFACT")
    print("-" * 100)
    for art in artifacts:
        quant = art.quant or "-"
        print(f"{art.index:>4}  {human_size(art.total_size):>12}  {art.artifact_type:<22}  {quant:<10}  {art.label}")
        for fname, fsize in art.files[:8]:
            print(f"      {human_size(fsize):>12}  {fname}")
        if len(art.files) > 8:
            print(f"      ... {len(art.files) - 8} more files")
        for note in art.notes:
            print(f"      note: {note}")
        print()

    mode = prompt_choice("Download mode", ["artifact_numbers", "pattern", "whole_repo", "skip"], "artifact_numbers")
    if mode == "skip":
        return mode, None, []
    if mode == "whole_repo":
        if not prompt_bool(f"Whole repo is {human_size(result.size_bytes)}. Continue?", False):
            return "skip", None, []
        return mode, None, []
    if mode == "pattern":
        pat = prompt("Pattern(s), comma separated", "*Q4_K_M.gguf")
        patterns = [p.strip() for p in pat.split(",") if p.strip()]
        return mode, patterns or None, []

    nums = parse_selection(prompt("Which artifact numbers? Examples: 1, 1,3, 2-4, all"), len(artifacts))
    selected = [artifacts[n - 1] for n in nums]
    if not selected:
        return "skip", None, []
    patterns: list[str] = []
    for art in selected:
        patterns.extend(art.allow_patterns())
    # Deduplicate while preserving order.
    seen: set[str] = set()
    patterns = [p for p in patterns if not (p in seen or seen.add(p))]
    return mode, patterns, selected


def choose_file_patterns_legacy(result: SearchResult) -> tuple[str, list[str] | None]:
    if result.kind == "dataset":
        mode = prompt_choice("Download mode", ["whole", "pattern", "skip"], "whole")
    else:
        mode = prompt_choice("Download mode", ["whole", "file_numbers", "pattern", "skip"], "file_numbers" if result.files else "whole")
    if mode == "skip":
        return mode, None
    if mode == "whole":
        return mode, None
    if mode == "pattern":
        default = "*.csv,*.jsonl,*.parquet" if result.kind == "dataset" else "*.gguf"
        pat = prompt("Pattern(s), comma separated", default)
        return mode, [p.strip() for p in pat.split(",") if p.strip()] or None
    if mode == "file_numbers":
        files = result.files
        if not files:
            print("No file list available. Falling back to whole repo.")
            return "whole", None
        print()
        print(f"Files for {result.repo_id}")
        for i, (name, size) in enumerate(files, 1):
            print(f"{i:>4}. {human_size(size):>12}  {name}")
            if i >= 300:
                print("... showing first 300 files")
                break
        nums = parse_selection(prompt("Which file numbers? Examples: 1, 1,3, 2-4, all"), min(len(files), 300))
        return mode, [files[n - 1][0] for n in nums] or None
    return mode, None

# -----------------------------------------------------------------------------
# Display and selection
# -----------------------------------------------------------------------------

def assign_indexes(results: list[SearchResult]) -> list[SearchResult]:
    for i, r in enumerate(results, 1):
        r.index = i
    return results



# -----------------------------------------------------------------------------
# Recommendation / duplicate helpers
# -----------------------------------------------------------------------------

def load_reputation_config() -> dict[str, Any]:
    default = {
        "known_good_owners": sorted(DEFAULT_KNOWN_GOOD_OWNERS),
        "known_malicious_owners": [],
        "warn_owners": [],
        "warn_name_terms": sorted(DEFAULT_WARN_NAME_TERMS),
    }
    if not REPUTATION_CACHE_PATH.exists():
        try:
            REPUTATION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            REPUTATION_CACHE_PATH.write_text(json.dumps(default, indent=2), encoding="utf-8")
        except Exception:
            pass
        return default
    try:
        data = json.loads(REPUTATION_CACHE_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for k, v in default.items():
                data.setdefault(k, v)
            return data
    except Exception:
        pass
    return default


def owner_of(repo_id: str) -> str:
    return repo_id.split("/", 1)[0].lower() if "/" in repo_id else repo_id.lower()


def normalize_owner_name(value: Any) -> str:
    text = str(value).strip().strip("\"'").lower()
    if not text:
        return ""
    text = re.sub(r"^@", "", text)
    text = re.sub(r"^(publisher:|owner:|author:|creator:)", "", text)
    parsed_repo = parse_hf_repo_id(text)
    if parsed_repo:
        return owner_of(parsed_repo)
    if "/" in text and not text.startswith("http"):
        text = text.split("/", 1)[0]
    text = re.sub(r"[^a-z0-9._-]+", "-", text).strip("-")
    return text


def family_key(repo_id: str) -> str:
    """Best-effort model family key for duplicate/mirror detection."""
    name = repo_id.split("/", 1)[-1].lower()
    name = re.sub(r"[-_](gguf|mlx|awq|gptq|bnb|4bit|8bit|fp8|bf16|fp16|f16|q[0-9][a-z0-9_]*|iq[0-9][a-z0-9_]*)($|[-_])", "-", name)
    name = re.sub(r"\b(gguf|mlx|awq|gptq|bnb|4bit|8bit|fp8|bf16|fp16|f16)\b", " ", name)
    name = re.sub(r"[^a-z0-9]+", " ", name)
    stop = {"model", "models", "hf", "the", "repo"}
    toks = [t for t in name.split() if t not in stop]
    return " ".join(toks[:8]) or name


def model_family_key(repo_id: str) -> str:
    """Broad model-family key used for hiding already-installed families.

    This intentionally collapses version, size, quantization, instruct/chat, and
    packaging variants. Examples:
      Qwen/Qwen2.5-Coder-32B-Instruct-GGUF -> qwen coder
      Qwen/Qwen3-Coder-30B-A3B-Instruct -> qwen coder
      deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct -> deepseek coder
    """
    owner = owner_of(repo_id)
    name = repo_id.split("/", 1)[-1].lower()
    combined = f"{owner} {name}"
    combined = combined.replace("__", " ")
    combined = re.sub(r"[^a-z0-9]+", " ", combined)
    tokens = combined.split()

    # Explicit family aliases for high-volume ecosystems. Keep these conservative
    # so excluding one family does not accidentally remove unrelated models.
    has_qwen = any(t == "qwen" or t.startswith("qwen") for t in tokens)
    if has_qwen:
        if any(t in tokens or any(x.startswith(t) for x in tokens) for t in {"coder", "code"}):
            return "qwen coder"
        if any(t in tokens for t in {"vl", "vision", "omni"}):
            return "qwen vision"
        return "qwen"
    has_deepseek = any(t == "deepseek" or t.startswith("deepseek") for t in tokens) or "deepseek-ai" in owner
    if has_deepseek:
        if any(t in tokens or any(x.startswith(t) for x in tokens) for t in {"coder", "code"}):
            return "deepseek coder"
        return "deepseek"
    if "llama" in tokens or owner in {"meta-llama", "codellama"}:
        if "code" in tokens or "codellama" in owner:
            return "codellama"
        return "llama"
    if "mistral" in tokens or owner == "mistralai":
        if "codestral" in tokens:
            return "codestral"
        return "mistral"
    if "gemma" in tokens or owner == "google":
        if "code" in tokens or "coder" in tokens:
            return "gemma code"
        return "gemma"
    if "phi" in tokens and owner == "microsoft":
        return "phi"

    stop = {
        "model", "models", "hf", "huggingface", "gguf", "mlx", "awq", "gptq", "bnb",
        "4bit", "8bit", "fp8", "bf16", "fp16", "f16", "q4", "q5", "q6", "q8",
        "instruct", "chat", "base", "it", "sft", "dpo", "rlhf", "preview", "latest",
        "v1", "v2", "v3", "v4", "v5", "a3b", "moe", "lite", "mini", "small", "large",
    }
    filtered = []
    for tok in tokens:
        if tok in stop:
            continue
        if re.fullmatch(r"v?\d+(\.\d+)*", tok):
            continue
        if re.fullmatch(r"\d+(b|m|k|x)?", tok):
            continue
        if re.fullmatch(r"q\d.*|iq\d.*", tok):
            continue
        filtered.append(tok)
    return " ".join(filtered[:3]) or owner or family_key(repo_id)


def model_family_groups(results: list[SearchResult]) -> dict[str, list[SearchResult]]:
    groups: dict[str, list[SearchResult]] = {}
    for r in results:
        if r.kind == "model":
            groups.setdefault(model_family_key(r.repo_id), []).append(r)
    return groups


def print_top_model_families(results: list[SearchResult], limit: int = 15) -> list[tuple[str, list[SearchResult]]]:
    groups = model_family_groups(results)
    ranked = sorted(
        groups.items(),
        key=lambda kv: (len(kv[1]), sum(x.downloads or 0 for x in kv[1]), sum(x.likes or 0 for x in kv[1])),
        reverse=True,
    )
    if not ranked:
        return []
    print()
    print(f"Top model families in these results (showing up to {min(limit, len(ranked))}):")
    print(f"{'#':>4}  {'COUNT':>5}  {'DOWNLOADS':>12}  {'LIKES':>7}  FAMILY  EXAMPLES")
    print("-" * 120)
    for i, (family, items) in enumerate(ranked[:limit], 1):
        downloads = sum(x.downloads or 0 for x in items)
        likes = sum(x.likes or 0 for x in items)
        examples = ", ".join(x.repo_id for x in sorted(items, key=lambda r: (r.downloads or 0, r.likes or 0), reverse=True)[:3])
        print(f"{i:>4}  {len(items):>5}  {downloads:>12}  {likes:>7}  {family:<20}  {examples}")
    return ranked


def matching_model_families(results: list[SearchResult], needle: str) -> set[str]:
    query = normalize_match_text(needle)
    if not query:
        return set()
    matches: set[str] = set()
    for r in results:
        if r.kind != "model":
            continue
        family = model_family_key(r.repo_id)
        haystacks = [
            normalize_match_text(r.repo_id),
            normalize_match_text(r.title or ""),
            normalize_match_text(family),
        ]
        if any(query in hay for hay in haystacks if hay):
            matches.add(family)
    return matches


def prompt_model_family_exclusion_terms() -> list[str]:
    print("Examples: qwen | qwen coder | deepseek | none")
    value = prompt("Families to exclude", "none").strip()
    return parse_model_family_exclusion_terms(value)


def parse_model_family_exclusion_terms(value: str) -> list[str]:
    if not value or value.lower() in {"none", "n", "no", "skip"}:
        return []
    terms: list[str] = []
    for part in re.split(r"[,;]+", value):
        item = part.strip()
        if item:
            terms.append(item)
    return terms


def apply_model_family_exclusions(results: list[SearchResult], terms: list[str], quiet_no_match: bool = False) -> list[SearchResult]:
    if not terms:
        return results
    exclude: set[str] = set()
    for term in terms:
        exclude.update(matching_model_families(results, term))
    if not exclude:
        if not quiet_no_match:
            print("No matching families were excluded.")
        return results
    filtered = [r for r in results if r.kind != "model" or model_family_key(r.repo_id) not in exclude]
    print(f"Excluded {len(results) - len(filtered)} result(s) from family/families: {', '.join(sorted(exclude))}")
    return filtered


def exclude_model_families_interactive(results: list[SearchResult]) -> list[SearchResult]:
    ranked = print_top_model_families(results)
    if not ranked:
        return results
    print()
    print("Exclude whole model families by number, family label, or model name.")
    print("Examples: 1,3 | qwen coder | qwen | none")
    value = prompt("Families to exclude", "none").strip()
    if not value or value.lower() in {"none", "n", "no", "skip"}:
        return results

    exclude: set[str] = set()
    by_number = {str(i): family for i, (family, _) in enumerate(ranked, 1)}
    by_name = {family.lower(): family for family, _ in ranked}
    for part in re.split(r"[,;]+", value):
        item = part.strip().lower()
        if not item:
            continue
        if item in by_number:
            exclude.add(by_number[item])
            continue
        matches = matching_model_families(results, item)
        if item in by_name:
            matches.add(by_name[item])
        exclude.update(matches)

    if not exclude:
        print("No matching families were excluded.")
        return results
    filtered = [r for r in results if r.kind != "model" or model_family_key(r.repo_id) not in exclude]
    print(f"Excluded {len(results) - len(filtered)} result(s) from family/families: {', '.join(sorted(exclude))}")
    return filtered


def annotate_recommendations(results: list[SearchResult]) -> None:
    cfg = load_reputation_config()
    reputation = load_owner_reputation_sets()
    good = reputation["known_good_owners"]
    bad = reputation["known_malicious_owners"]
    warn_owners = reputation["warn_owners"]
    warn_terms = {str(x).lower() for x in cfg.get("warn_name_terms", [])}

    groups: dict[str, list[SearchResult]] = {}
    for r in results:
        if r.kind == "model":
            key = family_key(r.repo_id)
            r.duplicate_family = key
            groups.setdefault(key, []).append(r)

    for group in groups.values():
        if len(group) > 1:
            for r in group:
                r.duplicate_group_size = len(group)
                others = [x.repo_id for x in group if x.repo_id != r.repo_id][:4]
                r.notes.append(f"possible duplicate/mirror family with {len(group) - 1} other repo(s): {', '.join(others)}")

    for r in results:
        owner = owner_of(r.repo_id)
        lower_name = r.repo_id.lower()
        rec = "Unknown"
        if r.leaderboard_rank is not None:
            rec = "Leaderboard"
        elif owner in bad:
            rec = "BlockOwner"
            r.notes.append("owner appears in local known_malicious_owners list")
        elif owner in warn_owners:
            rec = "WarnOwner"
            r.notes.append("owner appears in local warn_owners list")
        elif any(term and term in lower_name for term in warn_terms):
            rec = "WarnName"
            r.notes.append("repo name contains a term from local warn_name_terms")
        elif owner in good:
            rec = "KnownGood"
        elif (r.downloads or 0) < 50 and (r.likes or 0) < 2 and (r.size_bytes or 0) > 1024**3:
            rec = "LowSignal"
            r.notes.append("large repo with very low downloads/likes; review card and files carefully")
        if r.duplicate_group_size > 1 and rec not in {"BlockOwner", "WarnOwner", "WarnName"}:
            if rec == "Unknown":
                rec = "Duplicate Families"
            elif rec == "KnownGood":
                rec = "KnownGood/DF"
        r.recommendation = rec


def hide_duplicate_families(results: list[SearchResult]) -> list[SearchResult]:
    """Keep the strongest candidate in each likely duplicate family, preserving non-models."""
    reputation = load_owner_reputation_sets()
    good_owners = reputation["known_good_owners"]
    bad_owners = reputation["known_malicious_owners"]
    warn_owners = reputation["warn_owners"]
    best: dict[str, SearchResult] = {}
    hidden: dict[str, list[SearchResult]] = {}
    passthrough: list[SearchResult] = []
    for r in results:
        if r.kind != "model" or not r.duplicate_family:
            passthrough.append(r)
            continue
        key = r.duplicate_family
        owner = owner_of(r.repo_id)
        score = (
            0 if owner in bad_owners else 1,
            0 if owner in warn_owners else 1,
            1 if owner in good_owners else 0,
            1 if r.leaderboard_rank is not None else 0,
            r.downloads or 0,
            r.likes or 0,
            r.size_bytes or 0,
        )
        cur = best.get(key)
        if cur is None:
            best[key] = r
            continue
        cur_owner = owner_of(cur.repo_id)
        cur_score = (
            0 if cur_owner in bad_owners else 1,
            0 if cur_owner in warn_owners else 1,
            1 if cur_owner in good_owners else 0,
            1 if cur.leaderboard_rank is not None else 0,
            cur.downloads or 0,
            cur.likes or 0,
            cur.size_bytes or 0,
        )
        if score > cur_score:
            hidden.setdefault(key, []).append(cur)
            best[key] = r
        else:
            hidden.setdefault(key, []).append(r)
    kept = passthrough + list(best.values())
    for key, items in hidden.items():
        if key in best:
            names = [x.repo_id for x in items[:5]]
            best[key].notes.append(f"hidden {len(items)} likely duplicate repo(s): {', '.join(names)}")
    return kept


def duplicate_family_reputation_recommendation(results: list[SearchResult]) -> dict[str, Any]:
    merged = merge_search_results(results)
    groups: dict[str, list[SearchResult]] = {}
    for result in merged:
        if result.kind == "model":
            groups.setdefault(family_key(result.repo_id), []).append(result)
    duplicate_groups = {family: items for family, items in groups.items() if len(items) > 1}
    reputation = load_owner_reputation_sets()
    good_owners = reputation["known_good_owners"]
    bad_owners = reputation["known_malicious_owners"]
    warn_owners = reputation["warn_owners"]

    preferred: set[str] = set()
    risky: set[str] = set()
    caution: set[str] = set()
    for items in duplicate_groups.values():
        owners = {owner_of(item.repo_id) for item in items}
        preferred.update(owner for owner in owners if owner in good_owners)
        risky.update(owner for owner in owners if owner in bad_owners)
        caution.update(owner for owner in owners if owner in warn_owners and owner not in bad_owners)

    return {
        "duplicate_group_count": len(duplicate_groups),
        "preferred_owners": sorted(preferred),
        "risky_owners": sorted(risky),
        "warn_owners": sorted(caution),
        "default_hide": bool(preferred or risky or caution),
    }


def prompt_hide_duplicate_families_preference(results: list[SearchResult]) -> bool:
    suggestion = duplicate_family_reputation_recommendation(results)
    duplicate_group_count = int(suggestion["duplicate_group_count"])
    if duplicate_group_count <= 0:
        return False

    print()
    print(f"Found {duplicate_group_count} likely duplicate/mirror family group(s) in the current results.")
    preferred_owners = suggestion["preferred_owners"]
    risky_owners = suggestion["risky_owners"]
    warn_owners = suggestion["warn_owners"]
    if preferred_owners or risky_owners or warn_owners:
        print("Local author reputation hint:")
        if preferred_owners:
            print("  Known-good owners in duplicate families: " + ", ".join(preferred_owners[:8]))
        if risky_owners:
            print("  Owners flagged malicious in duplicate families: " + ", ".join(risky_owners[:8]))
        elif warn_owners:
            print("  Owners flagged warn/suspicious in duplicate families: " + ", ".join(warn_owners[:8]))
        if suggestion["default_hide"]:
            print("  Recommendation: yes, hide duplicates to prefer higher-trust owners.")
    return prompt_bool(
        "Hide likely duplicate/mirror repos by exact variant family?",
        bool(suggestion["default_hide"]),
    )


def result_size_label(result: SearchResult) -> str:
    if result.kind == "model" and result.visible_artifacts:
        sizes = [a.total_size for a in result.visible_artifacts if a.total_size]
        if sizes:
            lo = min(sizes)
            hi = max(sizes)
            if lo == hi:
                return human_size(lo)
            return f"{human_size(lo)}-{human_size(hi)}"
    return human_size(result.size_bytes)


def selection_queue_size_label(result: SearchResult) -> str:
    if result.selected_artifacts:
        sizes = [a.total_size for a in result.selected_artifacts if a.total_size]
        if sizes:
            lo = min(sizes)
            hi = max(sizes)
            if lo == hi:
                return human_size(lo)
            return f"{human_size(lo)}-{human_size(hi)}"
    return result_size_label(result)


def selection_queue_detail(result: SearchResult) -> str:
    if result.selected_artifacts:
        return ", ".join(f"{result.index}{alpha_label(a.index)} {a.label}" for a in result.selected_artifacts)
    if result.whole_repo_selected:
        return "WHOLE REPO"
    return ""


def clear_selection_state(result: SearchResult) -> None:
    result.selected_artifacts.clear()
    result.whole_repo_selected = False


def review_download_queue(results: list[SearchResult]) -> list[SearchResult]:
    queue = list(results)
    while queue:
        print()
        print("Selections queued for download:")
        for pos, result in enumerate(queue, 1):
            size_label = selection_queue_size_label(result)
            detail = selection_queue_detail(result)
            print(f"  {pos}. [repo {result.index}] {result.source}:{result.kind} {size_label} {result.repo_id}")
            if detail:
                print(f"      selected: {detail}")
        if not prompt_bool("Do you want to remove any selections before download?", False):
            return queue

        raw = read_input("Queue number(s) to remove, or 'all': ").strip().lower()
        if not raw or raw in {"none", "n", "no", "skip"}:
            print("No selections removed.")
            continue
        if raw == "all":
            for result in queue:
                clear_selection_state(result)
            print("Removed all queued selections.")
            return []

        nums = parse_selection(raw, len(queue))
        if not nums:
            print("No valid queue numbers entered. Nothing removed.")
            continue

        remove_set = set(nums)
        kept: list[SearchResult] = []
        removed_labels: list[str] = []
        for pos, result in enumerate(queue, 1):
            if pos in remove_set:
                removed_labels.append(result.repo_id)
                clear_selection_state(result)
            else:
                kept.append(result)
        queue = kept
        print(f"Removed {len(removed_labels)} selection(s): {', '.join(removed_labels)}")
    return []


def show_artifact_picker_for_repo(result: SearchResult) -> bool:
    """Inline artifact picker used when a repo number is selected from the result list."""
    if result.source != "hf" or result.kind != "model":
        return True
    refresh_hf_file_metadata(result)
    artifacts = artifacts_for_display(result)
    if not artifacts:
        print(f"Repo {result.index}: {result.repo_id} has no discovered direct model artifacts from file metadata.")
        print("Whole-repository fallback is disabled by default for HF model searches to avoid pulling full repos by accident.")
        return prompt_bool("Select whole repository anyway?", False)
    print()
    print(f"Repo {result.index}: {result.repo_id}")
    print(f"Discovered {len(artifacts)} downloadable artifact(s).")
    print(f"{'#':>4}  {'SIZE':>12}  {'TYPE':<22}  {'QUANT':<10}  ARTIFACT")
    print("-" * 100)
    for art in artifacts:
        quant = art.quant or "-"
        first = art.files[0][0] if art.files else art.label
        extra = f" +{len(art.files)-1} files" if len(art.files) > 1 else ""
        print(f"{art.index:>4}  {human_size(art.total_size):>12}  {art.artifact_type:<22}  {quant:<10}  {first}{extra}")
    print()
    mode = prompt_choice("Select from this repo", ["artifact_numbers", "whole_repo", "skip"], "artifact_numbers")
    if mode == "skip":
        return False
    if mode == "whole_repo":
        if prompt_bool(f"Whole repo is {human_size(result.size_bytes)}. Select whole repo?", False):
            result.whole_repo_selected = True
            return True
        return False
    nums = parse_selection(prompt("Which artifact numbers? Examples: 1, 1,3, 2-4, all"), len(artifacts))
    if not nums:
        return False
    for n in nums:
        art = artifacts[n - 1]
        if all(existing.files != art.files for existing in result.selected_artifacts):
            result.selected_artifacts.append(art)
            print(f"Selected: {result.index}{alpha_label(art.index)} {art.label} ({human_size(art.total_size)})")
    return True

def print_results_page(
    results: list[SearchResult],
    start: int,
    page_size: int,
    show_files: bool = True,
    top_files: int = 7,
    artifact_offsets: dict[int, int] | None = None,
) -> None:
    end = min(start + page_size, len(results))
    rec_width = 18
    print()
    print(f"Results {start + 1}-{end} of {len(results)}")
    print(f"{'#':>4}  {'SRC':<7}  {'KIND':<7}  {'FAMILY':<18}  {'LB':<16}  {'REC':<{rec_width}}  {'SIZE':>12}  {'DL':>10}  {'LIKES':>7}  {'TYPE':<22}  {'LICENSE':<14}  REPO")
    print("-" * 192)
    for r in results[start:end]:
        lb = r.leaderboard_label or "-"
        rec = r.recommendation or "-"
        fam = model_family_key(r.repo_id) if r.kind == "model" else "-"
        print(f"{r.index:>4}  {r.source:<7}  {r.kind:<7}  {fam:<18.18}  {lb:<16.16}  {rec:<{rec_width}.{rec_width}}  {result_size_label(r):>12}  {str(r.downloads if r.downloads is not None else '-'):>10}  {str(r.likes if r.likes is not None else '-'):>7}  {(r.pipeline or '-'):<22.22}  {(r.license or '-'):<14.14}  {r.repo_id}")
        if r.title:
            print(f"      title: {r.title}")
        if r.notes:
            for note in r.notes:
                print(f"      note: {note}")
        if r.selected_artifacts:
            labels = ", ".join(f"{r.index}{alpha_label(a.index)} {a.label}" for a in r.selected_artifacts)
            print(f"      selected: {labels}")
        elif r.whole_repo_selected:
            print("      selected: WHOLE REPO")
        if show_files:
            if r.source == "hf" and r.kind == "model":
                refresh_hf_file_metadata(r)
                artifacts = artifacts_for_display(r)
                if artifacts:
                    offset = 0
                    if artifact_offsets is not None:
                        max_offset = max(0, ((len(artifacts) - 1) // top_files) * top_files)
                        offset = min(max(0, artifact_offsets.get(r.index, 0)), max_offset)
                    shown_artifacts = artifacts[offset: offset + top_files]
                    print("      direct artifacts — select these with IDs like 1A/1B; repo number opens the artifact picker")
                    for art in shown_artifacts:
                        child_id = f"{r.index}{alpha_label(art.index)}"
                        quant = art.quant or "-"
                        first_file = art.files[0][0] if art.files else art.label
                        extra = f" +{len(art.files) - 1} files" if len(art.files) > 1 else ""
                        print(f"   {child_id + '.':<7} {human_size(art.total_size):>12}  {art.artifact_type:<20} {quant:<10} {first_file}{extra}")
                    if len(artifacts) > top_files:
                        shown_start = offset + 1
                        shown_end = min(len(artifacts), offset + top_files)
                        print(f"           showing artifacts {shown_start}-{shown_end} of {len(artifacts)}")
                        if shown_end < len(artifacts):
                            print(f"           ... {len(artifacts) - shown_end} more artifacts; press n{r.index} for more here or select repo {r.index} to see all")
                        elif offset > 0:
                            print(f"           ... press p{r.index} to go back or select repo {r.index} to see all")
                elif r.files:
                    print("      no direct GGUF/Core ML/etc. artifact detected in current file metadata; sample files:")
                    for fname, fsize in r.files[:top_files]:
                        print(f"      {human_size(fsize):>12}  {fname}")
                else:
                    print("      no HF file metadata available yet; selecting repo number will retry metadata before whole-repo fallback")
            elif r.files:
                for fname, fsize in r.files[:top_files]:
                    print(f"      {human_size(fsize):>12}  {fname}")
            print()


def paged_select(results: list[SearchResult], page_size: int = 20) -> list[SearchResult]:
    pos = 0
    selected: set[int] = set()
    excluded: set[int] = set()
    artifact_offsets: dict[int, int] = {}
    while True:
        browseable = [r for r in results if r.index not in excluded]
        unselected = [r for r in browseable if r.index not in selected]
        debug_log(
            "paged-select-loop",
            pos=pos,
            page_size=page_size,
            total_results=len(results),
            browseable_indexes=[r.index for r in browseable],
            unselected_indexes=[r.index for r in unselected],
            selected_indexes=sorted(selected),
            excluded_indexes=sorted(excluded),
            artifact_offsets=artifact_offsets,
        )
        if not browseable:
            if selected:
                print("No results remain visible. Continuing to the download step for the current selection cart.")
            else:
                print("No results remain visible.")
            break
        if not unselected and selected:
            print("All visible results are already in the selection cart. Continuing to the download step for the current selection cart.")
            break
        max_pos = max(0, ((len(browseable) - 1) // page_size) * page_size)
        pos = min(pos, max_pos)
        current_page = browseable[pos:min(pos + page_size, len(browseable))]
        artifact_page_targets: list[tuple[SearchResult, int]] = []
        for candidate in current_page:
            if candidate.source != "hf" or candidate.kind != "model":
                continue
            refresh_hf_file_metadata(candidate)
            artifacts = artifacts_for_display(candidate)
            if len(artifacts) <= 7:
                continue
            max_offset = max(0, ((len(artifacts) - 1) // 7) * 7)
            artifact_offsets[candidate.index] = min(max(0, artifact_offsets.get(candidate.index, 0)), max_offset)
            artifact_page_targets.append((candidate, max_offset))
        print_results_page(browseable, pos, page_size, artifact_offsets=artifact_offsets)
        if selected:
            print(f"Selection cart: {len(selected)} repo(s) selected. Selected repos stay visible and are marked in the list; use `cart` or `remove` to adjust.")
            print("Press Enter / type `done` to start downloading, or keep browsing the full results.")
        if pos + page_size >= len(browseable):
            if excluded:
                print(f"This is all of the currently visible models. {len(excluded)} result(s) are excluded from view.")
            else:
                print("This is all of the models the search found.")
        if artifact_page_targets:
            pageable_ids = ", ".join(str(r.index) for r, _ in artifact_page_targets)
            print(f"More artifact rows are available on this screen for repo(s): {pageable_ids}. Use n<repo> / p<repo>, for example n{artifact_page_targets[0][0].index}.")
            print("Commands: artifact IDs like 1A,2C | repo numbers open artifact picker | whole 1 | cart/remove/clear | xpub qwen | xf 7 / xfam qwen coder | n/p changes result pages | n2/p2 page artifact rows for repo 2 | Enter/done starts download | q")
        else:
            print("Commands: artifact IDs like 1A,2C | repo numbers open artifact picker | whole 1 | cart/remove/clear | xpub qwen | xf 7 / xfam qwen coder | n/p | Enter/done starts download | q")
        prompt_label = (
            "Add more repo/artifact IDs, or press Enter / type 'done' to start download"
            if selected else
            "Which repo number(s), artifact ID(s), or family exclusion do you want?"
        )
        cmd = read_input(f"{prompt_label} ").strip()
        cmd_lower = cmd.lower()
        debug_log("paged-select-command", cmd=cmd, cmd_lower=cmd_lower, pos=pos, selected_indexes=sorted(selected))
        if not cmd_lower and selected:
            break
        if cmd_lower in {"q", "quit", "exit"}:
            return []
        if cmd_lower in {"done", "d"}:
            break
        artifact_page_cmd = parse_artifact_page_command(cmd_lower)
        if artifact_page_cmd:
            direction, repo_num = artifact_page_cmd
            target_info = next(((target, target_max_offset) for target, target_max_offset in artifact_page_targets if target.index == repo_num), None)
            if target_info is None:
                print(f"Repo {repo_num} on this page does not have additional hidden artifact rows. Use the repo number to open its full artifact picker.")
                continue
            target, target_max_offset = target_info
            current_offset = artifact_offsets.get(repo_num, 0)
            debug_log(
                "artifact-page-command",
                direction=direction,
                repo_num=repo_num,
                current_offset=current_offset,
                target_max_offset=target_max_offset,
            )
            if direction == "next":
                if current_offset < target_max_offset:
                    artifact_offsets[repo_num] = min(target_max_offset, current_offset + 7)
                    print(f"Showing more artifact rows for repo {repo_num}.")
                else:
                    print(f"Already showing the last artifact rows for repo {repo_num}.")
            else:
                if current_offset > 0:
                    artifact_offsets[repo_num] = max(0, current_offset - 7)
                    print(f"Showing earlier artifact rows for repo {repo_num}.")
                else:
                    print(f"Already showing the first artifact rows for repo {repo_num}.")
            continue
        if cmd_lower in {"cart", "selected", "selection"}:
            cart = [r for r in results if r.index in selected]
            if not cart:
                print("Selection cart is empty.")
            else:
                print("Selection cart:")
                for r in cart:
                    if r.selected_artifacts:
                        labels = ", ".join(f"{r.index}{alpha_label(a.index)} {a.label}" for a in r.selected_artifacts)
                        print(f"  {r.index}. {r.repo_id}: {labels}")
                    elif r.whole_repo_selected:
                        print(f"  {r.index}. {r.repo_id}: WHOLE REPO")
                    else:
                        print(f"  {r.index}. {r.repo_id}")
            continue
        if cmd_lower.startswith("remove "):
            nums = parse_selection(cmd_lower.split(maxsplit=1)[1], len(results))
            for n in nums:
                if 1 <= n <= len(results):
                    selected.discard(n)
                    results[n - 1].selected_artifacts.clear()
                    results[n - 1].whole_repo_selected = False
            print(f"Selection cart now has {len(selected)} item(s).")
            continue
        if cmd_lower in {"clear", "clear-cart"}:
            for r in results:
                r.selected_artifacts.clear()
                r.whole_repo_selected = False
            selected.clear()
            print("Selection cart cleared.")
            continue
        if cmd_lower in {"n", "next"}:
            if pos >= max_pos:
                if excluded:
                    print(f"This is all of the currently visible models. {len(excluded)} result(s) are excluded from view.")
                else:
                    print("This is all of the models the search found.")
                if artifact_page_targets:
                    print(f"Use n<repo>/p<repo> for artifact rows on repo(s): {pageable_ids}.")
                else:
                    print("Use artifact IDs/repo numbers, press Enter for download, or 'q'.")
                debug_log("result-page-next-blocked", pos=pos, max_pos=max_pos, pageable_ids=pageable_ids if artifact_page_targets else [])
            else:
                pos = min(max_pos, pos + page_size)
                debug_log("result-page-next", new_pos=pos, max_pos=max_pos)
            continue
        if cmd_lower in {"p", "prev", "previous"}:
            if pos <= 0:
                if artifact_page_targets:
                    print(f"Already at the first result page. Use n<repo>/p<repo> for artifact rows on repo(s): {pageable_ids}.")
                else:
                    print("Already at the first page.")
                debug_log("result-page-prev-blocked", pos=pos, pageable_ids=pageable_ids if artifact_page_targets else [])
            else:
                pos = max(0, pos - page_size)
                debug_log("result-page-prev", new_pos=pos)
            continue
        if cmd_lower in {"families", "family", "top"}:
            print_top_model_families(browseable)
            continue
        if cmd_lower.startswith("xf ") or cmd_lower.startswith("exclude-family "):
            token = cmd_lower.split(maxsplit=1)[1].strip()
            fams: set[str] = set()
            for n in parse_selection(token, len(results)):
                if 1 <= n <= len(results) and results[n - 1].kind == "model":
                    fams.add(model_family_key(results[n - 1].repo_id))
            if fams:
                family_members = [r.index for r in results if r.kind == "model" and model_family_key(r.repo_id) in fams]
                excluded.update(family_members)
                print(f"Excluded family/families from selection view: {', '.join(sorted(fams))} ({len(family_members)} result(s))")
            else:
                print("No matching model family found for that repo number.")
            pos = 0
            continue
        if cmd_lower.startswith("xpub ") or cmd_lower.startswith("exclude-publisher ") or cmd_lower.startswith("exclude-owner "):
            token = cmd_lower.split(maxsplit=1)[1].strip()
            owners = {part.strip().lower().lstrip("@").split("/", 1)[0] for part in re.split(r"[,\s]+", token) if part.strip()}
            if owners:
                owner_members = [r.index for r in results if owner_of(r.repo_id) in owners or any(owner_of(r.repo_id).startswith(o) or o in owner_of(r.repo_id) for o in owners)]
                excluded.update(owner_members)
                print(f"Excluded publisher/owner(s) from selection view: {', '.join(sorted(owners))} ({len(owner_members)} result(s))")
            else:
                print("No publisher/owner tokens found.")
            pos = 0
            continue

        if cmd_lower.startswith("xfam "):
            needle = cmd_lower.split(maxsplit=1)[1].strip()
            matches = matching_model_families(results, needle)
            if matches:
                family_members = [r.index for r in results if r.kind == "model" and model_family_key(r.repo_id) in matches]
                excluded.update(family_members)
                print(f"Excluded family/families from selection view: {', '.join(sorted(matches))} ({len(family_members)} result(s))")
            else:
                print("No matching model family found for that family label/model name.")
            pos = 0
            continue

        if cmd_lower.startswith("whole "):
            nums = parse_selection(cmd_lower.split(maxsplit=1)[1], len(results))
            for n in nums:
                if 1 <= n <= len(results):
                    r = results[n - 1]
                    if prompt_bool(f"Select WHOLE repo {n}: {r.repo_id} ({human_size(r.size_bytes)})?", False):
                        selected.add(r.index)
            print("Added to cart. Press Enter or type 'done' to start the download step, or enter more IDs to add.")
            continue

        artifact_tokens = parse_repo_artifact_tokens(cmd, len(results))
        if artifact_tokens:
            debug_log("artifact-token-selection", artifact_tokens=artifact_tokens)
            for repo_num, art_num in artifact_tokens:
                r = results[repo_num - 1]
                artifacts = artifacts_for_display(r)
                if not artifacts:
                    refresh_hf_file_metadata(r)
                    artifacts = artifacts_for_display(r)
                if not artifacts:
                    print(f"Repo {repo_num} has no discovered direct artifacts. Whole-repo selection requires selecting repo number {repo_num} and confirming.")
                    continue
                if not (1 <= art_num <= len(artifacts)):
                    print(f"Artifact {repo_num}{alpha_label(art_num)} is out of range for repo {repo_num}; repo has {len(artifacts)} artifacts.")
                    continue
                art = artifacts[art_num - 1]
                if all(existing.files != art.files for existing in r.selected_artifacts):
                    r.selected_artifacts.append(art)
                selected.add(repo_num)
                print(f"Selected artifact {repo_num}{alpha_label(art_num)}: {r.repo_id} -> {art.label} ({human_size(art.total_size)})")
            print("Added to cart. Press Enter or type 'done' to start the download step, or enter more IDs to add.")
            continue

        if cmd_lower == "all":
            nums = [r.index for r in browseable if r.index not in selected]
        else:
            nums = parse_selection(cmd_lower, len(results))
        if nums:
            debug_log("repo-number-selection", nums=nums)
            for n in nums:
                if not (1 <= n <= len(results)):
                    continue
                r = results[n - 1]
                keep = show_artifact_picker_for_repo(r)
                if keep:
                    selected.add(r.index)
            print(f"Selection cart: {', '.join(str(n) for n in sorted(selected))}")
            print("Press Enter or type 'done' to start the download step, or enter more IDs to add.")
            continue
    return [r for r in results if r.index in selected]

# -----------------------------------------------------------------------------
# Local similarity scan
# -----------------------------------------------------------------------------

def normalize_name(s: str) -> str:
    s = s.lower().replace("__", "/")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    stop = {"gguf", "safetensors", "model", "models", "hf", "huggingface", "snapshots", "blobs", "main", "q4", "q5", "q6", "q8"}
    return " ".join(tok for tok in s.split() if tok not in stop)


def scan_local_candidates(target_names: Iterable[str], roots: list[Path]) -> dict[str, list[Path]]:
    needles = {name: set(normalize_name(name).split()) for name in target_names}
    hits: dict[str, list[Path]] = {name: [] for name in target_names}
    for root in roots:
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            try:
                if not (p.is_dir() or p.suffix.lower() in MODEL_FILE_EXTENSIONS):
                    continue
            except OSError:
                continue
            hay = set(normalize_name(str(p)).split())
            for name, toks in needles.items():
                if toks and len(toks & hay) >= min(3, len(toks)):
                    hits[name].append(p)
    return hits


def offer_local_scan(selected: list[SearchResult]) -> None:
    models = [r for r in selected if r.kind == "model"]
    if not models or not prompt_bool("Scan local model directories for same/similar names before download?", True):
        return
    roots = DEFAULT_LOCAL_MODEL_DIRS
    print("Scanning:")
    for r in roots:
        print(f"  {r}{'' if r.is_dir() else ' (missing)'}")
    hits = scan_local_candidates([r.repo_id for r in models], roots)
    print()
    print("Local similarity scan results:")
    for repo_id, paths in hits.items():
        print(f"\n{repo_id}")
        if not paths:
            print("  no obvious local match")
            continue
        for p in paths[:25]:
            try:
                size = p.stat().st_size if p.is_file() else 0
            except OSError:
                size = 0
            tag = human_size(size) if size else "dir"
            print(f"  [{tag:>10}] {p}")
        if len(paths) > 25:
            print(f"  ... {len(paths) - 25} more")

# -----------------------------------------------------------------------------
# Audit/security
# -----------------------------------------------------------------------------

def is_lfs_pointer(path: Path) -> bool:
    try:
        if path.stat().st_size > 4096:
            return False
        with path.open("rb") as f:
            return f.read(len(LFS_SIGNATURE)) == LFS_SIGNATURE
    except OSError:
        return False


def classify_message(msg: str) -> str:
    lower = msg.lower()
    if any(x in lower for x in ["missing expected", "size mismatch", "lfs pointer", "bad gguf", "missing safetensors shard", "missing gguf shard", "validation failed", "target missing"]):
        return "BLOCKER"
    if any(x in lower for x in ["scanner reported", "known malware", "malicious", "remote execution", "obfuscated", "unexpected executable binary"]):
        return "DANGER"
    if any(x in lower for x in ["pickle", ".pt", ".pth", ".pkl", "setup.py", "requirements.txt", "custom python", "modeling_", "tokenization_", "shell script"]):
        return "WARN"
    return "INFO"


def verify_download_integrity(target: Path, expected_files: list[tuple[str, int | None]] | None = None, allow_patterns: list[str] | None = None) -> list[dict[str, str]]:
    messages: list[str] = []
    if not target.exists():
        return [{"severity": "BLOCKER", "message": f"target missing: {target}"}]

    expected = expected_files or []
    if allow_patterns:
        expected = [(n, s) for n, s in expected if any(fnmatch.fnmatch(n, pat) for pat in allow_patterns)]

    for rel, expected_size in expected:
        p = target / rel
        if not p.exists():
            messages.append(f"missing expected file: {rel}")
            continue
        if expected_size is not None:
            try:
                actual = p.stat().st_size
                if actual != expected_size:
                    messages.append(f"size mismatch: {rel}: local {actual}, expected {expected_size}")
            except OSError as e:
                messages.append(f"stat failed: {rel}: {e}")

    for p in target.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(target)
        if is_lfs_pointer(p):
            messages.append(f"LFS pointer instead of real file: {rel}")
        suffix = p.suffix.lower()
        name = p.name.lower()
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if suffix == ".gguf":
            if size < 50 * 1024 * 1024:
                messages.append(f"suspiciously small GGUF: {rel} ({human_size(size)})")
            try:
                with p.open("rb") as f:
                    if f.read(4) != GGUF_MAGIC:
                        messages.append(f"bad GGUF magic: {rel}")
            except OSError as e:
                messages.append(f"cannot read GGUF: {rel}: {e}")
        elif suffix == ".safetensors":
            if size < 1024:
                messages.append(f"suspiciously small safetensors: {rel} ({human_size(size)})")
        elif suffix in {".pt", ".pth", ".pkl", ".pickle", ".bin"}:
            messages.append(f"pickle/deserialization risk file present: {rel}")
        elif suffix in {".sh", ".command"}:
            messages.append(f"shell script present; review before executing: {rel}")
        elif name in {"setup.py", "requirements.txt"} or name.startswith(("modeling_", "tokenization_")):
            messages.append(f"custom Python/dependency file present; review before trust_remote_code/import: {rel}")

    for idx in target.rglob("*.safetensors.index.json"):
        try:
            data = json.loads(idx.read_text())
            weight_map = data.get("weight_map", {})
            for shard in set(weight_map.values()):
                if not (idx.parent / shard).exists():
                    messages.append(f"missing safetensors shard referenced by {idx.relative_to(target)}: {shard}")
        except Exception as e:
            messages.append(f"cannot parse index {idx.relative_to(target)}: {type(e).__name__}: {e}")

    groups: dict[tuple[str, str], set[int]] = {}
    totals: dict[tuple[str, str], int] = {}
    for p in target.rglob("*.gguf"):
        m = re.search(r"(.+?)-0*(\d+)-of-0*(\d+)\.gguf$", p.name, re.I)
        if not m:
            continue
        base, part, total = m.group(1), int(m.group(2)), int(m.group(3))
        key = (str(p.parent), base)
        groups.setdefault(key, set()).add(part)
        totals[key] = total
    for key, parts in groups.items():
        total = totals[key]
        missing = sorted(set(range(1, total + 1)) - parts)
        if missing:
            messages.append(f"missing GGUF shard(s) for {key[1]}: {missing}")

    try:
        from safetensors import safe_open
        for st in target.rglob("*.safetensors"):
            try:
                with safe_open(str(st), framework="pt", device="cpu") as f:
                    _ = list(f.keys())[:1]
            except Exception as e:
                messages.append(f"safetensors validation failed: {st.relative_to(target)}: {type(e).__name__}: {e}")
    except Exception:
        pass

    return [{"severity": classify_message(m), "message": m} for m in messages]


def print_findings(findings: list[dict[str, str]]) -> None:
    if not findings:
        print("Audit: no findings.")
        return
    print("Audit findings:")
    for f in findings:
        print(f"  [{f['severity']}] {f['message']}")


def should_offer_delete(findings: list[dict[str, str]]) -> tuple[bool, str]:
    serious = [f["message"] for f in findings if f.get("severity") in {"DANGER", "BLOCKER"}]
    if not serious:
        return False, ""
    return True, "; ".join(serious[:5])


def delete_path_after_confirmation(path: Path, reason: str) -> bool:
    path = path.expanduser().resolve()
    if not path.exists():
        print(f"Nothing to delete; path does not exist: {path}")
        return False
    print()
    print("Potentially harmful or invalid download detected.")
    print(f"Path: {path}")
    print(f"Reason: {reason}")
    print("Deletion is permanent for normal filesystem use.")
    confirm = read_input("Delete this downloaded item? [y/N]: ").strip().lower()
    if confirm not in {"y", "yes"}:
        print("Keeping files.")
        return False
    if path.is_file() or path.is_symlink():
        path.unlink()
        print(f"Deleted file: {path}")
        return True
    if path.is_dir():
        shutil.rmtree(path)
        print(f"Deleted directory: {path}")
        return True
    print(f"Unsupported path type, not deleted: {path}")
    return False


def _venv_python(path: Path) -> Path | None:
    py = path / ".venv" / "bin" / "python"
    if py.is_file() and os.access(py, os.X_OK):
        return py
    return None


def _named_scanner_command(path: Path, target: Path) -> ScannerCommand | None:
    name = path.name.lower()
    if name == "modelaudit":
        py = _venv_python(path)
        if py:
            return ScannerCommand(
                "modelaudit",
                [str(py), "-m", "modelaudit", "scan", str(target), "--format", "json", "--max-size", "10GB", "--timeout", "900"],
                cwd=path,
            )
        exe = shutil.which("modelaudit")
        if exe:
            return ScannerCommand(
                "modelaudit",
                [exe, "scan", str(target), "--format", "json", "--max-size", "10GB", "--timeout", "900"],
            )
        return None

    if name in {"modelscan", "model-scan"}:
        project = path if name == "modelscan" else MODEL_TOOLS_DIR / "modelscan"
        py = _venv_python(project)
        if py:
            return ScannerCommand(
                name,
                [str(py), "-m", "modelscan.cli", "scan", "-p", str(target), "-r", "json"],
                cwd=project,
            )
        exe = shutil.which("modelscan")
        if exe:
            return ScannerCommand(name, [exe, "scan", "-p", str(target), "-r", "json"])
        return None

    if name == "modelguard":
        cli = path / "modelguard_cli.py"
        if cli.is_file():
            return ScannerCommand("ModelGuard", [sys.executable, str(cli), str(target)], cwd=path)
        return None

    if name == "palisade-scan":
        exe = shutil.which("palisade")
        if exe:
            return ScannerCommand("palisade-scan", [exe, "scan", str(target), "--format", "json"], cwd=path)
        return None

    if name == "skill-scanner":
        scan = path / "scan.sh"
        if scan.is_file() and os.access(scan, os.X_OK):
            return ScannerCommand("skill-scanner", [str(scan), "--workers", "1", "--quiet", str(target)], cwd=path)
        return None

    if name == "skillcheck":
        exe = path / "skillcheck"
        if exe.is_file() and os.access(exe, os.X_OK):
            return ScannerCommand("skillcheck", [str(exe), "--verbose", str(target)], cwd=path)
        wrapper = path / "start.sh"
        if wrapper.is_file() and os.access(wrapper, os.X_OK):
            return ScannerCommand("skillcheck", [str(wrapper), "--verbose", str(target)], cwd=path)
        return None

    return None


def find_tool_command(path: Path, target: Path) -> ScannerCommand | None:
    if not path.exists():
        return None
    named = _named_scanner_command(path, target)
    if named:
        return named
    if path.is_file():
        if os.access(path, os.X_OK):
            return ScannerCommand(path.name, [str(path), str(target)], cwd=path.parent)
        if path.suffix == ".py":
            return ScannerCommand(path.name, [sys.executable, str(path), str(target)], cwd=path.parent)
        return None
    if path.is_dir():
        candidates = [
            path / "main.py", path / "scan.py", path / "modelaudit.py", path / "modelscan.py",
            path / "skillcheck.py", path / "skill-scanner.py", path / "run.sh",
        ]
        for c in candidates:
            if c.exists():
                if c.suffix == ".py":
                    return ScannerCommand(c.name, [sys.executable, str(c), str(target)], cwd=path)
                if os.access(c, os.X_OK) or c.suffix == ".sh":
                    return ScannerCommand(c.name, [str(c), str(target)], cwd=path)
    return None


def run_external_security_tools(target: Path) -> list[dict[str, str]]:
    """Run external scanners against one path, streaming output instead of buffering it.

    Scanner failures are scanner WARNs, not model DANGERs. A model should only become
    DANGER/BLOCKER when a scanner output contains a clear high-risk finding.

    Honors `set_session_scan_preference()`: True → run silently; False → skip
    silently; None → fall back to per-download prompt (legacy).
    """
    findings: list[dict[str, str]] = []
    pref = get_session_scan_preference()
    if pref is False:
        print("Security scan: SKIP (session preference = never)")
        return findings
    if pref is None:
        # Legacy per-download prompt — only reached if no upfront preference
        # was set. Default Yes preserved for back-compat.
        if not prompt_bool("Run external security/audit tools against the staged download?", True):
            return findings
    # else: pref is True — run without prompting
    # Per-scanner outcome tracker. Status is one of:
    #   PASS    — rc==0
    #   HIGH    — rc!=0 with high-risk language in output
    #   FAIL    — rc!=0, plain non-zero exit (this is the "always exits 1" case)
    #   ERROR   — invocation/usage error (low-signal stderr)
    #   TIMEOUT — killed by watchdog
    #   EXC     — Python exception starting/wrapping the subprocess
    #   SKIP    — tool not found / duplicate command
    # (label, status, detail)
    scanner_outcomes: list[tuple[str, str, str]] = []
    seen_commands: set[tuple[str, ...]] = set()
    high_risk_re = re.compile(r"\b(malware|malicious|backdoor|rce|remote code execution|trojan|stealer|exploit)\b", re.I)
    low_signal_re = re.compile(r"(unrecognized arguments|usage:|error: argument|invalid option|unknown option)", re.I)
    for tool in DEFAULT_SECURITY_TOOLS:
        scanner = find_tool_command(tool, target)
        if not scanner:
            print(f"SKIP tool not found/usable: {tool}")
            scanner_outcomes.append((tool, "SKIP", "not found / not usable"))
            continue
        cmd_key = tuple(scanner.cmd)
        if cmd_key in seen_commands:
            print(f"SKIP duplicate scanner command: {scanner.label}")
            scanner_outcomes.append((scanner.label, "SKIP", "duplicate command"))
            continue
        seen_commands.add(cmd_key)
        print()
        print(f"Running read-only scanner command ({scanner.label}): {quote_cmd(scanner.cmd)}")
        risky_hits: list[str] = []
        recent_lines: list[str] = []
        saw_invocation_error = False
        # Per-scanner wall-clock cap (default 10 minutes; tune with env). Without
        # this, a scanner that hangs without producing stdout (e.g., reading
        # from stdin, blocked on a TUI prompt, or long-running) freezes the
        # whole audit step. The watchdog thread kills the process after the
        # timeout so the for-loop unblocks via EOF.
        scanner_timeout = env_int("MODEL_MANAGER_SCANNER_TIMEOUT_S", 600, minimum=10)
        try:
            proc = subprocess.Popen(
                scanner.cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(scanner.cwd) if scanner.cwd else None,
            )
            assert proc.stdout is not None

            killed_by_timeout = {"flag": False}
            import threading

            def _kill_after_timeout():
                killed_by_timeout["flag"] = True
                try:
                    proc.kill()
                except Exception:
                    pass

            timer = threading.Timer(scanner_timeout, _kill_after_timeout)
            timer.daemon = True
            timer.start()
            try:
                for line in proc.stdout:
                    clean = line.rstrip("\n")
                    print(clean)
                    if len(recent_lines) >= 80:
                        recent_lines.pop(0)
                    recent_lines.append(clean)
                    if low_signal_re.search(clean):
                        saw_invocation_error = True
                    if high_risk_re.search(clean):
                        risky_hits.append(clean[:300])
            finally:
                timer.cancel()

            if killed_by_timeout["flag"]:
                findings.append({"severity": "WARN", "message": f"scanner killed after {scanner_timeout}s timeout: {scanner.label} (set MODEL_MANAGER_SCANNER_TIMEOUT_S to extend)"})
                scanner_outcomes.append((scanner.label, "TIMEOUT", f"killed after {scanner_timeout}s"))
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                continue
            rc = proc.wait(timeout=10)
            if rc in (0, None):
                scanner_outcomes.append((scanner.label, "PASS", "rc=0"))
            else:
                if saw_invocation_error:
                    findings.append({"severity": "WARN", "message": f"scanner invocation error from {scanner.label}: rc={rc}"})
                    last = recent_lines[-1] if recent_lines else ""
                    scanner_outcomes.append((scanner.label, "ERROR", f"rc={rc} (usage/invocation; last line: {last[:100]})"))
                elif risky_hits:
                    findings.append({"severity": "HIGH", "message": f"scanner {scanner.label} returned rc={rc} with high-risk language; review output"})
                    scanner_outcomes.append((scanner.label, "HIGH", f"rc={rc}, {len(risky_hits)} risky hit(s)"))
                else:
                    findings.append({"severity": "WARN", "message": f"scanner reported non-zero exit from {scanner.label}: rc={rc}"})
                    last = recent_lines[-1] if recent_lines else ""
                    scanner_outcomes.append((scanner.label, "FAIL", f"rc={rc} (last line: {last[:100]})"))
            for hit in risky_hits[:5]:
                findings.append({"severity": "HIGH", "message": f"scanner output high-risk term from {scanner.label}: {hit}"})
        except subprocess.TimeoutExpired:
            findings.append({"severity": "WARN", "message": f"scanner timed out: {scanner.label}"})
            scanner_outcomes.append((scanner.label, "TIMEOUT", "subprocess.TimeoutExpired"))
            try:
                proc.kill()  # type: ignore[name-defined]
            except Exception:
                pass
        except Exception as e:
            findings.append({"severity": "WARN", "message": f"scanner failed {scanner.label}: {type(e).__name__}: {e}"})
            scanner_outcomes.append((scanner.label, "EXC", f"{type(e).__name__}: {e}"))

    # Always print a one-screen scanner summary so the user can see at a glance
    # which tools passed, failed, timed out, etc. The user's prior complaint
    # was that scanner exits-1 were buried in long output and they would hit
    # Enter past it without realizing the scan never ran clean.
    print()
    print("=" * 78)
    print("Scanner summary")
    print("=" * 78)
    if not scanner_outcomes:
        print("  (no scanners ran — all SKIPPED or none configured)")
    else:
        for label, status, detail in scanner_outcomes:
            print(f"  [{status:<7}] {label}{' — ' + detail if detail else ''}")
        ok = sum(1 for _, s, _ in scanner_outcomes if s == "PASS")
        bad = sum(1 for _, s, _ in scanner_outcomes if s in {"FAIL", "ERROR", "EXC"})
        risky = sum(1 for _, s, _ in scanner_outcomes if s == "HIGH")
        timed = sum(1 for _, s, _ in scanner_outcomes if s == "TIMEOUT")
        print(f"  totals: {ok} pass · {bad} fail · {risky} high-risk · {timed} timeout")
        if bad and not risky:
            print("  Note: failing scanners do NOT automatically block install — "
                  "they only generate WARNs. If they have been silently failing, "
                  "fix the tool / re-run with `MODEL_MANAGER_SCAN_AFTER_DOWNLOAD=never` "
                  "to skip them entirely while you debug.")
    print("=" * 78)
    return findings

# -----------------------------------------------------------------------------
# Download
# -----------------------------------------------------------------------------


def final_download_path(result: SearchResult, download_root: Path) -> Path:
    return download_root / ("huggingface" if result.source == "hf" else "kaggle") / result.kind / safe_folder_name(result.repo_id)


def partial_download_name(result: SearchResult) -> str:
    digest = hashlib.sha1(f"{result.source}:{result.kind}:{result.repo_id}".encode("utf-8")).hexdigest()[:10]
    return f"{safe_folder_name(result.repo_id)}.{digest}.partial"


def staging_download_path(result: SearchResult, download_root: Path) -> Path:
    incoming_root = Path(
        os.getenv("MODEL_MANAGER_INCOMING_DIR", str(_default_incoming_dir(download_root)))
    ).expanduser().resolve()
    provider_root = incoming_root / ("huggingface" if result.source == "hf" else "kaggle") / result.kind
    preferred = provider_root / partial_download_name(result)
    if preferred.exists():
        return preferred

    legacy_prefix = safe_folder_name(result.repo_id)
    try:
        legacy_candidates = sorted(
            provider_root.glob(f"{legacy_prefix}*.partial"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        legacy_candidates = []
    for candidate in legacy_candidates:
        if candidate.exists():
            return candidate
    return preferred


def looks_like_permanent_download_error(error: BaseException) -> bool:
    message = f"{type(error).__name__}: {error}".lower()
    permanent_markers = [
        "401",
        "403",
        "404",
        "repository not found",
        "revision not found",
        "gated repo",
        "invalid token",
        "authentication",
        "authorization",
        "permission denied",
        "not found",
    ]
    return any(marker in message for marker in permanent_markers)


def retry_delay_seconds(attempt_number: int) -> int:
    base = download_retry_base_delay_seconds()
    return min(base * (2 ** max(0, attempt_number - 1)), 60)


def describe_partial_download_state(path: Path) -> str:
    if not path.exists():
        return "no partial download exists yet"
    try:
        entries = list(path.iterdir()) if path.is_dir() else []
        return f"partial path exists with {len(entries)} item(s): {path}"
    except OSError:
        return f"partial path exists: {path}"


def merge_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            merge_tree(child, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                target.unlink()
            shutil.move(str(child), str(target))
    try:
        src.rmdir()
    except OSError:
        pass


def promote_staged_download(staged: Path, final: Path) -> Path | None:
    if not staged.exists():
        print(f"Staged download no longer exists; not installing: {staged}")
        return None
    print(f"Staged download passed review: {staged}")
    print(f"Final install path: {final}")
    if final.exists():
        if not prompt_bool("Final path already exists. Merge staged files into existing final path?", False):
            print("Keeping staged download in place for manual review.")
            return staged
        merge_tree(staged, final)
    else:
        final.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staged), str(final))
    print(f"Installed to: {final}")
    return final

def download_hf_with_hfdownloader(
    result: SearchResult,
    download_root: Path,
    allow_patterns: list[str] | None,
    artifacts: list[Artifact] | None,
) -> Path | None:
    binary = find_hfdownloader_binary()
    if not binary:
        print("hfdownloader binary not found. Run ./install_hfdownloader.sh or set MODEL_MANAGER_HFDOWNLOADER_BIN.")
        return None

    if allow_patterns is not None and not artifacts:
        print("hfdownloader skipped for custom glob patterns; using Python fallback to preserve exact pattern semantics.")
        return None
    if not hfdownloader_can_preserve_artifact_selection(artifacts):
        print("hfdownloader skipped for this artifact type; using Python fallback to avoid downloading unselected model files.")
        return None

    default_connections, default_active = recommend_hfdownloader_concurrency(result, artifacts)
    metrics = selected_download_metrics(result, artifacts)
    if metrics["total_size"] or metrics["file_count"]:
        size_label = human_size(metrics["total_size"]) if metrics["total_size"] else "unknown"
        file_label = str(metrics["file_count"] or "?")
        print(
            "Auto-tuned hfdownloader defaults from selected download: "
            f"size={size_label}, files={file_label}, "
            f"connections={default_connections}, max_active={default_active}"
        )
    connections = prompt_int_range("How many hfdownloader connections per file?", default_connections, 1, 64)
    max_active = prompt_int_range("How many concurrent file downloads?", default_active, 1, 64)

    target = staging_download_path(result, download_root)
    base = target.parent
    base.mkdir(parents=True, exist_ok=True)
    nested = base / result.repo_id
    safe_nested = base / safe_folder_name(result.repo_id)
    if nested.exists():
        print(f"Resuming existing hfdownloader partial download: {nested}")
    elif safe_nested.exists():
        print(f"Resuming existing hfdownloader partial download: {safe_nested}")
    elif target.exists():
        print(f"Existing staged partial download detected: {describe_partial_download_state(target)}")

    cmd = [
        str(binary),
        "download",
        result.repo_id,
        "--local-dir",
        str(base),
        "--connections",
        str(connections),
        "--max-active",
        str(max_active),
        "--verify",
        "size",
    ]
    if result.kind == "dataset":
        cmd.append("--dataset")

    filters = hfdownloader_filter_terms_from_artifacts(artifacts)
    if filters:
        cmd.extend(["--filters", ",".join(filters)])
    excludes = hfdownloader_exclude_terms_for_artifacts(artifacts)
    if excludes:
        cmd.extend(["--exclude", ",".join(excludes)])

    # Loud safety check: if we selected specific artifacts but ended up with
    # no `--filters` to constrain hfdownloader, that's the silent-whole-repo
    # bug we just fixed. Refuse rather than start a TB-scale pull.
    if artifacts and not filters:
        print()
        print("=" * 78)
        print("ABORT: artifact selection produced no hfdownloader filter terms.")
        print("=" * 78)
        for art in artifacts:
            print(f"  - {art.label}  (quant={art.quant!r}, type={art.artifact_type!r})")
        print()
        print("Without `--filters`, hfdownloader would pull the whole repo.")
        print("This usually means the Artifact lacks a recognizable quant tag.")
        print("Falling back to the Python downloader, which honors allow_patterns directly.")
        debug_log(
            "hfdownloader-filter-empty",
            repo_id=result.repo_id,
            artifact_count=len(artifacts),
            artifact_labels=[a.label for a in artifacts],
            artifact_quants=[a.quant for a in artifacts],
        )
        return None  # caller falls back to snapshot_download with allow_patterns

    debug_log(
        "hfdownloader-dispatch",
        repo_id=result.repo_id,
        kind=result.kind,
        target=str(target),
        selected_artifact_labels=[a.label for a in (artifacts or [])],
        selected_artifact_quants=[a.quant for a in (artifacts or [])],
        filter_terms=filters,
        exclude_terms=excludes,
        allow_patterns=allow_patterns,
        cmd=cmd,
    )

    endpoint = os.getenv("MODEL_MANAGER_HF_ENDPOINT") or os.getenv("HF_ENDPOINT")
    if endpoint:
        cmd.extend(["--endpoint", endpoint])

    env = os.environ.copy()
    if "HF_TOKEN" not in env and env.get("HUGGINGFACEHUB_API_TOKEN"):
        env["HF_TOKEN"] = env["HUGGINGFACEHUB_API_TOKEN"]

    print(f"Downloading HF {result.kind} with hfdownloader: {result.repo_id}")
    print(f"Target: {target}")
    print(
        "Download mode: Go hfdownloader, no Docker, writes real files to disk before post-download audit "
        f"(connections={connections}, max_active={max_active})"
    )
    if filters:
        print("Selected artifact filters:")
        for item in filters[:40]:
            print(f"  {item}")
        if len(filters) > 40:
            print(f"  ... {len(filters) - 40} more")
    if excludes:
        print("Safety excludes for unselected model formats: " + ", ".join(excludes))
    print(f"Running: {quote_cmd(cmd)}")
    attempts = download_retry_attempts()
    for attempt in range(1, attempts + 1):
        try:
            if attempt > 1:
                print(f"Retry attempt {attempt}/{attempts} with hfdownloader. Reusing any partial files already on disk.")
            subprocess.run(cmd, check=True, env=env)
            break
        except subprocess.CalledProcessError as e:
            debug_log(
                "hfdownloader-failed",
                repo_id=result.repo_id,
                source=result.source,
                kind=result.kind,
                cmd=cmd,
                returncode=e.returncode,
                attempt=attempt,
                attempts=attempts,
            )
            if attempt >= attempts:
                partial_hint = nested if nested.exists() else safe_nested if safe_nested.exists() else target if target.exists() else None
                if partial_hint is not None:
                    print(f"Partial download remains on disk at {partial_hint}. The next retry or rerun can resume from there.")
                print(f"hfdownloader failed after {attempts} attempt(s) with rc={e.returncode}. Falling back to huggingface_hub snapshot_download.")
                return None
            delay = retry_delay_seconds(attempt)
            print(
                f"hfdownloader attempt {attempt}/{attempts} failed with rc={e.returncode}. "
                f"Retrying in {delay}s and resuming from any partial files already downloaded."
            )
            time.sleep(delay)
    if nested.exists():
        return nested
    if safe_nested.exists():
        return safe_nested
    if target.exists():
        return target
    return target


# ──────────────────────────────────────────────────────────────────────
# Active downloads queue — persists across crashes/restarts
# ──────────────────────────────────────────────────────────────────────


def _load_active_downloads() -> list[dict]:
    """Return the list of currently-recorded in-flight downloads. Empty list
    if file missing or unreadable. Never raises."""
    try:
        if not ACTIVE_DOWNLOADS_PATH.is_file():
            return []
        data = json.loads(ACTIVE_DOWNLOADS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    except (OSError, json.JSONDecodeError):
        pass
    return []


def _save_active_downloads(records: list[dict]) -> None:
    """Atomic write so a partial save can never corrupt the file."""
    try:
        ACTIVE_DOWNLOADS_PATH.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix="active_downloads.", suffix=".tmp",
                                   dir=str(ACTIVE_DOWNLOADS_PATH.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
            os.replace(tmp, ACTIVE_DOWNLOADS_PATH)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except OSError:
        pass  # best-effort; never block the download itself


def _record_active_download(
    repo_id: str,
    kind: str,
    revision: str,
    allow_patterns: list[str] | None,
    download_root: Path,
    staging_path: Path,
    downloader: str,
) -> str:
    """Add a record. Returns the record id so the caller can clear it later."""
    rec_id = f"{int(time.time())}-{abs(hash((repo_id, downloader))) % 100000:05d}"
    record = {
        "id": rec_id,
        "repo_id": repo_id,
        "kind": kind,
        "revision": revision,
        "allow_patterns": list(allow_patterns) if allow_patterns else None,
        "download_root": str(download_root),
        "staging_path": str(staging_path),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "downloader": downloader,
    }
    records = _load_active_downloads()
    records.append(record)
    _save_active_downloads(records)
    return rec_id


def _clear_active_download(rec_id: str) -> None:
    records = _load_active_downloads()
    new = [r for r in records if r.get("id") != rec_id]
    if len(new) != len(records):
        _save_active_downloads(new)


def _staging_path_volume_unmounted(staging: Path) -> bool:
    """Return True when `staging` lives under `/Volumes/<name>/...` and that
    mount point is NOT currently mounted. Used by `_prune_stale_active_downloads`
    so we don't drop in-flight resume records while the drive happens to be
    unplugged at the moment we check. Plug the drive back in and the record
    is still resumable.

    Heuristic is macOS-specific (the script is macOS-targeted): any path with
    `/Volumes/<name>` as its first two segments where `os.path.ismount(<name>)`
    is False → considered unmounted. Returns False for boot-drive paths and
    for any failure to evaluate.
    """
    try:
        sp = Path(staging).resolve(strict=False)
    except (OSError, ValueError):
        return False
    parts = sp.parts
    # Must be /Volumes/<name>/... — fewer than 3 parts means we're at /Volumes
    # or above, which is never a per-drive mount point we care about here.
    if len(parts) < 3 or parts[0] != "/" or parts[1] != "Volumes":
        return False
    mount_point = Path("/") / parts[1] / parts[2]
    try:
        return not os.path.ismount(str(mount_point))
    except OSError:
        # Can't evaluate; err on the safe side and assume mounted (so prune
        # behavior is unchanged from before this function existed).
        return False


def _migrate_legacy_incoming_staging_paths(records: list[dict]) -> tuple[list[dict], int]:
    """Rewrite `staging_path` entries that point at the legacy
    `<DEFAULT_DOWNLOAD_DIR>/.incoming/...` location (when that's no longer
    the configured incoming dir). Why: the default `INCOMING_DOWNLOAD_DIR`
    used to be `<DEFAULT_DOWNLOAD_DIR>/.incoming`; it was moved to a
    sibling path to avoid blowing past LM Studio's 7,000-file scanner cap.
    A user who `mv`'d the directory will have queue records pointing at
    the old path — those records would otherwise be silently pruned as
    "stale" even though the partial bytes are still on disk under the
    new path. Returns (possibly-rewritten records, count of rewrites).

    Skipped if:
      - MODEL_MANAGER_INCOMING_DIR is set (user has explicit non-default config)
      - INCOMING_DOWNLOAD_DIR is the legacy default (no migration needed)
    """
    if os.environ.get("MODEL_MANAGER_INCOMING_DIR"):
        return records, 0
    legacy_prefix = (DEFAULT_DOWNLOAD_DIR / ".incoming").resolve()
    new_prefix = INCOMING_DOWNLOAD_DIR
    if legacy_prefix == new_prefix:
        return records, 0
    rewrites = 0
    out: list[dict] = []
    for rec in records:
        staging = rec.get("staging_path", "")
        if not staging:
            out.append(rec)
            continue
        try:
            sp = Path(staging)
            # Match only paths actually under the legacy default; don't touch
            # custom paths the user might have set per-download.
            if sp.is_relative_to(legacy_prefix):
                tail = sp.relative_to(legacy_prefix)
                candidate = new_prefix / tail
                if candidate.exists() and not sp.exists():
                    rec = {**rec, "staging_path": str(candidate)}
                    rewrites += 1
        except (ValueError, OSError):
            pass
        out.append(rec)
    return out, rewrites


def _prune_stale_active_downloads() -> list[dict]:
    """Drop records whose staging_path no longer exists. Return the remaining
    records (the still-resumable ones). Runs the legacy-incoming migration
    first so records that point at the pre-move `.incoming/` path get
    rewritten to the new sibling location before the existence check.

    Records whose staging_path lives on a currently-unmounted external
    volume are KEPT (not pruned) so a brief drive disconnect can't lose
    your in-flight downloads. Reconnect the drive and they're resumable
    on the next `modelmgr` run.
    """
    records = _load_active_downloads()
    records, rewrites = _migrate_legacy_incoming_staging_paths(records)
    if rewrites:
        print(
            f"Resume queue: migrated {rewrites} record(s) from legacy "
            f"`{DEFAULT_DOWNLOAD_DIR / '.incoming'}` to `{INCOMING_DOWNLOAD_DIR}`."
        )
        _save_active_downloads(records)
    alive: list[dict] = []
    unmounted_kept: list[str] = []
    for rec in records:
        staging = rec.get("staging_path", "")
        if not staging:
            continue
        sp = Path(staging)
        if sp.exists():
            alive.append(rec)
            continue
        if _staging_path_volume_unmounted(sp):
            alive.append(rec)
            try:
                unmounted_kept.append(str(Path("/") / sp.parts[1] / sp.parts[2]))
            except IndexError:
                pass
            continue
        # Genuinely missing on a mounted volume — prune.
    if unmounted_kept:
        unique_volumes = sorted(set(unmounted_kept))
        print(
            f"Resume queue: keeping {len(unmounted_kept)} record(s) whose staging dir "
            f"lives on a currently-unmounted volume: {', '.join(unique_volumes)}. "
            f"Reconnect the drive and re-run `modelmgr` to resume."
        )
    if len(alive) != len(records):
        _save_active_downloads(alive)
    return alive


# ---------------------------------------------------------------------------
# Session-level security-scan preference
# ---------------------------------------------------------------------------
# Set once at the start of a session by `prompt_session_scan_preference()` (or
# `--scan-after-download {ask,always,never}`), read by `run_external_security_tools`
# so the user is not prompted per-download. The user kept hitting Enter through
# the per-download prompt and triggering a scanner that has been intermittently
# exiting non-zero — this consolidates the choice and surfaces failures.
#
# Values:
#   None  → ask per-download (legacy behavior)
#   True  → always run scanners after each successful download
#   False → never run scanners
_SCAN_AFTER_DOWNLOAD_PREF: bool | None = None


def set_session_scan_preference(pref: bool | None) -> None:
    global _SCAN_AFTER_DOWNLOAD_PREF
    _SCAN_AFTER_DOWNLOAD_PREF = pref


def get_session_scan_preference() -> bool | None:
    return _SCAN_AFTER_DOWNLOAD_PREF


def _parse_scan_after_download_choice(value: str | None) -> bool | None | str:
    """Returns True / False / None / 'ask'. None means env wasn't set."""
    if value is None:
        return None
    v = value.strip().lower()
    if v in {"always", "yes", "y", "1", "true", "on", "run"}:
        return True
    if v in {"never", "no", "n", "0", "false", "off", "skip"}:
        return False
    if v in {"ask", "prompt"}:
        return "ask"
    return None


def prompt_session_scan_preference(
    explicit: bool | None | str = None,
) -> bool | None:
    """Choose ONCE per session whether to run external scanners after each
    successful download. Reads MODEL_MANAGER_SCAN_AFTER_DOWNLOAD as default
    (always / never / ask) if `explicit` is not provided. Returns the value
    set via `set_session_scan_preference`."""
    if explicit is None:
        explicit = _parse_scan_after_download_choice(
            os.environ.get("MODEL_MANAGER_SCAN_AFTER_DOWNLOAD")
        )
    if explicit is True:
        print("Security scan after each download: ALWAYS (run, no per-download prompt)")
        set_session_scan_preference(True)
        return True
    if explicit is False:
        print("Security scan after each download: NEVER (skip, no per-download prompt)")
        set_session_scan_preference(False)
        return False
    # explicit is "ask" or None — fall through to interactive
    print()
    print("Security scan preference for this session")
    print("  External scanners (modelaudit, modelscan, ModelGuard, skill-scanner) run AFTER each")
    print("  successful download. Pick once for the whole session — won't re-prompt per download.")
    print("  Override anytime via MODEL_MANAGER_SCAN_AFTER_DOWNLOAD={always,never,ask} or")
    print("  --scan-after-download {always,never,ask}.")
    choice = prompt_choice(
        "Run security scan after each successful download?",
        ["always", "never", "ask-per-download"],
        "always",
    )
    if choice == "always":
        set_session_scan_preference(True)
        return True
    if choice == "never":
        set_session_scan_preference(False)
        return False
    set_session_scan_preference(None)
    return None


# ---------------------------------------------------------------------------
# Startup warning: legacy `.incoming/` inside the download root
# ---------------------------------------------------------------------------
def warn_if_incoming_inside_download_root() -> None:
    """Older defaults put `INCOMING_DOWNLOAD_DIR` inside `DEFAULT_DOWNLOAD_DIR`.
    When DEFAULT_DOWNLOAD_DIR is also LM Studio's downloadsFolder, .incoming/
    can balloon past LM Studio's 7,000-file scanner cap (an interrupted RLHF
    dataset alone is 18k+ tiny shards) and silently break My Models. The
    default has been moved to a sibling, but a stale `.incoming/` from a
    pre-fix run will still sit inside the download root. Detect and warn
    on every startup until the user moves it. Suppress with
    MODEL_MANAGER_SUPPRESS_INCOMING_WARNING=1."""
    if os.environ.get("MODEL_MANAGER_SUPPRESS_INCOMING_WARNING", "").strip().lower() in {
        "1", "yes", "y", "true", "on",
    }:
        return
    legacy_incoming = DEFAULT_DOWNLOAD_DIR / ".incoming"
    if not legacy_incoming.exists() or not legacy_incoming.is_dir():
        return
    try:
        # Cheap check — count files up to 50, then bail. If 0, nothing to warn about.
        count = 0
        for _ in legacy_incoming.rglob("*"):
            count += 1
            if count >= 50:
                break
    except (PermissionError, OSError):
        count = -1
    if count == 0:
        return
    print()
    print("=" * 78)
    print("WARNING: legacy `.incoming/` directory found inside download root")
    print("=" * 78)
    print(f"  Path: {legacy_incoming}")
    print(f"  Files inside: {count}{'+' if count >= 50 else ''}")
    print()
    print("  If your download root doubles as LM Studio's downloadsFolder, this")
    print("  staging directory can blow past LM Studio's 7,000-file scanner cap")
    print("  and cause My Models to silently show 0 entries.")
    print()
    print("  The default staging path is now:")
    print(f"    {INCOMING_DOWNLOAD_DIR}")
    print("  (sibling of download root, NOT a child).")
    print()
    print("  Recommended cleanup (you run; this script never deletes user data):")
    print(f"    mv {legacy_incoming} {INCOMING_DOWNLOAD_DIR}")
    print(f"    # then in LM Studio, force a rescan:")
    print(f"    osascript -e 'quit app \"LM Studio\"'")
    print(f"    rm <REDACTED_PATH>")
    print(f"    open '/Applications/LM Studio.app'")
    print()
    print("  Suppress this warning: export MODEL_MANAGER_SUPPRESS_INCOMING_WARNING=1")
    print("=" * 78)
    print()


def offer_resume_active_downloads() -> None:
    """Called at the start of run_search_flow. Detects in-flight downloads
    from a prior session, prompts the user, and dispatches the resumes.
    Silent if there's nothing to resume."""
    records = _prune_stale_active_downloads()
    if not records:
        return
    print()
    print("─" * 78)
    print(f"Found {len(records)} unfinished download(s) from a previous session:")
    for i, rec in enumerate(records, 1):
        repo = rec.get("repo_id", "?")
        kind = rec.get("kind", "?")
        when = rec.get("started_at", "?")
        downloader = rec.get("downloader", "?")
        staging = rec.get("staging_path", "")
        size_str = ""
        try:
            sp = Path(staging)
            if sp.is_dir():
                total = sum(p.stat().st_size for p in sp.rglob("*") if p.is_file())
                size_str = f"  ({human_size(total)} on disk)"
        except OSError:
            pass
        print(f"  {i}. {repo}  [{kind}, via {downloader}, started {when}]{size_str}")
    print()
    print("Resume options:")
    print("  y   resume all")
    print("  N   skip (default — records remain for next time)")
    print("  q   forget all (delete records, partials stay on disk)")
    print("  1,3 resume just those by number")
    try:
        ans = input("Choice [N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if not ans or ans == "n":
        return
    if ans == "q":
        _save_active_downloads([])
        print("Cleared records. Partial files remain on disk for manual cleanup.")
        return
    if ans == "y":
        to_resume = list(records)
    else:
        try:
            picks = {int(t.strip()) for t in ans.split(",") if t.strip()}
        except ValueError:
            print("Unrecognized choice; skipping resume.")
            return
        to_resume = [records[i - 1] for i in sorted(picks) if 1 <= i <= len(records)]
        if not to_resume:
            print("No valid selections; skipping.")
            return

    for rec in to_resume:
        try:
            _resume_one_download(rec)
        except Exception as e:
            print(f"  ✗ resume failed for {rec.get('repo_id', '?')}: {type(e).__name__}: {e}")


def _resume_one_download(rec: dict) -> None:
    """Re-trigger a download from a recorded entry. Reuses the same code
    path as a fresh download; resumption happens via the staging dir."""
    repo_id = rec.get("repo_id")
    kind = rec.get("kind", "model")
    download_root_str = rec.get("download_root")
    if not repo_id or not download_root_str:
        return
    download_root = Path(download_root_str)
    allow_patterns = rec.get("allow_patterns")

    print()
    print(f"Resuming: {repo_id}")
    if kind == "dataset":
        result = build_exact_hf_result(repo_id, kind="dataset")
    else:
        result = build_exact_hf_result(repo_id, kind="model")
    if not result:
        print(f"  ✗ Could not look up {repo_id} on Hugging Face — skipping.")
        return
    annotate_results_with_leaderboard_cache([result])
    assign_indexes([result])

    target: Path | None = None
    try:
        if kind == "dataset":
            target = download_kaggle_result(result, download_root, allow_patterns) \
                if result.source == "kaggle" \
                else download_hf_result(result, download_root, allow_patterns, artifacts=[])
        else:
            target = download_hf_result(result, download_root, allow_patterns, artifacts=[])
    except Exception as e:
        print(f"  ✗ resume error: {type(e).__name__}: {e}")
        return

    if target:
        ok_to_install = post_download_audit(result, target, allow_patterns)
        if ok_to_install:
            final_target = final_download_path(result, download_root)
            installed = promote_staged_download(target, final_target)
            if installed and result.kind == "model":
                offer_prepare_models(installed, download_root)


def download_hf_result(
    result: SearchResult,
    download_root: Path,
    allow_patterns: list[str] | None,
    artifacts: list[Artifact] | None = None,
) -> Path | None:
    # Record this download in the persistent queue so a crash/restart can
    # surface it for resume. We compute the staging path the way the actual
    # downloaders do so the stored path matches what would be on disk.
    staging_path = staging_download_path(result, download_root)
    rec_id = _record_active_download(
        repo_id=result.repo_id,
        kind=result.kind or "model",
        revision="main",
        allow_patterns=allow_patterns,
        download_root=download_root,
        staging_path=staging_path,
        downloader=("hfdownloader" if (result.kind == "model" and hfdownloader_enabled()) else "snapshot_download"),
    )

    if result.kind == "model" and hfdownloader_enabled():
        target = download_hf_with_hfdownloader(result, download_root, allow_patterns, artifacts)
        if target is not None:
            _clear_active_download(rec_id)
            return target
        print("Falling back to huggingface_hub snapshot_download.")

    transfer_mode = os.getenv("MODEL_MANAGER_HF_TRANSFER_MODE", "fast").strip().lower()
    safe_mode = transfer_mode in {"safe", "low-memory", "low_memory", "conservative"}
    default_workers = recommend_hf_worker_count(result, artifacts, safe_mode=safe_mode)
    default_range_gets = recommend_hf_xet_range_gets(default_workers, safe_mode=safe_mode)
    metrics = selected_download_metrics(result, artifacts)
    if metrics["total_size"] or metrics["file_count"]:
        size_label = human_size(metrics["total_size"]) if metrics["total_size"] else "unknown"
        file_label = str(metrics["file_count"] or "?")
        print(
            "Auto-tuned Hugging Face defaults from selected download: "
            f"size={size_label}, files={file_label}, "
            f"workers={default_workers}, xet_range_gets={default_range_gets}"
        )
    worker_count = prompt_int_range("How many Hugging Face download workers?", default_workers, 1, 64)
    range_gets = recommend_hf_xet_range_gets(worker_count, safe_mode=safe_mode)
    transfer_cfg = configure_hf_download_environment(worker_count=worker_count, range_gets=range_gets)
    from huggingface_hub import snapshot_download
    target = staging_download_path(result, download_root)
    if target.exists():
        print(f"Resuming existing staged partial download: {describe_partial_download_state(target)}")
    target.mkdir(parents=True, exist_ok=True)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    ignore = HF_DATASET_IGNORE_MODEL_WEIGHTS if result.kind == "dataset" else None
    print(f"Downloading HF {result.kind}: {result.repo_id}")
    print(f"Target: {target}")
    print(
        f"Download mode: {transfer_cfg['mode']} hf_xet transfer, write to disk first, then run post-download audit "
        f"(max_workers={transfer_cfg['max_workers']}, "
        f"xet_high_performance={'on' if transfer_cfg['high_performance'] else 'off'}, "
        f"xet_range_gets={transfer_cfg['range_gets']})"
    )
    if transfer_cfg["mode"] == "fast" and not transfer_cfg["hf_xet_available"]:
        print('NOTE: hf_xet was not detected. Install with: python3 -m pip install -U "huggingface_hub[hf_xet]"')
    if allow_patterns:
        print("Selective patterns:")
        for p in allow_patterns[:40]:
            print(f"  {p}")
        if len(allow_patterns) > 40:
            print(f"  ... {len(allow_patterns) - 40} more")
    attempts = download_retry_attempts()
    for attempt in range(1, attempts + 1):
        try:
            if attempt > 1:
                print(f"Retry attempt {attempt}/{attempts}. Reusing the staged partial download at {target}.")
            HF_STUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=result.repo_id,
                repo_type=result.kind,
                local_dir=str(target),
                cache_dir=str(HF_STUB_CACHE_DIR),
                token=token,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore if allow_patterns is None else None,
                max_workers=int(transfer_cfg["max_workers"]),
            )
            break
        except Exception as e:
            debug_log(
                "snapshot-download-failed",
                repo_id=result.repo_id,
                source=result.source,
                kind=result.kind,
                target=str(target),
                attempt=attempt,
                attempts=attempts,
                exc_type=type(e).__name__,
                exc_message=str(e),
            )
            permanent = looks_like_permanent_download_error(e)
            if attempt >= attempts or permanent:
                if not permanent and target.exists():
                    print(f"Partial download remains on disk at {target}. The next retry or rerun can resume from there.")
                raise
            delay = retry_delay_seconds(attempt)
            print(
                f"snapshot_download attempt {attempt}/{attempts} failed: {type(e).__name__}: {e}"
            )
            print(f"Retrying in {delay}s and resuming from the partial download at {target}.")
            time.sleep(delay)
    _clear_active_download(rec_id)
    return target


def download_kaggle_result(result: SearchResult, download_root: Path, allow_patterns: list[str] | None) -> Path | None:
    target = staging_download_path(result, download_root)
    target.mkdir(parents=True, exist_ok=True)
    rec_id = _record_active_download(
        repo_id=result.repo_id,
        kind=result.kind or "dataset",
        revision="main",
        allow_patterns=allow_patterns,
        download_root=download_root,
        staging_path=target,
        downloader="kaggle",
    )
    print(f"Downloading Kaggle {result.kind}: {result.repo_id}")
    print(f"Target: {target}")
    if result.kind == "dataset":
        api = kaggle_api()
        if allow_patterns:
            files = kaggle_dataset_files(result.repo_id)
            matches = [name for name, _ in files if any(fnmatch.fnmatch(name, pat) for pat in allow_patterns)]
            if not matches:
                print("No Kaggle files matched the chosen pattern.")
                _clear_active_download(rec_id)
                return None
            for name in matches:
                api.dataset_download_file(result.repo_id, name, path=str(target), quiet=False)
            _clear_active_download(rec_id)
            return target
        api.dataset_download_files(result.repo_id, path=str(target), unzip=True, quiet=False)
        _clear_active_download(rec_id)
        return target

    try:
        import kagglehub
        fn = getattr(kagglehub, "model_download", None)
        if callable(fn):
            path = Path(fn(result.repo_id))
            if path.exists() and path != target:
                if path.is_dir():
                    shutil.copytree(path, target, dirs_exist_ok=True)
                else:
                    shutil.copy2(path, target / path.name)
            _clear_active_download(rec_id)
            return target
    except Exception as e:
        print(f"Kaggle model download failed or unavailable: {type(e).__name__}: {e}")
    print("Kaggle model download is unavailable in this environment/package version.")
    _clear_active_download(rec_id)
    return None



# -----------------------------------------------------------------------------
# Pre-download risk and completeness checks
# -----------------------------------------------------------------------------

BUILTIN_MALICIOUS_AI_TERMS = {
    "wormgpt", "fraudgpt", "darkbard", "wolfgpt", "demon-gpt", "demongpt",
    "poisongpt", "poison-gpt", "kawaiigpt", "kawaii-gpt", "xxxgpt",
    "malterminal", "lamehug", "evil-gpt", "evilgpt", "dark-llm", "darkllm",
}

BUILTIN_SECURITY_RESEARCH_TERMS = {
    "poison", "backdoor", "backdoored", "payload", "malware", "ransomware",
    "credential", "stealer", "phishing", "jailbreak", "bypass", "uncensored",
    "abliterated", "obliterated", "darkweb", "dark-web",
}


def normalize_match_text(value: str) -> str:
    text = str(value).lower()
    text = re.sub(r"([a-z])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)([a-z])", r"\1 \2", text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def candidate_risk_files() -> list[Path]:
    seen: set[Path] = set()
    files: list[Path] = []
    for p in RISK_INTEL_CANDIDATES:
        if p.exists() and p not in seen:
            files.append(p)
            seen.add(p)
    if MODEL_TOOLS_DIR.exists():
        for p in sorted(MODEL_TOOLS_DIR.iterdir()):
            if p in seen or not p.is_file():
                continue
            name = p.name.lower()
            if p.suffix.lower() not in {".json", ".csv", ".tsv", ".xlsx"}:
                continue
            if any(term in name for term in ["risk", "malicious", "malware", "avid", "mit", "threat", "model"]):
                files.append(p)
                seen.add(p)
    return files


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Risk intel: could not read JSON {path}: {type(e).__name__}: {e}")
        return []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ["rows", "findings", "models", "items", "data", "records"]:
            if isinstance(data.get(key), list):
                return [x for x in data[key] if isinstance(x, dict)]
        return [data]
    return []


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    delim = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            return [dict(row) for row in csv.DictReader(f, delimiter=delim)]
    except Exception as e:
        print(f"Risk intel: could not read table {path}: {type(e).__name__}: {e}")
        return []


_XLSX_NOISE_SHEET_PATTERNS = (
    # Substrings of sheet names that are NOT actual risk-data tabs in
    # taxonomy-style workbooks (MIT AI Risk Repository, similar). These
    # tabs hold prose, statistics, change logs, or table-of-contents
    # entries — loading their cells produces noisy fake "risk terms".
    "contents", "taxonomy", "explainer", "statistics",
    "compar",  # "Causal x Domain Taxonomy compar"
    "included", "considered",  # "Included resources", "Resources being considered"
    "change log", "changelog", "readme", "guide",
)


def _xlsx_sheet_is_noise(name: str) -> bool:
    n = (name or "").lower()
    return any(p in n for p in _XLSX_NOISE_SHEET_PATTERNS)


def _looks_like_header_row(values: tuple) -> bool:
    """Heuristic: a real header row has multiple short, distinct, non-banner
    string cells. Reject rows that are mostly empty, contain a single long
    sentence, start with warning glyphs, or include "Updated:" / "Last
    Modified" timestamps."""
    cells = [str(v).strip() if v is not None else "" for v in values]
    nonempty = [c for c in cells if c]
    if len(nonempty) < 3:
        return False
    for c in nonempty:
        if c.startswith(("⚠", "WARNING", "Watch ", "View ")):
            return False
        if "IMPORTANT NOTE" in c or "Updated:" in c or "Last Modified" in c:
            return False
        if len(c) > 60:  # long prose, almost certainly a banner
            return False
    return True


def load_xlsx_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import openpyxl  # type: ignore
    except Exception:
        print(f"Risk intel: {path.name} is .xlsx but openpyxl is not installed. Install with: python3 -m pip install openpyxl")
        return []
    rows: list[dict[str, Any]] = []
    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        for ws in wb.worksheets:
            if _xlsx_sheet_is_noise(ws.title):
                continue
            # Real header row may be preceded by banner/title/blank rows
            # (common in research workbooks). Scan up to 20 rows for the
            # first that looks like real column headers.
            iterator = ws.iter_rows(values_only=True)
            header: tuple | None = None
            scanned = 0
            for row_values in iterator:
                scanned += 1
                if _looks_like_header_row(row_values):
                    header = row_values
                    break
                if scanned >= 20:
                    break
            if header is None:
                continue
            keys = [str(x).strip() if x is not None else "" for x in header]
            if not any(keys):
                continue
            for values in iterator:
                row = {keys[i]: values[i] for i in range(min(len(keys), len(values))) if keys[i]}
                if any(v not in {None, ""} for v in row.values()):
                    row["_sheet"] = ws.title
                    rows.append(row)
    except Exception as e:
        print(f"Risk intel: could not read workbook {path}: {type(e).__name__}: {e}")
    return rows


def load_risk_intel_rows() -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    for path in candidate_risk_files():
        suffix = path.suffix.lower()
        if suffix == ".json":
            rows = load_json_rows(path)
        elif suffix in {".csv", ".tsv"}:
            rows = load_csv_rows(path)
        elif suffix == ".xlsx":
            rows = load_xlsx_rows(path)
        else:
            rows = []
        for row in rows:
            row["_source_file"] = str(path)
        all_rows.extend(rows)
    return all_rows


def load_risk_intel_rows_quiet() -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    xlsx_available = importlib.util.find_spec("openpyxl") is not None
    for path in candidate_risk_files():
        suffix = path.suffix.lower()
        if suffix == ".json":
            rows = load_json_rows(path)
        elif suffix in {".csv", ".tsv"}:
            rows = load_csv_rows(path)
        elif suffix == ".xlsx":
            if not xlsx_available:
                continue
            rows = load_xlsx_rows(path)
        else:
            rows = []
        for row in rows:
            row["_source_file"] = str(path)
        all_rows.extend(rows)
    return all_rows


# Cached weekly check for MIT AI Risk Repository updates. We watch the Google
# Sheets doc ID linked from the "Explore database" button on airisk.mit.edu —
# MIT swaps that ID when they version up the database, so a change is a strong
# signal that a newer release exists. All errors (network, parse, cache write)
# are swallowed silently — never blocks startup.
_AIRISK_URL = "https://airisk.mit.edu/"
_AIRISK_CACHE_PATH = CACHE_DIR / "airisk_mit_check.json"
_AIRISK_CHECK_INTERVAL_S = 7 * 24 * 3600
_AIRISK_EXPLORE_DB_RE = re.compile(
    r'<a\b[^>]*href="(https?://docs\.google\.com/spreadsheets/[^"]+)"[^>]*>'
    r'(?:\s|<[^>]+>)*Explore\s+database',
    re.IGNORECASE | re.DOTALL,
)
_AIRISK_DOC_ID_RE = re.compile(r"/d/([A-Za-z0-9_-]{20,})")


def check_airisk_mit_update_quietly() -> None:
    try:
        cache: dict[str, Any] = {}
        if _AIRISK_CACHE_PATH.exists():
            try:
                cache = json.loads(_AIRISK_CACHE_PATH.read_text())
            except (OSError, json.JSONDecodeError):
                cache = {}

        last_check = float(cache.get("last_check_epoch", 0))
        cached_doc_id = cache.get("explore_database_doc_id", "")
        now = time.time()

        if now - last_check < _AIRISK_CHECK_INTERVAL_S:
            return

        import urllib.request
        try:
            req = urllib.request.Request(
                _AIRISK_URL,
                headers={"User-Agent": "model_manager.py airisk-version-check"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read(500_000).decode("utf-8", errors="replace")
        except Exception:
            return

        link_match = _AIRISK_EXPLORE_DB_RE.search(html)
        if not link_match:
            return
        href = link_match.group(1)
        id_match = _AIRISK_DOC_ID_RE.search(href)
        if not id_match:
            return
        new_doc_id = id_match.group(1)

        cache["last_check_epoch"] = now
        cache["explore_database_url"] = href
        cache["explore_database_doc_id"] = new_doc_id
        try:
            _AIRISK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            _AIRISK_CACHE_PATH.write_text(json.dumps(cache, indent=2))
        except OSError:
            pass

        if cached_doc_id and cached_doc_id != new_doc_id:
            print(
                "NOTE: airisk.mit.edu 'Explore database' link changed — MIT may "
                "have released a newer AI Risk Repository."
            )
            print(f"  previous doc id: {cached_doc_id}")
            print(f"  current  doc id: {new_doc_id}")
            print(f"  Visit {_AIRISK_URL} to download and replace your local copy.")
    except Exception:
        return


def reputation_row_owners(row: dict[str, Any]) -> set[str]:
    lowered = {str(k).lower().strip(): v for k, v in row.items()}
    owners: set[str] = set()

    for key in ("owner", "publisher", "author", "creator", "organization", "org"):
        value = lowered.get(key)
        if value in {None, ""}:
            continue
        for part in re.split(r"[,\n;|]+", str(value)):
            owner = normalize_owner_name(part)
            if owner:
                owners.add(owner)

    for key in ("repo", "repo_id", "model", "model_id", "artifact", "url"):
        value = lowered.get(key)
        if value in {None, ""}:
            continue
        text = str(value).strip()
        if not text:
            continue
        parsed_repo = parse_hf_repo_id(text)
        if parsed_repo:
            owners.add(owner_of(parsed_repo))
            continue
        for token in re.split(r"[\s,;|]+", text):
            token = token.strip().strip("\"'")
            if "/" in token and not token.lower().startswith("http"):
                owner = normalize_owner_name(token)
                if owner:
                    owners.add(owner)
    return owners


def load_owner_reputation_sets() -> dict[str, set[str]]:
    global _OWNER_REPUTATION_CACHE
    if _OWNER_REPUTATION_CACHE is not None:
        return {key: set(value) for key, value in _OWNER_REPUTATION_CACHE.items()}

    cfg = load_reputation_config()
    good = {normalize_owner_name(x) for x in cfg.get("known_good_owners", [])}
    bad = {normalize_owner_name(x) for x in cfg.get("known_malicious_owners", [])}
    warn = {normalize_owner_name(x) for x in cfg.get("warn_owners", [])}
    good.discard("")
    bad.discard("")
    warn.discard("")

    for row in load_risk_intel_rows_quiet():
        owners = reputation_row_owners(row)
        if not owners:
            continue
        severity = risk_row_severity(row)
        if severity in {"DANGER", "BLOCKER"}:
            bad.update(owners)
        elif severity == "WARN":
            warn.update(owners)

    good -= bad
    warn -= bad
    _OWNER_REPUTATION_CACHE = {
        "known_good_owners": set(good),
        "known_malicious_owners": set(bad),
        "warn_owners": set(warn),
    }
    return {key: set(value) for key, value in _OWNER_REPUTATION_CACHE.items()}


_RISK_TERM_GENERIC_BLOCKLIST = {
    # MIT AI Risk Repository taxonomy / schema noise
    "paper", "papers", "additional evidence", "risk category", "risk subcategory",
    "category level", "domain", "subdomain", "causal factor",
    "intentional", "unintentional", "pre-deployment", "post-deployment",
    "human", "ai", "yes", "no", "n/a", "na", "true", "false",
    "tbd", "unknown", "see paper", "various",
    # Common spreadsheet noise
    "header", "footer", "sheet", "table", "row", "column",
}


def risk_row_terms(row: dict[str, Any]) -> list[str]:
    preferred = [
        "repo", "repo_id", "model", "model_id", "name", "artifact", "owner", "publisher",
        "tool", "alias", "indicator", "ioc", "keyword", "term", "title",
    ]
    terms: list[str] = []
    lowered = {str(k).lower().strip(): v for k, v in row.items()}
    for key in preferred:
        value = lowered.get(key)
        if value not in {None, ""}:
            terms.append(str(value))
    # Fallback: short textual cells from any column. Filter out plain English
    # words and generic schema noise — real indicators almost always contain
    # at least one non-letter character (digit, hyphen, slash, dot, colon)
    # because they're repo IDs, version strings, file names, or hashes.
    for key, value in row.items():
        if str(key).startswith("_") or value in {None, ""}:
            continue
        text = str(value).strip()
        if not (3 <= len(text) <= 140):
            continue
        if not any(ch.isalpha() for ch in text):
            continue
        # Real indicators (repo IDs, file names, version strings, hashes, URLs)
        # almost always contain a digit, slash, dot, or colon. Plain English
        # phrases — even compound ones like "Risk Sub-Category" — don't.
        # Hyphens and underscores are allowed in plain English so we don't
        # treat them as indicator markers.
        has_digit = any(ch.isdigit() for ch in text)
        has_strong_sep = any(ch in "/.:" for ch in text)
        long_enough = len(text) >= 60
        if not (has_digit or has_strong_sep or long_enough):
            continue
        if text.lower() in _RISK_TERM_GENERIC_BLOCKLIST:
            continue
        terms.append(text)
    # De-dup while preserving order.
    seen: set[str] = set()
    clean: list[str] = []
    for term in terms:
        t = term.strip()
        if not t or t.lower() in seen:
            continue
        seen.add(t.lower())
        clean.append(t)
    return clean


def risk_row_severity(row: dict[str, Any]) -> str:
    blob = normalize_match_text(" ".join(str(v) for k, v in row.items() if not str(k).startswith("_")))
    if any(x in blob for x in ["known malicious", "malicious", "malware", "backdoor", "block", "danger", "critical"]):
        return "DANGER"
    if any(x in blob for x in ["warn", "suspicious", "risk", "poison", "research", "dark web", "darkweb"]):
        return "WARN"
    return "INFO"


def selected_expected_files(result: SearchResult, allow_patterns: list[str] | None, artifacts: list[Artifact] | None) -> list[tuple[str, int | None]]:
    if artifacts:
        expected: list[tuple[str, int | None]] = []
        for art in artifacts:
            expected.extend(art.files)
        return expected
    if allow_patterns:
        matched: list[tuple[str, int | None]] = []
        for name, size in result.files:
            if any(fnmatch.fnmatch(name, pat) for pat in allow_patterns):
                matched.append((name, size))
        return matched
    return list(result.files)


_GGUF_FIRST_SHARD_RE = re.compile(r"-0*1-of-0*\d+\.gguf$", re.IGNORECASE)


def _print_ram_estimate_for_gguf_selection(
    result: SearchResult,
    expected_files: list[tuple[str, int | None]],
) -> None:
    """If any selected file is a GGUF, fetch its header via HF Range request,
    parse architecture metadata, and print a load-time RAM estimate at common
    context lengths. Silent (no error) if any step fails — purely advisory."""
    gguf_files = [(n, s) for n, s in expected_files if n.lower().endswith(".gguf")]
    if not gguf_files:
        return

    # Pick a header source: first shard if split, else largest single GGUF.
    header_source = None
    for name, size in gguf_files:
        if _GGUF_FIRST_SHARD_RE.search(name):
            header_source = name
            break
    if header_source is None:
        header_source = max(gguf_files, key=lambda x: x[1] or 0)[0]

    # Total weight = sum of selected GGUFs (handles sharded models)
    total_weight = sum(s or 0 for _, s in gguf_files)
    if total_weight == 0:
        return

    try:
        sys.path.insert(0, str(MODEL_MANAGER_SCRIPT_DIR))
        from gguf_inspect import (  # type: ignore
            fetch_gguf_header_bytes,
            parse_gguf_metadata,
            architecture_summary,
            estimate_load_ram,
            detect_machine_ram_bytes,
            format_size_gb,
        )
    except ImportError:
        return

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    print()
    print("Fetching GGUF header (~64 KB) for load-time RAM estimate…")
    header_bytes = fetch_gguf_header_bytes(
        repo_id=result.repo_id,
        filename=header_source,
        token=token,
    )
    if header_bytes is None:
        print("  Could not fetch GGUF header (network/auth/HF Range not allowed) — skipping estimate")
        return

    try:
        meta = parse_gguf_metadata(header_bytes)
    except Exception as e:
        print(f"  Could not parse GGUF metadata: {type(e).__name__}: {e}")
        return

    summary = architecture_summary(meta)
    if not summary.get("arch") or not summary.get("block_count"):
        print("  GGUF metadata incomplete — skipping estimate")
        return

    contexts = [4096, 8192, 16384, 32768]
    max_ctx = summary.get("context_length")
    if max_ctx and int(max_ctx) > 32768:
        contexts.append(int(max_ctx))
    contexts = sorted(set(contexts))

    estimates = estimate_load_ram(total_weight, summary, contexts)
    if not estimates:
        print("  Architecture missing required fields — cannot estimate RAM")
        return
    machine_ram = detect_machine_ram_bytes()

    print()
    print("Load-time RAM estimate")
    print("-" * 80)
    arch = summary["arch"]
    detail = f"Arch: {arch} ({summary['block_count']} layers, "
    detail += f"{summary.get('attention.head_count_kv') or summary.get('attention.head_count')} KV heads"
    if summary.get("expert_count"):
        detail += f", {summary['expert_count']} experts"
    detail += ")"
    print(detail)
    print(f"Weights: {human_size(total_weight)} ({len(gguf_files)} GGUF file{'s' if len(gguf_files) != 1 else ''})")
    if machine_ram:
        print(f"Machine RAM: {human_size(machine_ram)}")
    print()
    print(f"  {'Context':<10}{'fp16 KV':>10}{'q8 KV':>10}{'q4 KV':>10}    Verdict (q8 KV vs your RAM)")
    for ctx, row in estimates.items():
        q8 = row["q8"]
        verdict = ""
        if machine_ram:
            headroom_gb = (machine_ram - q8) / 1024**3
            if headroom_gb > 30:
                verdict = "✓ comfortable"
            elif headroom_gb > 10:
                verdict = f"⚠ tight (~{int(headroom_gb)} GB headroom)"
            elif headroom_gb > 0:
                verdict = "⚠ very tight"
            else:
                verdict = "✗ exceeds RAM"
        print(f"  {ctx:<10}{format_size_gb(row['fp16']):>10}{format_size_gb(row['q8']):>10}{format_size_gb(row['q4']):>10}    {verdict}")


def predownload_risk_and_completeness_check(
    result: SearchResult,
    allow_patterns: list[str] | None,
    artifacts: list[Artifact] | None,
    download_root: Path,
) -> bool:
    """Metadata-only check before spending hours downloading. Does not modify anything."""
    print()
    print("Pre-download check")
    print("-" * 80)
    print(f"Repo: {result.repo_id}")
    print(f"Source/kind: {result.source}/{result.kind}")
    print(f"Owner: {owner_of(result.repo_id)}")
    print(f"License: {result.license or 'not found'}")

    expected = selected_expected_files(result, allow_patterns, artifacts)
    expected_size = sum(size or 0 for _, size in expected)
    selected_label = "selected artifact(s)" if artifacts else ("selected patterns" if allow_patterns else "whole repo")
    print(f"Selection: {selected_label}")
    print(f"Expected selected files: {len(expected)}")
    print(f"Expected selected size: {human_size(expected_size or result.size_bytes)}")

    try:
        total, used, free = shutil.disk_usage(download_root)
        print(f"Download root free space: {human_size(free)}")
        if expected_size and free < expected_size * 1.10:
            print(f"BLOCKER: free space is below selected size + 10% safety margin ({human_size(int(expected_size * 1.10))}).")
            if not prompt_bool("Continue anyway?", False):
                return False
    except Exception as e:
        print(f"WARN: could not check free space: {type(e).__name__}: {e}")

    # Best-effort load-time RAM estimate (only for GGUF selections).
    # Fetches a ~64 KB Range from HF to read the GGUF header — silent on any failure.
    try:
        _print_ram_estimate_for_gguf_selection(result, expected)
    except Exception:
        pass

    findings: list[dict[str, str]] = []
    selected_names = [name for name, _ in expected]
    all_names = [name for name, _ in result.files]
    search_blob = normalize_match_text(" ".join([result.repo_id, result.title or ""] + selected_names + all_names[:50]))
    owner = owner_of(result.repo_id)

    reputation = load_owner_reputation_sets()
    bad_owners = reputation["known_malicious_owners"]
    warn_owners = reputation["warn_owners"]
    known_good = reputation["known_good_owners"]

    if owner in bad_owners:
        findings.append({"severity": "DANGER", "message": f"owner is in local known_malicious_owners: {owner}"})
    elif owner in warn_owners:
        findings.append({"severity": "WARN", "message": f"owner is in local warn_owners: {owner}"})
    elif owner in known_good:
        findings.append({"severity": "INFO", "message": f"owner appears in local known_good_owners: {owner}"})

    for term in sorted(BUILTIN_MALICIOUS_AI_TERMS):
        if normalize_match_text(term) and normalize_match_text(term) in search_blob:
            findings.append({"severity": "DANGER", "message": f"name/artifact matches built-in malicious-AI term: {term}"})
    for term in sorted(BUILTIN_SECURITY_RESEARCH_TERMS):
        nt = normalize_match_text(term)
        if nt and nt in search_blob:
            findings.append({"severity": "WARN", "message": f"name/artifact contains risk keyword: {term}"})

    risky_file_matches = [name for name in all_names if re.search(r"(^|/)(setup\.py|requirements\.txt|.*modeling.*\.py|.*tokenization.*\.py|.*configuration.*\.py|.*\.pkl|.*\.pickle|.*\.pt|.*\.pth|.*\.bin)$", name, re.I)]
    for name in risky_file_matches[:12]:
        sev = "WARN"
        if name.lower().endswith((".pkl", ".pickle", ".pt", ".pth", ".bin")):
            sev = "WARN"
        findings.append({"severity": sev, "message": f"repo contains code/deserialization-risk file: {name}"})
    if len(risky_file_matches) > 12:
        findings.append({"severity": "INFO", "message": f"repo contains {len(risky_file_matches) - 12} additional risky file(s) not shown"})

    # Split GGUF sanity from metadata only.
    if artifacts:
        for art in artifacts:
            if art.artifact_type == "gguf-split":
                parts = [name for name, _ in art.files]
                joined = " ".join(parts)
                m = re.search(r"of[-_\.](\d+)", joined, re.I)
                if m and len(parts) != int(m.group(1)):
                    findings.append({"severity": "BLOCKER", "message": f"split GGUF metadata looks incomplete for {art.label}: found {len(parts)} part(s), expected {m.group(1)}"})

    # Workbook / local sheet checks.
    rows = load_risk_intel_rows()
    matched_rows = []
    for row in rows:
        for term in risk_row_terms(row):
            nt = normalize_match_text(term)
            if not nt or len(nt) < 4:
                continue
            # Exact repo/owner/artifact-ish match, not fuzzy broad one-word matches.
            if nt in search_blob:
                matched_rows.append((risk_row_severity(row), term, row))
                break
    for severity, term, row in matched_rows[:20]:
        src = row.get("_source_file", "local risk workbook")
        sheet = row.get("_sheet")
        loc = f"{src}" + (f"::{sheet}" if sheet else "")
        findings.append({"severity": severity, "message": f"local risk intel match '{term}' from {loc}"})
    if len(matched_rows) > 20:
        findings.append({"severity": "INFO", "message": f"{len(matched_rows) - 20} additional local risk workbook match(es) not shown"})

    if findings:
        print("Findings:")
        for f in findings:
            print(f"  [{f['severity']}] {f['message']}")
    else:
        print("Findings: none from metadata/local risk intel.")

    serious = [f for f in findings if f["severity"] in {"DANGER", "BLOCKER"}]
    if serious:
        print()
        print("One or more DANGER/BLOCKER findings were found before download.")
        return prompt_bool("Download anyway?", False)

    if findings:
        return prompt_bool("Proceed with download despite WARN/INFO findings?", True)
    return True

def post_download_audit(result: SearchResult, target: Path, allow_patterns: list[str] | None) -> bool:
    findings = verify_download_integrity(target, result.files, allow_patterns)
    print_findings(findings)
    scanner_findings = run_external_security_tools(target)
    if scanner_findings:
        print_findings(scanner_findings)
    all_findings = findings + scanner_findings
    offer_delete, reason = should_offer_delete(all_findings)
    if offer_delete:
        delete_path_after_confirmation(target, reason)
        if not target.exists():
            return False
        print("Serious finding remains in staged download. Keeping it staged unless you explicitly continue.")
        return prompt_bool("Install staged download anyway?", False)
    if all_findings:
        print("No DANGER/BLOCKER findings requiring delete offer; staged files can be installed after review.")
        return prompt_bool("Install staged download despite WARN/INFO/HIGH findings?", True)
    return True


def download_selected(results: list[SearchResult], download_root_override: Path | None = None) -> None:
    if not results:
        return
    results = review_download_queue(results)
    if not results:
        print("No selections remain. Nothing to download.")
        return
    if download_root_override is not None:
        download_root = download_root_override.expanduser().resolve()
        print(f"Download root: {download_root}")
    else:
        download_root = Path(prompt("Download root", str(DEFAULT_DOWNLOAD_DIR))).expanduser().resolve()
    download_root.mkdir(parents=True, exist_ok=True)
    for r in results:
        print()
        print("=" * 110)
        print(f"Selected: {r.source}:{r.kind}:{r.repo_id}")
        print(f"License/card: {r.license or 'not found'}")
        print("Reminder: model/dataset license and commercial-use terms are upstream-controlled; review the card before redistribution or commercial use.")
        if r.source == "hf" and r.kind == "model" and r.selected_artifacts:
            artifacts = r.selected_artifacts
            patterns = []
            for art in artifacts:
                patterns.extend(art.allow_patterns())
            seen_patterns: set[str] = set()
            patterns = [pat for pat in patterns if not (pat in seen_patterns or seen_patterns.add(pat))]
            print("Using preselected artifact(s):")
            for art in artifacts:
                print(f"  - {art.label} ({human_size(art.total_size)})")
            mode = "artifact_numbers"
        elif r.source == "hf" and r.kind == "model" and r.whole_repo_selected:
            mode, patterns, artifacts = "whole_repo", None, []
        elif r.source == "hf" and r.kind == "model":
            mode, patterns, artifacts = choose_artifacts(r)
        else:
            mode, patterns = choose_file_patterns_legacy(r)
            artifacts = []
        if mode in {"skip", ""}:
            continue
        if not predownload_risk_and_completeness_check(r, patterns, artifacts, download_root):
            print("Skipped before download.")
            continue
        target: Path | None = None
        try:
            if r.source == "hf":
                target = download_hf_result(r, download_root, patterns, artifacts)
            elif r.source == "kaggle":
                target = download_kaggle_result(r, download_root, patterns)
        except Exception as e:
            debug_log(
                "download-failed",
                repo_id=r.repo_id,
                source=r.source,
                kind=r.kind,
                exc_type=type(e).__name__,
                exc_message=str(e),
                traceback=traceback.format_exc(),
            )
            print(f"Download failed: {type(e).__name__}: {e}")
            continue
        if target:
            ok_to_install = post_download_audit(r, target, patterns)
            if not ok_to_install:
                print(f"Not installing staged download. Staged path: {target if target.exists() else 'deleted'}")
                continue
            final_target = final_download_path(r, download_root)
            installed = promote_staged_download(target, final_target)
            if installed and r.kind == "model":
                offer_prepare_models(installed, download_root)

# -----------------------------------------------------------------------------
# Dataset sampling
# -----------------------------------------------------------------------------

def sample_hf_dataset(repo_id: str, n: int = 5) -> None:
    print(f"Sampling HF dataset: {repo_id}")
    try:
        from datasets import load_dataset
        ds = load_dataset(repo_id, split="train", streaming=True)
        for i, row in enumerate(ds):
            print(json.dumps(row, ensure_ascii=False, indent=2, default=str)[:4000])
            if i + 1 >= n:
                break
    except Exception as e:
        print(f"Streaming sample failed: {type(e).__name__}: {e}")


def sample_local_file(path: Path, n: int = 5) -> None:
    suffix = path.suffix.lower()
    print(f"Sample from {path}:")
    try:
        if suffix in {".csv", ".tsv"}:
            delim = "\t" if suffix == ".tsv" else ","
            with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.reader(f, delimiter=delim)
                for i, row in enumerate(reader, 1):
                    print(row[:30])
                    if i >= n:
                        break
        elif suffix in {".jsonl", ".txt"}:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f, 1):
                    print(line[:2000].rstrip())
                    if i >= n:
                        break
        elif suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(data, list):
                for row in data[:n]:
                    print(json.dumps(row, ensure_ascii=False, indent=2, default=str)[:2000])
            else:
                print(json.dumps(data, ensure_ascii=False, indent=2, default=str)[:4000])
        elif suffix == ".parquet":
            import pandas as pd
            print(pd.read_parquet(path).head(n).to_string())
        else:
            print(path.read_text(encoding="utf-8", errors="replace")[:4000])
    except Exception as e:
        print(f"Sample failed for {path}: {type(e).__name__}: {e}")


def sample_kaggle_dataset(ref: str, n: int = 5) -> None:
    print(f"Sampling Kaggle dataset: {ref}")
    files = kaggle_dataset_files(ref)
    if not files:
        print("Could not list Kaggle dataset files.")
        return
    candidates = [(name, size) for name, size in files if Path(name).suffix.lower() in DATA_SAMPLE_EXTENSIONS]
    if not candidates:
        print("No obvious small sampleable file found.")
        return
    name, size = sorted(candidates, key=lambda x: x[1] or 0)[0]
    print(f"Downloading sample file only: {name} ({human_size(size)})")
    with tempfile.TemporaryDirectory() as td:
        api = kaggle_api()
        try:
            api.dataset_download_file(ref, name, path=td, quiet=False)
        except Exception as e:
            print(f"Sample file download failed: {type(e).__name__}: {e}")
            return
        found = list(Path(td).rglob(Path(name).name))
        if not found:
            zips = list(Path(td).glob("*.zip"))
            if zips:
                shutil.unpack_archive(str(zips[0]), td)
                found = list(Path(td).rglob(Path(name).name))
        if found:
            sample_local_file(found[0], n=n)
        else:
            print("Sample file not found after download/unpack.")


def offer_dataset_sampling(results: list[SearchResult]) -> None:
    datasets = [r for r in results if r.kind == "dataset"]
    if not datasets or not prompt_bool("Sample any selected datasets before downloading?", True):
        return
    for r in datasets:
        if not prompt_bool(f"Sample {r.source}:{r.repo_id}?", True):
            continue
        n = prompt_int("Number of sample rows", 5)
        if r.source == "hf":
            sample_hf_dataset(r.repo_id, n=n)
        elif r.source == "kaggle":
            sample_kaggle_dataset(r.repo_id, n=n)

# -----------------------------------------------------------------------------
# Prep handoff
# -----------------------------------------------------------------------------

def script_accepts_flag(script: Path, flag: str) -> bool:
    try:
        proc = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, timeout=10)
        return flag in ((proc.stdout or "") + (proc.stderr or ""))
    except Exception:
        return False


def _find_first_existing(paths: list[Path]) -> Path | None:
    return next((p.expanduser().resolve() for p in paths if p.expanduser().is_file()), None)


def offer_model_conversion(target: Path, download_root: Path) -> None:
    if not prompt_bool("Run model conversion workflow after prep?", False):
        return
    script = _find_first_existing(CONVERSION_SCRIPT_CANDIDATES)
    if not script:
        print("model_conversion.py not found. Expected one of:")
        for p in CONVERSION_SCRIPT_CANDIDATES:
            print(f"  - {p.expanduser()}")
        return

    print("Conversion helper:", script)
    print("Downloaded model path:", target)
    print("Download root:", download_root)
    env = os.environ.copy()
    env["MODEL_MANAGER_DOWNLOAD_TARGET"] = str(target)
    env["MODEL_MANAGER_DOWNLOAD_ROOT"] = str(download_root)

    if prompt_bool("List conversion candidates only first?", True):
        cmd = [sys.executable, str(script), "--list-only"]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=False, env=env, cwd=str(download_root))
        if not prompt_bool("Run interactive conversion now?", False):
            return

    quant = prompt("Default GGUF quant for conversion", "Q4_K_M")
    workers = prompt_int("Conversion workers", 1)
    cmd = [sys.executable, str(script), "--quant", quant, "--workers", str(workers)]
    if prompt_bool("Skip audit inside conversion script?", True):
        cmd.append("--no-audit")
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=False, env=env, cwd=str(download_root))
    except Exception as e:
        print(f"Conversion step failed: {type(e).__name__}: {e}")


def offer_prepare_models(target: Path, download_root: Path | None = None) -> None:
    if not prompt_bool("Prepare this model for local/open-source hosting apps?", False):
        return
    script = _find_first_existing(PREP_SCRIPT_CANDIDATES)
    if not script:
        print("Prepare_models_for_All.py not found. Expected one of:")
        for p in PREP_SCRIPT_CANDIDATES:
            print(f"  - {p.expanduser()}")
        return

    target = target.expanduser().resolve()
    download_root = (download_root or target.parent).expanduser().resolve()

    print("Known prep orchestrator:", script)
    print("Downloaded model path:", target)
    print("Download root:", download_root)
    print("Supported prep targets when matching per-app scripts exist:")
    print("  " + ", ".join(PREP_APP_ORDER))
    print("Notes:")
    print("  - LM Studio is treated as the canonical layout/hub step.")
    print("  - Ollama should run before AnythingLLM because AnythingLLM can depend on Ollama listing.")
    print("  - AIStudio / AI Navigator DB injection should run late; quit AI Navigator first.")
    print("  - LocallyAI, LocalAI, Apollo, and OffGrid require corresponding Prepare_models_for_<App>.py scripts.")

    apps = prompt("Only these apps, comma-separated, or blank for all", "")
    dry = prompt_bool("Dry run first?", True)
    cmd = [sys.executable, str(script)]
    if dry:
        cmd.append("--dry-run")
    if apps:
        cmd += ["--only", apps]
    if script_accepts_flag(script, "--continue-on-error"):
        cmd.append("--continue-on-error")
    if script_accepts_flag(script, "--scripts-dir"):
        cmd += ["--scripts-dir", str(script.parent)]

    # Be download-location aware when the prep script supports it. Older prep scripts
    # may not define these flags, so environment variables are always supplied too.
    for flag in ("--model-dir", "--models-dir", "--input-dir", "--download-dir"):
        if script_accepts_flag(script, flag):
            cmd += [flag, str(target)]
            break
    if script_accepts_flag(script, "--download-root"):
        cmd += ["--download-root", str(download_root)]

    env = os.environ.copy()
    env["MODEL_MANAGER_DOWNLOAD_TARGET"] = str(target)
    env["MODEL_MANAGER_DOWNLOAD_ROOT"] = str(download_root)
    env["MODEL_MANAGER_PREP_APPS"] = apps or ",".join(PREP_APP_ORDER)

    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=False, env=env, cwd=str(download_root))
    except Exception as e:
        print(f"Prepare step failed: {type(e).__name__}: {e}")

    offer_model_conversion(target, download_root)

# -----------------------------------------------------------------------------
# Leaderboards
# -----------------------------------------------------------------------------

def show_leaderboard_sources(
    categories_override: list[str] | None = None,
    search_hf_spaces: bool | None = None,
    manual_cache_update: bool | None = None,
    refresh_cache: bool | None = None,
    refresh_limit: int | None = None,
) -> None:
    valid_categories = ["general", "coding", "visual", "security", "embedding"]
    if categories_override:
        categories = [c for c in categories_override if c in valid_categories]
        if not categories:
            categories = ["coding"]
    else:
        categories = prompt_multi_choice("Leaderboard category", valid_categories, "coding")
    print()
    for category in categories:
        print(f"Useful current leaderboard sources for {category}:")
        for name, url in LEADERBOARD_SOURCES[category]:
            print(f"  - {name}: {url}")
        print()
    if (search_hf_spaces if search_hf_spaces is not None else prompt_bool("Also search Hugging Face Spaces for matching leaderboard spaces?", True)):
        try:
            api = hf_api()
            terms_by_category = {
                "coding": "coding leaderboard",
                "visual": "vision leaderboard",
                "security": "security leaderboard",
                "embedding": "embedding leaderboard",
                "general": "LLM leaderboard",
            }
            seen_spaces: set[str] = set()
            for category in categories:
                terms = terms_by_category[category]
                spaces = list(api.list_spaces(search=terms, limit=20))
                print()
                print(f"HF Spaces search: {terms}")
                for s in spaces:
                    sid = getattr(s, "id", None) or str(s)
                    if sid in seen_spaces:
                        continue
                    seen_spaces.add(sid)
                    likes = getattr(s, "likes", None)
                    updated = getattr(s, "lastModified", None)
                    print(f"  - {sid} | likes={likes} | updated={updated} | https://huggingface.co/spaces/{sid}")
        except Exception as e:
            print(f"HF Spaces search failed: {type(e).__name__}: {e}")
    print()
    print(f"Leaderboard cache path: {LEADERBOARD_CACHE_PATH}")
    do_manual = manual_cache_update if manual_cache_update is not None else prompt_bool("Add/update cached leaderboard model repo IDs manually?", False)
    if do_manual:
        for category in categories:
            add_manual_leaderboard_cache(category)
    do_refresh = refresh_cache if refresh_cache is not None else prompt_bool("Refresh best-effort cached leaderboard model IDs from HF searches?", False)
    if do_refresh:
        limit = refresh_limit if refresh_limit is not None else prompt_int("Cache limit per category", 20)
        refresh_leaderboard_cache_best_effort(categories, limit=limit)

# -----------------------------------------------------------------------------
# Main flows
# -----------------------------------------------------------------------------

def collect_search_results(
    provider_search_terms: list[str],
    source: str,
    kind: str,
    limit: int,
    hf_model_candidate_limit: int | None = None,
    pre_excluded_publishers: set[str] | None = None,
    pre_excluded_terms: set[str] | None = None,
    pre_excluded_family_terms: list[str] | None = None,
) -> list[SearchResult]:
    results: list[SearchResult] = []
    for term in provider_search_terms:
        if source in {"huggingface", "both"} and kind in {"models", "both"}:
            if hf_model_candidate_limit and hf_model_candidate_limit > limit:
                print(f"Searching Hugging Face models: {term} (deep scan up to {hf_model_candidate_limit} candidates)")
            else:
                print(f"Searching Hugging Face models: {term}")
            results.extend(
                search_hf_models(
                    term,
                    limit,
                    candidate_limit=hf_model_candidate_limit,
                    pre_excluded_publishers=pre_excluded_publishers,
                    pre_excluded_terms=pre_excluded_terms,
                    pre_excluded_family_terms=pre_excluded_family_terms,
                )
            )
        if source in {"huggingface", "both"} and kind in {"datasets", "both"}:
            print(f"Searching Hugging Face datasets: {term}")
            results.extend(search_hf_datasets(term, limit))
        if source in {"kaggle", "both"} and kind in {"datasets", "both"}:
            print(f"Searching Kaggle datasets: {term}")
            try:
                results.extend(search_kaggle_datasets(term, limit))
            except Exception as e:
                print(f"Kaggle dataset search failed for {term}: {type(e).__name__}: {e}")
        if source in {"kaggle", "both"} and kind in {"models", "both"}:
            print(f"Searching Kaggle models best-effort: {term}")
            try:
                kg_models = search_kaggle_models(term, limit)
                if not kg_models:
                    print("No Kaggle model results or Kaggle model search unavailable.")
                results.extend(kg_models)
            except Exception as e:
                print(f"Kaggle model search failed for {term}: {type(e).__name__}: {e}")
    return results


def filter_and_rank_search_results(
    results: list[SearchResult],
    kind: str,
    selected_artifact_types: set[str],
    model_size_range: tuple[int | None, int | None],
    exclude_multipart_models: bool,
    excluded_publishers: set[str],
    excluded_terms: set[str],
    excluded_family_terms: list[str],
    hide_duplicate_families_pref: bool,
    quiet_family_no_match: bool = False,
    excluded_existing_repo_ids: set[str] | None = None,
) -> list[SearchResult]:
    results = merge_search_results(results)
    results = apply_publisher_exclusions(results, excluded_publishers)
    results = apply_term_exclusions(results, excluded_terms)
    if excluded_existing_repo_ids:
        results = apply_existing_repo_exclusions(results, excluded_existing_repo_ids)
    if kind in {"models", "both"}:
        results = filter_results_by_artifact_types(results, selected_artifact_types)
        results = filter_results_by_model_size(results, model_size_range)
        results = filter_results_by_multipart_models(results, exclude_multipart_models)
    annotate_results_with_leaderboard_cache(results)
    annotate_recommendations(results)
    results = apply_model_family_exclusions(results, excluded_family_terms, quiet_no_match=quiet_family_no_match)
    if results and hide_duplicate_families_pref:
        before = len(results)
        results = hide_duplicate_families(results)
        after = len(results)
        if after < before:
            print(f"Hidden {before - after} likely duplicate/mirror repo(s). Kept strongest candidate per exact variant family.")
    results.sort(key=lambda r: (r.leaderboard_rank is None, r.leaderboard_rank or 10**9, r.recommendation not in {"KnownGood", "KnownGood/DF", "Leaderboard"}, -(r.downloads or 0), -(r.size_bytes or 0)))
    assign_indexes(results)
    return results


def deep_hf_candidate_limit(limit: int) -> int:
    return min(max(limit * 20, 200), 500)


def run_search_flow(options: argparse.Namespace | None = None) -> None:
    check_airisk_mit_update_quietly()
    warn_if_incoming_inside_download_root()
    # Set the session-level scanner preference once, up front. Honors
    # --scan-after-download {ask,always,never} CLI flag and
    # MODEL_MANAGER_SCAN_AFTER_DOWNLOAD env var.
    explicit_scan_pref: bool | None | str = None
    if options is not None:
        cli_value = getattr(options, "scan_after_download", None)
        explicit_scan_pref = _parse_scan_after_download_choice(cli_value)
    prompt_session_scan_preference(explicit_scan_pref)
    offer_resume_active_downloads()
    download_root_override = None
    if options is not None and getattr(options, "download_root", None):
        download_root_override = Path(options.download_root).expanduser().resolve()
    debug_log(
        "run-search-flow-start",
        options_supplied=options is not None,
        download_root_override=str(download_root_override) if download_root_override else None,
        specific_repo=getattr(options, "specific_repo", None) if options is not None else None,
        search_query=getattr(options, "search_query", None) if options is not None else None,
    )

    specific_repo_arg = getattr(options, "specific_repo", None) if options is not None else None
    if specific_repo_arg or (options is None and prompt_bool("Do you have a single specific model or dataset to download?", False)):
        value = specific_repo_arg or prompt("Paste Hugging Face repo ID or URL", "")
        repo_id = parse_hf_repo_id(value)
        if not repo_id:
            print("Could not parse a Hugging Face repo ID. Use owner/name or a huggingface.co URL.")
            return
        kind = getattr(options, "specific_kind", None) if options is not None else None
        if kind not in {"model", "dataset"}:
            kind = prompt_choice("Specific item type", ["model", "dataset"], "model")
        result = build_exact_hf_result(repo_id, kind=kind)
        if not result:
            return
        annotate_results_with_leaderboard_cache([result])
        assign_indexes([result])
        print_results_page([result], 0, 1)
        selected = paged_select([result], page_size=1)
        if not selected:
            print("No selection.")
            return
        offer_local_scan(selected)
        offer_dataset_sampling(selected)
        download_selected(selected, download_root_override)
        return

    while True:
        if options is not None and getattr(options, "search_query", None) is not None:
            query = options.search_query
        else:
            query = prompt('Search term. Supports OR or commas, e.g. code OR cybersecurity or code, coding', "")
        if not query:
            print("No query entered.")
            if options is None and prompt_bool("Try another search?", True):
                continue
            return
        raw_terms = split_boolean_query(query)
        # Partition into exact repo-ID lookups vs name-based search terms.
        # A term matching `owner/repo` shape is treated as a direct HF lookup
        # (build_exact_hf_result) rather than a substring search — HF's search
        # endpoint doesn't index the slash, so e.g. `meta-llama/Llama-Prompt-
        # Guard-2-86M` would otherwise return zero hits even though the repo
        # exists. Quoted repo IDs work too; split_boolean_query already strips
        # the quotes.
        direct_repo_ids: list[str] = []
        search_terms: list[str] = []
        for term in raw_terms:
            parsed = parse_hf_repo_id(term)
            if parsed and "/" in parsed:
                direct_repo_ids.append(parsed)
            else:
                search_terms.append(term)
        if direct_repo_ids and search_terms:
            print(f"Detected {len(direct_repo_ids)} direct repo ID(s) + {len(search_terms)} search term(s):")
            print("  Direct (exact fetch): " + ", ".join(direct_repo_ids))
            print("  Search terms: " + ", ".join(search_terms))
        elif direct_repo_ids:
            print(f"Resolved {len(direct_repo_ids)} direct Hugging Face repo ID(s) — fetching exactly these (skipping name-based search):")
            for rid in direct_repo_ids:
                print(f"  - {rid}")
        elif len(search_terms) > 1:
            print(f"Split search into {len(search_terms)} separate searches:")
            for term in search_terms:
                print(f"  - {term}")
        kind = getattr(options, "search_kind", None) if options is not None else None
        if kind not in {"models", "datasets", "both"}:
            kind = prompt_choice("Search for", ["models", "datasets", "both"], "models")
        # Fetch direct repo IDs eagerly so we can short-circuit later if the
        # user typed only repo IDs (skip all the search-term expansion and
        # provider round-tripping).
        direct_results: list[SearchResult] = []
        if direct_repo_ids:
            # For "both", try as model first; if that fails fall back to dataset.
            direct_kinds_to_try: list[str]
            if kind == "datasets":
                direct_kinds_to_try = ["dataset"]
            elif kind == "models":
                direct_kinds_to_try = ["model"]
            else:
                direct_kinds_to_try = ["model", "dataset"]
            for rid in direct_repo_ids:
                fetched: SearchResult | None = None
                for k in direct_kinds_to_try:
                    fetched = build_exact_hf_result(rid, kind=k)
                    if fetched is not None:
                        break
                if fetched is None:
                    print(f"  ✗ direct fetch failed for {rid} (not a valid HF model or dataset)")
                    continue
                fetched.direct_lookup = True
                direct_results.append(fetched)
            if direct_results:
                print(f"  ✓ {len(direct_results)}/{len(direct_repo_ids)} direct repo lookup(s) succeeded.")
        selected_artifact_types: set[str] = set()
        model_size_range: tuple[int | None, int | None] = (None, None)
        exclude_multipart_models = False
        if kind in {"models", "both"}:
            selected_from_args = parse_artifact_type_filter_value(getattr(options, "artifact_types", None)) if options is not None else None
            selected_artifact_types = selected_from_args if selected_from_args is not None else choose_artifact_type_filters()
            size_from_args = parse_model_size_range_or_default(getattr(options, "model_size_range", None)) if options is not None else None
            model_size_range = size_from_args if size_from_args is not None else prompt_model_size_range()
            exclude_multipart_arg = getattr(options, "exclude_multipart_models", None) if options is not None else None
            if exclude_multipart_arg is not None:
                exclude_multipart_models = bool(exclude_multipart_arg)
            elif multipart_artifact_filter_relevant(selected_artifact_types):
                exclude_multipart_models = prompt_exclude_multipart_models(selected_artifact_types)
        if options is not None and getattr(options, "exclude_publishers", None) is not None:
            excluded_publishers = parse_excluded_publishers_value(options.exclude_publishers)
        else:
            excluded_publishers = prompt_excluded_publishers()
        if options is not None and getattr(options, "exclude_terms", None) is not None:
            excluded_terms = parse_excluded_terms_value(options.exclude_terms)
        else:
            excluded_terms = prompt_excluded_terms()
        existing_dirs: list[Path] = []
        if options is not None and getattr(options, "exclude_existing_dirs", None) is not None:
            existing_dirs = parse_existing_dirs_value(options.exclude_existing_dirs)
        else:
            existing_dirs = prompt_existing_dirs()
        excluded_existing_repo_ids: set[str] = set()
        if existing_dirs:
            print("Scanning existing-content directories for already-downloaded repos...")
            excluded_existing_repo_ids = collect_existing_repo_ids(existing_dirs)
            if excluded_existing_repo_ids:
                print(
                    f"Will hide {len(excluded_existing_repo_ids)} already-present repo(s) "
                    f"from results."
                )
            else:
                print("  No recognizable existing repo dirs found.")
        source = getattr(options, "search_source", None) if options is not None else None
        if source not in {"huggingface", "kaggle", "both"}:
            source = prompt_choice("Search source", ["huggingface", "kaggle", "both"], "huggingface")
        limit = getattr(options, "result_limit", None) if options is not None else None
        if limit is None:
            limit = prompt_int("How many results per source/type/search term?", 50)
        limit = max(1, int(limit))
        page_size = getattr(options, "page_size", None) if options is not None else None
        if page_size is None:
            page_size = prompt_int("Display batch size", 20)
        page_size = max(1, int(page_size))
        hide_duplicate_families_pref: bool | None = False
        excluded_family_terms: list[str] = []
        if kind in {"models", "both"}:
            hide_arg = getattr(options, "hide_duplicate_families", None) if options is not None else None
            hide_duplicate_families_pref = hide_arg if hide_arg is not None else None
            if options is not None and getattr(options, "exclude_families", None) is not None:
                excluded_family_terms = parse_model_family_exclusion_terms(options.exclude_families)
            else:
                excluded_family_terms = prompt_model_family_exclusion_terms()

        if kind in {"models", "both"}:
            # Expand each base search term independently so the cap is per-base,
            # not global. With "Kimi-K2.6, DeepSeek, Grok" and a cap of 5, this
            # produces 5 variants for each — not 5 total dominated by one base.
            seen_terms: set[str] = set()
            provider_search_terms = []
            pre_cap_total = 0
            capped_any = False
            for base in search_terms:
                expanded = expand_search_terms_for_model_name_variants([base])
                expanded = expand_search_terms_for_artifacts(expanded, selected_artifact_types)
                pre_cap_total += len(expanded)
                if len(expanded) > _HF_SEARCH_MAX_TERMS:
                    expanded = expanded[:_HF_SEARCH_MAX_TERMS]
                    capped_any = True
                for term in expanded:
                    if term not in seen_terms:
                        seen_terms.add(term)
                        provider_search_terms.append(term)
            if capped_any and pre_cap_total > len(provider_search_terms):
                print(
                    f"Capping expanded search terms from {pre_cap_total} to "
                    f"{len(provider_search_terms)} ({_HF_SEARCH_MAX_TERMS} per base term; "
                    f"MODEL_MANAGER_HF_SEARCH_MAX_TERMS to override)."
                )
        else:
            provider_search_terms = list(search_terms)
        if provider_search_terms != search_terms:
            print("Expanded search with model naming variants and artifact filename terms:")
            for term in provider_search_terms[:20]:
                print(f"  - {term}")
            if len(provider_search_terms) > 20:
                print(f"  ... {len(provider_search_terms) - 20} more")
        debug_log(
            "search-config",
            search_terms=search_terms,
            provider_search_terms=provider_search_terms[:20],
            provider_search_term_count=len(provider_search_terms),
            kind=kind,
            source=source,
            limit=limit,
            page_size=page_size,
            selected_artifact_types=sorted(selected_artifact_types),
            model_size_range=size_range_label(model_size_range),
            exclude_multipart_models=exclude_multipart_models,
            excluded_publishers=sorted(excluded_publishers),
            excluded_terms=sorted(excluded_terms),
            excluded_family_terms=excluded_family_terms,
            hide_duplicate_families=hide_duplicate_families_pref if hide_duplicate_families_pref is not None else "prompt_after_search",
        )

        # Run the name-based search only if there are non-direct terms left.
        # Otherwise (user typed only repo IDs) skip the round-trip entirely.
        if provider_search_terms:
            raw_results = collect_search_results(provider_search_terms, source, kind, limit)
        else:
            raw_results = []
        # Direct lookups go FIRST so merge_search_results keeps them as the
        # primary record (preserving the direct_lookup flag) when a duplicate
        # also shows up through the name-search.
        if direct_results:
            raw_results = list(direct_results) + raw_results
        raw_hit_count = len(raw_results)
        merged_hit_count = len(merge_search_results(raw_results))
        if kind in {"models", "both"} and hide_duplicate_families_pref is None:
            hide_duplicate_families_pref = prompt_hide_duplicate_families_preference(raw_results)
        results = filter_and_rank_search_results(
            raw_results,
            kind,
            selected_artifact_types,
            model_size_range,
            exclude_multipart_models,
            excluded_publishers,
            excluded_terms,
            excluded_family_terms,
            bool(hide_duplicate_families_pref),
            excluded_existing_repo_ids=excluded_existing_repo_ids,
        )
        debug_log(
            "search-results",
            raw_hit_count=raw_hit_count,
            merged_hit_count=merged_hit_count,
            displayed_count=len(results),
            displayed_indexes=[r.index for r in results[:20]],
            displayed_repo_ids=[r.repo_id for r in results[:20]],
            excluded_existing_repo_id_count=len(excluded_existing_repo_ids),
        )
        if raw_hit_count or merged_hit_count or results:
            summary_parts = [f"Showing {len(results)} result(s)"]
            if merged_hit_count != len(results):
                summary_parts.append(f"from {merged_hit_count} merged repo hit(s)")
            if raw_hit_count != merged_hit_count:
                summary_parts.append(f"from {raw_hit_count} raw hit(s)")
            print("Search summary: " + "; ".join(summary_parts) + ".")

        # Deep scan is a NAME-based fallback — only meaningful when there are
        # search terms left. If the user typed only direct repo IDs and a
        # direct fetch failed (e.g. wrong owner spelling), the deep scan
        # won't help — skip it. Otherwise re-fetch into raw_results AND
        # re-prepend direct lookups so they survive the deep-scan path too.
        if source in {"huggingface", "both"} and kind in {"models", "both"} and len(results) < page_size and provider_search_terms:
            deep_limit = deep_hf_candidate_limit(limit)
            if deep_limit > limit:
                print()
                print(
                    f"Only {len(results)} result(s) survived the filters. "
                    f"Deepening Hugging Face model scan to {deep_limit} candidates per search term before giving up."
                )
                deep_terms = artifact_focused_search_terms(provider_search_terms, search_terms)
                raw_results = collect_search_results(
                    deep_terms,
                    "huggingface",
                    "models",
                    limit,
                    hf_model_candidate_limit=deep_limit,
                    pre_excluded_publishers=excluded_publishers,
                    pre_excluded_terms=excluded_terms,
                    pre_excluded_family_terms=excluded_family_terms,
                )
                if direct_results:
                    raw_results = list(direct_results) + raw_results
                raw_hit_count = len(raw_results)
                merged_hit_count = len(merge_search_results(raw_results))
                results = filter_and_rank_search_results(
                    raw_results,
                    "models",
                    selected_artifact_types,
                    model_size_range,
                    exclude_multipart_models,
                    excluded_publishers,
                    excluded_terms,
                    excluded_family_terms,
                    bool(hide_duplicate_families_pref),
                    quiet_family_no_match=True,
                    excluded_existing_repo_ids=excluded_existing_repo_ids,
                )
                debug_log(
                    "search-results-deep-scan",
                    raw_hit_count=raw_hit_count,
                    merged_hit_count=merged_hit_count,
                    displayed_count=len(results),
                    displayed_indexes=[r.index for r in results[:20]],
                    displayed_repo_ids=[r.repo_id for r in results[:20]],
                )
                if raw_hit_count or merged_hit_count or results:
                    summary_parts = [f"Showing {len(results)} result(s) after deep scan"]
                    if merged_hit_count != len(results):
                        summary_parts.append(f"from {merged_hit_count} merged repo hit(s)")
                    if raw_hit_count != merged_hit_count:
                        summary_parts.append(f"from {raw_hit_count} raw hit(s)")
                    print("Search summary: " + "; ".join(summary_parts) + ".")
        if not results:
            print("No results.")
            print("Try a broader term, a different artifact type, or another source.")
            if options is None and prompt_bool("Search again?", True):
                continue
            return
        selected = paged_select(results, page_size=page_size)
        if not selected:
            print("No selection.")
            if options is None and prompt_bool("Start another search?", True):
                continue
            return
        offer_local_scan(selected)
        offer_dataset_sampling(selected)
        download_selected(selected, download_root_override)
        return


def run_local_audit_only(dry_run: bool | None = None) -> None:
    script_candidates = [
        MODEL_MANAGER_SCRIPT_DIR / "model_audit.py",
        Path("<Your Model Directory>/model_audit.py"),
        Path("./model_audit.py"),
        Path.home() / "model_tools" / "model_audit.py",
        Path.home() / "model_audit.py",
    ]
    script = next((p for p in script_candidates if p.is_file()), None)
    if script:
        cmd = [sys.executable, str(script)]
        if (dry_run if dry_run is not None else prompt_bool("Dry run?", True)):
            cmd.append("--dry-run")
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=False)
        return
    print("model_audit.py not found. Running built-in local similarity scan instead.")
    query = prompt("Model name/search term")
    hits = scan_local_candidates([query], DEFAULT_LOCAL_MODEL_DIRS)
    for name, paths in hits.items():
        print(name)
        for p in paths[:100]:
            print(f"  {p}")


def launch_native_gui() -> None:
    script = Path(__file__).expanduser().resolve().parent / "model_manager_gui.swift"
    swift = shutil.which("swift")
    if not swift:
        print("Swift was not found on PATH; cannot launch the native macOS GUI.")
        return
    if not script.is_file():
        print(f"GUI launcher not found: {script}")
        return
    cmd = [swift, str(script), "--download-root", str(DEFAULT_DOWNLOAD_DIR)]
    print(f"Launching native macOS GUI: {quote_cmd(cmd)}")
    subprocess.run(cmd, check=False)


def main() -> int:
    configure_hf_download_environment()
    print(f"Model Manager {MODEL_MANAGER_VERSION}")
    print(f"Active script: {Path(__file__).expanduser().resolve()}")
    print(f"Sibling tool directory (checked first for prep/audit/scanners): {MODEL_MANAGER_SCRIPT_DIR}")
    ap = argparse.ArgumentParser(
        description="Interactive HF/Kaggle model+dataset search/download manager; optional subprocess helpers for prep, conversion, audit, and scanners.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Optional subprocess integrations (only when you choose them or use --audit/--gui):\n"
        "  Prep:        Prepare_models_for_All.py (MODEL_MANAGER_SCRIPT_DIR, then MODEL_TOOLS_DIR, …)\n"
        "  Conversion:  model_conversion.py (same search order)\n"
        "  Local audit: model_audit.py (--audit; same search order)\n"
        "  Scanners:    modelaudit, modelscan, ModelGuard, skillcheck, … under script dir then MODEL_TOOLS_DIR\n"
        "  Downloads:   hfdownloader if MODEL_MANAGER_HFDOWNLOADER is enabled; else huggingface_hub\n"
        "  GUI:         model_manager_gui.swift next to this script (requires swift)\n"
        "MODEL_MANAGER_TOOLS_DIR defaults to <REDACTED_PATH>; sibling directory is always this script's folder.",
    )
    ap.add_argument("--search", action="store_true", help="Start search flow immediately")
    ap.add_argument("--audit", action="store_true", help="Run local audit flow immediately")
    ap.add_argument("--leaderboards", action="store_true", help="Show leaderboard sources/search immediately")
    ap.add_argument("--gui", action="store_true", help="Launch the native macOS GUI")
    ap.add_argument("--download-root", help="Download root to use when a search flow reaches download selection")
    ap.add_argument("--specific-repo", help="Run search flow for a specific Hugging Face repo ID or URL")
    ap.add_argument("--specific-kind", choices=["model", "dataset"], default="model", help="Specific repo type")
    ap.add_argument("--search-query", help="Search term(s), with OR or comma splitting supported")
    ap.add_argument("--search-kind", choices=["models", "datasets", "both"], help="Search for models, datasets, or both")
    ap.add_argument("--artifact-types", help="Model artifact types to include, e.g. gguf,coreml or any")
    ap.add_argument("--model-size-range", help='Model artifact size range, e.g. "200 MB - 2 TB" or any')
    ap.add_argument("--exclude-multipart-models", dest="exclude_multipart_models", action="store_true", default=None, help="Hide multi-part model artifacts such as gguf-split or safetensors-sharded")
    ap.add_argument("--include-multipart-models", dest="exclude_multipart_models", action="store_false", help="Allow multi-part model artifacts such as gguf-split or safetensors-sharded")
    ap.add_argument("--exclude-publishers", help="Comma-separated authors/publishers to exclude, or none")
    ap.add_argument("--exclude-terms", help="Comma-separated terms/tags to exclude, or none")
    ap.add_argument("--exclude-families", help="Comma-separated model families to exclude, or none")
    ap.add_argument(
        "--exclude-existing-dirs",
        help=(
            "Colon- or comma-separated directories whose already-downloaded "
            "<owner>/<repo>, <owner>__<repo>, models--<owner>--<repo>, and "
            "datasets--<owner>--<repo> entries will be hidden from results. "
            "Defaults from MODEL_MANAGER_EXISTING_DIRS env var."
        ),
    )
    ap.add_argument(
        "--scan-after-download",
        choices=["ask", "always", "never"],
        help=(
            "Whether to run external security scanners (modelaudit, modelscan, "
            "ModelGuard, skill-scanner) after each successful download. Set "
            "ONCE per session — no per-download prompt. Defaults from env var "
            "MODEL_MANAGER_SCAN_AFTER_DOWNLOAD={ask,always,never}; falls back "
            "to interactive prompt if unset."
        ),
    )
    ap.add_argument("--search-source", choices=["huggingface", "kaggle", "both"], help="Search source")
    ap.add_argument("--result-limit", type=int, help="Results per source/type/search term")
    ap.add_argument("--page-size", type=int, help="Displayed result batch size")
    ap.add_argument("--hide-duplicate-families", dest="hide_duplicate_families", action="store_true", default=None, help="Hide likely duplicate/mirror repos")
    ap.add_argument("--show-duplicate-families", dest="hide_duplicate_families", action="store_false", help="Show likely duplicate/mirror repos")
    ap.add_argument("--audit-dry-run", dest="audit_dry_run", action="store_true", default=None, help="Run local audit in dry-run mode")
    ap.add_argument("--audit-live", dest="audit_dry_run", action="store_false", help="Run local audit without --dry-run")
    ap.add_argument("--leaderboard-categories", help="Comma-separated leaderboard categories: general,coding,visual,security,embedding")
    ap.add_argument("--skip-hf-spaces", action="store_true", help="Do not search Hugging Face Spaces while showing leaderboards")
    ap.add_argument("--manual-leaderboard-cache", action="store_true", help="Prompt to manually add/update cached leaderboard model IDs")
    ap.add_argument("--refresh-leaderboard-cache", action="store_true", help="Refresh best-effort cached leaderboard model IDs")
    ap.add_argument("--leaderboard-cache-limit", type=int, help="Cache refresh limit per category")
    ap.add_argument("--debug", action="store_true", help="Print picker/search debug traces to stderr")
    ap.add_argument("--debug-log", help="Optional file path for debug traces when --debug is enabled")
    args = ap.parse_args()
    configure_debug_mode(args.debug or DEBUG_ENABLED, args.debug_log or (str(DEBUG_LOG_PATH) if DEBUG_LOG_PATH else None))
    debug_log("main-args", args=vars(args))
    if DEBUG_ENABLED:
        print("Debug mode enabled.")
        if DEBUG_LOG_PATH is not None:
            print(f"Debug log: {DEBUG_LOG_PATH}")

    try:
        if args.search:
            run_search_flow(args)
            return 0
        if args.audit:
            run_local_audit_only(args.audit_dry_run)
            return 0
        if args.leaderboards:
            categories = None
            if args.leaderboard_categories:
                categories = [normalize_match_text(x) for x in re.split(r"[,;]+", args.leaderboard_categories) if normalize_match_text(x)]
            leaderboard_options_supplied = any([
                args.leaderboard_categories,
                args.skip_hf_spaces,
                args.manual_leaderboard_cache,
                args.refresh_leaderboard_cache,
                args.leaderboard_cache_limit is not None,
            ])
            show_leaderboard_sources(
                categories_override=categories,
                search_hf_spaces=(not args.skip_hf_spaces) if leaderboard_options_supplied else None,
                manual_cache_update=args.manual_leaderboard_cache if leaderboard_options_supplied else None,
                refresh_cache=args.refresh_leaderboard_cache if leaderboard_options_supplied else None,
                refresh_limit=args.leaderboard_cache_limit,
            )
            return 0
        if args.gui:
            launch_native_gui()
            return 0

        while True:
            print()
            choice = prompt_choice("What do you want to do?", ["search/download", "local audit", "leaderboards", "native GUI", "quit"], "search/download")
            if choice == "search/download":
                run_search_flow()
            elif choice == "local audit":
                run_local_audit_only()
            elif choice == "leaderboards":
                show_leaderboard_sources()
            elif choice == "native GUI":
                launch_native_gui()
            else:
                return 0
    except InputAborted:
        print()
        print("Interactive input was closed. Exiting without making changes.")
        print("Re-run `model_manager.py` in a terminal session to continue prompts like result selection or delete confirmation.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
