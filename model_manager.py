#!/usr/bin/env python3
"""
model_manager_v1.8_family_filter.py

Interactive model/dataset finder and downloader for Hugging Face and Kaggle.

Core design:
  - Repos are search results.
  - Downloadable model artifacts inside a repo are discovered and selected separately.
  - Downloads are read-only/audit-only by default.
  - Destructive action is never automatic; deletion is offered only after DANGER/BLOCKER findings.

Optional dependencies:
  python3 -m pip install --upgrade huggingface_hub datasets kaggle kagglehub pandas pyarrow safetensors

Auth:
  HF_TOKEN or HUGGINGFACEHUB_API_TOKEN for gated/private Hugging Face repos.
  ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY for Kaggle.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

MODEL_TOOLS_DIR = Path(os.getenv("MODEL_MANAGER_TOOLS_DIR", str(Path(__file__).resolve().parent))).expanduser().resolve()

DEFAULT_LOCAL_MODEL_DIRS = [
    Path("/Volumes/ModelStorage/models"),
    Path("/Volumes/ModelStorage/models-flat"),
    Path("/Volumes/ModelStorage/.cache/huggingface"),
    Path.home() / ".cache" / "huggingface",
    Path("/Volumes/ModelStorage/.cache/modelscope"),
]

DEFAULT_SECURITY_TOOLS = [
    MODEL_TOOLS_DIR / "modelaudit",
    MODEL_TOOLS_DIR / "modelscan",
    MODEL_TOOLS_DIR / "model-scan",
    MODEL_TOOLS_DIR / "ModelGuard",
    MODEL_TOOLS_DIR / "palisade-scan",
    MODEL_TOOLS_DIR / "skillcheck",
    MODEL_TOOLS_DIR / "skill-scanner",
    Path("~/modelaudit/").expanduser(),
    Path("~/modelscan").expanduser(),
    Path("~/modelscan/").expanduser(),
    Path("~/skillcheck/").expanduser(),
    Path("~/skill-scanner").expanduser(),
    Path("~/skill-scanner/").expanduser(),
]

PREP_SCRIPT_CANDIDATES = [
    MODEL_TOOLS_DIR / "Prepare_models_for_All.py",
    Path("/Volumes/ModelStorage/models-flat/Prepare_models_for_All.py"),
    Path("./Prepare_models_for_All.py"),
    Path.home() / "Prepare_models_for_All.py",
]

CONVERSION_SCRIPT_CANDIDATES = [
    MODEL_TOOLS_DIR / "model_conversion.py",
    Path("/Volumes/ModelStorage/models-flat/model_conversion.py"),
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
DEFAULT_DOWNLOAD_DIR = Path(os.getenv("MODEL_MANAGER_DOWNLOAD_DIR", str(Path.home() / "models"))).expanduser().resolve()
CACHE_DIR = Path(os.getenv("MODEL_MANAGER_CACHE_DIR", str(Path.home() / ".cache" / "model_manager"))).expanduser().resolve()
LEADERBOARD_CACHE_PATH = CACHE_DIR / "leaderboard_cache.json"
REPUTATION_CACHE_PATH = CACHE_DIR / "publisher_reputation.json"
DEFAULT_HF_DOWNLOAD_MAX_WORKERS = 8
DEFAULT_HF_XET_RANGE_GETS = 16
DEFAULT_HFDOWNLOADER_CONNECTIONS = 16
DEFAULT_HFDOWNLOADER_MAX_ACTIVE = 4
HFDOWNLOADER_SAFE_FILTER_ARTIFACT_TYPES = {"gguf", "gguf-split", "safetensors", "safetensors-sharded"}
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

# Local pre-download risk intelligence. The script searches these files first, then any
# likely risk workbook found in ~/model_tools. Supported formats: .json, .csv, .tsv, .xlsx
# (.xlsx requires openpyxl).
RISK_INTEL_CANDIDATES = [
    MODEL_TOOLS_DIR / "model_risk_intel.xlsx",
    MODEL_TOOLS_DIR / "model_risk_intel.csv",
    MODEL_TOOLS_DIR / "model_risk_intel.json",
    MODEL_TOOLS_DIR / "malicious_ai_models.xlsx",
    MODEL_TOOLS_DIR / "malicious_ai_models.csv",
    MODEL_TOOLS_DIR / "ai_model_risk_workbook.xlsx",
]

# Conservative defaults. Edit ~/.cache/model_manager/publisher_reputation.json for your own allow/warn/block lists.
DEFAULT_KNOWN_GOOD_OWNERS = {
    "qwen", "deepseek-ai", "microsoft", "codellama", "meta-llama", "mistralai",
    "google", "google-bert", "nomic-ai", "jinaai", "sentence-transformers",
    "unsloth", "bartowski", "lmstudio-community", "ggml-org", "mlx-community",
}
DEFAULT_WARN_NAME_TERMS = {
    "uncensored", "unsensored", "abliterated", "obliterated", "jailbreak",
    "backdoor", "backdoored", "poisoned", "eval-bypass", "bypass",
}
DEFAULT_EXCLUDED_AUTHORS = ["DavidAU", "TheBloke", "mradermacher", "bartowski"]
DEFAULT_EXCLUDED_TERMS = ["uncensored", "nsfw", "roleplay", "erotic", "jailbreak"]
MODEL_MANAGER_VERSION = "v1.8.1-artifact-picker-not-filter"
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
    print("Examples: 200 MB - 2 TB | 4 GB - 80 GB | any")
    default = "200 MB - 2 TB"
    while True:
        value = prompt("Model size range", default)
        parsed = parse_size_range(value)
        if parsed is not None:
            lo, hi = parsed
            if lo is None and hi is None:
                print("Model size filter: any")
            else:
                print(f"Model size filter: {human_size(lo)} - {human_size(hi)}")
            return parsed
        print("Invalid size range. Include units, for example: 200 MB - 2 TB")


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


def hf_xet_available() -> bool:
    return importlib.util.find_spec("hf_xet") is not None or shutil.which("hf-xet") is not None


def configure_hf_download_environment(worker_count: int | None = None) -> dict[str, int | bool | str]:
    """Configure Hugging Face transfers before `huggingface_hub` is imported.

    Default mode uses Hugging Face's fast Xet transfer path. Set
    MODEL_MANAGER_HF_TRANSFER_MODE=safe to lower concurrency for fragile runs.
    """
    mode = os.getenv("MODEL_MANAGER_HF_TRANSFER_MODE", "fast").strip().lower()
    safe_mode = mode in {"safe", "low-memory", "low_memory", "conservative"}
    default_workers = 1 if safe_mode else DEFAULT_HF_DOWNLOAD_MAX_WORKERS
    default_range_gets = 2 if safe_mode else DEFAULT_HF_XET_RANGE_GETS
    max_workers = worker_count or env_int("MODEL_MANAGER_HF_MAX_WORKERS", default_workers, minimum=1)
    range_gets = worker_count or env_int("MODEL_MANAGER_HF_XET_RANGE_GETS", default_range_gets, minimum=1)

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
    if not artifacts:
        return []
    terms: list[str] = []
    seen: set[str] = set()
    for art in artifacts:
        for name, _ in art.files:
            base = Path(name).name.strip()
            if not base:
                continue
            lower = base.lower()
            if lower in seen:
                continue
            seen.add(lower)
            terms.append(base)
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


def prompt(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{label}{suffix}: ").strip()
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
        if r.source != "hf":
            r.notes.append("artifact type not confirmed; provider does not expose reliable file metadata")
            kept.append(r)
            continue
        refresh_hf_file_metadata(r)
        artifacts = discover_artifacts(r)
        matching = [a for a in artifacts if artifact_matches_selected_type(a, selected_types)]
        if matching:
            for i, art in enumerate(matching, 1):
                art.index = i
            r.visible_artifacts = matching
            # Keep only selected artifact families visible/selectable by pruning files to matching artifacts plus common companions.
            allowed_names = {name for art in matching for name, _ in art.files}
            companion_names = {name for name, _ in r.files if any(fnmatch.fnmatch(name, pat) for pat in COMMON_MODEL_COMPANIONS)}
            keep_names = allowed_names | companion_names
            r.files = [(name, size) for name, size in r.files if name in keep_names]
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
                for i, art in enumerate(matching, 1):
                    art.index = i
                r.visible_artifacts = matching
                allowed_names = {name for art in matching for name, _ in art.files}
                companion_names = {name for name, _ in r.files if any(fnmatch.fnmatch(name, pat) for pat in COMMON_MODEL_COMPANIONS)}
                r.files = [(name, size) for name, size in r.files if name in allowed_names or name in companion_names]
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
        value = input(f"Leaderboard #{rank}: ").strip()
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
        models = list(api.list_models(search=query, limit=fetch_limit, sort="downloads", full=True))
    except TypeError:
        models = list(api.list_models(search=query, limit=fetch_limit, full=True))
    results: list[SearchResult] = []
    for m in models:
        repo_id = getattr(m, "modelId", None) or getattr(m, "id", None)
        if not repo_id:
            continue
        if candidate_excluded_before_metadata(
            repo_id,
            m,
            excluded_publishers=pre_excluded_publishers,
            excluded_terms=pre_excluded_terms,
            excluded_family_terms=pre_excluded_family_terms,
        ):
            continue
        notes: list[str] = []
        try:
            info = api.model_info(repo_id=repo_id, files_metadata=True)
            total, files = _repo_files_from_info(info)
            lic = _card_license(info)
        except Exception as e:
            total, files, lic = None, [], None
            notes.append(f"metadata error: {type(e).__name__}: {e}")
        results.append(SearchResult(
            index=0,
            source="hf",
            kind="model",
            repo_id=repo_id,
            size_bytes=total,
            downloads=getattr(m, "downloads", None),
            likes=getattr(m, "likes", None),
            pipeline=getattr(m, "pipeline_tag", None),
            updated=str(getattr(m, "lastModified", None) or ""),
            license=lic,
            url=f"https://huggingface.co/{repo_id}",
            files=files,
            notes=notes,
            raw=m,
        ))
        if candidate_limit is None and len(results) >= limit:
            break
    return results


def search_hf_datasets(query: str, limit: int) -> list[SearchResult]:
    api = hf_api()
    try:
        datasets = list(api.list_datasets(search=query, limit=limit, sort="downloads", full=True))
    except Exception:
        from huggingface_hub import list_datasets
        datasets = list(list_datasets(search=query, limit=limit))
    results: list[SearchResult] = []
    for ds in datasets:
        repo_id = getattr(ds, "id", None) or getattr(ds, "repo_id", None)
        if not repo_id:
            continue
        notes: list[str] = []
        try:
            info = api.dataset_info(repo_id=repo_id, files_metadata=True)
            total, files = _repo_files_from_info(info)
            lic = _card_license(info)
        except Exception as e:
            total, files, lic = None, [], None
            notes.append(f"metadata error: {type(e).__name__}: {e}")
        results.append(SearchResult(
            index=0,
            source="hf",
            kind="dataset",
            repo_id=repo_id,
            size_bytes=total,
            downloads=getattr(ds, "downloads", None),
            likes=getattr(ds, "likes", None),
            updated=str(getattr(ds, "last_modified", None) or getattr(ds, "lastModified", None) or ""),
            license=lic,
            url=f"https://huggingface.co/datasets/{repo_id}",
            files=files,
            notes=notes,
            raw=ds,
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
    good = {str(x).lower() for x in cfg.get("known_good_owners", [])}
    bad = {str(x).lower() for x in cfg.get("known_malicious_owners", [])}
    warn_owners = {str(x).lower() for x in cfg.get("warn_owners", [])}
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
            rec = "DupFamily" if rec == "Unknown" else rec
        r.recommendation = rec


def hide_duplicate_families(results: list[SearchResult]) -> list[SearchResult]:
    """Keep the strongest candidate in each likely duplicate family, preserving non-models."""
    best: dict[str, SearchResult] = {}
    hidden: dict[str, list[SearchResult]] = {}
    passthrough: list[SearchResult] = []
    for r in results:
        if r.kind != "model" or not r.duplicate_family:
            passthrough.append(r)
            continue
        key = r.duplicate_family
        score = (
            1 if r.leaderboard_rank is not None else 0,
            1 if owner_of(r.repo_id) in DEFAULT_KNOWN_GOOD_OWNERS else 0,
            r.downloads or 0,
            r.likes or 0,
            r.size_bytes or 0,
        )
        cur = best.get(key)
        if cur is None:
            best[key] = r
            continue
        cur_score = (
            1 if cur.leaderboard_rank is not None else 0,
            1 if owner_of(cur.repo_id) in DEFAULT_KNOWN_GOOD_OWNERS else 0,
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

def print_results_page(results: list[SearchResult], start: int, page_size: int, show_files: bool = True, top_files: int = 7) -> None:
    end = min(start + page_size, len(results))
    print()
    print(f"Results {start + 1}-{end} of {len(results)}")
    print(f"{'#':>4}  {'SRC':<7}  {'KIND':<7}  {'FAMILY':<18}  {'LB':<16}  {'REC':<12}  {'SIZE':>12}  {'DL':>10}  {'LIKES':>7}  {'TYPE':<22}  {'LICENSE':<14}  REPO")
    print("-" * 185)
    for r in results[start:end]:
        lb = r.leaderboard_label or "-"
        rec = r.recommendation or "-"
        fam = model_family_key(r.repo_id) if r.kind == "model" else "-"
        print(f"{r.index:>4}  {r.source:<7}  {r.kind:<7}  {fam:<18.18}  {lb:<16.16}  {rec:<12.12}  {result_size_label(r):>12}  {str(r.downloads if r.downloads is not None else '-'):>10}  {str(r.likes if r.likes is not None else '-'):>7}  {(r.pipeline or '-'):<22.22}  {(r.license or '-'):<14.14}  {r.repo_id}")
        if r.title:
            print(f"      title: {r.title}")
        if r.notes:
            for note in r.notes:
                print(f"      note: {note}")
        if show_files:
            if r.source == "hf" and r.kind == "model":
                refresh_hf_file_metadata(r)
                artifacts = artifacts_for_display(r)
                if artifacts:
                    print("      direct artifacts — select these with IDs like 1A/1B; repo number opens the artifact picker")
                    for art in artifacts[:top_files]:
                        child_id = f"{r.index}{alpha_label(art.index)}"
                        quant = art.quant or "-"
                        first_file = art.files[0][0] if art.files else art.label
                        extra = f" +{len(art.files) - 1} files" if len(art.files) > 1 else ""
                        print(f"   {child_id + '.':<7} {human_size(art.total_size):>12}  {art.artifact_type:<20} {quant:<10} {first_file}{extra}")
                    if len(artifacts) > top_files:
                        print(f"           ... {len(artifacts) - top_files} more artifacts; select repo {r.index} to see all")
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
    while True:
        visible = [r for r in results if r.index not in selected and r.index not in excluded]
        if not visible:
            print("No unselected results remain.")
            break
        pos = min(pos, max(0, len(visible) - page_size))
        print_results_page(visible, pos, page_size)
        print("Commands: artifact IDs like 1A,2C | repo numbers 1,3 or 2-6 | xpub qwen exclude publisher | xf 7 / xfam qwen coder exclude family | n next | p previous | all | done | q")
        cmd = input("Which repo number(s), artifact ID(s), or family exclusion do you want? ").strip()
        cmd_lower = cmd.lower()
        if cmd_lower in {"q", "quit", "exit"}:
            return []
        if cmd_lower in {"done", "d"}:
            break
        if cmd_lower in {"n", "next"}:
            max_pos = max(0, len(visible) - page_size)
            if pos >= max_pos:
                print("Already at the last page. Use artifact IDs/repo numbers, 'done', or 'q'.")
            else:
                pos = min(max_pos, pos + page_size)
            continue
        if cmd_lower in {"p", "prev", "previous"}:
            if pos <= 0:
                print("Already at the first page.")
            else:
                pos = max(0, pos - page_size)
            continue
        if cmd_lower in {"families", "family", "top"}:
            print_top_model_families([r for r in results if r.index not in selected])
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
            if prompt_bool("Done selecting?", False):
                break
            pos = 0
            continue

        artifact_tokens = parse_repo_artifact_tokens(cmd, len(results))
        if artifact_tokens:
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
            if prompt_bool("Done selecting?", False):
                break
            pos = 0
            continue

        if cmd_lower == "all":
            nums = [r.index for r in visible]
        else:
            nums = parse_selection(cmd_lower, len(results))
        if nums:
            for n in nums:
                if not (1 <= n <= len(results)):
                    continue
                r = results[n - 1]
                if r.index in selected:
                    continue
                keep = show_artifact_picker_for_repo(r)
                if keep:
                    selected.add(r.index)
            print(f"Selected repos/artifacts: {', '.join(str(n) for n in sorted(selected))}")
            if prompt_bool("Done selecting?", False):
                break
            pos = 0
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
    confirm = input("Delete this downloaded item? [y/N]: ").strip().lower()
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
    findings: list[dict[str, str]] = []
    if not prompt_bool("Run external security/audit tools against the download?", True):
        return findings
    seen_commands: set[tuple[str, ...]] = set()
    for tool in DEFAULT_SECURITY_TOOLS:
        scanner = find_tool_command(tool, target)
        if not scanner:
            print(f"SKIP tool not found/usable: {tool}")
            continue
        cmd_key = tuple(scanner.cmd)
        if cmd_key in seen_commands:
            print(f"SKIP duplicate scanner command: {scanner.label}")
            continue
        seen_commands.add(cmd_key)
        print()
        print(f"Running read-only scanner command ({scanner.label}): {quote_cmd(scanner.cmd)}")
        try:
            proc = subprocess.run(scanner.cmd, text=True, capture_output=True, timeout=1800, cwd=str(scanner.cwd) if scanner.cwd else None)
            output = (proc.stdout or "") + "\n" + (proc.stderr or "")
            print(output[-4000:] if output.strip() else f"Tool exit code: {proc.returncode}")
            if proc.returncode not in (0, None):
                findings.append({"severity": "DANGER", "message": f"scanner reported non-zero exit from {scanner.label}: rc={proc.returncode}"})
            if re.search(r"malware|malicious|backdoor|rce|remote code execution", output, re.I):
                findings.append({"severity": "DANGER", "message": f"scanner output contains high-risk language from {scanner.label}"})
        except subprocess.TimeoutExpired:
            findings.append({"severity": "WARN", "message": f"scanner timed out: {scanner.label}"})
        except Exception as e:
            findings.append({"severity": "WARN", "message": f"scanner failed {scanner.label}: {type(e).__name__}: {e}"})
    return findings

# -----------------------------------------------------------------------------
# Download
# -----------------------------------------------------------------------------

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

    default_connections = env_int("MODEL_MANAGER_HFD_CONNECTIONS", DEFAULT_HFDOWNLOADER_CONNECTIONS, minimum=1)
    default_active = env_int("MODEL_MANAGER_HFD_MAX_ACTIVE", DEFAULT_HFDOWNLOADER_MAX_ACTIVE, minimum=1)
    connections = prompt_int_range("How many hfdownloader connections per file?", default_connections, 1, 64)
    max_active = prompt_int_range("How many concurrent file downloads?", default_active, 1, 64)

    base = download_root / "huggingface" / result.kind
    target = base / result.repo_id
    base.mkdir(parents=True, exist_ok=True)

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
    subprocess.run(cmd, check=True, env=env)
    return target


def download_hf_result(
    result: SearchResult,
    download_root: Path,
    allow_patterns: list[str] | None,
    artifacts: list[Artifact] | None = None,
) -> Path | None:
    if result.kind == "model" and hfdownloader_enabled():
        target = download_hf_with_hfdownloader(result, download_root, allow_patterns, artifacts)
        if target is not None:
            return target
        print("Falling back to huggingface_hub snapshot_download.")

    default_workers = env_int("MODEL_MANAGER_HF_MAX_WORKERS", DEFAULT_HF_DOWNLOAD_MAX_WORKERS, minimum=1)
    worker_count = prompt_int_range("How many Hugging Face download workers?", default_workers, 1, 64)
    transfer_cfg = configure_hf_download_environment(worker_count=worker_count)
    from huggingface_hub import snapshot_download
    target = download_root / "huggingface" / result.kind / safe_folder_name(result.repo_id)
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
    snapshot_download(
        repo_id=result.repo_id,
        repo_type=result.kind,
        local_dir=str(target),
        token=token,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore if allow_patterns is None else None,
        max_workers=int(transfer_cfg["max_workers"]),
    )
    return target


def download_kaggle_result(result: SearchResult, download_root: Path, allow_patterns: list[str] | None) -> Path | None:
    target = download_root / "kaggle" / result.kind / safe_folder_name(result.repo_id)
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Kaggle {result.kind}: {result.repo_id}")
    print(f"Target: {target}")
    if result.kind == "dataset":
        api = kaggle_api()
        if allow_patterns:
            files = kaggle_dataset_files(result.repo_id)
            matches = [name for name, _ in files if any(fnmatch.fnmatch(name, pat) for pat in allow_patterns)]
            if not matches:
                print("No Kaggle files matched the chosen pattern.")
                return None
            for name in matches:
                api.dataset_download_file(result.repo_id, name, path=str(target), quiet=False)
            return target
        api.dataset_download_files(result.repo_id, path=str(target), unzip=True, quiet=False)
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
            return target
    except Exception as e:
        print(f"Kaggle model download failed or unavailable: {type(e).__name__}: {e}")
    print("Kaggle model download is unavailable in this environment/package version.")
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
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


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
            iterator = ws.iter_rows(values_only=True)
            try:
                header = next(iterator)
            except StopIteration:
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
    # Fallback: short textual cells from any column, useful for messy spreadsheets.
    for key, value in row.items():
        if str(key).startswith("_") or value in {None, ""}:
            continue
        text = str(value).strip()
        if 3 <= len(text) <= 140 and any(ch.isalpha() for ch in text):
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

    findings: list[dict[str, str]] = []
    selected_names = [name for name, _ in expected]
    all_names = [name for name, _ in result.files]
    search_blob = normalize_match_text(" ".join([result.repo_id, result.title or ""] + selected_names + all_names[:50]))
    owner = owner_of(result.repo_id)

    cfg = load_reputation_config()
    bad_owners = {str(x).lower() for x in cfg.get("known_malicious_owners", [])}
    warn_owners = {str(x).lower() for x in cfg.get("warn_owners", [])}
    known_good = {str(x).lower() for x in cfg.get("known_good_owners", [])}

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

def post_download_audit(result: SearchResult, target: Path, allow_patterns: list[str] | None) -> None:
    findings = verify_download_integrity(target, result.files, allow_patterns)
    print_findings(findings)
    scanner_findings = run_external_security_tools(target)
    if scanner_findings:
        print_findings(scanner_findings)
    all_findings = findings + scanner_findings
    offer_delete, reason = should_offer_delete(all_findings)
    if offer_delete:
        delete_path_after_confirmation(target, reason)
    elif all_findings:
        print("No DANGER/BLOCKER findings requiring delete offer; keeping files.")


def download_selected(results: list[SearchResult], download_root_override: Path | None = None) -> None:
    if not results:
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
            print(f"Download failed: {type(e).__name__}: {e}")
            continue
        if target:
            post_download_audit(r, target, patterns)
            if r.kind == "model":
                offer_prepare_models(target, download_root)

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
    excluded_publishers: set[str],
    excluded_terms: set[str],
    excluded_family_terms: list[str],
    hide_duplicate_families_pref: bool,
    quiet_family_no_match: bool = False,
) -> list[SearchResult]:
    results = merge_search_results(results)
    results = apply_publisher_exclusions(results, excluded_publishers)
    results = apply_term_exclusions(results, excluded_terms)
    if kind in {"models", "both"}:
        results = filter_results_by_artifact_types(results, selected_artifact_types)
        results = filter_results_by_model_size(results, model_size_range)
    annotate_results_with_leaderboard_cache(results)
    annotate_recommendations(results)
    results = apply_model_family_exclusions(results, excluded_family_terms, quiet_no_match=quiet_family_no_match)
    if results and hide_duplicate_families_pref:
        before = len(results)
        results = hide_duplicate_families(results)
        after = len(results)
        if after < before:
            print(f"Hidden {before - after} likely duplicate/mirror repo(s). Kept strongest candidate per exact variant family.")
    results.sort(key=lambda r: (r.leaderboard_rank is None, r.leaderboard_rank or 10**9, r.recommendation not in {"KnownGood", "Leaderboard"}, -(r.downloads or 0), -(r.size_bytes or 0)))
    assign_indexes(results)
    return results


def deep_hf_candidate_limit(limit: int) -> int:
    return min(max(limit * 20, 200), 500)


def run_search_flow(options: argparse.Namespace | None = None) -> None:
    download_root_override = None
    if options is not None and getattr(options, "download_root", None):
        download_root_override = Path(options.download_root).expanduser().resolve()

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
        search_terms = split_boolean_query(query)
        if len(search_terms) > 1:
            print(f"Split search into {len(search_terms)} separate searches:")
            for term in search_terms:
                print(f"  - {term}")
        kind = getattr(options, "search_kind", None) if options is not None else None
        if kind not in {"models", "datasets", "both"}:
            kind = prompt_choice("Search for", ["models", "datasets", "both"], "models")
        selected_artifact_types: set[str] = set()
        model_size_range: tuple[int | None, int | None] = (None, None)
        if kind in {"models", "both"}:
            selected_from_args = parse_artifact_type_filter_value(getattr(options, "artifact_types", None)) if options is not None else None
            selected_artifact_types = selected_from_args if selected_from_args is not None else choose_artifact_type_filters()
            size_from_args = parse_model_size_range_or_default(getattr(options, "model_size_range", None)) if options is not None else None
            model_size_range = size_from_args if size_from_args is not None else prompt_model_size_range()
        if options is not None and getattr(options, "exclude_publishers", None) is not None:
            excluded_publishers = parse_excluded_publishers_value(options.exclude_publishers)
        else:
            excluded_publishers = prompt_excluded_publishers()
        if options is not None and getattr(options, "exclude_terms", None) is not None:
            excluded_terms = parse_excluded_terms_value(options.exclude_terms)
        else:
            excluded_terms = prompt_excluded_terms()
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
        hide_duplicate_families_pref = False
        excluded_family_terms: list[str] = []
        if kind in {"models", "both"}:
            hide_arg = getattr(options, "hide_duplicate_families", None) if options is not None else None
            hide_duplicate_families_pref = hide_arg if hide_arg is not None else prompt_bool("Hide likely duplicate/mirror repos by exact variant family?", True)
            if options is not None and getattr(options, "exclude_families", None) is not None:
                excluded_family_terms = parse_model_family_exclusion_terms(options.exclude_families)
            else:
                excluded_family_terms = prompt_model_family_exclusion_terms()

        provider_search_terms = expand_search_terms_for_artifacts(search_terms, selected_artifact_types) if kind in {"models", "both"} else search_terms
        if provider_search_terms != search_terms:
            print("Expanded search with artifact filename terms:")
            for term in provider_search_terms[:20]:
                print(f"  - {term}")
            if len(provider_search_terms) > 20:
                print(f"  ... {len(provider_search_terms) - 20} more")

        raw_results = collect_search_results(provider_search_terms, source, kind, limit)
        results = filter_and_rank_search_results(
            raw_results,
            kind,
            selected_artifact_types,
            model_size_range,
            excluded_publishers,
            excluded_terms,
            excluded_family_terms,
            hide_duplicate_families_pref,
        )

        if source in {"huggingface", "both"} and kind in {"models", "both"} and len(results) < page_size:
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
                results = filter_and_rank_search_results(
                    raw_results,
                    "models",
                    selected_artifact_types,
                    model_size_range,
                    excluded_publishers,
                    excluded_terms,
                    excluded_family_terms,
                    hide_duplicate_families_pref,
                    quiet_family_no_match=True,
                )
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
        Path("/Volumes/ModelStorage/models-flat/model_audit.py"),
        Path("./model_audit.py"),
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
    ap = argparse.ArgumentParser(description="Interactive HF/Kaggle model+dataset search/download manager")
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
    ap.add_argument("--exclude-publishers", help="Comma-separated authors/publishers to exclude, or none")
    ap.add_argument("--exclude-terms", help="Comma-separated terms/tags to exclude, or none")
    ap.add_argument("--exclude-families", help="Comma-separated model families to exclude, or none")
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
    args = ap.parse_args()

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


if __name__ == "__main__":
    raise SystemExit(main())
