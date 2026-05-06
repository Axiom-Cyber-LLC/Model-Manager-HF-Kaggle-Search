#!/usr/bin/env python3
"""
model_manager_v1.6.py

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
import json
import os
import re
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

DEFAULT_LOCAL_MODEL_DIRS = [
    Path("/Volumes/ModelStorage/models"),
    Path("/Volumes/ModelStorage/models-flat"),
    Path("/Volumes/ModelStorage/.cache/huggingface"),
    Path.home() / ".cache" / "huggingface",
    Path("/Volumes/ModelStorage/.cache/modelscope"),
]

DEFAULT_SECURITY_TOOLS = [
    Path("~/modelaudit/").expanduser(),
    Path("~/modelscan").expanduser(),
    Path("~/modelscan/").expanduser(),
    Path("~/skillcheck/").expanduser(),
    Path("~/skill-scanner").expanduser(),
    Path("~/skill-scanner/").expanduser(),
]

PREP_SCRIPT_CANDIDATES = [
    Path("/Volumes/ModelStorage/models-flat/Prepare_models_for_All.py"),
    Path("./Prepare_models_for_All.py"),
    Path.home() / "Prepare_models_for_All.py",
]

DEFAULT_DOWNLOAD_DIR = Path(os.getenv("MODEL_MANAGER_DOWNLOAD_DIR", str(Path.home() / "model_downloads"))).expanduser().resolve()
CACHE_DIR = Path(os.getenv("MODEL_MANAGER_CACHE_DIR", str(Path.home() / ".cache" / "model_manager"))).expanduser().resolve()
LEADERBOARD_CACHE_PATH = CACHE_DIR / "leaderboard_cache.json"
REPUTATION_CACHE_PATH = CACHE_DIR / "publisher_reputation.json"

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
MODEL_MANAGER_VERSION = "v1.6"
MODEL_FILE_EXTENSIONS = {".gguf", ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx", ".mlmodel"}
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
      code | cybersecurity | Threat Detection
      code ; cybersecurity ; Threat Detection

    This intentionally treats only OR / | / ; as split operators. AND/NOT are
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

            if ch in {"|", ";"}:
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


def search_hf_models(query: str, limit: int) -> list[SearchResult]:
    api = hf_api()
    try:
        models = list(api.list_models(search=query, limit=limit, sort="downloads", full=True))
    except TypeError:
        models = list(api.list_models(search=query, limit=limit, full=True))
    results: list[SearchResult] = []
    for m in models:
        repo_id = getattr(m, "modelId", None) or getattr(m, "id", None)
        if not repo_id:
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


def search_kaggle_models(query: str, limit: int) -> list[SearchResult]:
    """Best effort; Kaggle model APIs vary heavily by installed package version."""
    results: list[SearchResult] = []
    try:
        import kagglehub
        for fn_name in ("model_list", "models_list", "list_models", "search_models"):
            fn = getattr(kagglehub, fn_name, None)
            if not callable(fn):
                continue
            try:
                items = fn(search=query) if fn_name != "search_models" else fn(query)
                for item in list(items)[:limit]:
                    handle = str(getattr(item, "handle", None) or getattr(item, "ref", None) or getattr(item, "id", None) or item)
                    notes = []
                    if "huggingface" in handle.lower() or "hugging-face" in handle.lower():
                        notes.append("possible Hugging Face mirror/integration")
                    results.append(SearchResult(index=0, source="kaggle", kind="model", repo_id=handle, title=str(item), url=f"https://www.kaggle.com/models/{handle}", notes=notes, raw=item))
                if results:
                    return results
            except Exception:
                pass
    except Exception:
        pass

    if shutil.which("kaggle"):
        try:
            proc = subprocess.run(["kaggle", "models", "list", "-s", query], capture_output=True, text=True, timeout=60)
            for line in (proc.stdout or "").splitlines():
                ln = line.strip()
                if not ln or ln.lower().startswith(("ref", "warning", "usage")):
                    continue
                first = ln.split()[0]
                if "/" not in first:
                    continue
                notes = []
                if "huggingface" in ln.lower() or "hugging-face" in ln.lower():
                    notes.append("possible Hugging Face mirror/integration")
                results.append(SearchResult(index=0, source="kaggle", kind="model", repo_id=first, title=ln, url=f"https://www.kaggle.com/models/{first}", notes=notes))
                if len(results) >= limit:
                    break
        except Exception as e:
            print(f"Kaggle model search failed: {type(e).__name__}: {e}")
    return results


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

    artifacts = discover_artifacts(result)
    if not artifacts:
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


def show_artifact_picker_for_repo(result: SearchResult) -> bool:
    """Inline artifact picker used when a repo number is selected from the result list."""
    if result.source != "hf" or result.kind != "model":
        return True
    artifacts = discover_artifacts(result)
    if not artifacts:
        return True
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
    print(f"{'#':>4}  {'SRC':<7}  {'KIND':<7}  {'LB':<16}  {'REC':<12}  {'SIZE':>12}  {'DL':>10}  {'LIKES':>7}  {'TYPE':<22}  {'LICENSE':<14}  REPO")
    print("-" * 165)
    for r in results[start:end]:
        lb = r.leaderboard_label or "-"
        rec = r.recommendation or "-"
        print(f"{r.index:>4}  {r.source:<7}  {r.kind:<7}  {lb:<16.16}  {rec:<12.12}  {human_size(r.size_bytes):>12}  {str(r.downloads if r.downloads is not None else '-'):>10}  {str(r.likes if r.likes is not None else '-'):>7}  {(r.pipeline or '-'):<22.22}  {(r.license or '-'):<14.14}  {r.repo_id}")
        if r.title:
            print(f"      title: {r.title}")
        if r.notes:
            for note in r.notes:
                print(f"      note: {note}")
        if show_files and r.files:
            if r.source == "hf" and r.kind == "model":
                artifacts = discover_artifacts(r)
                if artifacts:
                    for art in artifacts[:top_files]:
                        child_id = f"{r.index}{alpha_label(art.index)}"
                        quant = art.quant or "-"
                        first_file = art.files[0][0] if art.files else art.label
                        extra = f" +{len(art.files) - 1} files" if len(art.files) > 1 else ""
                        print(f"   {child_id + '.':<7} {human_size(art.total_size):>12}  {art.artifact_type:<20} {quant:<10} {first_file}{extra}")
                    if len(artifacts) > top_files:
                        print(f"           ... {len(artifacts) - top_files} more artifacts; select repo {r.index} to see all")
                else:
                    for fname, fsize in r.files[:top_files]:
                        print(f"      {human_size(fsize):>12}  {fname}")
            else:
                for fname, fsize in r.files[:top_files]:
                    print(f"      {human_size(fsize):>12}  {fname}")
            print()


def paged_select(results: list[SearchResult], page_size: int = 20) -> list[SearchResult]:
    pos = 0
    selected: set[int] = set()
    while True:
        visible = [r for r in results if r.index not in selected]
        if not visible:
            print("No unselected results remain.")
            break
        pos = min(pos, max(0, len(visible) - page_size))
        print_results_page(visible, pos, page_size)
        print("Commands: repo numbers like 1,3,5 or 2-6 | artifact IDs like 1A,2C | n next | p previous | all | done | q")
        cmd = input("Which repo number(s) or artifact ID(s) do you want to select/download? ").strip()
        cmd_lower = cmd.lower()
        if cmd_lower in {"q", "quit", "exit"}:
            return []
        if cmd_lower in {"done", "d"}:
            break
        if cmd_lower in {"n", "next"}:
            pos = min(max(0, len(visible) - page_size), pos + page_size)
            continue
        if cmd_lower in {"p", "prev", "previous"}:
            pos = max(0, pos - page_size)
            continue

        artifact_tokens = parse_repo_artifact_tokens(cmd, len(results))
        if artifact_tokens:
            for repo_num, art_num in artifact_tokens:
                r = results[repo_num - 1]
                artifacts = discover_artifacts(r)
                if not artifacts:
                    print(f"Repo {repo_num} has no discovered artifacts; selecting whole repo for later choice.")
                    selected.add(repo_num)
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


def find_tool_command(path: Path, target: Path) -> list[str] | None:
    if not path.exists():
        return None
    if path.is_file():
        if os.access(path, os.X_OK):
            return [str(path), str(target)]
        if path.suffix == ".py":
            return [sys.executable, str(path), str(target)]
        return None
    if path.is_dir():
        candidates = [
            path / "main.py", path / "scan.py", path / "modelaudit.py", path / "modelscan.py",
            path / "skillcheck.py", path / "skill-scanner.py", path / "run.sh",
        ]
        for c in candidates:
            if c.exists():
                if c.suffix == ".py":
                    return [sys.executable, str(c), str(target)]
                if os.access(c, os.X_OK) or c.suffix == ".sh":
                    return [str(c), str(target)]
        for c in path.iterdir():
            if c.is_file() and os.access(c, os.X_OK):
                return [str(c), str(target)]
    return None


def run_external_security_tools(target: Path) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if not prompt_bool("Run external security/audit tools against the download?", True):
        return findings
    for tool in DEFAULT_SECURITY_TOOLS:
        cmd = find_tool_command(tool, target)
        if not cmd:
            print(f"SKIP tool not found/usable: {tool}")
            continue
        print()
        print(f"Running read-only scanner command: {' '.join(cmd)}")
        try:
            proc = subprocess.run(cmd, text=True, capture_output=True, timeout=1800)
            output = (proc.stdout or "") + "\n" + (proc.stderr or "")
            print(output[-4000:] if output.strip() else f"Tool exit code: {proc.returncode}")
            if proc.returncode not in (0, None):
                findings.append({"severity": "DANGER", "message": f"scanner reported non-zero exit from {tool}: rc={proc.returncode}"})
            if re.search(r"malware|malicious|backdoor|rce|remote code execution", output, re.I):
                findings.append({"severity": "DANGER", "message": f"scanner output contains high-risk language from {tool}"})
        except subprocess.TimeoutExpired:
            findings.append({"severity": "WARN", "message": f"scanner timed out: {tool}"})
        except Exception as e:
            findings.append({"severity": "WARN", "message": f"scanner failed {tool}: {type(e).__name__}: {e}"})
    return findings

# -----------------------------------------------------------------------------
# Download
# -----------------------------------------------------------------------------

def download_hf_result(result: SearchResult, download_root: Path, allow_patterns: list[str] | None) -> Path | None:
    from huggingface_hub import snapshot_download
    target = download_root / "huggingface" / result.kind / safe_folder_name(result.repo_id)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    ignore = HF_DATASET_IGNORE_MODEL_WEIGHTS if result.kind == "dataset" else None
    print(f"Downloading HF {result.kind}: {result.repo_id}")
    print(f"Target: {target}")
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


def download_selected(results: list[SearchResult]) -> None:
    if not results:
        return
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
        target: Path | None = None
        try:
            if r.source == "hf":
                target = download_hf_result(r, download_root, patterns)
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


def offer_prepare_models(target: Path, download_root: Path | None = None) -> None:
    if not prompt_bool("Prepare this model for local/open-source hosting apps?", False):
        return
    script = next((p for p in PREP_SCRIPT_CANDIDATES if p.expanduser().is_file()), None)
    if not script:
        print("Prepare_models_for_All.py not found in expected locations.")
        return
    script = script.expanduser().resolve()
    target = target.expanduser().resolve()
    download_root = (download_root or target.parent).expanduser().resolve()

    print("Known prep orchestrator:", script)
    print("Downloaded model path:", target)
    print("Download root:", download_root)
    print("Common targets: LM Studio, Ollama, AnythingLLM, GPT4All, Jan, AIStudio, plus your added per-app scripts if present.")
    print("Additional apps you mentioned may need corresponding Prepare_models_for_<App>.py support: Locally AI, Apollo.app, LocalAI, Off Grid.")
    apps = prompt("Only these apps, comma-separated, or blank for all", "")
    dry = prompt_bool("Dry run first?", True)
    cmd = [sys.executable, str(script)]
    if dry:
        cmd.append("--dry-run")
    if apps:
        cmd += ["--only", apps]

    # Be download-location aware when the prep script supports it. Your current
    # Prepare_models_for_All.py does not define these flags, so we also pass env vars.
    for flag in ("--model-dir", "--models-dir", "--input-dir", "--download-dir"):
        if script_accepts_flag(script, flag):
            cmd += [flag, str(target)]
            break
    if script_accepts_flag(script, "--download-root"):
        cmd += ["--download-root", str(download_root)]

    env = os.environ.copy()
    env["MODEL_MANAGER_DOWNLOAD_TARGET"] = str(target)
    env["MODEL_MANAGER_DOWNLOAD_ROOT"] = str(download_root)

    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=False, env=env, cwd=str(download_root))
    except Exception as e:
        print(f"Prepare step failed: {type(e).__name__}: {e}")

# -----------------------------------------------------------------------------
# Leaderboards
# -----------------------------------------------------------------------------

def show_leaderboard_sources() -> None:
    categories = prompt_multi_choice("Leaderboard category", ["general", "coding", "visual", "security", "embedding"], "coding")
    print()
    for category in categories:
        print(f"Useful current leaderboard sources for {category}:")
        for name, url in LEADERBOARD_SOURCES[category]:
            print(f"  - {name}: {url}")
        print()
    if prompt_bool("Also search Hugging Face Spaces for matching leaderboard spaces?", True):
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
    if prompt_bool("Add/update cached leaderboard model repo IDs manually?", False):
        for category in categories:
            add_manual_leaderboard_cache(category)
    if prompt_bool("Refresh best-effort cached leaderboard model IDs from HF searches?", False):
        refresh_leaderboard_cache_best_effort(categories, limit=prompt_int("Cache limit per category", 20))

# -----------------------------------------------------------------------------
# Main flows
# -----------------------------------------------------------------------------

def run_search_flow() -> None:
    if prompt_bool("Do you have a single specific model or dataset to download?", False):
        value = prompt("Paste Hugging Face repo ID or URL", "")
        repo_id = parse_hf_repo_id(value)
        if not repo_id:
            print("Could not parse a Hugging Face repo ID. Use owner/name or a huggingface.co URL.")
            return
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
        download_selected(selected)
        return

    query = prompt("Search term, partial match is OK. Simple OR is supported", "")
    if not query:
        print("No query entered.")
        return
    search_terms = split_boolean_query(query)
    if len(search_terms) > 1:
        print(f"Split OR query into {len(search_terms)} separate searches:")
        for term in search_terms:
            print(f"  - {term}")
    kind = prompt_choice("Search for", ["models", "datasets", "both"], "models")
    source = prompt_choice("Search source", ["huggingface", "kaggle", "both"], "huggingface")
    limit = prompt_int("How many results per source/type/search term?", 50)
    page_size = prompt_int("Display batch size", 20)

    results: list[SearchResult] = []
    for term in search_terms:
        if source in {"huggingface", "both"} and kind in {"models", "both"}:
            print(f"Searching Hugging Face models: {term}")
            results.extend(search_hf_models(term, limit))
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

    results = merge_search_results(results)
    annotate_results_with_leaderboard_cache(results)
    annotate_recommendations(results)
    if results and prompt_bool("Hide likely duplicate/mirror repos by model family?", True):
        before = len(results)
        results = hide_duplicate_families(results)
        after = len(results)
        if after < before:
            print(f"Hidden {before - after} likely duplicate/mirror repo(s). Kept strongest candidate per family.")
    results.sort(key=lambda r: (r.leaderboard_rank is None, r.leaderboard_rank or 10**9, r.recommendation not in {"KnownGood", "Leaderboard"}, -(r.downloads or 0), -(r.size_bytes or 0)))
    assign_indexes(results)
    if not results:
        print("No results.")
        return
    selected = paged_select(results, page_size=page_size)
    if not selected:
        print("No selection.")
        return
    offer_local_scan(selected)
    offer_dataset_sampling(selected)
    download_selected(selected)


def run_local_audit_only() -> None:
    script_candidates = [
        Path("/Volumes/ModelStorage/models-flat/model_audit.py"),
        Path("./model_audit.py"),
        Path.home() / "model_audit.py",
    ]
    script = next((p for p in script_candidates if p.is_file()), None)
    if script:
        cmd = [sys.executable, str(script)]
        if prompt_bool("Dry run?", True):
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


def main() -> int:
    print(f"Model Manager {MODEL_MANAGER_VERSION}")
    print(f"Active script: {Path(__file__).expanduser().resolve()}")
    ap = argparse.ArgumentParser(description="Interactive HF/Kaggle model+dataset search/download manager")
    ap.add_argument("--search", action="store_true", help="Start search flow immediately")
    ap.add_argument("--audit", action="store_true", help="Run local audit flow immediately")
    ap.add_argument("--leaderboards", action="store_true", help="Show leaderboard sources/search immediately")
    args = ap.parse_args()

    if args.search:
        run_search_flow()
        return 0
    if args.audit:
        run_local_audit_only()
        return 0
    if args.leaderboards:
        show_leaderboard_sources()
        return 0

    while True:
        print()
        choice = prompt_choice("What do you want to do?", ["search/download", "local audit", "leaderboards", "quit"], "search/download")
        if choice == "search/download":
            run_search_flow()
        elif choice == "local audit":
            run_local_audit_only()
        elif choice == "leaderboards":
            show_leaderboard_sources()
        else:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
