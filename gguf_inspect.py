"""
gguf_inspect.py — read GGUF architecture metadata and estimate RAM at load time.

Two surfaces:
  1. parse_gguf_metadata(file_path | bytes) -> dict of key→value
  2. estimate_load_ram(file_size_bytes, metadata, context_lengths) -> table

GGUF format reference:
  https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""
from __future__ import annotations

import io
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, BinaryIO

GGUF_MAGIC = b"GGUF"

# GGUF value types (gguf_value_type enum)
_GGUF_TYPE_UINT8 = 0
_GGUF_TYPE_INT8 = 1
_GGUF_TYPE_UINT16 = 2
_GGUF_TYPE_INT16 = 3
_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_INT32 = 5
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_BOOL = 7
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
_GGUF_TYPE_UINT64 = 10
_GGUF_TYPE_INT64 = 11
_GGUF_TYPE_FLOAT64 = 12

_FIXED_TYPES = {
    _GGUF_TYPE_UINT8: ("<B", 1),
    _GGUF_TYPE_INT8:  ("<b", 1),
    _GGUF_TYPE_UINT16: ("<H", 2),
    _GGUF_TYPE_INT16:  ("<h", 2),
    _GGUF_TYPE_UINT32: ("<I", 4),
    _GGUF_TYPE_INT32:  ("<i", 4),
    _GGUF_TYPE_FLOAT32: ("<f", 4),
    _GGUF_TYPE_BOOL:   ("<?", 1),
    _GGUF_TYPE_UINT64: ("<Q", 8),
    _GGUF_TYPE_INT64:  ("<q", 8),
    _GGUF_TYPE_FLOAT64: ("<d", 8),
}

# Keys we want to extract per architecture. The architecture name is read from
# `general.architecture`, then we ask for `<arch>.<key>`.
_WANTED_PER_ARCH = (
    "block_count",
    "embedding_length",
    "context_length",
    "feed_forward_length",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.value_length",
    "rope.dimension_count",
    "expert_count",
    "expert_used_count",
)


class _Reader:
    def __init__(self, src: BinaryIO):
        self.src = src

    def _read(self, n: int) -> bytes:
        b = self.src.read(n)
        if len(b) != n:
            raise EOFError(f"short read: wanted {n}, got {len(b)}")
        return b

    def fixed(self, value_type: int) -> Any:
        fmt, size = _FIXED_TYPES[value_type]
        return struct.unpack(fmt, self._read(size))[0]

    def string(self) -> str:
        length = struct.unpack("<Q", self._read(8))[0]
        if length > 1_000_000:
            raise ValueError(f"implausible string length {length}")
        return self._read(length).decode("utf-8", errors="replace")

    def value(self, value_type: int) -> Any:
        if value_type in _FIXED_TYPES:
            return self.fixed(value_type)
        if value_type == _GGUF_TYPE_STRING:
            return self.string()
        if value_type == _GGUF_TYPE_ARRAY:
            sub_type = struct.unpack("<I", self._read(4))[0]
            length = struct.unpack("<Q", self._read(8))[0]
            if length > 1_000_000:
                raise ValueError(f"implausible array length {length}")
            return [self.value(sub_type) for _ in range(length)]
        raise ValueError(f"unknown GGUF value type: {value_type}")


def parse_gguf_metadata(source) -> dict[str, Any]:
    """
    Parse GGUF header + metadata KV pairs. `source` may be:
      - a Path / str pointing to a local file
      - bytes / bytearray / memoryview holding the file's first ~tens of KB
      - a binary file-like object

    Returns a dict mapping key → value. Includes synthetic keys:
      _gguf_version, _tensor_count, _metadata_kv_count
    """
    if isinstance(source, (bytes, bytearray, memoryview)):
        f = io.BytesIO(bytes(source))
        opened = False
    elif isinstance(source, (str, os.PathLike)):
        f = open(source, "rb")
        opened = True
    else:
        f = source
        opened = False

    try:
        rd = _Reader(f)
        magic = rd._read(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"not a GGUF file (magic={magic!r})")
        version = struct.unpack("<I", rd._read(4))[0]
        tensor_count = struct.unpack("<Q", rd._read(8))[0]
        kv_count = struct.unpack("<Q", rd._read(8))[0]

        out: dict[str, Any] = {
            "_gguf_version": version,
            "_tensor_count": tensor_count,
            "_metadata_kv_count": kv_count,
        }
        for _ in range(kv_count):
            try:
                key = rd.string()
                vtype = struct.unpack("<I", rd._read(4))[0]
                out[key] = rd.value(vtype)
            except (EOFError, ValueError):
                # Truncated header — return what we have
                break
        return out
    finally:
        if opened:
            f.close()


def architecture_summary(meta: dict[str, Any]) -> dict[str, Any]:
    """
    Pick out architecture-shape fields from a parsed metadata dict.

    Returns: arch, block_count, embedding_length, head_count, head_count_kv,
             head_dim, context_length, expert_count, expert_used_count.
    Missing keys are None.
    """
    arch = meta.get("general.architecture")
    keys: dict[str, Any] = {"arch": arch}

    if not arch:
        return keys

    for k in _WANTED_PER_ARCH:
        keys[k] = meta.get(f"{arch}.{k}")

    embedding = keys.get("embedding_length")
    head_count = keys.get("attention.head_count")
    head_dim = keys.get("attention.key_length") or keys.get("rope.dimension_count")
    if not head_dim and embedding and head_count:
        try:
            head_dim = int(embedding) // int(head_count)
        except (TypeError, ZeroDivisionError):
            head_dim = None
    keys["head_dim"] = head_dim

    return keys


# Bytes per element for KV cache when stored in a given precision
_KV_BYTES = {
    "fp16": 2,
    "q8":   1.0625,   # ~8.5 bits/elem with scale overhead
    "q4":   0.5625,   # ~4.5 bits/elem with scale overhead
}


def estimate_load_ram(
    file_size_bytes: int,
    summary: dict[str, Any],
    context_lengths: list[int],
) -> dict[int, dict[str, float]] | None:
    """
    For each context length, estimate RAM in BYTES for {fp16, q8, q4} KV cache.

    Formula:
      kv_cache  = n_layers * n_kv_heads * (key_dim + value_dim) * kv_bytes * ctx
      compute   ≈ 0.15 * weights   (rule-of-thumb activation/scratch overhead)
      total     = weights + kv_cache + compute

    Some architectures (e.g., deepseek2) have asymmetric K and V dims; we
    read them separately when present. Falls back to head_dim for both.

    Returns None if essential architecture fields are missing.
    """
    n_layers = summary.get("block_count")
    n_kv_heads = summary.get("attention.head_count_kv") or summary.get("attention.head_count")
    head_dim = summary.get("head_dim")
    if not (n_layers and n_kv_heads and head_dim):
        return None

    key_dim = summary.get("attention.key_length") or head_dim
    value_dim = summary.get("attention.value_length") or head_dim

    n_layers = int(n_layers)
    n_kv_heads = int(n_kv_heads)
    key_dim = int(key_dim)
    value_dim = int(value_dim)
    weights = int(file_size_bytes)
    compute_overhead = int(weights * 0.15)

    out: dict[int, dict[str, float]] = {}
    for ctx in context_lengths:
        row: dict[str, float] = {}
        for tag, kv_bytes in _KV_BYTES.items():
            kv_total = n_layers * n_kv_heads * (key_dim + value_dim) * kv_bytes * ctx
            row[tag] = float(weights + compute_overhead + kv_total)
        out[ctx] = row
    return out


def detect_machine_ram_bytes() -> int | None:
    """Return total system RAM in bytes (macOS sysctl / Linux /proc/meminfo).
    Returns None if undetectable."""
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5, check=True,
            )
            return int(out.stdout.strip())
        except (subprocess.SubprocessError, ValueError, OSError):
            return None
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb * 1024
        except (OSError, ValueError):
            return None
    return None


def format_size_gb(n_bytes: float) -> str:
    return f"{n_bytes / (1024**3):.1f} GB"


def format_size_compact(n_bytes: float) -> str:
    if n_bytes >= 1024**4:
        return f"{n_bytes / (1024**4):.1f}TB"
    return f"{n_bytes / (1024**3):.1f}GB"


# ──────────────────────────────────────────────────────────────────────
# Pre-download header fetch (so we can estimate RAM before we have the
# whole multi-GB file on disk).
# ──────────────────────────────────────────────────────────────────────

def fetch_gguf_header_bytes(
    repo_id: str,
    filename: str,
    revision: str = "main",
    token: str | None = None,
    n_bytes: int = 65536,
    timeout: float = 30.0,
) -> bytes | None:
    """
    Fetch the first n_bytes of a GGUF on Hugging Face via a Range request.
    Follows the LFS CDN redirect. Returns the bytes on success, None on any
    failure (network, 404, gated, range not satisfiable, etc.).
    """
    try:
        import urllib.request
        url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
        headers = {
            "Range": f"bytes=0-{n_bytes - 1}",
            "User-Agent": "model_manager.py gguf-inspect",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            return data
    except Exception:
        return None
