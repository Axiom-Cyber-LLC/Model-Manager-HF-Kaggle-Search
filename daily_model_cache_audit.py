#!/usr/bin/env python3
"""
daily_model_cache_audit.py

Daily read-only audit for explicitly allowed model/cache directories only.

It reports:
  - files with macOS quarantine flag
  - new/modified files in allowed model/cache directories
  - largest recent files in allowed directories

It does NOT:
  - scan /Volumes/ModelStorage as a whole drive
  - audit cloud folders
  - delete files
  - quarantine files
  - remove quarantine xattrs
  - modify files
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path


# Explicit allowlist only. Do not add /Volumes/ModelStorage here.
SCAN_DIRS = [
    Path("/Volumes/ModelStorage/models"),
    Path("/Volumes/ModelStorage/models-flat"),
    Path("/Volumes/ModelStorage/.cache/huggingface"),
    Path("/Volumes/ModelStorage/.cache/modelscope"),
    Path.home() / ".cache" / "huggingface",
]

REPORT_DIR = Path.home() / "model_quarantine_reports"

LOOKBACK_HOURS = 24
LARGEST_RECENT_LIMIT = 200
LARGEST_QUARANTINE_LIMIT = 200

SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".Trash",
    ".Spotlight-V100",
    ".fseventsd",
    ".TemporaryItems",
}

PROTECTED_PATH_PREFIXES = [
    Path("/"),
    Path("/System"),
    Path("/Library"),
    Path("/private"),
    Path("/usr"),
    Path("/bin"),
    Path("/sbin"),
    Path("/Applications"),
    Path.home() / "Library" / "Mobile Documents",
    Path.home() / "Library" / "CloudStorage",
    Path.home() / "Dropbox",
]


def human_size(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "-"

    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(num_bytes)

    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024

    return f"{value:.2f} PB"


def is_explicitly_allowed_root(path: Path) -> bool:
    resolved = path.expanduser().resolve()
    allowed_roots = [p.expanduser().resolve() for p in SCAN_DIRS]

    return any(resolved == root for root in allowed_roots)


def validate_scan_dirs() -> None:
    for root in SCAN_DIRS:
        resolved = root.expanduser().resolve()

        # Block accidental whole-drive or system-wide scans.
        forbidden_exact = {
            Path("/"),
            Path("/Volumes"),
            Path("/Volumes/ModelStorage"),
            Path.home(),
            Path.home() / "Library",
            Path.home() / "Library" / "CloudStorage",
            Path.home() / "Library" / "Mobile Documents",
        }

        if resolved in forbidden_exact:
            raise RuntimeError(f"Refusing unsafe scan root: {resolved}")

        # Explicit cloud-folder block.
        cloud_roots = [
            Path.home() / "Library" / "CloudStorage",
            Path.home() / "Library" / "Mobile Documents",
            Path.home() / "Dropbox",
        ]
        for cloud_root in cloud_roots:
            try:
                resolved.relative_to(cloud_root.expanduser().resolve())
                raise RuntimeError(f"Refusing cloud-synced scan root: {resolved}")
            except ValueError:
                pass


def has_quarantine_xattr(path: Path) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["xattr", "-p", "com.apple.quarantine", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        return False, f"xattr_error:{type(exc).__name__}:{exc}"

    if result.returncode == 0:
        return True, result.stdout.strip()

    return False, ""


def gatekeeper_status() -> str:
    try:
        result = subprocess.run(
            ["spctl", "--status"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        return f"unknown: {type(exc).__name__}: {exc}"

    return result.stdout.strip() or "unknown"


def walk_files(root: Path):
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]

            for filename in filenames:
                yield Path(dirpath) / filename
    except PermissionError:
        return


def main() -> int:
    validate_scan_dirs()

    now = datetime.now()
    cutoff = now - timedelta(hours=LOOKBACK_HOURS)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    scanned_files = 0
    missing_dirs = []
    quarantine_findings = []
    recent_files = []
    growth_by_scan_root: dict[str, int] = {}

    seen_paths: set[str] = set()
    gk_status = gatekeeper_status()

    for root in SCAN_DIRS:
        root = root.expanduser()

        if not root.exists():
            missing_dirs.append(str(root))
            continue

        if not is_explicitly_allowed_root(root):
            raise RuntimeError(f"Refusing non-allowlisted scan root: {root}")

        root_key = str(root)

        for path in walk_files(root):
            path_key = str(path)
            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)

            scanned_files += 1

            try:
                st = path.stat()
            except OSError:
                continue

            size = int(st.st_size)
            modified_dt = datetime.fromtimestamp(st.st_mtime)
            created_dt = datetime.fromtimestamp(st.st_birthtime) if hasattr(st, "st_birthtime") else None

            flagged, q_value = has_quarantine_xattr(path)
            if flagged:
                quarantine_findings.append({
                    "path": str(path),
                    "size_bytes": size,
                    "size_human": human_size(size),
                    "modified": modified_dt.isoformat(timespec="seconds"),
                    "created": created_dt.isoformat(timespec="seconds") if created_dt else None,
                    "quarantine_value": q_value,
                })

            is_recent = modified_dt >= cutoff or (created_dt is not None and created_dt >= cutoff)
            if is_recent:
                growth_by_scan_root[root_key] = growth_by_scan_root.get(root_key, 0) + size

                recent_files.append({
                    "path": str(path),
                    "size_bytes": size,
                    "size_human": human_size(size),
                    "modified": modified_dt.isoformat(timespec="seconds"),
                    "created": created_dt.isoformat(timespec="seconds") if created_dt else None,
                    "scan_root": root_key,
                })

    recent_files.sort(key=lambda item: item["size_bytes"], reverse=True)
    quarantine_findings.sort(key=lambda item: item["size_bytes"], reverse=True)

    total_recent_bytes = sum(item["size_bytes"] for item in recent_files)
    growth_rows = [
        {
            "scan_root": root,
            "bytes": size,
            "size_human": human_size(size),
        }
        for root, size in sorted(growth_by_scan_root.items(), key=lambda kv: kv[1], reverse=True)
    ]

    stamp = now.strftime("%Y-%m-%d")
    json_path = REPORT_DIR / f"model_cache_audit_{stamp}.json"
    txt_path = REPORT_DIR / f"model_cache_audit_{stamp}.txt"
    recent_csv_path = REPORT_DIR / f"model_cache_recent_files_{stamp}.csv"
    quarantine_csv_path = REPORT_DIR / f"model_cache_quarantine_{stamp}.csv"

    summary = {
        "generated_at": now.isoformat(timespec="seconds"),
        "lookback_hours": LOOKBACK_HOURS,
        "gatekeeper_status": gk_status,
        "scanned_files": scanned_files,
        "recent_files_count": len(recent_files),
        "recent_files_total_bytes": total_recent_bytes,
        "recent_files_total_human": human_size(total_recent_bytes),
        "quarantine_flagged_files": len(quarantine_findings),
        "scan_dirs": [str(p.expanduser()) for p in SCAN_DIRS],
        "missing_dirs": missing_dirs,
        "whole_drive_scan": False,
        "read_only": True,
    }

    json_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "growth_by_scan_root": growth_rows,
                "largest_recent_files": recent_files[:LARGEST_RECENT_LIMIT],
                "quarantine_findings": quarantine_findings[:LARGEST_QUARANTINE_LIMIT],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with recent_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "size_bytes", "size_human", "modified", "created", "scan_root"],
        )
        writer.writeheader()
        writer.writerows(recent_files)

    with quarantine_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "size_bytes", "size_human", "modified", "created", "quarantine_value"],
        )
        writer.writeheader()
        writer.writerows(quarantine_findings)

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("Daily model/cache audit\n")
        f.write("=" * 32 + "\n")
        f.write(f"Generated: {summary['generated_at']}\n")
        f.write(f"Lookback hours: {LOOKBACK_HOURS}\n")
        f.write(f"Gatekeeper status: {gk_status}\n")
        f.write("Whole-drive scan: disabled\n")
        f.write("Mode: read-only\n")
        f.write(f"Scanned files: {scanned_files}\n")
        f.write(f"Recent files: {len(recent_files)}\n")
        f.write(f"Recent total size: {human_size(total_recent_bytes)}\n")
        f.write(f"Quarantine flagged files: {len(quarantine_findings)}\n")

        if missing_dirs:
            f.write("\nMissing allowed directories:\n")
            for d in missing_dirs:
                f.write(f"  - {d}\n")

        f.write("\nGrowth by allowed scan root:\n")
        if growth_rows:
            for row in growth_rows:
                f.write(f"  {row['size_human']:>12}  {row['scan_root']}\n")
        else:
            f.write("  No recent growth detected.\n")

        f.write(f"\nLargest recent files, top {min(LARGEST_RECENT_LIMIT, len(recent_files))}:\n")
        if recent_files:
            for item in recent_files[:LARGEST_RECENT_LIMIT]:
                f.write(f"  {item['size_human']:>12}  {item['modified']}  {item['path']}\n")
        else:
            f.write("  No recent files detected.\n")

        f.write(f"\nQuarantine flagged files, top {min(LARGEST_QUARANTINE_LIMIT, len(quarantine_findings))}:\n")
        if quarantine_findings:
            for item in quarantine_findings[:LARGEST_QUARANTINE_LIMIT]:
                f.write(f"  {item['size_human']:>12}  {item['modified']}  {item['path']}\n")
                f.write(f"                quarantine: {item['quarantine_value']}\n")
        else:
            f.write("  No files with com.apple.quarantine found.\n")

    print(f"Gatekeeper status: {gk_status}")
    print("Whole-drive scan: disabled")
    print(f"Scanned files: {scanned_files}")
    print(f"Recent files: {len(recent_files)}")
    print(f"Recent total size: {human_size(total_recent_bytes)}")
    print(f"Quarantine flagged files: {len(quarantine_findings)}")
    print(f"Report: {txt_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
