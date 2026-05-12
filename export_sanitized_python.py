#!/usr/bin/env python3
"""
export_sanitized_python.py

Create sanitized export copies of Python scripts. Originals are left unchanged.

This version is careful about path redaction:
  - It replaces path-looking strings only.
  - It does not globally replace ordinary words.
  - It catches:
      <REDACTED_PATH>
      <REDACTED_PATH>
      <REDACTED_PATH>
      <REDACTED_PATH>
      <REDACTED_PATH>
      <REDACTED_PATH>
      <REDACTED_PATH>
  - It also replaces explicitly provided --old-dir paths.
  - It preserves Python code structure and only changes matched path text.

Supported usage:
  python3 export_sanitized_python.py Prepare_models_for_Lmstudio.py
  python3 export_sanitized_python.py --source Prepare_models_for_Lmstudio.py
  python3 export_sanitized_python.py --source . --export-dir ./sanitized_export
  python3 export_sanitized_python.py --source . --old-dir "<Your Model Directory>"

Original files are never modified.
"""

# Defer annotation evaluation so PEP 604 union syntax (X | Y) used in
# function signatures works even when this script is run with the macOS
# system Python 3.9. The type alias `Replacement` below uses typing.Union
# explicitly because aliases are evaluated at runtime regardless of this
# future import.
from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Tuple, Union


DEFAULT_PLACEHOLDER = "<Your Model Directory>"
DEFAULT_PATH_PLACEHOLDER = "<REDACTED_PATH>"

DEFAULT_PERSONAL_VALUES = [
    "<REDACTED>",
]

DEFAULT_MODEL_DIR_PATTERNS = [
    "<Your Model Directory>",
    "<Your Model Directory>",
    "<Your Model Directory>",
    "<Your Model Directory>/huggingface/hub",
    "<Your Model Directory>",
]

SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "site-packages",
    "dist",
    "build",
    "sanitized_export",
}

# PEP 604 union syntax (X | Y) at module scope requires Python 3.10+;
# /usr/bin/python3 on macOS is still 3.9 in some setups. Use typing.Union
# / typing.Tuple so the sanitizer runs on either interpreter.
Replacement = Tuple[re.Pattern, Union[str, Callable[[re.Match], str]]]


def redact_secret_assignment(match: re.Match) -> str:
    variable_name = match.group("name")
    quote = match.group("quote")
    spacing = match.group("spacing") or " = "
    return f"{variable_name}{spacing}{quote}<REDACTED_SECRET>{quote}"


def build_path_regexes(path_placeholder: str) -> list[Replacement]:
    """
    Redact only path-shaped values.

    These patterns intentionally stop at characters that normally terminate a
    path in Python source: quotes, whitespace, comma, closing bracket/paren,
    backtick, semicolon, or angle bracket.

    They do not replace arbitrary text elsewhere in the code.
    """
    path_end = r"""(?=$|[\s"'`,)\]\};<>])"""

    return [
        # <REDACTED_PATH> or <REDACTED_PATH>
        (
            re.compile(rf"<REDACTED_PATH>"'`,)\]\}};<>]+{path_end}"),
            path_placeholder,
        ),
        (
            re.compile(rf"<REDACTED_PATH>"'`,)\]\}};<>]+{path_end}"),
            path_placeholder,
        ),

        # macOS user paths, case-tolerant for /Users and /users.
        (
            re.compile(rf"/[Uu]sers/[^\s\"'`,)\]\}};<>]+{path_end}"),
            path_placeholder,
        ),

        # Shell-style home paths.
        (
            re.compile(rf"<REDACTED_PATH>"'`,)\]\}};<>]+{path_end}"),
            path_placeholder,
        ),
        (
            re.compile(rf"\<REDACTED_PATH>"'`,)\]\}};<>]+{path_end}"),
            path_placeholder,
        ),
        (
            re.compile(rf"\$\{{HOME\}}/[^\s\"'`,)\]\}};<>]+{path_end}"),
            path_placeholder,
        ),
    ]


def build_replacements(
    old_dirs: list[str],
    model_placeholder: str,
    path_placeholder: str,
    extra_secrets: list[str],
    include_default_model_dirs: bool,
) -> list[Replacement]:
    replacements: list[Replacement] = []

    # Exact model-directory replacements first. These use the model placeholder.
    dirs_to_replace: list[str] = []

    if include_default_model_dirs:
        dirs_to_replace.extend(DEFAULT_MODEL_DIR_PATTERNS)

    dirs_to_replace.extend(old_dirs)

    seen_dirs: set[str] = set()

    for directory in dirs_to_replace:
        directory = directory.strip()
        if not directory or directory in seen_dirs:
            continue

        seen_dirs.add(directory)
        replacements.append((re.compile(re.escape(directory)), model_placeholder))

    # Then redact any remaining path-shaped local paths. These use path placeholder.
    replacements.extend(build_path_regexes(path_placeholder))

    # Explicit personal values only. Do NOT redact generic first/last names by default,
    # because that can accidentally change code identifiers, comments, model names, etc.
    explicit_values = DEFAULT_PERSONAL_VALUES + extra_secrets
    seen_values: set[str] = set()

    for value in explicit_values:
        value = value.strip()
        if not value or value.lower() in seen_values:
            continue

        seen_values.add(value.lower())
        replacements.append((re.compile(re.escape(value), re.IGNORECASE), "<REDACTED>"))

    # Email addresses.
    replacements.append(
        (
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
            "<REDACTED_EMAIL>",
        )
    )

    # US-style phone numbers.
    replacements.append(
        (
            re.compile(r"\b(?:\+1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"),
            "<REDACTED_PHONE>",
        )
    )

    # Common Python secret assignments.
    replacements.append(
        (
            re.compile(
                r"""
                \b
                (?P<name>
                    api[_-]?key |
                    secret |
                    token |
                    access[_-]?token |
                    refresh[_-]?token |
                    password |
                    passwd |
                    private[_-]?key
                )
                \b
                (?P<spacing>\s*=\s*)
                (?P<quote>["'])
                .*?
                (?P=quote)
                """,
                re.IGNORECASE | re.VERBOSE,
            ),
            redact_secret_assignment,
        )
    )

    return replacements


def should_skip_path(path: Path, export_dir: Path) -> bool:
    if any(part in SKIP_DIR_NAMES for part in path.parts):
        return True

    try:
        path.resolve().relative_to(export_dir.resolve())
        return True
    except ValueError:
        return False
    except OSError:
        return False


def iter_python_files(source: Path, export_dir: Path):
    if source.is_file():
        if source.suffix == ".py" and not should_skip_path(source, export_dir):
            yield source
        return

    for path in source.rglob("*.py"):
        if path.is_file() and not should_skip_path(path, export_dir):
            yield path


def sanitize_text(text: str, replacements: list[Replacement]) -> tuple[str, int]:
    sanitized = text
    matched_patterns = 0

    for pattern, replacement in replacements:
        before = sanitized
        sanitized = pattern.sub(replacement, sanitized)

        if sanitized != before:
            matched_patterns += 1

    return sanitized, matched_patterns


def read_text_utf8(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def safe_relative_path(path: Path, source: Path) -> Path:
    if source.is_file():
        return Path(path.name)

    try:
        return path.resolve().relative_to(source.resolve())
    except ValueError:
        return Path(path.name)


def export_file(
    source_file: Path,
    source_root: Path,
    export_dir: Path,
    replacements: list[Replacement],
) -> dict:
    original_text = read_text_utf8(source_file)

    record = {
        "source": str(source_file),
        "exported": None,
        "status": "unknown",
        "replacement_groups_matched": 0,
    }

    if original_text is None:
        record["status"] = "skipped_non_utf8"
        return record

    sanitized_text, matched_patterns = sanitize_text(original_text, replacements)

    relative = safe_relative_path(source_file, source_root)
    destination = export_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(sanitized_text, encoding="utf-8")

    try:
        shutil.copystat(source_file, destination)
    except OSError:
        pass

    record["exported"] = str(destination)
    record["status"] = "exported_changed" if sanitized_text != original_text else "exported_unchanged"
    record["replacement_groups_matched"] = matched_patterns

    return record


def write_manifest(
    export_dir: Path,
    source: Path,
    records: list[dict],
    model_placeholder: str,
    path_placeholder: str,
    old_dirs: list[str],
    include_default_model_dirs: bool,
) -> Path:
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "export_dir": str(export_dir),
        "model_placeholder": model_placeholder,
        "path_placeholder": path_placeholder,
        "include_default_model_dirs": include_default_model_dirs,
        "explicit_old_dirs": old_dirs,
        "total_files": len(records),
        "exported_files": sum(1 for r in records if str(r.get("status", "")).startswith("exported")),
        "changed_files": sum(1 for r in records if r.get("status") == "exported_changed"),
        "skipped_files": sum(1 for r in records if str(r.get("status", "")).startswith("skipped")),
        "records": records,
    }

    manifest_path = export_dir / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def normalize_accidental_single_dash_args(argv: list[str]) -> list[str]:
    option_names = {
        "-source": "--source",
        "-export-dir": "--export-dir",
        "-old-dir": "--old-dir",
        "-placeholder": "--placeholder",
        "-path-placeholder": "--path-placeholder",
        "-extra-secret": "--extra-secret",
        "-clean-export-dir": "--clean-export-dir",
        "-no-default-model-dirs": "--no-default-model-dirs",
    }

    return [option_names.get(arg, arg) for arg in argv]


def main() -> int:
    import sys

    normalized_argv = normalize_accidental_single_dash_args(sys.argv[1:])

    parser = argparse.ArgumentParser(
        description="Export sanitized copies of Python scripts without modifying originals."
    )

    parser.add_argument(
        "positional_source",
        nargs="?",
        type=Path,
        help="Optional source Python file or directory.",
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Python file or directory to export.",
    )

    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("sanitized_export"),
        help="Destination folder for sanitized copies. Default: ./sanitized_export.",
    )

    parser.add_argument(
        "--old-dir",
        action="append",
        default=[],
        help='Exact model directory path to replace. Can be repeated. Example: --old-dir "<Your Model Directory>"',
    )

    parser.add_argument(
        "--no-default-model-dirs",
        action="store_true",
        help="Do not automatically replace built-in <REDACTED_PATH> model/cache paths.",
    )

    parser.add_argument(
        "--placeholder",
        default=DEFAULT_PLACEHOLDER,
        help='Replacement text for known model directories. Default: "<Your Model Directory>".',
    )

    parser.add_argument(
        "--path-placeholder",
        default=DEFAULT_PATH_PLACEHOLDER,
        help='Replacement text for other local paths. Default: "<REDACTED_PATH>".',
    )

    parser.add_argument(
        "--extra-secret",
        action="append",
        default=[],
        help="Additional exact personal/private value to redact. Can be repeated.",
    )

    parser.add_argument(
        "--clean-export-dir",
        action="store_true",
        help="Delete the export directory before writing the new export.",
    )

    args = parser.parse_args(normalized_argv)

    source_arg = args.source or args.positional_source or Path(".")
    source = source_arg.expanduser().resolve()
    export_dir = args.export_dir.expanduser().resolve()

    if not source.exists():
        print(f"ERROR: source does not exist: {source}")
        return 1

    if args.clean_export_dir and export_dir.exists():
        try:
            source.relative_to(export_dir)
            print("ERROR: refusing to delete an export directory that contains the source.")
            return 1
        except ValueError:
            pass

        shutil.rmtree(export_dir)

    export_dir.mkdir(parents=True, exist_ok=True)

    replacements = build_replacements(
        old_dirs=args.old_dir,
        model_placeholder=args.placeholder,
        path_placeholder=args.path_placeholder,
        extra_secrets=args.extra_secret,
        include_default_model_dirs=not args.no_default_model_dirs,
    )

    records: list[dict] = []

    print(f"Source: {source}")
    print(f"Export directory: {export_dir}")
    print("Original files will not be modified.")
    print()

    for py_file in iter_python_files(source, export_dir):
        record = export_file(
            source_file=py_file,
            source_root=source,
            export_dir=export_dir,
            replacements=replacements,
        )
        records.append(record)

        status = record["status"]
        exported = record["exported"] or ""
        print(f"{status}: {py_file}")
        if exported:
            print(f"  -> {exported}")

    manifest_path = write_manifest(
        export_dir=export_dir,
        source=source,
        records=records,
        model_placeholder=args.placeholder,
        path_placeholder=args.path_placeholder,
        old_dirs=args.old_dir,
        include_default_model_dirs=not args.no_default_model_dirs,
    )

    print()
    print("Export complete.")
    print(f"Files scanned: {len(records)}")
    print(f"Files exported: {sum(1 for r in records if str(r.get('status', '')).startswith('exported'))}")
    print(f"Files changed during export: {sum(1 for r in records if r.get('status') == 'exported_changed')}")
    print(f"Manifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
