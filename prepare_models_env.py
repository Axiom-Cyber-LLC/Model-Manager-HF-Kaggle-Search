from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def _clean_path(value: str | None) -> Path | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    return Path(text).expanduser().resolve()


def prep_environment_paths(
    model_dir: Path | None = None,
    download_root: Path | None = None,
) -> dict[str, Path | None]:
    target = model_dir.expanduser().resolve() if model_dir is not None else _clean_path(os.getenv("MODEL_MANAGER_DOWNLOAD_TARGET"))
    root = download_root.expanduser().resolve() if download_root is not None else _clean_path(os.getenv("MODEL_MANAGER_DOWNLOAD_ROOT"))
    return {
        "download_target": target,
        "download_root": root,
    }


def export_prep_environment(
    model_dir: Path | None = None,
    download_root: Path | None = None,
) -> dict[str, Path | None]:
    paths = prep_environment_paths(model_dir=model_dir, download_root=download_root)
    if paths["download_target"] is not None:
        os.environ["MODEL_MANAGER_DOWNLOAD_TARGET"] = str(paths["download_target"])
    if paths["download_root"] is not None:
        os.environ["MODEL_MANAGER_DOWNLOAD_ROOT"] = str(paths["download_root"])
    return paths


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            resolved = path.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def manager_scan_roots() -> list[Path]:
    paths = prep_environment_paths()
    target = paths["download_target"]
    root = paths["download_root"]
    extras: list[Path] = []

    if root is not None:
        extras.extend(
            [
                root,
                root / "huggingface" / "model",
                root / "kaggle" / "model",
                root / "local",
            ]
        )

    if target is not None:
        target_dir = target if target.is_dir() else target.parent
        extras.extend(
            [
                target_dir,
                target_dir.parent,
            ]
        )

    existing = [path for path in extras if path is not None]
    return _dedupe_paths(existing)


def extend_scan_roots(default_roots: Iterable[Path]) -> list[Path]:
    return _dedupe_paths(list(default_roots) + manager_scan_roots())


def default_manager_models_dir(default: Path) -> Path:
    paths = prep_environment_paths()
    target = paths["download_target"]
    root = paths["download_root"]

    if target is not None:
        target_dir = target if target.is_dir() else target.parent
        if target_dir.parent.name in {"model", "dataset"}:
            return target_dir.parent
        if target_dir.name in {"model", "models-flat", "local"}:
            return target_dir
        if target_dir.parent.exists():
            return target_dir.parent
        return target_dir

    if root is not None:
        preferred = root / "huggingface" / "model"
        if preferred.exists():
            return preferred
        if root.exists():
            return root

    return default.expanduser().resolve()
