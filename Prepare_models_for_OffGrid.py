#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, re, shutil, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HOME = Path.home()
DEFAULT_ROOTS = [
    Path("/Volumes/ModelStorage/models"),
    Path("/Volumes/ModelStorage/models-flat"),
    Path("/Volumes/ModelStorage/models/huggingface/model"),
    Path("/Volumes/ModelStorage/models-flat/local"),
    Path("/Volumes/ModelStorage/.cache/huggingface"),
    HOME / ".cache" / "huggingface",
    Path("/Volumes/ModelStorage/.cache/modelscope"),
    Path("/Volumes/ModelStorage/.cache/model_manager"),
    HOME / "model_downloads" / "huggingface" / "model",
    HOME / "Library" / "Application Support" / "nomic.ai" / "GPT4All",
    HOME / ".lmstudio" / "models",
    Path("~/skill-scanner/scan-results/20260503T074319Z"),
    Path("~/skill-scanner/scan-results/20260503T074334Z"),
]
SKIP_DIR_NAMES = {"blobs", ".locks", "refs", ".studio_links", ".git", "__pycache__"}
MIN_GGUF_BYTES = 50 * 1024 * 1024
GGUF_MAGIC = b"GGUF"
_print_lock = threading.Lock()

def log(msg: str) -> None:
    with _print_lock:
        print(msg)

def sanitize(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]", "-", name)
    name = re.sub(r"-+", "-", name)
    return name.strip("-") or "model"

def is_probably_model(path: Path) -> bool:
    try:
        if path.stat().st_size < MIN_GGUF_BYTES:
            return False
        with path.open("rb") as f:
            return f.read(4) == GGUF_MAGIC
    except OSError:
        return False

def iter_ggufs(roots: list[Path]):
    seen=set()
    for root in roots:
        if not root.exists():
            continue
        for cur, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in SKIP_DIR_NAMES]
            curp=Path(cur)
            for name in files:
                if not name.lower().endswith(".gguf"):
                    continue
                p=curp/name
                try:
                    rp=p.resolve()
                except OSError:
                    rp=p
                if rp in seen:
                    continue
                seen.add(rp)
                if is_probably_model(p):
                    yield p

def flat_name(path: Path) -> str:
    parts = path.parts
    # Preserve author/model signal when possible.
    if len(parts) >= 3:
        parent = sanitize(path.parent.name)
        grand = sanitize(path.parent.parent.name)
        if grand not in {"models", "local", "model", "huggingface", "models-flat"}:
            return f"{grand}__{parent}__{sanitize(path.name)}"
        return f"{parent}__{sanitize(path.name)}"
    return sanitize(path.name)

def ensure_symlink(src: Path, dst: Path, dry_run: bool) -> tuple[str, str]:
    if dst.exists() or dst.is_symlink():
        try:
            if dst.resolve() == src.resolve():
                return "exists", str(dst)
        except OSError:
            pass
        return "skip-collision", str(dst)
    if dry_run:
        return "would-link", f"{dst} -> {src}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst)
    return "linked", str(dst)

def clean_dangling(target: Path, dry_run: bool) -> int:
    n=0
    if not target.exists():
        return 0
    for p in target.rglob("*"):
        if p.is_symlink() and not p.exists():
            n += 1
            log(("WOULD REMOVE" if dry_run else "REMOVE") + f" dangling {p}")
            if not dry_run:
                p.unlink(missing_ok=True)
    return n

def run_linker(app_name: str, target: Path, args: argparse.Namespace, nested: bool=False) -> int:
    if args.clean:
        clean_dangling(target, args.dry_run)
    roots = list(args.gguf_root or []) or DEFAULT_ROOTS
    models = list(iter_ggufs(roots))
    log(f"Prepare models for {app_name} — {'DRY RUN' if args.dry_run else 'LIVE'}")
    log(f"target: {target}")
    log(f"models found: {len(models)}")
    if args.dry_run:
        log("no filesystem changes will be made")
    counts={}
    def one(src: Path):
        if nested:
            model_id = sanitize(src.stem)
            dst = target / model_id / src.name
        else:
            dst = target / flat_name(src)
        return ensure_symlink(src, dst, args.dry_run)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for status, detail in ex.map(one, models):
            counts[status]=counts.get(status,0)+1
            if status in {"linked", "would-link", "skip-collision"}:
                log(f"  {status:<14} {detail}")
    log("summary: " + ", ".join(f"{k}={v}" for k,v in sorted(counts.items())) if counts else "summary: no models")
    return 0

def build_parser(desc: str, default_target: Path):
    ap=argparse.ArgumentParser(description=desc)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--clean", action="store_true", help="remove dangling symlinks under target first")
    ap.add_argument("--gguf-root", type=Path, action="append", default=[], help="extra/override GGUF scan root; repeatable")
    ap.add_argument("--target", type=Path, default=default_target)
    ap.add_argument("--workers", type=int, default=16)
    return ap


def main() -> int:
    ap = build_parser('Prepare local GGUF models for OffGrid/offline runners by symlinking them into an OffGrid model directory.', HOME / 'Library' / 'Application Support' / 'OffGrid' / 'models')
    args = ap.parse_args()
    return run_linker('OffGrid', args.target, args, nested=False)

if __name__ == "__main__":
    raise SystemExit(main())
