#!/usr/bin/env python3
"""
Prepare_models_for_All.py
-------------------------
Orchestrator: runs `model_audit` once, then runs every per-app prepare script
in dependency order with shared flags (--workers, --dry-run).

Run order (each app's data layout depends on the previous step's output):
  1.  Prepare_models_for_Lmstudio.py    (canonical layout / hub)
  2.  Prepare_models_for_Ollama.py      (registers GGUFs from the layout)
  3.  Prepare_models_for_AnythingLLM.py (depends on Ollama listing being live)
  4.  Prepare_models_for_GPT4All.py     (symlinks)
  5.  Prepare_models_for_Jan.py         (symlinks)
  6.  Prepare_models_for_AIStudio.py    (DB inject — last; AI Nav must be quit)
  7.  Prepare_models_for_LocallyAI.py   (symlinks, if script exists)
  8.  Prepare_models_for_LocalAI.py     (symlinks/config, if script exists)
  9.  Prepare_models_for_Apollo.py      (symlinks, if script exists)
  10. Prepare_models_for_OffGrid.py     (symlinks, if script exists)

Children run WITHOUT --audit (we did it once up front).

Usage:
    python3 Prepare_models_for_All.py                # full sequence
    python3 Prepare_models_for_All.py --dry-run
    python3 Prepare_models_for_All.py --workers 8
    python3 Prepare_models_for_All.py --no-audit
    python3 Prepare_models_for_All.py --only Ollama,GPT4All
    python3 Prepare_models_for_All.py --skip AIStudio
    python3 Prepare_models_for_All.py --continue-on-error
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Find model_audit (it lives next to the prepare scripts)
_AUDIT_PATHS = [
    Path(__file__).resolve().parent,
    Path("/Volumes/ModelStorage/models-flat"),
]
for _p in _AUDIT_PATHS:
    if (_p / "model_audit.py").is_file() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    import model_audit
    _HAS_AUDIT = True
except ImportError:
    _HAS_AUDIT = False


DEFAULT_SCRIPTS_DIR = Path(__file__).resolve().parent

# Order matters — see module docstring.
APP_ORDER = [
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


def _csv(s: str) -> set[str]:
    return {tok.strip().lower() for tok in s.split(",") if tok.strip()}


def _resolve_app_set(only: str | None, skip: str | None) -> list[str]:
    apps = list(APP_ORDER)
    if only:
        wanted = _csv(only)
        apps = [a for a in apps if a.lower() in wanted]
    if skip:
        unwanted = _csv(skip)
        apps = [a for a in apps if a.lower() not in unwanted]
    return apps


def _script_accepts(script: Path, flag: str) -> bool:
    """Probe `script --help` once and look for the flag in its help text."""
    try:
        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return flag in (proc.stdout or "")


def _run_one(script: Path, args: argparse.Namespace) -> int:
    """Run a single prepare script as a subprocess. Return its exit code."""
    cmd = [sys.executable, str(script)]
    if args.dry_run and _script_accepts(script, "--dry-run"):
        cmd.append("--dry-run")
    if _script_accepts(script, "--workers"):
        cmd += ["--workers", str(args.workers)]
    print()
    print("─" * 78)
    print(f"▶ {script.name}")
    print("─" * 78)
    print(f"  command: {' '.join(cmd)}")
    t0 = time.time()
    rc = subprocess.call(cmd)
    elapsed = time.time() - t0
    status = "OK" if rc == 0 else f"FAIL (rc={rc})"
    print(f"  → {status} in {elapsed:.1f}s")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--scripts-dir", type=Path, default=DEFAULT_SCRIPTS_DIR,
                    help=f"Where the per-app Prepare_models_for_X.py scripts "
                         f"live. Default: {DEFAULT_SCRIPTS_DIR}")
    ap.add_argument("--workers", type=int, default=8,
                    help="Forwarded to each child script as --workers (default 8)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Forward --dry-run to children; audit also runs in dry-run")
    ap.add_argument("--no-audit", action="store_true",
                    help="Skip the model_audit step at the start")
    ap.add_argument("--non-interactive-audit", action="store_true",
                    help="Run audit but never prompt to delete (report only)")
    ap.add_argument("--skip-duplicates-audit", action="store_true",
                    help="Skip the SHA-256 duplicate-detection pass in audit (faster)")
    ap.add_argument("--only", type=str, default=None,
                    help="Comma-list of app names to include (Lmstudio,Ollama,…)")
    ap.add_argument("--skip", type=str, default=None,
                    help="Comma-list of app names to skip")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="Keep running subsequent steps even if one fails")
    args = ap.parse_args()

    print("=" * 78)
    print(f"Prepare_models_for_All — {'DRY RUN' if args.dry_run else 'LIVE'}")
    print("=" * 78)

    if not args.scripts_dir.is_dir():
        print(f"ERROR: scripts dir does not exist: {args.scripts_dir}")
        return 2

    apps = _resolve_app_set(args.only, args.skip)
    if not apps:
        print("ERROR: --only/--skip filtered out every app.")
        return 2
    print(f"  scripts dir: {args.scripts_dir}")
    print(f"  apps:        {', '.join(apps)}")
    print(f"  workers:     {args.workers}")
    print()

    # 1) Audit step
    if not args.no_audit:
        if not _HAS_AUDIT:
            print(f"WARNING: model_audit.py not importable from {_AUDIT_PATHS}; "
                  "skipping audit step.")
        else:
            print("Step 1 — running model audit…")
            report = model_audit.run_audit(
                workers=args.workers,
                skip_duplicates=args.skip_duplicates_audit,
            )
            model_audit.interactive_prompt(
                report, dry_run=args.dry_run,
                non_interactive=args.non_interactive_audit,
            )

    # 2) Run each prepare script
    print()
    print("Step 2 — running per-app prepare scripts…")
    results: list[tuple[str, int, float]] = []
    t_start = time.time()
    for app in apps:
        script = args.scripts_dir / f"Prepare_models_for_{app}.py"
        if not script.is_file():
            print(f"  SKIP {app}: {script} not found")
            results.append((app, 127, 0.0))
            # Extended targets are optional unless specifically requested and continue-on-error is off.
            optional_extended = {"LocallyAI", "LocalAI", "Apollo", "OffGrid"}
            if app not in optional_extended and not args.continue_on_error:
                print("  Aborting. Use --continue-on-error to ignore missing scripts.")
                return 2
            continue
        t0 = time.time()
        rc = _run_one(script, args)
        results.append((app, rc, time.time() - t0))
        if rc != 0 and not args.continue_on_error:
            print(f"\nAborting due to {app} failure (rc={rc}). "
                  "Use --continue-on-error to keep going.")
            break

    # 3) Summary
    total_elapsed = time.time() - t_start
    print()
    print("=" * 78)
    print(f"Summary — {total_elapsed:.1f}s total")
    print("=" * 78)
    for app, rc, elapsed in results:
        tag = "OK  " if rc == 0 else f"FAIL"
        print(f"  {tag}  {app:<14}  rc={rc:<4}  {elapsed:.1f}s")
    failures = sum(1 for _, rc, _ in results if rc != 0)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
