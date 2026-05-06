#!/usr/bin/env python3
"""
model_conversion.py
-------------------
Interactive safetensors → GGUF converter for the local-model tree.

Flow:
  1. Run the shared model_audit step first (skip with --no-audit).
  2. Discover candidate dirs: HF-style folders with config.json + .safetensors
     and no existing .gguf sibling.
  3. Print numbered table; offer "0. Convert them all".
  4. Per selected model, ask quant (default Q4_K_M; --quant overrides).
  5. Run convert_hf_to_gguf.py → llama-quantize.
  6. After conversions complete, prompt to register via Prepare_models_for_All.py.

Usage:
    python3 model_conversion.py                       # interactive
    python3 model_conversion.py --list-only           # candidate table only
    python3 model_conversion.py --select 0             # convert all candidates non-interactively
    python3 model_conversion.py --select 3             # convert candidate #3 non-interactively
    python3 model_conversion.py --quant Q5_K_M        # default quant
    python3 model_conversion.py --workers 1           # parallel conversions
    python3 model_conversion.py --no-audit            # skip audit step
"""
from __future__ import annotations
import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# model_audit lives next to this script in the public tool checkout; the
# ModelStorage path is only a user-configurable fallback.
_AUDIT_PATHS = [
    Path(__file__).resolve().parent,
    Path("/Volumes/ModelStorage/models-flat"),
]
for _p in _AUDIT_PATHS:
    if (_p / "model_audit.py").is_file() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    import model_audit
except ImportError as e:
    sys.stderr.write(f"ERROR: model_audit.py not importable from "
                     f"{_AUDIT_PATHS}: {e}\n")
    sys.exit(2)


# ── Defaults ─────────────────────────────────────────────────────────

HOME = Path.home()
DEFAULT_SCAN_ROOTS = [
    Path("/Volumes/ModelStorage/models"),
    Path("/Volumes/ModelStorage/models-flat"),
    Path("/Volumes/ModelStorage/.cache/huggingface"),
    HOME / ".cache" / "huggingface",
    Path("/Volumes/ModelStorage/.cache/modelscope"),
]
DEFAULT_QUANT = "Q4_K_M"
DEFAULT_OUT_DTYPE = "f16"  # convert_hf_to_gguf.py outtype before quantizing
DEFAULT_ORCHESTRATOR = Path(__file__).resolve().with_name("Prepare_models_for_All.py")
SUPPORTED_QUANTS = (
    "F32", "F16", "BF16",
    "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S",
    "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q2_K",
    "IQ4_XS", "IQ3_M", "IQ3_S", "IQ2_M", "IQ2_S",
)

CONVERT_BIN = "/opt/homebrew/bin/convert_hf_to_gguf.py"
QUANTIZE_BIN = "/opt/homebrew/bin/llama-quantize"
SKIP_DIR_NAMES = {"blobs", ".locks", "refs", ".studio_links", ".git", "__pycache__"}
SHARDED_RE = re.compile(r"model-\d+-of-\d+\.safetensors$")


# ── Candidate discovery ──────────────────────────────────────────────


@dataclass
class Candidate:
    path: Path
    publisher: str
    model: str
    total_safetensors_bytes: int
    n_safetensors: int
    has_existing_gguf: bool
    existing_gguf_quants: list[str]


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} PB"


def _publisher_model(d: Path) -> tuple[str, str]:
    """Best-effort (publisher, model) from path layout."""
    for anc in d.parents:
        if anc.name.startswith("models--"):
            bits = anc.name[len("models--"):].split("--", 1)
            if len(bits) == 2 and all(bits):
                return bits[0], bits[1]
    parts = d.parts
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return "local", d.name


_QUANT_RE = re.compile(
    r"[-_](Q\d[_\w]*|IQ\d[_\w]*|F16|F32|BF16|FP16)\.gguf$",
    re.IGNORECASE,
)


def _detect_quant(name: str) -> str:
    m = _QUANT_RE.search(name)
    return m.group(1).upper() if m else "?"


def _looks_like_safetensors_dir(d: Path) -> bool:
    """An HF-style safetensors directory: config.json + at least one .safetensors."""
    if not d.is_dir():
        return False
    if not (d / "config.json").is_file():
        return False
    return any(d.glob("*.safetensors"))


def _has_blocking_format(d: Path) -> bool:
    """If MLX-only (no PyTorch/HF tokenizer files), convert_hf_to_gguf can't load it.
    The single most reliable check: does it have a tokenizer.json or tokenizer.model?"""
    return any(
        (d / f).is_file()
        for f in ("tokenizer.json", "tokenizer.model", "vocab.json")
    )


def discover_candidates(roots: list[Path]) -> list[Candidate]:
    seen: set[Path] = set()
    out: list[Candidate] = []
    for root in roots:
        if not root.is_dir():
            continue
        # Walk subdirs that contain a config.json
        for cfg in root.rglob("config.json"):
            d = cfg.parent
            try:
                rel = d.relative_to(root).parts
            except ValueError:
                rel = ()
            if any(part in SKIP_DIR_NAMES for part in rel):
                continue
            if any(part.startswith(".") and part not in (".", "..")
                   for part in rel[:-1]):
                continue
            try:
                real = d.resolve()
            except OSError:
                real = d
            if real in seen:
                continue
            seen.add(real)
            if not _looks_like_safetensors_dir(d):
                continue
            if not _has_blocking_format(d):
                # No tokenizer artifacts — convert_hf_to_gguf would refuse
                continue
            safetensors = list(d.glob("*.safetensors"))
            existing_ggufs = list(d.glob("*.gguf"))
            try:
                total_st = sum(f.stat().st_size for f in safetensors)
            except OSError:
                continue
            publisher, model = _publisher_model(d)
            out.append(Candidate(
                path=d,
                publisher=publisher,
                model=model,
                total_safetensors_bytes=total_st,
                n_safetensors=len(safetensors),
                has_existing_gguf=bool(existing_ggufs),
                existing_gguf_quants=sorted({_detect_quant(g.name) for g in existing_ggufs}),
            ))
    out.sort(key=lambda c: (c.has_existing_gguf, c.publisher.lower(), c.model.lower()))
    return out


def print_table(cands: list[Candidate]) -> None:
    print()
    print(f"{'#':>3}  {'publisher/model':<55}  {'size':>10}  shards  existing")
    print("-" * 100)
    for i, c in enumerate(cands, 1):
        existing = ",".join(c.existing_gguf_quants) if c.has_existing_gguf else "—"
        name = f"{c.publisher}/{c.model}"
        if len(name) > 55:
            name = name[:52] + "..."
        print(f"{i:>3}  {name:<55}  {_human(c.total_safetensors_bytes):>10}  "
              f"{c.n_safetensors:>5}  {existing}")
    print()
    print("  0  Convert them all")
    print()


# ── Selection parsing ────────────────────────────────────────────────


def parse_selection(s: str, n_max: int) -> list[int]:
    """Parse '0' / '3' / '1,3,5' / '2-5' / '1,4-7,9' into 1-based indices."""
    s = s.strip().lower()
    if not s:
        return []
    if s == "0" or s in ("all", "*"):
        return list(range(1, n_max + 1))
    out: set[int] = set()
    for tok in s.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-", 1)
            try:
                lo = int(a); hi = int(b)
            except ValueError:
                raise ValueError(f"bad range: {tok!r}")
            for i in range(lo, hi + 1):
                if 1 <= i <= n_max:
                    out.add(i)
        else:
            try:
                i = int(tok)
            except ValueError:
                raise ValueError(f"bad index: {tok!r}")
            if 1 <= i <= n_max:
                out.add(i)
    return sorted(out)


# ── Conversion ───────────────────────────────────────────────────────


def _check_tools() -> tuple[bool, str]:
    if not Path(CONVERT_BIN).is_file():
        return False, f"converter not found at {CONVERT_BIN}"
    if not Path(QUANTIZE_BIN).is_file():
        return False, f"quantizer not found at {QUANTIZE_BIN}"
    # Test the gguf import that convert_hf_to_gguf needs
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import gguf"],
            capture_output=True, timeout=10,
        )
        if proc.returncode != 0:
            return False, ("the python interpreter that runs "
                           f"{CONVERT_BIN} cannot import the `gguf` module. "
                           "Install with:  pip install gguf")
    except (subprocess.SubprocessError, OSError) as e:
        return False, f"could not check gguf module: {e}"
    return True, ""


def _convert_one(c: Candidate, quant: str, dry_run: bool) -> tuple[Candidate, str, str]:
    """
    Return (candidate, status, message). status in {ok, exists, skipped, failed}.
    """
    out_stem = f"{c.model}-{quant}"
    f16_path = c.path / f"{c.model}-f16.gguf"
    final_path = c.path / f"{out_stem}.gguf"

    if final_path.is_file() and final_path.stat().st_size > 50 * 1024 * 1024:
        return c, "exists", f"already at {final_path.name}"

    if dry_run:
        return c, "ok", (f"[DRY] would run convert_hf_to_gguf.py on {c.path} "
                        f"→ {f16_path.name} → quantize to {final_path.name}")

    # 1) Convert to f16 (if not already)
    if not f16_path.is_file() or f16_path.stat().st_size < 50 * 1024 * 1024:
        cmd = [sys.executable, CONVERT_BIN, str(c.path),
               "--outfile", str(f16_path), "--outtype", DEFAULT_OUT_DTYPE]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        except subprocess.TimeoutExpired:
            return c, "failed", "convert_hf_to_gguf.py timed out (>1h)"
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-400:]
            return c, "failed", f"convert_hf_to_gguf failed: {tail}"

    # 2) Quantize
    if quant.upper() in ("F16", "FP16"):
        # No quantization needed; rename
        try:
            f16_path.rename(final_path)
        except OSError as e:
            return c, "failed", f"rename failed: {e}"
        return c, "ok", f"produced {final_path.name}"

    cmd = [QUANTIZE_BIN, str(f16_path), str(final_path), quant]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    except subprocess.TimeoutExpired:
        return c, "failed", "llama-quantize timed out (>2h)"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-400:]
        return c, "failed", f"llama-quantize failed: {tail}"

    return c, "ok", f"produced {final_path.name}"


def run_conversions(cands: list[Candidate], quant: str, workers: int,
                    dry_run: bool) -> list[Path]:
    """Convert each candidate at the requested quant. Returns produced GGUFs."""
    if not cands:
        return []
    print(f"\nConverting {len(cands)} model(s) at quant={quant}, workers={workers}…")
    if workers > 2 and not dry_run:
        print("WARNING: each llama-quantize already saturates the cores. "
              "workers > 2 will likely THRASH. Continuing in 5s — Ctrl-C to abort.")
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("Aborted.")
            return []

    produced: list[Path] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {pool.submit(_convert_one, c, quant, dry_run): c for c in cands}
        for fut in as_completed(futs):
            c, status, msg = fut.result()
            tag = {"ok": "OK   ", "exists": "SKIP ", "skipped": "SKIP ",
                   "failed": "FAIL "}[status]
            print(f"  {tag}{c.publisher}/{c.model}  →  {msg}")
            if status == "ok":
                # Recompute the path
                gguf = c.path / f"{c.model}-{quant}.gguf"
                if gguf.is_file():
                    produced.append(gguf)
    return produced


# ── Main ─────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--list-only", action="store_true",
                    help="Print candidate table and exit (no conversion, no audit)")
    ap.add_argument("--no-audit", action="store_true",
                    help="Skip the model_audit step at start")
    ap.add_argument("--non-interactive-audit", action="store_true",
                    help="Run audit but never prompt to delete (report only)")
    ap.add_argument("--quant", default=DEFAULT_QUANT,
                    help=f"GGUF quant target. Default {DEFAULT_QUANT}. "
                         f"Use comma-list for multiple, e.g. Q4_K_M,Q5_K_M.")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel conversions. Default 1 (each conversion is "
                         "already CPU-saturating; >2 thrashes).")
    ap.add_argument("--root", type=Path, action="append", default=[],
                    help=f"Scan root (repeatable). Default: "
                         f"{', '.join(str(r) for r in DEFAULT_SCAN_ROOTS)}")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print convert/quantize commands without running")
    ap.add_argument("--select", default=None,
                    help="Non-interactive selection: 0/all, a number, comma-list, or range like 2-5")
    ap.add_argument("--yes-prepare", action="store_true",
                    help="Run the post-conversion prepare-all/register step without prompting")
    ap.add_argument("--force", action="store_true",
                    help="Re-convert even if a sibling .gguf at the target quant exists")
    ap.add_argument("--orchestrator",
                    default=str(DEFAULT_ORCHESTRATOR),
                    help="Path to orchestrator (used by the post-conversion register prompt)")
    args = ap.parse_args()

    print("=" * 78)
    print(f"Safetensors → GGUF converter — {'DRY RUN' if args.dry_run else 'LIVE'}")
    print("=" * 78)

    roots = args.root or DEFAULT_SCAN_ROOTS

    # 1) Audit step
    if not args.no_audit and not args.list_only:
        print("\nStep 1/4 — running model audit…")
        report = model_audit.run_audit(roots=roots, workers=8,
                                       skip_duplicates=False)
        model_audit.interactive_prompt(
            report, dry_run=args.dry_run,
            non_interactive=args.non_interactive_audit,
        )

    # 2) Discover
    print(f"\n{'Step 2/4' if not args.list_only else 'Listing'} — discovering "
          f"safetensors candidates…")
    cands = discover_candidates(roots)
    if not cands:
        print("No safetensors-only candidates found.")
        return 0

    print_table(cands)

    if args.list_only:
        return 0

    # 3) Tool sanity check
    ok, msg = _check_tools()
    if not ok:
        print(f"\nERROR: {msg}")
        return 3

    # 4) Selection
    print("Step 3/4 — selecting models to convert.")
    if args.select is not None:
        sel = str(args.select).strip()
        print(f"  non-interactive selection: {sel}")
    else:
        try:
            sel = input(
                f"Which model(s) would you like to convert? "
                f"[0=all, 1-{len(cands)}, comma-list, ranges like 3-7] > "
            )
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return 1
    try:
        idxs = parse_selection(sel, len(cands))
    except ValueError as e:
        print(f"  bad selection: {e}")
        return 1
    if not idxs:
        print("  nothing selected.")
        return 0
    chosen = [cands[i - 1] for i in idxs]
    if not args.force:
        # Filter out ones with existing matching quant
        quants = [q.strip().upper() for q in args.quant.split(",")]
        keep = []
        for c in chosen:
            for q in quants:
                if q not in c.existing_gguf_quants:
                    keep.append(c)
                    break
        if len(keep) < len(chosen):
            print(f"  {len(chosen) - len(keep)} model(s) already have the "
                  f"requested quant — skipping (use --force to re-convert).")
        chosen = keep
    print(f"  selected: {len(chosen)} model(s)")
    for c in chosen:
        print(f"    - {c.publisher}/{c.model}")

    # 5) Convert
    print("\nStep 4/4 — converting…")
    quants = [q.strip().upper() for q in args.quant.split(",")]
    for q in quants:
        if q not in SUPPORTED_QUANTS:
            print(f"WARNING: quant {q!r} not in known list ({', '.join(SUPPORTED_QUANTS)}); "
                  "passing through to llama-quantize anyway.")
    produced: list[Path] = []
    for q in quants:
        produced.extend(run_conversions(chosen, q, args.workers, args.dry_run))

    if not produced:
        print("\nNo new GGUFs produced.")
        return 0

    print()
    print(f"Produced {len(produced)} GGUF(s):")
    for g in produced:
        print(f"  {g}")

    # 6) Optional register
    print()
    try:
        ans = input("Run prepare-all now to register the new GGUF(s)? [y/N] > "
                    ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return 0
    if ans in ("y", "yes"):
        if not Path(args.orchestrator).is_file():
            print(f"  orchestrator not found at {args.orchestrator}")
            return 0
        cmd = [sys.executable, args.orchestrator, "--no-audit",
               "--workers", str(max(1, args.workers))]
        print(f"  exec: {' '.join(cmd)}")
        os.execv(sys.executable, cmd)

    return 0


if __name__ == "__main__":
    sys.exit(main())
