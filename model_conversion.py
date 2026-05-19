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
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# model_audit lives next to this script's intended home; until everything is in
# one tree, fall back to the SamsungSSDE copy.
_AUDIT_PATHS = [
    Path("<Your Model Directory>"),
    Path(__file__).resolve().parent,
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
    Path("<Your Model Directory>"),
    Path("<Your Model Directory>"),
    Path("<REDACTED_PATH>"),
    HOME / ".cache" / "huggingface",
    Path("<Your Model Directory>"),
]
DEFAULT_QUANT = "Q8_0"  # was Q4_K_M; Q8_0 is the safe default for LLM eval quality.
DEFAULT_OUT_DTYPE = "f16"  # convert_hf_to_gguf.py outtype before quantizing
SUPPORTED_QUANTS = (
    "F32", "F16", "BF16",
    "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S",
    "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q2_K",
    "IQ4_XS", "IQ3_M", "IQ3_S", "IQ2_M", "IQ2_S",
)
# Common typos and aliases mapped to the correct llama-quantize name.
QUANT_ALIASES = {
    "Q8_K_M": "Q8_0",     # No K variant for Q8; user likely meant Q8_0.
    "Q8_K": "Q8_0",
    "Q4KM": "Q4_K_M",
    "Q5KM": "Q5_K_M",
    "Q6KM": "Q6_K",
    "FP16": "F16",
    "FP32": "F32",
}

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
    # NEW (2026-05-11): compatibility classification + inode dedup
    # compat: "ok" | "mlx-quant" | "classifier" | "sentence-transformer" | "unknown"
    compat: str = "ok"
    compat_reason: str = ""
    # (st_dev, st_ino) of the largest *.safetensors file in `path`. Used to
    # dedupe candidates that point at the same physical bytes via different
    # display directories (e.g., LM Studio hub layout + flat layout sharing
    # hardlinks). None when the candidate has no safetensors files we can stat.
    primary_inode: tuple[int, int] | None = None


# ── Compatibility detection ──────────────────────────────────────────


# model_type values that produce embeddings/classifiers, not generative LLMs.
_ENCODER_TYPES = {
    "bert", "roberta", "distilbert", "albert", "electra", "xlm-roberta",
    "mpnet", "deberta", "deberta-v2", "deberta-v3", "convbert",
    "longformer", "reformer", "big_bird",
}

# `architectures` substrings that mean "this has a non-LLM task head" —
# convert_hf_to_gguf either can't handle them or produces a useless GGUF.
_CLASSIFIER_ARCH_TOKENS = (
    "forsequenceclassification",
    "fortokenclassification",
    "formultiplechoice",
    "forquestionanswering",
    "formaskedlm",
)


def _detect_compat(d: Path) -> tuple[str, str]:
    """Classify whether `d` will produce a usable LLM GGUF.

    Returns (compat, reason). compat is one of:
      ok                   — generative LLM that convert_hf_to_gguf supports
      mlx-quant            — MLX-quantized; convert_hf_to_gguf needs FP16 source
      classifier           — has a task head (BERT-for-classification, etc.)
      sentence-transformer — embedding model; needs special outtype/arch support
      unknown              — couldn't read config.json or no model_type clue
    """
    cfg_path = d / "config.json"
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unknown", "could not read config.json"

    # MLX-quantized models leave a quantization block with bits/group_size.
    quant_cfg = cfg.get("quantization")
    if isinstance(quant_cfg, dict) and ("bits" in quant_cfg or "group_size" in quant_cfg):
        return "mlx-quant", f"MLX quantized ({quant_cfg.get('bits', '?')}-bit)"

    # Sentence-transformers checkpoints carry their own config siblings.
    if (d / "config_sentence_transformers.json").is_file() or \
       (d / "sentence_bert_config.json").is_file():
        return "sentence-transformer", "sentence-transformers config present"

    archs_raw = cfg.get("architectures") or []
    archs_lower = [str(a).lower() for a in archs_raw]
    for arch in archs_lower:
        for tok in _CLASSIFIER_ARCH_TOKENS:
            if tok in arch:
                return "classifier", f"task head: {arch}"

    model_type = str(cfg.get("model_type") or "").lower()
    if model_type in _ENCODER_TYPES:
        # No classifier head detected above, but encoder architectures
        # without a generative head are usually sentence-transformers.
        return "sentence-transformer", f"{model_type} encoder model"

    return "ok", ""


def _primary_safetensors_inode(safetensors: list[Path]) -> tuple[int, int] | None:
    """Return (st_dev, st_ino) of the largest safetensors file, for dedup."""
    if not safetensors:
        return None
    candidates: list[tuple[int, Path]] = []
    for p in safetensors:
        try:
            candidates.append((p.stat().st_size, p))
        except OSError:
            continue
    if not candidates:
        return None
    candidates.sort(reverse=True)
    try:
        s = candidates[0][1].stat()
        return (s.st_dev, s.st_ino)
    except OSError:
        return None


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
    seen_paths: set[Path] = set()
    # Dedup by canonical inode of the largest safetensors file. Hardlinked
    # mirrors (e.g., LM Studio hub layout + flat install layout pointing at
    # the same bytes) previously appeared as separate entries because the
    # directory paths differed even though the model bytes were identical.
    seen_inodes: set[tuple[int, int]] = set()
    out: list[Candidate] = []
    duplicates_skipped = 0
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
            if real in seen_paths:
                continue
            seen_paths.add(real)
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
            primary_inode = _primary_safetensors_inode(safetensors)
            if primary_inode is not None and primary_inode in seen_inodes:
                duplicates_skipped += 1
                continue
            if primary_inode is not None:
                seen_inodes.add(primary_inode)
            compat, compat_reason = _detect_compat(d)
            publisher, model = _publisher_model(d)
            out.append(Candidate(
                path=d,
                publisher=publisher,
                model=model,
                total_safetensors_bytes=total_st,
                n_safetensors=len(safetensors),
                has_existing_gguf=bool(existing_ggufs),
                existing_gguf_quants=sorted({_detect_quant(g.name) for g in existing_ggufs}),
                compat=compat,
                compat_reason=compat_reason,
                primary_inode=primary_inode,
            ))
    if duplicates_skipped:
        print(f"  Deduped {duplicates_skipped} candidate(s) that hardlink to the same bytes.")
    # Sort: compatible first, then existing-gguf last, then alphabetical.
    out.sort(key=lambda c: (
        c.compat != "ok",          # ok first
        c.has_existing_gguf,        # not-yet-converted first
        c.publisher.lower(),
        c.model.lower(),
    ))
    return out


_COMPAT_TAG = {
    "ok": "ok",
    "mlx-quant": "MLX",
    "classifier": "CLS",
    "sentence-transformer": "ST",
    "unknown": "?",
}


def print_table(cands: list[Candidate]) -> None:
    print()
    print(f"{'#':>3}  {'publisher/model':<55}  {'size':>10}  shards  {'compat':<5}  existing")
    print("-" * 110)
    for i, c in enumerate(cands, 1):
        existing = ",".join(c.existing_gguf_quants) if c.has_existing_gguf else "—"
        name = f"{c.publisher}/{c.model}"
        if len(name) > 55:
            name = name[:52] + "..."
        tag = _COMPAT_TAG.get(c.compat, "?")
        print(f"{i:>3}  {name:<55}  {_human(c.total_safetensors_bytes):>10}  "
              f"{c.n_safetensors:>5}  {tag:<5}  {existing}")
    # Footnote any incompatibles so the user can decide whether to skip them.
    incompat = [c for c in cands if c.compat != "ok"]
    if incompat:
        print()
        print("Compat key:  ok = generative LLM (will convert)")
        print("             MLX = MLX-quantized weights; convert_hf_to_gguf needs FP16 source — will fail")
        print("             CLS = task head (classifier/QA); produces unusable GGUF — will fail")
        print("             ST  = sentence-transformer / encoder-only; embedding GGUF only if supported")
        print("             ?   = couldn't read config.json")
        print()
        print(f"  {len(incompat)} of {len(cands)} candidates flagged as likely-incompatible:")
        for c in incompat:
            print(f"    [{_COMPAT_TAG[c.compat]:<3}] {c.publisher}/{c.model} — {c.compat_reason}")
    print()
    print("  0  Convert them all")
    print()


def filter_incompatible_interactive(cands: list[Candidate]) -> list[Candidate]:
    """If any candidates are flagged incompatible, ask whether to drop them
    from the selection menu. Returns the (possibly trimmed) candidate list.
    Non-interactive callers (no TTY) skip the prompt and keep all candidates."""
    incompat = [c for c in cands if c.compat != "ok"]
    if not incompat:
        return cands
    if not sys.stdin.isatty():
        return cands
    try:
        ans = input(
            f"Hide the {len(incompat)} likely-incompatible candidate(s) from "
            f"the selection menu? [Y/n] > "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return cands
    if ans in ("", "y", "yes"):
        kept = [c for c in cands if c.compat == "ok"]
        print(f"  Hidden {len(cands) - len(kept)}; {len(kept)} candidate(s) remain.")
        return kept
    return cands


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
    ap.add_argument("--allow-unknown-quant", action="store_true",
                    help="Skip the interactive confirm when --quant is not in the known list "
                         "(useful for new llama-quantize options).")
    ap.add_argument("--include-incompatible", action="store_true",
                    help="Don't auto-hide candidates flagged as MLX-quantized, classifier, "
                         "sentence-transformer, or unknown architecture. Default behavior is "
                         "to ask interactively; this flag forces 'include all'.")
    ap.add_argument("--orchestrator",
                    default="<Your Model Directory>/Prepare_models_for_All.py",
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

    # Offer to hide candidates flagged as likely-incompatible (MLX-quantized,
    # classifier heads, sentence-transformers, unknown arch). This is the
    # most common cause of "I picked 18, got 18 FAILs" complaints. Skip the
    # prompt with --include-incompatible.
    if not args.include_incompatible:
        cands = filter_incompatible_interactive(cands)
        if not cands:
            print("All candidates were filtered out; nothing to convert.")
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
    quants_raw = [q.strip().upper() for q in args.quant.split(",")]
    quants: list[str] = []
    for q in quants_raw:
        # Auto-resolve common typos / aliases so a Q8_K_M doesn't reach
        # llama-quantize and blow up after a long convert step.
        if q in QUANT_ALIASES:
            mapped = QUANT_ALIASES[q]
            print(f"  quant alias: {q!r} → {mapped!r}")
            q = mapped
        if q not in SUPPORTED_QUANTS:
            # Offer a "did you mean" guess from the known list.
            guess = difflib.get_close_matches(q, SUPPORTED_QUANTS, n=1, cutoff=0.4)
            hint = f"  Did you mean: {guess[0]!r}?" if guess else ""
            print(f"WARNING: quant {q!r} is not in the known list.")
            if hint:
                print(hint)
            print(f"  Known: {', '.join(SUPPORTED_QUANTS)}")
            if sys.stdin.isatty() and not args.allow_unknown_quant:
                try:
                    ans = input(
                        f"  Pass {q!r} through to llama-quantize anyway? [y/N] > "
                    ).strip().lower()
                except (EOFError, KeyboardInterrupt):
                    ans = ""
                if ans not in ("y", "yes"):
                    print(f"  Skipping unknown quant {q!r}.")
                    continue
            else:
                print(f"  Passing {q!r} through (--allow-unknown-quant or non-TTY).")
        quants.append(q)
    if not quants:
        print("  No valid quants to run; aborting.")
        return 0
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
