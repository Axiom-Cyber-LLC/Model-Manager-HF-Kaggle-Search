# Model Manager HF/Kaggle Search

Local tools for finding, downloading, auditing, and preparing AI model artifacts.

The active script is `model_manager.py`. It downloads models to disk first, then runs post-download scanners from this tool folder. It does not use Docker for scanner execution and does not start a web GUI.

## Public Repository Hygiene

This public repository was rebuilt from a sanitized export. Do not commit local model caches, scan results, chat exports, binary model files, scanner checkouts, virtual environments, archives, spreadsheets, or machine-specific paths.

## Recent changes

### 2026-05-08 (risk-intel noise filter)

**`model_manager.py`**
- Pre-download risk-intel matching against the local AI Risk Repository / similar workbooks no longer floods INFO findings with generic words like `Paper`, `Risk Category`, etc. Three layered fixes in `load_xlsx_rows` and `risk_row_terms`:
  1. **Sheet allow-list**: workbook tabs whose names match noise patterns (`contents`, `taxonomy`, `explainer`, `statistics`, `compar`, `included`, `considered`, `change log`, `readme`, `guide`) are skipped. Only the actual data tab(s) are loaded.
  2. **Header-row scan**: research workbooks often prepend a banner / "Updated:" / "View video" row before the real columns. The loader now scans up to 20 rows for one that looks like real headers (≥3 short non-banner cells) and uses *that* as the column names. Previously, a banner row was treated as headers, scrambling all column lookups.
  3. **Stricter fallback term filter**: terms extracted from non-preferred columns must contain a digit, slash, dot, or colon — or be ≥60 chars. Plain English compound words (e.g. "Risk Sub-Category") no longer slip through. A small blocklist of taxonomy-noise terms is also applied.

### 2026-05-08 (scanner watchdog)

**`model_manager.py`**
- Post-download scanner step now has a **per-scanner watchdog**. Previously a scanner that hung without producing stdout (e.g., `skillcheck` waiting on stdin or any TUI prompt) blocked the for-loop reading stdout indefinitely — the existing `proc.wait(timeout=10)` only fires *after* the loop completes, so it never kicked in. Two fixes: (a) `stdin=subprocess.DEVNULL` is now passed so scanners cannot block on stdin, and (b) a `threading.Timer` watchdog kills any scanner that exceeds `MODEL_MANAGER_SCANNER_TIMEOUT_S` (default 600 = 10 minutes). Killed scanners surface as a WARN finding ("scanner killed after Ns timeout") and the audit continues with the next tool. Override the timeout per shell (`export MODEL_MANAGER_SCANNER_TIMEOUT_S=1800`) or one-off (`MODEL_MANAGER_SCANNER_TIMEOUT_S=1800 modelmgr ...`).

### 2026-05-08 (still later)

**`Prepare_models_for_Lmstudio.py`**
- New: **read-only second scan over `--lmstudio-dir`** (the LM Studio downloadsFolder). After the existing scan over `--input` (`models-flat`), the prep now also walks `<lmstudio-dir>/<publisher>/<repo>/<files>` and adds to READY any model that's hardlinked there. Closes the loop on the previous migration changes — newly-downloaded models that land in `lmstudio_dir` via the migrations now appear in the READY listing instead of being silently absent. Read-only by design (does not touch or restructure files), and dedupes against the first scan by display name.

### 2026-05-08 (later still)

**`Prepare_models_for_Lmstudio.py`**
- New: **`--migrate-underscore-flat`** (default on) — finds `<owner>__<repo>/` directories under `HF_CACHE_DIRS` (the install layout `model_manager.py` uses for non-cache snapshot downloads) and hardlinks files into `<lmstudio_dir>/<owner>/<repo>/`. Closes the gap where freshly-downloaded models landed in a layout LM Studio's downloadsFolder scanner did not recognize, leaving them invisible despite being on disk. Use `--no-migrate-underscore-flat` to skip. Hardlinks on same volume → instant, zero extra disk.

The prep script now runs three migrations by default, each handling one of the layouts a downloaded model can be in:

| Source layout | Migration step |
|---|---|
| `models--owner--repo/snapshots/<rev>/` (HF hub cache) | `--migrate-hf-cache` |
| `<owner>__<repo>/<files>` (model_manager flat install) | `--migrate-underscore-flat` |
| `<owner>/<repo>/<files>` (already publisher/repo) | `--mirror-models-flat` |

### 2026-05-08 (later)

**`model_manager.py`**
- Search-term expansion is now **per-base-term capped** (default 5 per base, env var `MODEL_MANAGER_HF_SEARCH_MAX_TERMS`). Previously a global cap could silently drop entire base terms when multiple were given (`Kimi, DeepSeek, Grok` → all variants of Kimi only). Now each base you type gets its own slice of variants.
- HF API metadata fetches during search are now **parallelized** (`MODEL_MANAGER_HF_SEARCH_WORKERS`, default 12). Replaces a sequential per-result `model_info()` loop. End-to-end search time drops ~10× on multi-term queries.
- `MODEL_MANAGER_HF_SEARCH_DELAY_MS` defaults to **0** (no throttle). Throttle is only useful for ANONYMOUS callers; with an HF account — free OR Pro — `HF_TOKEN` raises rate limits enough that the throttle is unnecessary. The 429 retry helper still absorbs any genuine rate-limit hit. Override per-shell with `export MODEL_MANAGER_HF_SEARCH_DELAY_MS=150` (or one-off `MODEL_MANAGER_HF_SEARCH_DELAY_MS=150 modelmgr ...`) on anonymous machines that hit 429s.
- `hfdownloader` (Go binary, multipart/chunked-parallel) is now the **default download path** for models. `MODEL_MANAGER_HFDOWNLOADER=0` opts out per-run. Datasets, artifact selections it cannot preserve cleanly, and any download error all fall back to `huggingface_hub.snapshot_download` automatically.
- New CLI: **size-range filter accepts one-sided bounds** — `<8 GB`, `>50 GB`, `<=2 TB`, `>=100 MB`, `under 8 GB`, `over 50 GB`, `at least 100 GB`. Two-sided ranges and `any` still work.
- New: pre-download **load-time RAM estimate** for GGUF selections. Fetches just the GGUF header (~64 KB Range request from HF), parses architecture metadata, and prints a context × KV-precision table with verdict against your detected machine RAM (macOS sysctl). Silently skipped on non-GGUF or fetch failure — never blocks downloads.
- `DEFAULT_EXCLUDED_AUTHORS` is now empty (was `[DavidAU, TheBloke, mradermacher, bartowski]`). The previous list contradicted `DEFAULT_KNOWN_GOOD_OWNERS` and was not risk-justified. Per-search exclusions are still entered at the prompt or via the existing flag.

**`Prepare_models_for_Lmstudio.py`**
- New: **publisher/repo/ tree mirror** (`--mirror-models-flat`, default on) — hardlinks any `<publisher>/<repo>/<files>` trees from `--input` (the manager's working dir) into `--lmstudio-dir` (the LM Studio `downloadsFolder` read from `~/.lmstudio/settings.json`). Fixes the case where models the prep script lists as READY were invisible to LM Studio because the two paths had drifted apart.
- New: **`--clean-symlinks`** — walks LM Studio model dirs for broken weight-file symlinks (target no longer exists). Prints `rm` commands grouped by parent dir; never deletes anything itself.
- New: **`--interactive-broken`** — after the BROKEN report prints, prompt per-entry for `[r]esume / [d]elete / [s]kip / [q]uit`. Resume calls `huggingface_hub.snapshot_download` for that repo. Delete requires explicit "Are you sure?" confirmation, then queues a copy-pasteable `rm -rf` block at the end (does not auto-delete).
- BROKEN report now includes the on-disk path under each entry on a `path:` continuation line.
- `HF_CACHE_DIRS` now also honors `HF_HUB_CACHE` and `HF_HOME` env vars at startup, so moving the HF cache to a new location no longer requires editing this list.
- READY listing now appends a per-entry **load-time RAM hint** for GGUF entries (`≈ N GB to load @ 8K (Q8 KV)`), parsed from the local GGUF header. Silently omitted when metadata can't be read.

**`Prepare_models_for_Ollama.py`**
- `--clean-orphans` now actually does what its docstring says — walks every Ollama manifest, parses its config + layer blob digests, and removes registrations whose backing blobs no longer exist on disk. Previously it only matched image/diffusion name patterns. Image-pattern removal is preserved as a first pass.

**`gguf_inspect.py`** (new)
- Self-contained GGUF metadata parser, RAM estimator, and machine-RAM detector. Used by `model_manager.py`'s pre-download estimate and `Prepare_models_for_Lmstudio.py`'s READY hint. Includes a `fetch_gguf_header_bytes()` helper that fetches the first ~64 KB of a GGUF via HTTP Range request (follows HF's LFS CDN redirect) so RAM estimates can run before any large-file download begins.

### 2026-05-08

**`Prepare_models_for_Lmstudio.py`**
- HF cache migration now scans every cache root in `HF_CACHE_DIRS` for `models--owner--repo/` directories and hardlinks the latest snapshot's files into LM Studio's `downloadsFolder` (read from `~/.lmstudio/settings.json`, with a sensible fallback). Migration is default-on now; use `--no-migrate-hf-cache` to skip. The destination is overridable via the new `--lmstudio-dir` flag. Hardlinks are zero-extra-disk on the same volume; source directories are left intact unless `--cleanup-hf-source` is passed.
- New `--resume-broken` flag: detects HF hub-cache repos that have only a `refs/main` stub (interrupted or never-completed downloads) and resumes them via `huggingface_hub.snapshot_download` — idempotent, honors `HF_TOKEN`, supports parallel repos via `--resume-broken-workers N`. Combine with `--dry-run` to see per-repo size estimates without downloading.
- New `--clean-orphan-stubs` flag: scans every HF cache root (plus `~/.cache/model_manager/hf_refs_stubs/`) for `models--*` dirs that contain only `refs/` and no `blobs/` or `snapshots/` — bookkeeping residue from `huggingface_hub.snapshot_download(local_dir=...)`. Prints copy-pasteable `rm -rf` commands grouped by parent directory; never deletes anything itself.

**`model_manager.py`**
- `snapshot_download` is now invoked with `cache_dir=~/.cache/model_manager/hf_refs_stubs/`, steering its always-written `refs/<rev>` stubs into a throwaway location so they no longer pollute the user's main `HF_HUB_CACHE`.
- HF API throttling and 429 handling: `MODEL_MANAGER_HF_SEARCH_DELAY_MS` (default 150 ms) inserts a small sleep between metadata fetches during search. A new retry helper catches HTTP 429 responses, honors the `Retry-After` header, and retries up to 3 times. Wrapped around `list_models`, `model_info`, `list_datasets`, and `dataset_info`.
- Search-term expansion is now capped at `MODEL_MANAGER_HF_SEARCH_MAX_TERMS` (default 5). The previous combination of name-variant expansion and artifact-suffix expansion could produce 25+ search terms per query, multiplying API load.
- `DEFAULT_EXCLUDED_AUTHORS` is now an empty list. The previous list (`DavidAU`, `TheBloke`, `mradermacher`, `bartowski`) contradicted `DEFAULT_KNOWN_GOOD_OWNERS` (which already included `bartowski`) and was not risk-justified. Per-search exclusions can still be entered at the prompt or via the `--exclude-publishers` flag.
- Weekly cached check for AI Risk Repository updates: at search start, the manager extracts the Google Sheets doc ID from the "Explore database" button on `airisk.mit.edu` and compares it to the cached value. When the ID changes (signal that MIT versioned up the database), a one-line nag prints the previous and current IDs and the URL to re-download. Cache state lives in `~/.cache/model_manager/airisk_mit_check.json`. All network and parse errors are swallowed silently — the check never blocks startup.

## Core Setup

```bash
cd ~/ModelTools
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python model_manager.py --help
```

Launch the native GUI:

```bash
modelmgr --gui
```

The GUI includes tabs for the full `modelmgr` menu, guided search/download options, direct Hugging Face downloads, local audit/prep/conversion, and leaderboards. It exposes search source, artifact type, model size, author/tag/family exclusions, duplicate handling, result batching, direct downloader filters/excludes/endpoint/concurrency, audit dry-run mode, prepare-app filters, conversion quant/workers/selection, and leaderboard cache controls. Prompt-heavy flows open in Terminal so result selection, artifact picking, scanner prompts, delete confirmations, prep, and conversion stay visible. The direct Hugging Face downloader panel stays inside the native GUI.

See [System Overview](docs/system-overview.md) for a diagram of the search, download, scan, prepare, and conversion flow.

Optional environment variables:

```bash
export HF_TOKEN="..."
export MODEL_MANAGER_DOWNLOAD_DIR="$HOME/models"
export MODEL_MANAGER_TOOLS_DIR="$HOME/ModelTools"
```

Kaggle downloads require `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` and `KAGGLE_KEY`.

## Scanner Repositories

`model_manager.py` looks for these scanner folders directly under `~/ModelTools`:

```bash
cd ~/ModelTools
git clone https://github.com/promptfoo/modelaudit.git modelaudit
git clone https://github.com/protectai/modelscan.git modelscan
git clone https://github.com/rizzit17/ModelGuard.git ModelGuard
git clone https://github.com/highflame-ai/palisade-scan.git palisade-scan
git clone https://github.com/hc-sc-ocdo-bdpd/model-scan.git model-scan
git clone https://github.com/mondoohq/skillcheck.git skillcheck
git clone https://github.com/cisco-ai-defense/skill-scanner.git skill-scanner
```

Expected local scanner behavior:

- `modelaudit`: runs from `modelaudit/.venv` when present, using `python -m modelaudit scan`.
- `modelscan`: runs from `modelscan/.venv` when present, using `python -m modelscan.cli scan`.
- `ModelGuard`: runs through `ModelGuard/modelguard_cli.py`, a terminal-only wrapper in this workspace. It does not start the upstream Streamlit UI.
- `palisade-scan`: uses the `palisade` CLI from `PATH` with `palisade-scan` as the working folder. If the CLI is not installed, the scanner is skipped.
- `model-scan`: kept as a reference/demo repo. The manager intentionally maps scanning to the safer `modelscan` CLI instead of running demo files that may unpickle models.
- `skillcheck`: runs the local `skillcheck/skillcheck` binary when present.
- `skill-scanner`: runs `skill-scanner/scan.sh` when present.

`skillcheck` and `skill-scanner` are supplemental agent-skill scanners, not full model-file scanners. Treat their results as additional signal, not as proof that a model is safe.

## Scanner Setup Notes

Use each scanner repo's own setup instructions. A typical local setup is:

```bash
cd ~/ModelTools/modelaudit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

```bash
cd ~/ModelTools/modelscan
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

```bash
cd ~/ModelTools/skill-scanner
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

```bash
cd ~/ModelTools/skillcheck
make build
```

For Palisade, install the local `palisade` CLI according to upstream instructions. The manager does not run the Palisade GitHub Action container.

## Attribution And Licenses

Keep each scanner repo's `LICENSE`, `NOTICE`, and third-party notice files with the clone. MIT and Apache-2.0 projects generally require retaining copyright and license notices in copies or substantial portions of the software.

Scanner attribution recorded for this workspace:

- Assisted by ModelAudit from Promptfoo, Inc. (`promptfoo/modelaudit`), MIT License. Keep `modelaudit/LICENSE` and `modelaudit/THIRD_PARTY_NOTICES.md`.
- Assisted by ModelScan from Protect AI (`protectai/modelscan`), Apache License 2.0. Keep `modelscan/LICENSE`.
- Assisted by ModelGuard (`rizzit17/ModelGuard`). No root license file was present in the local checkout; verify upstream terms before redistribution or vendoring.
- Assisted by Palisade scan tooling from Highflame (`highflame-ai/palisade-scan`). No root license file was present in the local checkout; verify upstream terms before redistribution or vendoring. The Azure DevOps extension subfolder declares ISC.
- Assisted by model-scan (`hc-sc-ocdo-bdpd/model-scan`), MIT License. Keep `model-scan/LICENSE`.
- Assisted by skillcheck from Mondoo (`mondoohq/skillcheck`), Apache License 2.0. Keep `skillcheck/LICENSE`.
- Assisted by Cisco AI Skill Scanner from Cisco Systems, Inc. and its affiliates (`cisco-ai-defense/skill-scanner`), Apache License 2.0. Keep `skill-scanner/LICENSE`.

This README is not legal advice. Review upstream licenses before distributing scanner code, scanner binaries, model files, or commercial bundles.

## Safety Notes

- Downloads are written to disk before post-download scanning.
- Destructive cleanup is not automatic; delete prompts are only offered after high-risk findings.
- Do not load, unpickle, or execute untrusted model files manually while investigating scanner output.
- A clean scanner result is not a guarantee of safety. Treat scanner results as part of a broader review process.

┌─────────────────────────────────────────────────────────────────────────────┐
│                               modelmgr                                      │
│                                                                             │
│  ~/bin/modelmgr                                                             │
│      └── python3 /Users/bowmanbt/model_tools/model_manager.py               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Main Menu                                      │
│                                                                             │
│  1. Search / download                                                       │
│  2. Local audit / repair                                                    │
│  3. Leaderboards                                                            │
│  4. Conversion workflow                                                     │
│  5. Prepare models for apps                                                 │
│  6. Quit                                                                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         1. Search / Download                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Artifact Type Picker                                │
│                                                                             │
│  Default selected:                                                          │
│    [x] GGUF                                                                 │
│    [x] Core ML: .mlmodel / .mlpackage                                       │
│                                                                             │
│  Optional only when explicitly selected:                                    │
│    [ ] MLX                                                                  │
│    [ ] ONNX                                                                 │
│    [ ] Safetensors                                                          │
│    [ ] Keras / TensorFlow                                                   │
│    [ ] Raw PyTorch / pickle-risk formats                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Search Query Layer                                  │
│                                                                             │
│  Examples:                                                                  │
│    code                                                                     │
│    code NOT bartowski                                                       │
│    code NOT --owner:bartowski                                               │
│    code -owner:bartowski -owner:unsloth                                     │
│                                                                             │
│  Filters split into:                                                        │
│    - owner / publisher excludes                                             │
│    - model family excludes                                                  │
│    - repo/name text excludes                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Provider Search                                     │
│                                                                             │
│  Hugging Face                                                               │
│    - model search                                                           │
│    - filename search: .gguf, .mlmodel, .mlpackage                           │
│    - repo metadata fetch                                                    │
│    - sibling file discovery                                                 │
│                                                                             │
│  Kaggle                                                                     │
│    - best-effort model/dataset search                                       │
│    - marked as artifact-unconfirmed when file metadata is unavailable       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Result Filtering                                    │
│                                                                             │
│  1. Remove excluded owners/publishers                                       │
│  2. Remove excluded repo/name text                                          │
│  3. Remove unsupported artifact types                                       │
│  4. Group into model families                                               │
│  5. Let user exclude entire model families                                  │
│                                                                             │
│  Example family excludes:                                                   │
│    qwen coder                                                               │
│    qwen                                                                     │
│    deepseek coder                                                           │
│    mradermacher                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         Result / Artifact Picker                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Repo row                                                                   │
│                                                                             │
│    1. unsloth/Qwen3-Coder-Next-GGUF                                         │
│                                                                             │
│  Artifact rows                                                              │
│                                                                             │
│    1A. Qwen3-Coder-Next-Q4_K_M.gguf                                         │
│    1B. Qwen3-Coder-Next-Q5_K_M.gguf                                         │
│    1C. Qwen3-Coder-Next-Q8_0.gguf                                          │
│                                                                             │
│  Preferred behavior:                                                        │
│    - selecting 1 opens artifact picker                                      │
│    - selecting 1A adds artifact to cart                                     │
│    - whole repo requires explicit command: whole 1                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Selection Cart                                 │
│                                                                             │
│  Commands:                                                                  │
│    1A,2C          add direct artifacts                                      │
│    1              open artifacts for repo 1                                 │
│    whole 1        force whole-repo selection                                │
│    xpub qwen      exclude publisher/owner                                   │
│    xfam qwen      exclude family                                            │
│    cart           show selected items                                       │
│    remove 2       remove item from cart                                     │
│    clear          empty cart                                                │
│    n / p          next / previous page                                      │
│    done           start download                                            │
│    q              quit selection                                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           Download Root                                     │
│                                                                             │
│  /Volumes/SamsungSSDE/models                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Staging Area                                      │
│                                                                             │
│  /Volumes/SamsungSSDE/models/.incoming                                      │
│                                                                             │
│  Purpose:                                                                   │
│    - stream model files directly to disk                                    │
│    - avoid holding multi-GB files in RAM                                    │
│    - isolate partial/incomplete/risky downloads                             │
│    - scan before final install                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Final Layout                                      │
│                                                                             │
│  Hugging Face models:                                                       │
│    /Volumes/SamsungSSDE/models/huggingface/model/<owner>__<repo>/           │
│                                                                             │
│  Hugging Face datasets:                                                     │
│    /Volumes/SamsungSSDE/models/huggingface/dataset/<owner>__<dataset>/      │
│                                                                             │
│  Kaggle models:                                                             │
│    /Volumes/SamsungSSDE/models/kaggle/model/<owner>__<model>/               │
│                                                                             │
│  Kaggle datasets:                                                           │
│    /Volumes/SamsungSSDE/models/kaggle/dataset/<owner>__<dataset>/           │
└─────────────────────────────────────────────────────────────────────────────┘

┌───────────────┐
│ User selects  │
│ artifacts     │
└───────┬───────┘
        ▼
┌────────────────────────────────────────────────────┐
│ Stream download to .incoming                       │
│                                                    │
│ Never:                                             │
│   - load full model into memory                    │
│   - capture binary output into shell variables     │
│   - capture huge scanner output into RAM           │
└───────┬────────────────────────────────────────────┘
        ▼
┌────────────────────────────────────────────────────┐
│ Built-in staged audit                              │
│                                                    │
│ Checks:                                            │
│   - GGUF magic/header                              │
│   - missing split shards                           │
│   - LFS pointer files                              │
│   - suspicious tiny files                          │
│   - risky scripts/code                             │
│   - pickle/PyTorch risk files                      │
└───────┬────────────────────────────────────────────┘
        ▼
┌────────────────────────────────────────────────────┐
│ External scanners, staged path only                │
│                                                    │
│ Do not scan all model roots after every download.  │
└───────┬────────────────────────────────────────────┘
        ▼
┌───────────────────────────────┬───────────────────────────────┐
│ PASS                          │ WARN / HIGH / BLOCKER         │
│                               │                               │
│ move from .incoming           │ ask:                          │
│ to final model path           │   - delete staged download    │
│                               │   - quarantine staged download│
│                               │   - keep staged download      │
│                               │   - continue anyway           │
└───────────────────────────────┴───────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         Safety Stack                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. Pre-download checks                                                     │
│                                                                             │
│  - owner / publisher reputation                                             │
│  - known-good author = INFO only                                            │
│  - suspicious repo names                                                    │
│  - selected artifact type risk                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. Built-in model integrity checks                                         │
│                                                                             │
│  - GGUF header validation                                                   │
│  - GGUF shard completeness                                                  │
│  - LFS pointer detection                                                    │
│  - tiny file detection                                                      │
│  - missing safetensors shards                                               │
│  - risky extension detection                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. model_audit.py                                                          │
│                                                                             │
│  Local model hygiene:                                                       │
│    - duplicate models                                                       │
│    - dangling symlinks                                                      │
│    - corrupt GGUFs                                                          │
│    - orphan shards                                                          │
│    - stale app links                                                        │
│                                                                             │
│  Important behavior:                                                        │
│    dangling symlink → rebuild symlink first, not delete model               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. External scanners                                                       │
│                                                                             │
│  ModelAudit                                                                 │
│    - unsafe deserialization                                                 │
│    - suspicious code                                                        │
│    - secrets                                                                │
│    - network indicators                                                     │
│    - CVE patterns                                                           │
│                                                                             │
│  modelscan                                                                  │
│    - independent model-file scan                                            │
│                                                                             │
│  skill-scanner                                                              │
│    - README / skill-like content findings                                   │
│    - must be interpreted carefully                                          │
│                                                                             │
│  Microsoft Defender                                                         │
│    - AV/malware/exploit correlation                                         │
│    - only meaningful if alert path matches model path                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┬──────────────────────────────────────────────────────────────┐
│ Severity     │ Expected behavior                                            │
├──────────────┼──────────────────────────────────────────────────────────────┤
│ INFO         │ display only                                                 │
│ LOW          │ display, no blocking                                         │
│ WARN         │ ask/review, no delete prompt by default                      │
│ MEDIUM       │ review recommended                                           │
│ HIGH         │ review strongly; do not auto-delete unless clearly relevant  │
│ DANGER       │ deletion/quarantine prompt allowed                           │
│ BLOCKER      │ block install unless user overrides                          │
│ CRITICAL     │ block install; delete/quarantine recommended                 │
└──────────────┴──────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           Local Audit                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Scan known model roots                                                     │
│                                                                             │
│  /Volumes/SamsungSSDE/models                                                │
│  /Volumes/SamsungSSDE/models-flat                                           │
│  /Volumes/SamsungSSDE/models/huggingface/model                              │
│  /Volumes/SamsungSSDE/models-flat/local                                     │
│  /Users/bowmanbt/Library/Application Support/nomic.ai/GPT4All               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Findings                                                                   │
│                                                                             │
│  Dangling symlink                                                           │
│    ├─ search model roots for same filename                                  │
│    ├─ if found: rebuild symlink                                             │
│    └─ if not found: delete broken link only                                 │
│                                                                             │
│  Broken model                                                               │
│    ├─ missing GGUF shards                                                   │
│    ├─ invalid GGUF header                                                   │
│    ├─ LFS pointer instead of model                                          │
│    └─ suspicious tiny artifact                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         Conversion Workflow                                 │
│                                                                             │
│  /Users/bowmanbt/model_tools/model_conversion.py                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Safetensors discovery                                                      │
│                                                                             │
│  Finds candidate HF-style folders containing safetensors.                   │
│                                                                             │
│  Important: safetensors are not default search targets.                     │
│  Conversion is explicit only.                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  User chooses candidates                                                    │
│                                                                             │
│    0 / all                                                                  │
│    3                                                                        │
│    1,3                                                                      │
│    2-4                                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  convert_hf_to_gguf.py                                                      │
│  llama-quantize                                                             │
│  gguf Python module                                                         │
│                                                                             │
│  Verified interpreter:                                                      │
│    /opt/homebrew/opt/python@3.14/bin/python3.14                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                       Prepare Models for Apps                               │
│                                                                             │
│  /Users/bowmanbt/model_tools/Prepare_models_for_All.py                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Per-app scripts                                                            │
│                                                                             │
│  Prepare_models_for_Lmstudio.py                                             │
│  Prepare_models_for_Ollama.py                                               │
│  Prepare_models_for_AnythingLLM.py                                          │
│  Prepare_models_for_GPT4All.py                                              │
│  Prepare_models_for_Jan.py                                                  │
│  Prepare_models_for_AINavigator.py                                          │
│  Prepare_models_for_AIStudio.py                                             │
│  Prepare_models_for_LocallyAI.py                                            │
│  Prepare_models_for_LocalAI.py                                              │
│  Prepare_models_for_Apollo.py                                               │
│  Prepare_models_for_OffGrid.py                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  App integration behavior                                                   │
│                                                                             │
│  Preferred:                                                                 │
│    - symlink GGUFs                                                          │
│    - write app-specific config references                                   │
│    - rebuild broken links                                                   │
│                                                                             │
│  Avoid:                                                                     │
│    - copying huge model files into every app folder                         │
│    - deleting app/model data because of one stale symlink                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐
│ modelmgr     │
└──────┬───────┘
       ▼
┌─────────────────────┐
│ Search/download     │
└──────┬──────────────┘
       ▼
┌─────────────────────┐
│ Pick artifact types │
│ default: GGUF/CoreML│
└──────┬──────────────┘
       ▼
┌─────────────────────┐
│ Search HF/Kaggle    │
│ filename-aware      │
└──────┬──────────────┘
       ▼
┌─────────────────────┐
│ Apply NOT filters   │
│ owner/family/name   │
└──────┬──────────────┘
       ▼
┌─────────────────────┐
│ Artifact picker     │
│ 1A / 1B / 2C        │
└──────┬──────────────┘
       ▼
┌─────────────────────┐
│ Selection cart      │
│ download on done    │
└──────┬──────────────┘
       ▼
┌─────────────────────┐
│ Stream to .incoming │
│ no big memory use   │
└──────┬──────────────┘
       ▼
┌─────────────────────┐
│ Scan staged path    │
│ not all roots       │
└──────┬──────────────┘
       ▼
┌──────────────┬────────────────┐
│ Clean        │ Risk/warning   │
│              │                │
│ move final   │ ask remediation│
└──────┬───────┴────────────────┘
       ▼
┌─────────────────────┐
│ Prepare app links   │
│ optional            │
└──────┬──────────────┘
       ▼
┌─────────────────────┐
│ Local audit/repair  │
│ periodic hygiene    │
└─────────────────────┘

Design summary

The updated framework should treat Model Manager as a defensive model intake and operations console.

It should not be just a downloader.

It should:

1. Search only useful model types by default.
2. Prefer direct artifacts over whole repositories.
3. Let you exclude publishers, families, and repo-name terms clearly.
4. Keep a selection cart until you type done.
5. Stream downloads to disk, not memory.
6. Stage downloads before final install.
7. Scan only the staged path after download.
8. Move clean files into /Volumes/SamsungSSDE/models.
9. Repair app symlinks instead of deleting model data.
10. Keep conversion and app prep modular.
11. Separate model safety alerts from unrelated Defender/FileProvider/security-research noise.
