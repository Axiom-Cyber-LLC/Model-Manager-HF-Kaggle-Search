# Model Manager HF/Kaggle Search

Local tools for finding, downloading, auditing, and preparing AI model artifacts.

The active script is `model_manager.py`. It downloads models to disk first, then runs post-download scanners from this tool folder. It does not use Docker for scanner execution and does not start a web GUI.

## Public Repository Hygiene

This public repository was rebuilt from a sanitized export. Do not commit local model caches, scan results, chat exports, binary model files, scanner checkouts, virtual environments, archives, spreadsheets, or machine-specific paths.

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
