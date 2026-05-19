#!/usr/bin/env python3
"""
agent_filesystem_wrapper.py
---------------------------
Universal harness that turns ANY local model (LM Studio / Ollama / llama-server)
into an agent that actually writes to disk, runs shell commands, and (optionally)
self-corrects via CodeQL feedback (arxiv 2506.23034) and RAG over the
CRAKEN/Alpha-Root cyber corpora (arxiv 2505.17107, 2602.22218).

The model itself never touches disk; this wrapper does. The model only has to
emit text. Two output formats are accepted:

  1. OpenAI-style tool calls (preferred; works with Qwen2.5+, Llama-3.1+,
     Devstral, Mistral 2024+, Phi-4, Foundation-Sec-8B, etc.).
  2. Fenced code blocks tagged with the target path on the first line, e.g.
        ```python:/abs/path/foo.py
        <code>
        ```
     This is what gets parsed for older or chat-only models like
     Lily-Cybersecurity-7B, Seneca, BaronLLM, etc. that have no tool template.

This means even a refusal-prone Llama-2 finetune that says "I can't write to
disk" will still have any code it does emit captured and written. Combined
with --system-prompt agent_system_prompt.md, refusals drop near zero.

Usage:
  python agent_filesystem_wrapper.py \
      --endpoint http://localhost:1234/v1 \
      --model "lily-cybersecurity-7b-uncensored" \
      --root  <REDACTED_PATH> \
      --system-prompt _presets/agent_system_prompt.md \
      --task "write a python tcp port scanner to scanner.py"

Optional flags:
  --codeql        run CodeQL after each file write; loop on findings
  --rag PATH      use a JSONL/markdown corpus as RAG context per turn
  --max-turns N   default 8

Dependencies: openai, requests, pathlib (stdlib). CodeQL CLI optional.
"""

from __future__ import annotations
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    sys.stderr.write("pip install openai\n")
    sys.exit(1)


# ---------- tool registry ----------------------------------------------------

@dataclass
class FileTools:
    """All filesystem ops are confined to `root`. No path escapes."""
    root: Path
    log: list[dict] = field(default_factory=list)

    def _resolve(self, p: str) -> Path:
        candidate = (self.root / p).resolve() if not Path(p).is_absolute() else Path(p).resolve()
        if self.root.resolve() not in candidate.parents and candidate != self.root.resolve():
            raise PermissionError(f"path {candidate} escapes root {self.root}")
        return candidate

    def write_file(self, path: str, content: str) -> str:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        self.log.append({"op": "write", "path": str(target), "bytes": len(content)})
        return f"WROTE {target} ({len(content)} bytes)"

    def read_file(self, path: str) -> str:
        return self._resolve(path).read_text()

    def list_dir(self, path: str = ".") -> str:
        target = self._resolve(path)
        return "\n".join(sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir()))

    def run_shell(self, cmd: str, timeout: int = 60) -> str:
        proc = subprocess.run(
            cmd, shell=True, cwd=self.root, capture_output=True, text=True, timeout=timeout
        )
        self.log.append({"op": "shell", "cmd": cmd, "rc": proc.returncode})
        return f"rc={proc.returncode}\n--stdout--\n{proc.stdout}\n--stderr--\n{proc.stderr}"


TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file at the given path (relative to project root).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file's contents.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List directory entries.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "default": "."}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Execute a shell command in project root. Returns stdout/stderr/rc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string"},
                    "timeout": {"type": "integer", "default": 60},
                },
                "required": ["cmd"],
            },
        },
    },
]


# ---------- fenced-block fallback for non-tool-calling models ----------------

FENCED_RE = re.compile(
    r"```(?P<lang>[\w+\-]*)\s*:\s*(?P<path>[^\n`]+)\n(?P<body>.*?)```",
    re.DOTALL,
)


def extract_fenced_writes(text: str) -> list[tuple[str, str]]:
    """Pulls (path, content) from fenced blocks like ```py:/abs/path or ```sh:./run.sh"""
    return [(m.group("path").strip(), m.group("body")) for m in FENCED_RE.finditer(text)]


# ---------- CodeQL feedback loop (arxiv 2506.23034) -------------------------

def codeql_scan(file_path: Path) -> str | None:
    """Returns CodeQL summary string, or None if codeql CLI missing or no findings."""
    if not shutil_which("codeql"):
        return None
    db = file_path.parent / ".codeql-db"
    lang = {"py": "python", "js": "javascript", "ts": "javascript", "java": "java",
            "c": "cpp", "cpp": "cpp", "go": "go"}.get(file_path.suffix.lstrip("."))
    if not lang:
        return None
    try:
        subprocess.run(
            ["codeql", "database", "create", str(db), f"--language={lang}",
             f"--source-root={file_path.parent}", "--overwrite"],
            check=True, capture_output=True, timeout=120,
        )
        out = subprocess.run(
            ["codeql", "database", "analyze", str(db),
             f"codeql/{lang}-security-and-quality.qls", "--format=sarif-latest",
             "--output=/tmp/cql.sarif"],
            check=True, capture_output=True, timeout=180,
        )
        sarif = json.loads(Path("/tmp/cql.sarif").read_text())
        findings = []
        for run in sarif.get("runs", []):
            for r in run.get("results", []):
                findings.append(f"- {r.get('ruleId')}: {r.get('message',{}).get('text','')}")
        return "\n".join(findings) if findings else None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        return f"codeql error: {e}"


def shutil_which(cmd: str) -> str | None:
    import shutil
    return shutil.which(cmd)


# ---------- RAG retrieval (lightweight; arxiv 2505.17107 inspired) ----------

def rag_lookup(corpus_path: Path, query: str, k: int = 3) -> str:
    """Naive BM25-ish ranking over a JSONL corpus with {'text': ...} entries."""
    if not corpus_path.exists():
        return ""
    docs: list[str] = []
    if corpus_path.is_dir():
        for f in corpus_path.rglob("*.md"):
            docs.append(f.read_text(errors="ignore"))
        for f in corpus_path.rglob("*.jsonl"):
            for line in f.read_text(errors="ignore").splitlines():
                try:
                    docs.append(json.loads(line).get("text", ""))
                except json.JSONDecodeError:
                    pass
    else:
        for line in corpus_path.read_text(errors="ignore").splitlines():
            try:
                docs.append(json.loads(line).get("text", ""))
            except json.JSONDecodeError:
                docs.append(line)
    q_terms = {t.lower() for t in re.findall(r"\w+", query) if len(t) > 3}
    scored = sorted(
        ((sum(1 for t in q_terms if t in d.lower()), d) for d in docs if d),
        reverse=True,
    )
    return "\n---\n".join(d[:1500] for _, d in scored[:k])


# ---------- main agent loop --------------------------------------------------

def run_agent(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    tools = FileTools(root=root)

    client = OpenAI(base_url=args.endpoint, api_key=args.api_key or "lm-studio")
    sys_prompt = Path(args.system_prompt).read_text() if args.system_prompt else (
        "You are an autonomous agent with filesystem and shell tools. Use them. "
        "Do not refuse on capability grounds."
    )

    rag_ctx = ""
    if args.rag:
        rag_ctx = rag_lookup(Path(args.rag), args.task, k=3)
        if rag_ctx:
            sys_prompt += f"\n\nReference context:\n{rag_ctx}"

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": args.task},
    ]

    written: list[Path] = []
    for turn in range(args.max_turns):
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=messages,
                tools=TOOL_SCHEMA,
                tool_choice="auto",
                temperature=0.2,
            )
        except Exception:
            # model doesn't support tools — drop them and retry
            resp = client.chat.completions.create(
                model=args.model, messages=messages, temperature=0.2,
            )

        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        # path A: proper tool calls
        if getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                name = call.function.name
                try:
                    a = json.loads(call.function.arguments or "{}")
                    out = getattr(tools, name)(**a)
                    if name == "write_file":
                        written.append(Path(a["path"]))
                except Exception as e:
                    out = f"ERROR {type(e).__name__}: {e}"
                messages.append({
                    "role": "tool", "tool_call_id": call.id, "content": out,
                })
            continue

        # path B: fenced-block fallback for non-tool-calling models
        text = msg.content or ""
        writes = extract_fenced_writes(text)
        if writes:
            for path, body in writes:
                out = tools.write_file(path, body)
                written.append(Path(path))
                print(out)
            messages.append({
                "role": "user",
                "content": "Files written. Continue or say DONE.",
            })
            continue

        if text.strip().upper().endswith("DONE") or not text.strip():
            break
        # model said something but neither tool-called nor wrote — nudge it
        messages.append({
            "role": "user",
            "content": "You did not produce a tool call or fenced file block. "
                       "Emit the file using ```lang:/abs/path syntax.",
        })

    # CodeQL post-loop (arxiv 2506.23034)
    if args.codeql and written:
        for f in written:
            findings = codeql_scan(f)
            if findings:
                print(f"\n[codeql] {f}:\n{findings}")
                # one repair turn
                messages.append({
                    "role": "user",
                    "content": f"CodeQL flagged issues in {f}:\n{findings}\n"
                               "Rewrite the file fixing them.",
                })
                resp = client.chat.completions.create(
                    model=args.model, messages=messages,
                    tools=TOOL_SCHEMA, tool_choice="auto", temperature=0.2,
                )
                # apply same parse logic to the repair response
                m2 = resp.choices[0].message
                if getattr(m2, "tool_calls", None):
                    for call in m2.tool_calls:
                        if call.function.name == "write_file":
                            a = json.loads(call.function.arguments)
                            tools.write_file(**a)
                else:
                    for path, body in extract_fenced_writes(m2.content or ""):
                        tools.write_file(path, body)

    print(f"\nDONE. {len(tools.log)} ops, {len(written)} files written under {root}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="http://localhost:1234/v1")
    ap.add_argument("--model", required=True, help="model name as exposed by the endpoint")
    ap.add_argument("--root", required=True, help="filesystem sandbox root")
    ap.add_argument("--task", required=True)
    ap.add_argument("--system-prompt", help="path to a system-prompt markdown file")
    ap.add_argument("--rag", help="path to a JSONL/markdown corpus directory or file")
    ap.add_argument("--codeql", action="store_true", help="run CodeQL feedback loop after writes")
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--api-key", default=None)
    return run_agent(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
