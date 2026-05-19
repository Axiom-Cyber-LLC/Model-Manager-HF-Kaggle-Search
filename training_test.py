#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def prompt_for_path(prompt_text: str) -> Path:
    while True:
        raw = input(prompt_text).strip().strip('"').strip("'")

        if not raw:
            print("Please enter a file path.")
            continue

        path = Path(raw).expanduser()

        if path.exists() and path.is_file():
            return path

        print(f"File not found: {path}")


def choose_format() -> str:
    formats = {
        "1": "alpaca",
        "2": "messages",
        "3": "prompt_completion",
        "4": "text",
    }

    print("\nChoose dataset format:")
    print("1. Alpaca JSONL: instruction/input/output")
    print("2. Messages JSONL: messages")
    print("3. Prompt/completion JSONL: prompt/completion")
    print("4. Text JSONL: text")

    while True:
        choice = input("Format [1]: ").strip() or "1"

        if choice in formats:
            return formats[choice]

        print("Invalid choice. Enter 1, 2, 3, or 4.")


def validate_alpaca(row: dict, line_no: int) -> list[str]:
    errors = []
    required = {"instruction", "input", "output"}

    missing = required - set(row.keys())
    if missing:
        errors.append(f"Line {line_no}: missing keys: {sorted(missing)}")
        return errors

    if not str(row["instruction"]).strip():
        errors.append(f"Line {line_no}: empty instruction")

    if row["input"] is None:
        errors.append(f"Line {line_no}: input must be a string. Use empty string if unused.")

    if not str(row["output"]).strip():
        errors.append(f"Line {line_no}: empty output")

    return errors


def validate_messages(row: dict, line_no: int) -> list[str]:
    errors = []

    if "messages" not in row:
        return [f"Line {line_no}: missing key: messages"]

    messages = row["messages"]

    if not isinstance(messages, list) or not messages:
        return [f"Line {line_no}: messages must be a non-empty list"]

    roles = []

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"Line {line_no}: message {idx} is not an object")
            continue

        role = msg.get("role")
        content = msg.get("content")

        roles.append(role)

        if role not in {"system", "user", "assistant"}:
            errors.append(f"Line {line_no}: message {idx} has invalid role: {role}")

        if not isinstance(content, str) or not content.strip():
            errors.append(f"Line {line_no}: message {idx} has empty content")

    if "assistant" not in roles:
        errors.append(f"Line {line_no}: no assistant message found")

    return errors


def validate_prompt_completion(row: dict, line_no: int) -> list[str]:
    errors = []

    required = {"prompt", "completion"}
    missing = required - set(row.keys())

    if missing:
        errors.append(f"Line {line_no}: missing keys: {sorted(missing)}")
        return errors

    if not str(row["prompt"]).strip():
        errors.append(f"Line {line_no}: empty prompt")

    if not str(row["completion"]).strip():
        errors.append(f"Line {line_no}: empty completion")

    return errors


def validate_text(row: dict, line_no: int) -> list[str]:
    if "text" not in row:
        return [f"Line {line_no}: missing key: text"]

    if not isinstance(row["text"], str) or not row["text"].strip():
        return [f"Line {line_no}: empty text"]

    return []


def validate_jsonl(path: Path, dataset_format: str) -> int:
    bad = 0
    total = 0

    validators = {
        "alpaca": validate_alpaca,
        "messages": validate_messages,
        "prompt_completion": validate_prompt_completion,
        "text": validate_text,
    }

    validator = validators[dataset_format]

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            total += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Line {line_no}: invalid JSON: {e}")
                bad += 1
                continue

            if not isinstance(row, dict):
                print(f"Line {line_no}: JSONL record must be an object")
                bad += 1
                continue

            errors = validator(row, line_no)

            if errors:
                bad += 1
                for error in errors:
                    print(error)

    print("\nValidation summary")
    print("------------------")
    print(f"File: {path}")
    print(f"Format: {dataset_format}")
    print(f"Checked records: {total}")
    print(f"Bad records: {bad}")

    if total == 0:
        print("FAIL: file contains zero valid JSONL records.")
        return 1

    if bad:
        print("FAIL: dataset has format issues.")
        return 1

    print("PASS: dataset format looks valid.")
    return 0


def main() -> int:
    if len(sys.argv) >= 2:
        path = Path(sys.argv[1]).expanduser()

        if not path.exists() or not path.is_file():
            print(f"Dataset file not found: {path}")
            return 2
    else:
        path = prompt_for_path("Enter path to training JSONL file: ")

    if len(sys.argv) >= 3:
        dataset_format = sys.argv[2].strip().lower()
        allowed = {"alpaca", "messages", "prompt_completion", "text"}

        if dataset_format not in allowed:
            print(f"Invalid format: {dataset_format}")
            print(f"Allowed formats: {', '.join(sorted(allowed))}")
            return 2
    else:
        dataset_format = choose_format()

    return validate_jsonl(path, dataset_format)


if __name__ == "__main__":
    raise SystemExit(main())
