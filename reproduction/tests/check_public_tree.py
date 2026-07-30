#!/usr/bin/env python3
"""Check the publishable tree for common hygiene and portability failures."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
TEXT_SUFFIXES = {".md", ".py", ".sh", ".json", ".toml", ".txt", ".yml", ".yaml"}
FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".joblib",
    ".key",
    ".log",
    ".p12",
    ".parquet",
    ".pem",
    ".pfx",
    ".pt",
    ".pth",
    ".safetensors",
}
CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
SENSITIVE = {
    "absolute macOS user path": re.compile(r"/Users/[^/\s]+/"),
    "absolute Linux home path": re.compile(r"/home/[^/\s]+/"),
    "private key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "assigned token or key": re.compile(
        r"(?i)\b(?:access[_-]?token|token|api[_-]?key|secret)\s*[:=]\s*"
        r"[\"']?[^\s\"'<>]{6,}"
    ),
    "credential-bearing URL": re.compile(
        r"(?i)https?://[^\s)]*(?:access_token|token|secret|api[_-]?key)="
        r"[^&\s)]+"
    ),
    "assigned password": re.compile(
        r"(?i)\bpassword\s*[:=]\s*[\"']?[^\s\"'<>]{6,}"
    ),
}
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history",
        action="store_true",
        help="Also reject forbidden file types reachable from Git history.",
    )
    return parser.parse_args()


def candidate_files() -> list[Path]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    )
    paths = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        path = REPOSITORY / raw.decode("utf-8")
        if path.is_file():
            paths.append(path)
    return sorted(set(paths))


def notebook_text(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    strings: list[str] = []

    def collect(value: object) -> None:
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)

    collect(notebook)
    return "\n".join(strings)


def check_text(path: Path, text: str, failures: list[str]) -> None:
    relative = path.relative_to(REPOSITORY).as_posix()
    if CJK.search(text):
        failures.append(f"non-English CJK text: {relative}")
    if path.resolve() != Path(__file__).resolve():
        for label, pattern in SENSITIVE.items():
            if pattern.search(text):
                failures.append(f"{label}: {relative}")
    if path.suffix == ".py":
        try:
            ast.parse(text, filename=relative)
        except SyntaxError as error:
            failures.append(f"Python syntax: {relative}: {error}")
    elif path.suffix == ".json":
        try:
            json.loads(text)
        except json.JSONDecodeError as error:
            failures.append(f"JSON syntax: {relative}: {error}")


def check_markdown_links(path: Path, text: str, failures: list[str]) -> None:
    relative = path.relative_to(REPOSITORY).as_posix()
    for target in MARKDOWN_LINK.findall(text):
        clean = target.strip().split("#", 1)[0]
        if not clean or clean.startswith(("http://", "https://", "mailto:")):
            continue
        clean = clean.strip("<>")
        resolved = (path.parent / clean).resolve()
        try:
            resolved.relative_to(REPOSITORY)
        except ValueError:
            failures.append(f"link escapes repository: {relative} -> {target}")
            continue
        if not resolved.exists():
            failures.append(f"broken local link: {relative} -> {target}")


def check_shell(path: Path, failures: list[str]) -> None:
    result = subprocess.run(
        ["bash", "-n", str(path)],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        failures.append(
            f"shell syntax: {path.relative_to(REPOSITORY)}: {result.stderr.strip()}"
        )


def check_history(failures: list[str]) -> None:
    output = subprocess.run(
        ["git", "rev-list", "--objects", "--all"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    checked_blobs: set[str] = set()
    for line in output.splitlines():
        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue
        object_id, relative = parts
        suffix = Path(relative).suffix.lower()
        if suffix in FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden historical artifact: {relative}")

        if object_id in checked_blobs:
            continue
        checked_blobs.add(object_id)
        object_type = subprocess.run(
            ["git", "cat-file", "-t", object_id],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if object_type != "blob":
            continue
        size = int(
            subprocess.run(
                ["git", "cat-file", "-s", object_id],
                cwd=REPOSITORY,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        if size > 10 * 1024 * 1024:
            failures.append(f"historical file exceeds 10 MiB: {relative}")
            continue
        if suffix not in TEXT_SUFFIXES and suffix != ".ipynb":
            continue
        if relative == "reproduction/tests/check_public_tree.py":
            continue
        content = subprocess.run(
            ["git", "cat-file", "-p", object_id],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
        ).stdout
        if b"\0" in content[:8192]:
            continue
        text = content.decode("utf-8", errors="replace")
        if CJK.search(text):
            failures.append(f"non-English CJK text in history: {relative}")
        for label, pattern in SENSITIVE.items():
            if pattern.search(text):
                failures.append(f"{label} in history: {relative}")


def main() -> None:
    args = parse_args()
    files = candidate_files()
    failures: list[str] = []
    for path in files:
        relative = path.relative_to(REPOSITORY).as_posix()
        suffix = path.suffix.lower()
        if suffix in FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden public artifact: {relative}")
        if path.stat().st_size > 10 * 1024 * 1024:
            failures.append(f"file exceeds 10 MiB: {relative}")

        if suffix == ".ipynb":
            try:
                text = notebook_text(path)
            except json.JSONDecodeError as error:
                failures.append(f"notebook JSON syntax: {relative}: {error}")
            else:
                check_text(path, text, failures)
        elif suffix in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8")
            check_text(path, text, failures)
            if suffix == ".md":
                check_markdown_links(path, text, failures)
        if suffix == ".sh":
            check_shell(path, failures)

    authors = subprocess.run(
        ["git", "log", "--format=%an <%ae>"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if re.search(r"(?i)\b(?:gpt|claude|openai|anthropic)\b", authors):
        failures.append("AI identity found in Git author metadata")
    if args.history:
        check_history(failures)

    if failures:
        formatted = "\n".join(f"- {failure}" for failure in sorted(set(failures)))
        raise SystemExit(f"Public-tree checks failed:\n{formatted}")
    print(f"Public-tree checks passed for {len(files)} files.")


if __name__ == "__main__":
    main()
