#!/usr/bin/env python3
"""
Comment out every line with '//' in source files under a directory (recursively).

Targets file extensions: .h, .hpp, .cpp, .inl, .cu, .cuh
- Optional: skip backup creation with --no-backup
"""

import argparse
from pathlib import Path

EXTENSIONS = {".h", ".hpp", ".cpp", ".inl", ".cu", ".cuh"}


def looks_binary(sample: bytes) -> bool:
    """Simple heuristic to detect binary files."""
    if b"\x00" in sample:
        return True
    text_chars = bytearray(range(32, 127)) + b"\n\r\t\b\f"
    weird = sum(ch not in text_chars for ch in sample)
    return weird > max(8, len(sample) * 0.10)


def process_file(path: Path, skip_already=True, dry_run=False, create_backup=True, encoding="utf-8"):
    """Comment out all lines in the file with //."""
    raw = path.read_bytes()
    if looks_binary(raw[:4096]):
        print(f"[skip-binary] {path}")
        return

    text = raw.decode(encoding, errors="ignore")
    lines = text.splitlines(keepends=True)

    changed = False
    out_lines = []
    for line in lines:
        stripped = line.lstrip()
        leading = line[: len(line) - len(stripped)]
        if skip_already and stripped.startswith("//"):
            new_line = line
        else:
            new_line = f"{leading}//{stripped}"
            changed = True
        out_lines.append(new_line)

    if not changed:
        print(f"[no-change]  {path}")
        return

    if dry_run:
        print(f"[dry-run]    {path}")
        return

    # Optional backup
    if create_backup:
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            bak.write_bytes(raw)

    path.write_text("".join(out_lines), encoding=encoding, errors="ignore")
    print(f"[commented]  {path}" + ("" if not create_backup else "  (backup created)"))


def comment_all(root_dir: Path, skip_already=True, dry_run=False, create_backup=True, encoding="utf-8"):
    """Recursively comment all target files."""
    if not root_dir.exists():
        raise SystemExit(f"Path not found: {root_dir}")

    count = 0
    for p in root_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTENSIONS:
            process_file(p, skip_already, dry_run, create_backup, encoding)
            count += 1

    print(f"\nDone. Scanned {count} file(s).")


def main():
    ap = argparse.ArgumentParser(description="Comment out all lines with // in C/C++/CUDA headers/sources.")
    ap.add_argument("root", type=Path, help="Root directory to scan recursively")
    ap.add_argument("--no-skip-already", action="store_true",
                    help="Do NOT skip lines that already start with // (will double-comment).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Scan and report without modifying files.")
    ap.add_argument("--no-backup", action="store_true",
                    help="Do NOT create .bak backup files.")
    ap.add_argument("--encoding", default="utf-8",
                    help="File encoding for read/write (default: utf-8).")
    args = ap.parse_args()

    comment_all(
        root_dir=args.root,
        skip_already=not args.no_skip_already,
        dry_run=args.dry_run,
        create_backup=not args.no_backup,
        encoding=args.encoding
    )


if __name__ == "__main__":
    target_dir = Path("../src/temp")
    dry_run_mode = False
    skip_already_commented = False


    comment_all(
        root_dir=target_dir,
        skip_already=skip_already_commented,
        dry_run=dry_run_mode,
        encoding="utf-8"
    )
