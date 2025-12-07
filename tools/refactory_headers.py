#!/usr/bin/env python3
"""
Refactor #include directives using subdirectory names under include/atlas (no hardcoding):
- Build a map: {basename -> 'atlas/.../basename'} by scanning include/atlas recursively.
- Replace any #include "xxx.h" or #include <xxx.h> with #include <atlas/.../xxx.h> if found.
"""

import argparse
import pathlib
import re
import sys

# Regex to match C/C++ include lines
INCLUDE_RE = re.compile(r'^\s*#\s*include\s*([<"])\s*([^">]+)\s*([">])')

# Which file extensions to scan in the target root
DEFAULT_EXTS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".ipp", ".inl", ".tpp"}

# Which extensions count as "header-like" under include/atlas when building the map
HEADER_EXTS = {".h", ".hh", ".hpp", ".hxx", ".ipp", ".inl", ".tpp"}


def collect_headers(atlas_root: pathlib.Path) -> dict[str, str]:
    """
    Recursively collect all headers under include/atlas/** and build a map:
        basename -> 'atlas/.../basename'
    No duplicate checking per user's instruction.
    """
    if not atlas_root.is_dir():
        print(f"[ERROR] '{atlas_root}' does not exist.", file=sys.stderr)
        sys.exit(1)

    name_to_canonical: dict[str, str] = {}
    # Walk recursively and map every header-like file
    for p in atlas_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in HEADER_EXTS:
            rel = p.relative_to(atlas_root).as_posix()  # e.g., 'atlas/foo/bar/baz.h'
            name_to_canonical[p.name] = rel
    return name_to_canonical


def process_file(path: pathlib.Path, name_map: dict[str, str], write: bool) -> bool:
    """
    Scan a single file; rewrite include lines if the basename exists in name_map.
    Returns True if file content changed.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")

    lines = text.splitlines(keepends=True)
    changed = False
    out_lines = []

    for line in lines:
        m = INCLUDE_RE.match(line)
        if not m:
            out_lines.append(line)
            continue

        # Extract the "path" part and reduce to basename
        raw_path = m.group(2)
        basename = pathlib.Path(raw_path).name

        # If this basename is one of our known headers, rewrite
        canonical = name_map.get(basename)
        if canonical:
            new_line = f'#include <atlas/{canonical}>\n'
            if new_line != line:
                out_lines.append(new_line)
                changed = True
            else:
                out_lines.append(line)
        else:
            out_lines.append(line)

    if write and changed:
        # Preserve original encoding guess (utf-8 preferred)
        try:
            path.write_text("".join(out_lines), encoding="utf-8")
        except UnicodeEncodeError:
            path.write_text("".join(out_lines), encoding="latin-1")

    return changed


def main():
    ap = argparse.ArgumentParser(
        description="Refactor #include to <atlas/...> using actual subdirectory names under include/atlas (no hardcoding)."
    )
    ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path("."),
                    help="Directory to scan for source files (where includes will be rewritten).")
    ap.add_argument("--include-dir", type=pathlib.Path, default=pathlib.Path("include"),
                    help="Include root that contains the 'atlas' directory.")
    ap.add_argument("--ext", nargs="*", default=sorted(DEFAULT_EXTS),
                    help="File extensions to scan in --root (default: common C/C++).")
    ap.add_argument("--write", action="store_true",
                    help="Apply changes in-place. Without this flag, runs as a dry-run.")
    args = ap.parse_args()

    atlas_dir = args.include_dir.resolve()
    root = args.root.resolve()
    exts = {e if e.startswith(".") else "." + e for e in args.ext}

    # Step 1: Build {basename -> 'atlas/.../basename'} map by scanning include/atlas/**
    name_map = collect_headers(atlas_dir)
    if not name_map:
        print("[ERROR] No headers found under include/atlas.", file=sys.stderr)
        sys.exit(2)

    # Step 2: Walk target root and rewrite
    total_files = 0
    changed_files = 0

    for p in root.rglob("*"):
        if p.is_file() and p.suffix in exts:
            total_files += 1
            if process_file(p, name_map, write=args.write):
                changed_files += 1

    mode = "WRITE" if args.write else "DRY-RUN"
    print(f"[DONE] Mode: {mode}")
    print(f"[DONE] Scanned files: {total_files}")
    print(f"[DONE] Files changed: {changed_files}")


if __name__ == "__main__":
    main()
