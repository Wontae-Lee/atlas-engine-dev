"""
Lint Atlas execution-space annotations.

Two checks, both grounded in docs/architecture/06-conventions.md:

1. device-inline (the reason -rdc stays off):
   ATLAS_DEVICE / ATLAS_ALL_DEVICE functions must be defined inline in a header.
   An out-of-line definition of such a function in a .cu translation unit cannot
   be called from device code in another .cu, so it is flagged. Kernel-local
   functors (defined in-class, no `Type::` qualifier) and device lambdas
   (`ATLAS_ALL_DEVICE(...)`) are intentionally not flagged.

2. nodiscard-first:
   ATLAS_NODISCARD expands to [[nodiscard]], which only appertains to the
   function when it leads the declaration. If an execution-space macro precedes
   it the attribute binds to the return type and is silently dropped on the CUDA
   build. It must come before ATLAS_ALL_DEVICE / ATLAS_HOST / ATLAS_DEVICE /
   static / ATLAS_FORCE_INLINE.

Usage:
    python tools/check_device_inline.py            # scan include/ and src/
    python tools/check_device_inline.py <files...> # scan given files (pre-commit)

Exits 1 when any violation is found, 0 otherwise.
"""

import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN_ROOTS = ["include", "src"]
HEADER_EXT = (".h", ".hpp", ".cuh")
SOURCE_EXT = (".cu",)

# nodiscard-first: an execution-space / inline / static specifier before NODISCARD.
NODISCARD_MISORDER = re.compile(
    r"\b(ATLAS_ALL_DEVICE|ATLAS_DEVICE|ATLAS_HOST|ATLAS_FORCE_INLINE|static)\b"
    r".*\bATLAS_NODISCARD\b"
)

# device-inline: a line that leads with a device execution-space specifier
# (optionally after ATLAS_NODISCARD) and is not a lambda (`macro(`).
DEVICE_LEAD = re.compile(
    r"^\s*(?:ATLAS_NODISCARD\s+)?(?:ATLAS_ALL_DEVICE|ATLAS_DEVICE)\b(?!\s*\()"
)

# a qualified out-of-line definition head: Type::method( or ns::Type::~method(
QUALIFIED_DEF = re.compile(
    r"\b[A-Za-z_]\w*::(?:~?[A-Za-z_]\w*|operator[^\s(]*)\s*\("
)


def iter_files(paths):
    """Yield source files to scan, skipping external/ and non-source files."""
    for path in paths:
        if os.path.isdir(path):
            for subdir, _, files in os.walk(path):
                if "external" in subdir.split(os.sep):
                    continue
                for name in files:
                    if name.endswith(HEADER_EXT + SOURCE_EXT):
                        yield os.path.join(subdir, name)
        elif path.endswith(HEADER_EXT + SOURCE_EXT) and "external" not in path.split(os.sep):
            yield path


def check_nodiscard_first(path, lines):
    """Flag declarations where ATLAS_NODISCARD is not the leading specifier."""
    violations = []
    for i, line in enumerate(lines, start=1):
        code = line.split("//", 1)[0]
        if "ATLAS_NODISCARD" in code and NODISCARD_MISORDER.search(code):
            violations.append((path, i, line.strip(),
                               "ATLAS_NODISCARD must come first"))
    return violations


def check_device_inline(path, lines):
    """Flag device functions defined out-of-line in a .cu translation unit."""
    if not path.endswith(SOURCE_EXT):
        return []
    violations = []
    for i, line in enumerate(lines):
        if not DEVICE_LEAD.match(line):
            continue
        window = " ".join(lines[i:i + 3])
        if QUALIFIED_DEF.search(window):
            violations.append((path, i + 1, line.strip(),
                               "device function defined out-of-line in .cu "
                               "(must be header-inline; Atlas builds without -rdc)"))
    return violations


def main(argv):
    targets = argv[1:] or [os.path.join(REPO_ROOT, r) for r in SCAN_ROOTS]

    violations = []
    for path in iter_files(targets):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        violations.extend(check_nodiscard_first(path, lines))
        violations.extend(check_device_inline(path, lines))

    if not violations:
        print("check_device_inline: OK")
        return 0

    for path, lineno, text, message in violations:
        rel = os.path.relpath(path, REPO_ROOT)
        print(f"{rel}:{lineno}: {message}\n    {text}")
    print(f"\ncheck_device_inline: {len(violations)} violation(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
