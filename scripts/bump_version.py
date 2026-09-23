#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
CMAKE_PATH = ROOT / "CMakeLists.txt"
PYPROJECT_PATH = ROOT / "pyproject.toml"
CITATION_PATH = ROOT / "CITATION.cff"
README_PATH = ROOT / "README.md"
SEMANTIC_VERSION = re.compile(
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
)
README_VERSION_PATTERN = re.compile(
    r"(?P<prefix>^\[CITATION\.cff\]\(CITATION\.cff\) contains the citation metadata for version )"
    r"(?P<version>[0-9]+\.[0-9]+\.[0-9]+)[.:]",
    re.MULTILINE,
)


class MetadataError(RuntimeError):
    pass


def require_single(pattern: re.Pattern[str], text: str, description: str) -> re.Match[str]:
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise MetadataError(f"Expected one {description}, found {len(matches)}")
    return matches[0]


def update_cmake(text: str, version: str) -> str:
    project_pattern = re.compile(r"project\s*\([^)]*\)", re.DOTALL)
    project = require_single(project_pattern, text, "CMake project(...) block")
    version_pattern = re.compile(r"(?P<prefix>\bVERSION\s+)(?P<version>[0-9]+\.[0-9]+\.[0-9]+)\b")
    require_single(version_pattern, project.group(0), "CMake project VERSION")
    updated_project = version_pattern.sub(rf"\g<prefix>{version}", project.group(0), count=1)
    return text[: project.start()] + updated_project + text[project.end() :]


def update_pyproject(text: str, version: str) -> str:
    project_pattern = re.compile(r"^\[project\]\s*$.*?(?=^\[|\Z)", re.MULTILINE | re.DOTALL)
    project = require_single(project_pattern, text, "[project] table")
    version_pattern = re.compile(
        r'(?P<prefix>^version\s*=\s*")(?P<version>[^"\n]+)(?P<suffix>"\s*$)',
        re.MULTILINE,
    )
    require_single(version_pattern, project.group(0), "[project].version")
    updated_project = version_pattern.sub(
        rf"\g<prefix>{version}\g<suffix>", project.group(0), count=1
    )
    return text[: project.start()] + updated_project + text[project.end() :]


def update_citation(text: str, version: str, release_date: str) -> str:
    version_pattern = re.compile(
        r'(?P<prefix>^version:\s*")(?P<version>[^"\n]+)(?P<suffix>"\s*$)',
        re.MULTILINE,
    )
    date_pattern = re.compile(
        r'(?P<prefix>^date-released:\s*")(?P<date>[^"\n]+)(?P<suffix>"\s*$)',
        re.MULTILINE,
    )
    doi_pattern = re.compile(r"^doi:\s*.*(?:\n|\Z)", re.MULTILINE)
    require_single(version_pattern, text, "CITATION.cff version")
    require_single(date_pattern, text, "CITATION.cff date-released")
    if len(doi_pattern.findall(text)) > 1:
        raise MetadataError("Expected at most one top-level CITATION.cff doi field")
    text = version_pattern.sub(rf"\g<prefix>{version}\g<suffix>", text, count=1)
    text = date_pattern.sub(rf"\g<prefix>{release_date}\g<suffix>", text, count=1)
    return doi_pattern.sub("", text, count=1)


def update_readme(text: str, version: str) -> str:
    current = require_single(README_VERSION_PATTERN, text, "README.md citation version")
    remainder = text[current.end() :]
    if remainder.startswith("\n[doi:"):
        doi = re.match(r"\n\[doi:[^\]\n]+\]\(https://doi\.org/[^)\n]+\)\.", remainder)
        if doi is None:
            raise MetadataError("Invalid README.md release DOI link")
        remainder = remainder[doi.end() :]
    return text[: current.start()] + current.group("prefix") + version + "." + remainder


def read_versions() -> dict[str, str]:
    cmake_text = CMAKE_PATH.read_text()
    cmake_project = require_single(
        re.compile(r"project\s*\([^)]*\)", re.DOTALL),
        cmake_text,
        "CMake project(...) block",
    ).group(0)
    cmake_version = require_single(
        re.compile(r"\bVERSION\s+([0-9]+\.[0-9]+\.[0-9]+)\b"),
        cmake_project,
        "CMake project VERSION",
    ).group(1)
    python_project = require_single(
        re.compile(r"^\[project\]\s*$.*?(?=^\[|\Z)", re.MULTILINE | re.DOTALL),
        PYPROJECT_PATH.read_text(),
        "[project] table",
    ).group(0)
    python_version = require_single(
        re.compile(r'^version\s*=\s*"([^"\n]+)"\s*$', re.MULTILINE),
        python_project,
        "[project].version",
    ).group(1)
    citation_version = require_single(
        re.compile(r'^version:\s*["\']?([^"\'\s]+)', re.MULTILINE),
        CITATION_PATH.read_text(),
        "CITATION.cff version",
    ).group(1)
    readme_version = require_single(
        README_VERSION_PATTERN,
        README_PATH.read_text(),
        "README.md citation version",
    ).group("version")
    return {
        "CMakeLists.txt": cmake_version,
        "pyproject.toml": python_version,
        "CITATION.cff": citation_version,
        "README.md": readme_version,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update Atlas release metadata")
    parser.add_argument("version", help="new release version in X.Y.Z form")
    args = parser.parse_args()
    if SEMANTIC_VERSION.fullmatch(args.version) is None:
        parser.error("version must use strict X.Y.Z semantic-version form")
    return args


def main() -> None:
    args = parse_args()
    current_versions = read_versions()
    if len(set(current_versions.values())) != 1:
        raise MetadataError(f"Current release versions do not match: {current_versions}")
    if next(iter(current_versions.values())) == args.version:
        raise MetadataError(f"{args.version} is already the current release version")
    originals = {
        CMAKE_PATH: CMAKE_PATH.read_text(),
        PYPROJECT_PATH: PYPROJECT_PATH.read_text(),
        CITATION_PATH: CITATION_PATH.read_text(),
        README_PATH: README_PATH.read_text(),
    }
    updated = {
        CMAKE_PATH: update_cmake(originals[CMAKE_PATH], args.version),
        PYPROJECT_PATH: update_pyproject(originals[PYPROJECT_PATH], args.version),
        CITATION_PATH: update_citation(
            originals[CITATION_PATH], args.version, date.today().isoformat()
        ),
        README_PATH: update_readme(originals[README_PATH], args.version),
    }

    try:
        for path, content in updated.items():
            path.write_text(content)
        versions = read_versions()
        if set(versions.values()) != {args.version}:
            raise MetadataError(f"Version verification failed: {versions}")
    except Exception:
        for path, content in originals.items():
            path.write_text(content)
        raise

    print(f"Updated Atlas release metadata to {args.version}")
    for path in updated:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
