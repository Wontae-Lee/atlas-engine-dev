import os
import re
from typing import List, Dict, Any, Optional

# -----------------------------
# Core utilities
# -----------------------------

def _apply_substitutions(text: str, rules: List[Dict[str, Any]], ext: str) -> (str, bool):
    """Apply regex substitutions for a given extension."""
    changed = False
    for r in rules:
        if ext not in r.get("exts", []):
            continue
        flags = r.get("flags", 0)
        new_text, n = re.subn(r["pattern"], r["repl"], text, flags=flags)
        if n > 0:
            text = new_text
            changed = True
    return text, changed


def _normalize_spacing_lines(lines: List[str], rules: List[Dict[str, Any]], ext: str) -> (List[str], bool):
    """
    Ensure a fixed number of blank lines between two single-line regex patterns.
    Each rule applies only if ext is included in rule["exts"].
    """
    if not rules:
        return lines, False

    changed_total = False
    for rule in rules:
        if ext not in rule.get("exts", []):
            continue

        from_pat = re.compile(rule["from"])
        to_pat   = re.compile(rule["to"])
        want     = int(rule.get("blank_lines", 1))

        new_lines = []
        i = 0
        changed = False

        while i < len(lines):
            line = lines[i]
            new_lines.append(line)

            if from_pat.match(line.rstrip("\n")):
                j = i + 1
                blank_idxs = []
                while j < len(lines) and lines[j].strip() == "":
                    blank_idxs.append(j)
                    j += 1

                if j < len(lines) and to_pat.match(lines[j]):
                    cur_blank = len(blank_idxs)
                    if cur_blank != want:
                        # remove the blanks we just appended
                        for _ in range(cur_blank):
                            if new_lines:
                                new_lines.pop()
                        # insert desired count
                        new_lines.extend("\n" for _ in range(want))
                        changed = True
            i += 1

        lines = new_lines
        changed_total = changed_total or changed

    return lines, changed_total


def _apply_formatting(text: str, rules: Dict[str, Any]) -> (str, bool):
    """
    Formatting toggles:
      - trim_trailing_ws: remove trailing spaces/tabs on every line
      - ensure_eof_newline: ensure the file ends with exactly one '\n'
      - unify_line_endings: remove '\r' to normalize CRLF -> LF
    """
    if not rules:
        return text, False

    changed = False

    if rules.get("unify_line_endings", False):
        if "\r" in text:
            text = text.replace("\r", "")
            changed = True

    if rules.get("trim_trailing_ws", False):
        parts = text.splitlines(keepends=True)
        trimmed = []
        touched = False
        for p in parts:
            # keep newline as-is if present, strip only spaces/tabs before it
            if p.endswith("\n"):
                stripped = re.sub(r"[ \t]+\n$", "\n", p)
            else:
                stripped = re.sub(r"[ \t]+$", "", p)
            if stripped != p:
                touched = True
            trimmed.append(stripped)
        if touched:
            text = "".join(trimmed)
            changed = True

    if rules.get("ensure_eof_newline", False):
        if not text.endswith("\n"):
            text += "\n"
            changed = True

    return text, changed


# -----------------------------
# Public API (single-call entry)
# -----------------------------

def fix_code_in_dir(
        root_dir: str,
        *,
        extensions: Optional[List[str]] = None,
        word_rules_before: Optional[List[Dict[str, Any]]] = None,
        spacing_rules: Optional[List[Dict[str, Any]]] = None,
        word_rules_after: Optional[List[Dict[str, Any]]] = None,
        format_rules: Optional[Dict[str, Any]] = None,
) -> None:
    """
    One-call fixer that applies:
      1) word_rules_before (regex substitutions)
      2) spacing_rules     (blank-line normalization between patterns)
      3) word_rules_after  (regex substitutions)
      4) format_rules      (simple formatting toggles)

    Each rule for words/spacing must include "exts": [".h", ".hpp", ...]
    so they apply only to selected extensions.

    If 'extensions' is None, it is auto-collected from all rules' "exts".
    """
    word_rules_before = word_rules_before or []
    spacing_rules = spacing_rules or []
    word_rules_after = word_rules_after or []
    format_rules = format_rules or {}

    # Auto-collect extensions if not provided
    if extensions is None:
        exts = set()
        for r in word_rules_before:
            exts.update(r.get("exts", []))
        for r in spacing_rules:
            exts.update(r.get("exts", []))
        for r in word_rules_after:
            exts.update(r.get("exts", []))
        extensions = sorted(exts)

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            if ext not in extensions:
                continue

            path = os.path.join(dirpath, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception as e:
                print(f"Skipped {path} (read error: {e})")
                continue

            changed = False

            # 1) word rules (before)
            if word_rules_before:
                text, c = _apply_substitutions(text, word_rules_before, ext)
                changed = changed or c

            # 2) spacing rules (line-based)
            if spacing_rules:
                lines = text.splitlines(keepends=True)
                lines, c = _normalize_spacing_lines(lines, spacing_rules, ext)
                text = "".join(lines)
                changed = changed or c

            # 3) word rules (after)
            if word_rules_after:
                text, c = _apply_substitutions(text, word_rules_after, ext)
                changed = changed or c

            # 4) format rules (final pass)
            if format_rules:
                text, c = _apply_formatting(text, format_rules)
                changed = changed or c

            if changed:
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(text)
                    print(f"[UPDATED] {path}")
                except Exception as e:
                    print(f"Skipped {path} (write error: {e})")


# -----------------------------
# Example usage (edit below)
# -----------------------------
if __name__ == "__main__":
    target_dir = "../include/atlas"

    # BEFORE substitutions (run first)
    WORD_RULES_BEFORE = [
        # Example: convert tabs to 4 spaces in headers
        # {"exts": [".h", ".hpp"], "pattern": r"\t", "repl": "    ", "flags": 0},
    ]

    # Spacing rules (your previous two rules included here)
    SPACING_RULES = [
        # .hpp: ensure exactly one blank line between '}' and 'template'
        {"exts": [".hpp"], "from": r'^\s*}\s*(//.*)?$', "to": r'^\s*template\b', "blank_lines": 1},
        # .h: ensure exactly one blank line between ';' and the next comment
        {"exts": [".h"], "from": r'.*;\s*(//.*)?$', "to": r'^\s*(//|/\*)', "blank_lines": 1},
    ]

    # AFTER substitutions (run after spacing)
    WORD_RULES_AFTER = [
        # Example: collapse 3+ blank lines to 2 (safety net)
        # {"exts": [".h", ".hpp"], "pattern": r"(?:\n[ \t]*){3,}", "repl": "\n\n", "flags": 0},
    ]

    # Simple formatting toggles
    FORMAT_RULES = {
        "unify_line_endings": True,   # remove '\r' (CRLF -> LF)
        "trim_trailing_ws": True,     # remove trailing spaces/tabs per line
        "ensure_eof_newline": True,   # file ends with exactly one '\n'
    }

    fix_code_in_dir(
        target_dir,
        # If you want to force extensions, set e.g. extensions=[".h", ".hpp"]
        extensions=None,
        word_rules_before=WORD_RULES_BEFORE,
        spacing_rules=SPACING_RULES,
        word_rules_after=WORD_RULES_AFTER,
        format_rules=FORMAT_RULES,
    )
