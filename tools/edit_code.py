import argparse
import os
import re
from pathlib import Path


DEFAULT_EXTENSIONS = (".hpp", ".cu")


def remove_comments(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    return re.sub(r"//.*", "", code)


def remove_leading_spaces(code: str) -> str:
    return "\n".join(line.lstrip() for line in code.splitlines())


def add_blank_line_after_member_functions(code: str) -> str:
    lines = code.splitlines()
    edited_lines = []
    brace_depth = 0

    for index, line in enumerate(lines):
        stripped = line.strip()
        depth_before = brace_depth

        brace_depth += line.count("{")
        brace_depth -= line.count("}")

        edited_lines.append(line.rstrip())

        is_function_close = stripped == "}" and depth_before > 1 and brace_depth == 1
        has_next_line = index + 1 < len(lines)
        next_line_is_blank = has_next_line and lines[index + 1].strip() == ""

        if is_function_close and has_next_line and not next_line_is_blank:
            edited_lines.append("")

    return "\n".join(edited_lines)


def edit_code(code: str) -> str:
    code = remove_comments(code)
    code = remove_leading_spaces(code)
    code = add_blank_line_after_member_functions(code)
    return "\n".join(line.rstrip() for line in code.splitlines())


def should_process(path: Path, extensions: tuple[str, ...]) -> bool:
    return path.is_file() and path.suffix in extensions


def edit_file(path: Path) -> bool:
    try:
        original = path.read_text(encoding="utf-8")
        edited = edit_code(original)

        if edited != original:
            path.write_text(edited, encoding="utf-8")

        print(f"Processed: {path}")
        return True
    except Exception as error:
        print(f"Skipped {path} (Error: {error})")
        return False


def edit_directory(root_dir: Path, extensions: tuple[str, ...]) -> None:
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            path = Path(dirpath) / filename

            if should_process(path, extensions):
                edit_file(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove comments, strip leading spaces, and insert one space after closing braces."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="../include/atlas",
        help="File or directory to edit. Defaults to ../include/atlas.",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=list(DEFAULT_EXTENSIONS),
        help="File extensions to process when path is a directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.path)
    extensions = tuple(args.extensions)

    if path.is_file():
        edit_file(path)
        return

    edit_directory(path, extensions)


if __name__ == "__main__":
    main()
