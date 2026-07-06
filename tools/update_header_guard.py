import os
import re


def ensure_pragma_once(content):
    """
    Make sure the file begins with '#pragma once'.
    Remove existing header guards if present.
    """
    lines = content.splitlines()

    # Remove header guards (#ifndef / #define)
    cleaned = []
    skip = False
    for line in lines:
        if re.match(r"^\s*#ifndef\b", line):
            skip = True
            continue
        if skip and re.match(r"^\s*#define\b", line):
            continue
        if skip and re.match(r"^\s*#endif\b", line):
            skip = False
            continue
        cleaned.append(line)

    cleaned_text = "\n".join(cleaned).strip()

    # Ensure pragma once at top
    if not cleaned_text.startswith("#pragma once"):
        cleaned_text = "#pragma once\n\n" + cleaned_text

    return cleaned_text + "\n"


def update_to_pragma_once(filepath):
    """
    Load a file, remove header guards, enforce '#pragma once', and save.
    """
    os.path.dirname(filepath)
    os.path.basename(filepath)

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    new_content = ensure_pragma_once(content)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"[Updated] {filepath}")


def batch_update_to_pragma_once(root_dir):
    """
    Recursively update all .h, .hpp, .inl files under root_dir to use '#pragma once'.
    """
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('.h', '.hpp', '.inl')):
                filepath = os.path.join(subdir, file)
                update_to_pragma_once(filepath)


if __name__ == "__main__":
    # ATLAS include headers
    PROJECT_ROOT = "../include"
    batch_update_to_pragma_once(PROJECT_ROOT)
