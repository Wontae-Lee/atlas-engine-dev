import os
import re

def path_to_guard(filepath, project_name, include_root="include"):
    """
    Generate a header guard string based on file path and project name.
    Only the path relative to include_root is used.
    """
    abspath = os.path.abspath(filepath)
    include_root_abs = os.path.abspath(include_root)

    if abspath.startswith(include_root_abs):
        relpath = abspath[len(include_root_abs):].lstrip(os.sep)
    else:
        relpath = os.path.relpath(filepath)

    guard = relpath.replace(os.sep, '_').replace('.', '_').upper()
    guard = guard.replace('-', '_')

    if guard.startswith(project_name.upper() + "_"):
        full_guard = f"INCLUDE_{guard}"
    else:
        full_guard = f"INCLUDE_{project_name.upper()}_{guard}"

    return full_guard

def replace_header_guard(content, new_guard):
    """
    Replace the existing header guard (#ifndef/#define ... #endif).
    Ensures the last #endif line is replaced with a clean '#endif'.
    """
    # Replace #ifndef and #define (only the first occurrence)
    pattern = r"(?P<ifndef>#ifndef\s+[A-Za-z0-9_]+)\s+(?P<define>#define\s+[A-Za-z0-9_]+)"
    new_guard_block = f"#ifndef {new_guard}\n#define {new_guard}"
    content = re.sub(pattern, new_guard_block, content, count=1)

    # Split into lines to handle the last #endif precisely
    lines = content.splitlines()
    for i in range(len(lines) - 1, -1, -1):  # search from bottom
        if re.match(r"^\s*#endif\b", lines[i]):
            lines[i] = "#endif"  # replace with clean #endif
            break

    return "\n".join(lines) + "\n"

def update_header_guard(filepath, project_name, include_root="include"):
    """
    Update the header guard in the given file and print info.
    """
    directory = os.path.dirname(filepath)
    filename = os.path.basename(filepath)

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    new_guard = path_to_guard(filepath, project_name, include_root)
    new_content = replace_header_guard(content, new_guard)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"[Directory] {directory if directory else '.'}")
    print(f"[Filename]  {filename}")
    print(f"[Guard]     {new_guard}")
    print()

def batch_update_header_guards(root_dir, project_name, include_root="include"):
    """
    Recursively update header guards for all .h and .hpp files under root_dir.
    """
    for subdir, _, files in os.walk(root_dir):
        for file in files:

            if file.endswith(('.h', '.hpp', '.inl')):
                filepath = os.path.join(subdir, file)
                update_header_guard(filepath, project_name, include_root)

if __name__ == "__main__":
    PROJECT_ROOT = "../include"
    PROJECT_NAME = "ATLAS"
    batch_update_header_guards(PROJECT_ROOT, PROJECT_NAME, include_root=PROJECT_ROOT)

    PROJECT_ROOT = "../src/third-party"
    PROJECT_NAME = "VIZKIT"
    batch_update_header_guards(PROJECT_ROOT, PROJECT_NAME, include_root=PROJECT_ROOT)