import os


def remove_leading_spaces_in_dir(root_dir: str, extensions=None):
    """
    Remove leading spaces from each line of all files under a directory (recursively).

    Args:
        root_dir (str): Target directory path.
        extensions (list[str] | None): List of file extensions to process (e.g. [".cpp", ".h"]).
                                       If None, all files will be processed.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:

            if extensions and not any(filename.endswith(ext) for ext in extensions):
                continue

            file_path = os.path.join(dirpath, filename)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                new_lines = [line.lstrip() for line in lines]

                with open(file_path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)

                print(f"Processed: {file_path}")
            except Exception as e:
                print(f"Skipped {file_path} (Error: {e})")


if __name__ == "__main__":
    target_dir = "../include/atlas/math/matrix"
    remove_leading_spaces_in_dir(target_dir, extensions=[".cpp", ".h", ".hpp", ".cu"])
