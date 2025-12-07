import os


def generate_header(header_path, include_root, project_name, header_name):
    """
    Generate atlas.h header that aggregates all headers in include/atlas.

    - Collects all .hpp and .cuh files recursively
    - Excludes atlas.h and math.h
    - Writes #include lines in alphabetical order
    - Uses <atlas/...> form for includes
    - Files under atlas/vizkit/* are wrapped in:
        #ifdef ATLAS_ENABLE_VIZKIT
            ...
        #endif
    """

    root_header = os.path.join(os.path.dirname(__file__), "../")
    header = os.path.abspath(os.path.join(root_header, header_path))

    with open(header, "w") as header_file:
        # Header guard start
        header_file.write(f"#ifndef INCLUDE_{project_name.upper()}_{project_name.upper()}_H\n")
        header_file.write(f"#define INCLUDE_{project_name.upper()}_{project_name.upper()}_H\n\n")

        include_root = os.path.abspath(os.path.join(root_header, include_root))

        collected_normal = []
        collected_vizkit = []

        for path, dirs, files in os.walk(include_root):
            for file in files:
                if (file.endswith(".h") or file.endswith(".cuh")) \
                        and file != f"{project_name}.h" \
                        and file != "math.h":

                    rel_path = os.path.relpath(os.path.join(path, file), include_root)
                    rel_path = rel_path.replace("\\", "/")  # Windows normalize

                    # classify vizkit
                    if rel_path.startswith("vizkit/"):
                        collected_vizkit.append(rel_path)
                    else:
                        collected_normal.append(rel_path)

        # normal includes
        for rel_path in sorted(collected_normal, key=lambda s: s.lower()):
            if header_name == "":
                header_file.write(f"#include <{project_name}/{rel_path}>\n")
            else:
                header_file.write(f"#include <{project_name}/{header_name}/{rel_path}>\n")

        # vizkit includes under macro
        if collected_vizkit:
            header_file.write("\n#ifdef ATLAS_ENABLE_VIZKIT\n")
            for rel_path in sorted(collected_vizkit, key=lambda s: s.lower()):
                if header_name == "":
                    header_file.write(f"#include <{project_name}/{rel_path}>\n")
                else:
                    header_file.write(f"#include <{project_name}/{header_name}/{rel_path}>\n")
            header_file.write("#endif\n")

        header_file.write("\n#endif")


if __name__ == "__main__":
    generate_header("include/atlas/atlas.h", "include/atlas/", "atlas", "")
