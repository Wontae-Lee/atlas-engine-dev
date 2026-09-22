# ============================================================
# Atlas examples
#
# Entered unconditionally from the top-level CMakeLists.txt; builds C++ example
# programs only when ATLAS_EXAMPLES is enabled. The Python examples under
# examples/python/ need no build step (they import the `atlas` extension).
# ============================================================
if (NOT ATLAS_EXAMPLES)
    return()
endif ()

add_subdirectory(cpp)

if (ATLAS_INTERACTIVE)
    add_subdirectory(interactive)
endif ()
