# ============================================================
# Atlas examples
#
# Entered from the top-level CMakeLists.txt when ATLAS_EXAMPLES is enabled. The
# Python examples under examples/python/ need no build step (they import the
# `atlas` extension).
# ============================================================
add_subdirectory(cpp)

if (ATLAS_INTERACTIVE AND ATLAS_INTERACTIVE_RENDERING)
    add_subdirectory(interactive)
endif ()
