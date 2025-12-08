# =====================================================
# Example modules (under src/examples)
# =====================================================

# Base path for examples
set(ATLAS_EXAMPLES_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# Always add: memory, geometry, solver
add_subdirectory("${ATLAS_EXAMPLES_DIR}/memory")
add_subdirectory("${ATLAS_EXAMPLES_DIR}/geometry")
add_subdirectory("${ATLAS_EXAMPLES_DIR}/solver")

# Optional: BVH examples (no vizkit dependency — always safe)
add_subdirectory("${ATLAS_EXAMPLES_DIR}/spatial/bounding_volume_hierarchy")

# Optional: Vizkit examples
if (ATLAS_USE_VIZKIT)
    add_subdirectory("${ATLAS_EXAMPLES_DIR}/visualization")
endif ()