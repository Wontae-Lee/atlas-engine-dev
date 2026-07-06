# Atlas library targets: the atlas-core interface, the compiled engine library,
# and optional logging.

# atlas-core carries public includes, backend defs, and transitive deps.
# Header-only consumers link atlas::core; consumers of compiled code link the
# `atlas` library below, which re-exports atlas::core.
message(STATUS "[ATLAS] Engine core: ENABLED")

set(ATLAS_CORE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library(atlas-core INTERFACE)
add_library(atlas::core ALIAS atlas-core)

target_include_directories(atlas-core
        INTERFACE
        "${ATLAS_CORE_INCLUDE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/external/tinyobj"
        "${CMAKE_CURRENT_SOURCE_DIR}/external/lyra/include"
        # Thrust ships with the CUDA toolkit; exposing its include dirs lets
        # host-compiled consumers (e.g. the Python bindings) see the same
        # Thrust headers as nvcc TUs.
        "${CUDAToolkit_INCLUDE_DIRS}"
)

target_compile_definitions(atlas-core
        INTERFACE
        ${ATLAS_BACKEND_COMPILE_DEFINITIONS}
)

# tinyobj is referenced by public atlas-core headers, so propagate it.
target_link_libraries(atlas-core
        INTERFACE
        tinyobjloader
        atlas::serialization
        TBB::tbb
        CUDA::cudart
)

# atlas — compiled engine library from the .cu definitions under src/atlas
# (mirrors include/atlas). nvcc compiles all of it; the Thrust device system
# decides whether GPU code is generated.
file(GLOB_RECURSE ATLAS_ENGINE_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/atlas/*.cu"
)

if (ATLAS_ENGINE_SOURCES STREQUAL "")
    message(STATUS "[ATLAS] Engine library: no sources under src/atlas yet; skipping.")
else ()
    add_library(atlas STATIC ${ATLAS_ENGINE_SOURCES})
    add_library(atlas::atlas ALIAS atlas)

    target_link_libraries(atlas PUBLIC atlas::core)

    # atlas <-> atlas::serialization is a static-library cycle: the engine's
    # Universe::save() calls into serialization, and serialization reads back
    # Universe getters / UniverseState types from the engine. Declaring the
    # reverse edge lets CMake repeat both archives on the link line so the
    # linker resolves the cycle regardless of symbol order (without it, a
    # consumer only links whichever direction its first-seen symbols happen to
    # need). atlas::core already pulls serialization the other way.
    target_link_libraries(atlas-serialization PRIVATE atlas)

    # Allow linking the static engine into shared objects (e.g. the Python module).
    set_target_properties(atlas PROPERTIES POSITION_INDEPENDENT_CODE ON)

    message(STATUS "[ATLAS] Engine library: ENABLED")
endif ()

# atlas-logging — compiled logging library. When enabled, atlas-core exports
# ATLAS_ENABLE_LOGGING and the atlas::logging link, so linking atlas::core alone
# gives downstream users logging support.
if (ATLAS_LOGGING)
    message(STATUS "[ATLAS] Logging: ENABLED")

    add_library(atlas-logging STATIC
            "${CMAKE_CURRENT_SOURCE_DIR}/src/logging/logging.cpp"
    )
    add_library(atlas::logging ALIAS atlas-logging)

    target_include_directories(atlas-logging
            PUBLIC
            "${ATLAS_CORE_INCLUDE_DIR}"
    )

    target_compile_definitions(atlas-core INTERFACE ATLAS_ENABLE_LOGGING)
    target_compile_definitions(atlas-logging PUBLIC ATLAS_ENABLE_LOGGING)

    target_link_libraries(atlas-core INTERFACE atlas::logging)
else ()
    message(STATUS "[ATLAS] Logging: DISABLED")
endif ()
