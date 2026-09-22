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
)

if (ATLAS_USE_NVCC)
    # CUDA 13 packages CCCL (including Thrust) below include/cccl, while older toolkits
    # place Thrust directly below include. Host-compiled consumers need both layouts.
    set(ATLAS_CUDA_PUBLIC_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
    foreach (ATLAS_CUDA_INCLUDE_DIR IN LISTS CUDAToolkit_INCLUDE_DIRS)
        if (EXISTS "${ATLAS_CUDA_INCLUDE_DIR}/cccl")
            list(APPEND ATLAS_CUDA_PUBLIC_INCLUDE_DIRS "${ATLAS_CUDA_INCLUDE_DIR}/cccl")
        endif ()
    endforeach ()
    target_include_directories(atlas-core INTERFACE ${ATLAS_CUDA_PUBLIC_INCLUDE_DIRS})
endif ()

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
)

if (ATLAS_USE_NVCC)
    target_link_libraries(atlas-core INTERFACE CUDA::cudart)
endif ()

# atlas — compiled engine library from the .cu definitions under src/atlas
# (mirrors include/atlas). TBB uses the native C++ compiler by default; CUDA
# and ATLAS_HOST_COMPILER=nvcc use nvcc.
file(GLOB_RECURSE ATLAS_ENGINE_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/atlas/*.cu"
)

# Without nvcc the .cu suffix is just a name. The sources hold no CUDA-only syntax --
# every kernel is a parallel_for over an ATLAS_ALL_DEVICE lambda, and those annotations
# vanish outside __CUDACC__ -- so the host compiler builds them as ordinary C++.
if (NOT ATLAS_USE_NVCC AND NOT ATLAS_ENGINE_SOURCES STREQUAL "")
    set_source_files_properties(${ATLAS_ENGINE_SOURCES} PROPERTIES LANGUAGE CXX)
endif ()

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
            "${CMAKE_CURRENT_SOURCE_DIR}/src/atlas/logging/logging.cpp"
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
