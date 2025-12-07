# =====================================================
# atlas-core library
#   - Core engine + math + CUDA kernels
#   - Optionally compiled with vizkit / GL support
# =====================================================
message(STATUS "[ATLAS] Engine core: ENABLED")

# Public include root (installed / exported headers)
set(ATLAS_CORE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
set(ATLAS_CORE_PUBLIC_DIR "${ATLAS_CORE_INCLUDE_DIR}/atlas")

# Header set (public API)
file(GLOB_RECURSE atlas_core_headers
        CONFIGURE_DEPENDS
        "${ATLAS_CORE_PUBLIC_DIR}/*.h"
)

# Source set (implementation, C++ + CUDA)
file(GLOB_RECURSE atlas_core_sources
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

# Main core library target
add_library(atlas-core ${atlas_core_headers} ${atlas_core_sources})
add_library(atlas::core ALIAS atlas-core)

# C++ standard + include path for consumers
target_compile_features(atlas-core PUBLIC cxx_std_20)
target_include_directories(atlas-core PUBLIC "${ATLAS_CORE_INCLUDE_DIR}")

# =====================================================
# Core CPU-side dependencies (always required)
# =====================================================
find_package(TBB REQUIRED)
target_link_libraries(atlas-core
        PUBLIC
        TBB::tbb
        tinyobjloader
)

# =====================================================
# Optional vizkit / OpenGL stack
#   - Controlled by top-level option ATLAS_USE_VIZKIT
#   - Exposes ATLAS_ENABLE_VIZKIT to the code when enabled
# =====================================================
if (ATLAS_USE_VIZKIT)
    # Add a compile definition that code can use:
    # #ifdef ATLAS_ENABLE_VIZKIT ... #endif
    target_compile_definitions(atlas-core
            PUBLIC
            ATLAS_ENABLE_VIZKIT
    )

    # GL / windowing stack
    find_package(glfw3 REQUIRED)
    find_package(OpenGL REQUIRED)
    find_package(GLUT REQUIRED)
    find_package(GLEW REQUIRED)

    target_link_libraries(atlas-core
            PUBLIC
            glfw
            GLEW::GLEW
            GL
            GLU
            glut
    )
endif ()

# =====================================================
# Optional CUDA runtime
#   - Controlled by top-level option ATLAS_USE_CUDA
#   - Core CUDA code (kernels, Thrust, etc.) live in atlas_core_sources
# =====================================================
if (ATLAS_USE_CUDA)
    find_package(CUDAToolkit QUIET)

    if (TARGET CUDA::cudart)
        target_link_libraries(atlas-core PUBLIC CUDA::cudart)
    endif ()
endif ()
