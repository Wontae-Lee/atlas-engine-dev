# ================================================================
# CMake configuration for the atlas-engine-dev project
#   - Defines a header-only visualization toolkit library
#     exposed as the target: atlas::vizkit
#   - Core dependency: Intel TBB
#   - Optional dependencies: OpenGL / GLFW / GLEW / GLUT, CUDA
# ================================================================
cmake_minimum_required(VERSION 3.20)

# ----------------------------------------------------------------
# Project declaration
#
# LANGUAGES CXX:
#   - This project only uses C++ at the CMake level.
#   - CUDA is handled as an optional runtime dependency (via CUDAToolkit),
#     but this particular target is a header-only C++ library.
# ----------------------------------------------------------------
project(atlas-engine-dev LANGUAGES CXX)

# ----------------------------------------------------------------
# C++ standard configuration
#
# CMAKE_CXX_STANDARD 20:
#   - Request C++20 as the language standard for all C++ targets
#     unless they override it explicitly.
#
# CMAKE_CXX_STANDARD_REQUIRED ON:
#   - Enforce that the compiler must support C++20; do not silently
#     fall back to an older standard.
#
# CMAKE_CXX_EXTENSIONS OFF:
#   - Disable compiler-specific language extensions (e.g. GNU++20),
#     ensuring more portable and standard-compliant code.
# ----------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ----------------------------------------------------------------
# Configuration options
#
# ATLAS_USE_VIZKIT:
#   - Controls whether the OpenGL / windowing stack is enabled.
#   - When ON, additional dependencies (GLFW, GLEW, GLUT, OpenGL)
#     are required and ATLAS_ENABLE_VIZKIT is defined for consumers.
#
# ATLAS_USE_CUDA:
#   - Controls whether CUDA runtime support is linked.
#   - This target remains header-only; CUDA is treated as a runtime
#     library dependency (e.g. CUDA::cudart) rather than enabling
#     CUDA language compilation.
# ----------------------------------------------------------------
option(ATLAS_USE_VIZKIT "Enable Vizkit (OpenGL/windowing) stack" ON)
option(ATLAS_USE_CUDA "Enable CUDA runtime support" OFF)

# ----------------------------------------------------------------
# Public include directories
#
# ATLAS_VIZEKIT_INCLUDE_DIR:
#   - Root include directory for all public headers of the vizkit
#     component of this project.
#
# ATLAS_VIZEKIT_PUBLIC_DIR:
#   - Namespace root inside the include tree (e.g. include/atlas).
#   - Used mainly for IDE convenience and header discovery; the
#     actual include path exported to consumers is ATLAS_VIZEKIT_INCLUDE_DIR.
# ----------------------------------------------------------------
set(ATLAS_VIZEKIT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(ATLAS_VIZEKIT_PUBLIC_DIR "${ATLAS_VIZEKIT_INCLUDE_DIR}/atlas")

# ----------------------------------------------------------------
# Header discovery (for IDE integration / tooling)
#
# file(GLOB_RECURSE ...):
#   - Collects all .h files under the public include tree.
#   - This does not affect the build of the header-only library
#     itself (INTERFACE targets have no compiled sources), but
#     helps IDEs display the header list in their project trees.
#
# NOTE:
#   - CONFIGURE_DEPENDS tells CMake to re-run configuration when
#     the set of matching header files changes, which is useful
#     while actively developing the public API.
# ----------------------------------------------------------------
file(GLOB_RECURSE atlas_vizkit_headers
        CONFIGURE_DEPENDS
        "${ATLAS_VIZEKIT_PUBLIC_DIR}/*.h"
)

# ----------------------------------------------------------------
# Header-only library target
#
# add_library(atlas-vizkit INTERFACE):
#   - Declares an INTERFACE library, meaning:
#       * It has no compiled object files of its own.
#       * It is simply a collection of usage requirements
#         (include directories, compile definitions, linked
#         libraries) that are propagated to consumers.
#
# add_library(atlas::vizkit ALIAS atlas-vizkit):
#   - Provides a namespaced alias so other targets can link
#     against the library using the canonical form:
#       target_link_libraries(<consumer> PRIVATE atlas::vizkit)
# ----------------------------------------------------------------
add_library(atlas-vizkit INTERFACE)
add_library(atlas::vizkit ALIAS atlas-vizkit)

# ----------------------------------------------------------------
# Include directories for the header-only target
#
# target_include_directories(... INTERFACE ...):
#   - INTERFACE scope means:
#       * These include paths are NOT used to build atlas-vizkit
#         itself (it has no sources).
#       * They ARE propagated to any target that links against
#         atlas::vizkit.
#
# Consumers will be able to include headers such as:
#   #include <atlas/…>
# as long as they link to atlas::vizkit.
# ----------------------------------------------------------------
target_include_directories(atlas-vizkit
        INTERFACE
        "${ATLAS_VIZEKIT_INCLUDE_DIR}"
)

# ================================================================
# Core CPU-side dependencies
#   - Intel oneTBB (Threading Building Blocks)
# ================================================================
# ----------------------------------------------------------------
# find_package(TBB REQUIRED):
#   - Locates a system installation of oneTBB.
#   - Fails configuration if TBB is not found.
#   - Exposes the imported target TBB::tbb on success.
# ----------------------------------------------------------------
find_package(TBB REQUIRED)

# ----------------------------------------------------------------
# Link TBB as an INTERFACE dependency
#
# target_link_libraries(atlas-vizkit INTERFACE TBB::tbb):
#   - Any consumer of atlas::vizkit automatically links against TBB.
#   - This also ensures that any TBB-related compile options are
#     propagated to dependent targets.
# ----------------------------------------------------------------
target_link_libraries(atlas-vizkit
        INTERFACE
        TBB::tbb
)

# ================================================================
# Optional Vizkit / OpenGL stack
#   - Controlled by ATLAS_USE_VIZKIT
#   - When enabled:
#       * ATLAS_ENABLE_VIZKIT is defined for consumers.
#       * OpenGL and windowing libraries are added as dependencies.
# ================================================================
if (ATLAS_USE_VIZKIT)
    # ------------------------------------------------------------
    # Public compile definition for consumers
    #
    # ATLAS_ENABLE_VIZKIT:
    #   - Can be used in user code to conditionally compile
    #     visualization-related features, e.g.:
    #
    #       #ifdef ATLAS_ENABLE_VIZKIT
    #           // Vizkit / OpenGL code
    #       #endif
    # ------------------------------------------------------------
    target_compile_definitions(atlas-vizkit
            INTERFACE
            ATLAS_ENABLE_VIZKIT
    )

    # ------------------------------------------------------------
    # OpenGL / windowing dependencies
    #
    # glfw3:
    #   - Cross-platform window and input library.
    #
    # OpenGL / GLU:
    #   - Core OpenGL rendering API and the OpenGL Utility Library.
    #
    # GLUT:
    #   - Utility toolkit for managing windows and rendering contexts.
    #
    # GLEW:
    #   - OpenGL Extension Wrangler, used to load OpenGL extensions
    #     at runtime.
    #
    # NOTE:
    #   - These find_package calls are REQUIRED when ATLAS_USE_VIZKIT
    #     is ON; configuration will fail if any of them is missing.
    # ------------------------------------------------------------
    find_package(glfw3 REQUIRED)
    find_package(OpenGL REQUIRED)
    find_package(GLUT REQUIRED)
    find_package(GLEW REQUIRED)

    # ------------------------------------------------------------
    # Propagate OpenGL / windowing libraries to consumers
    #
    # All link libraries are added with INTERFACE scope so that:
    #   - atlas-vizkit remains a header-only target.
    #   - Any consumer that links atlas::vizkit also links these
    #     graphics libraries and inherits the necessary compile
    #     and link options.
    # ------------------------------------------------------------
    target_link_libraries(atlas-vizkit
            INTERFACE
            glfw
            GLEW::GLEW
            GL
            GLU
            glut
    )
endif ()

# ================================================================
# Optional CUDA runtime support
#   - Controlled by ATLAS_USE_CUDA
#   - When enabled and CUDA::cudart is available:
#       * The CUDA runtime library is added as an INTERFACE
#         dependency of atlas::vizkit.
#   - The target itself is still header-only; no CUDA sources
#     are compiled here.
# ================================================================
if (ATLAS_USE_CUDA)
    # ------------------------------------------------------------
    # Attempt to find the CUDA toolkit quietly
    #
    # QUIET:
    #   - Avoids noisy output when CUDA is not installed.
    #   - The presence of the imported target CUDA::cudart is
    #     checked explicitly below to decide whether to link it.
    # ------------------------------------------------------------
    find_package(CUDAToolkit QUIET)

    # ------------------------------------------------------------
    # Conditionally link the CUDA runtime (if available)
    #
    # CUDA::cudart:
    #   - Imported target representing the CUDA runtime library.
    #
    # INTERFACE scope:
    #   - Any consumer of atlas::vizkit will automatically link
    #     against the CUDA runtime when ATLAS_USE_CUDA is ON and
    #     CUDA::cudart is present.
    # ------------------------------------------------------------
    if (TARGET CUDA::cudart)
        target_link_libraries(atlas-vizkit
                INTERFACE
                CUDA::cudart
        )
    endif ()
endif ()
