# ============================================================
# Atlas Python bindings (nanobind)
#
# This directory (src/python/atlas) builds the `atlas` Python
# extension module from the vendored nanobind submodule. It is
# configured only when ATLAS_PYTHON is enabled (guarded in the
# top-level CMakeLists.txt).
#
# The module links against atlas::core so the bindings share the
# exact same headers, backend-selection macros, and dependencies
# as the C++ library. The compiled artifact is named `atlas` so it
# can be imported directly with `import atlas`.
# ============================================================

# nanobind needs the Python interpreter and the Development.Module
# component (the limited set of headers required to build extension
# modules, without a full Python library dependency).
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

# Pull in the vendored nanobind sources. Adding the submodule as a
# subdirectory exposes the nanobind_add_module() helper.
add_subdirectory("${CMAKE_SOURCE_DIR}/external/nanobind" nanobind)

# The bindings are split into module.cpp (NB_MODULE) plus one bindings_*.cpp per
# engine module group. Glob them so a new group needs no edit here.
file(GLOB ATLAS_PYTHON_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

# Build the extension module.
#   STABLE_ABI : build against the CPython stable ABI when available
#                so a single wheel works across Python versions.
#   NB_STATIC  : link the nanobind runtime statically into the module.
nanobind_add_module(atlas_python
        STABLE_ABI
        NB_STATIC
        ${ATLAS_PYTHON_SOURCES}
)

# The importable module name is `atlas`, independent of the CMake
# target name used above.
set_target_properties(atlas_python PROPERTIES OUTPUT_NAME atlas)

target_link_libraries(atlas_python PRIVATE atlas::core)

# The runtime bindings (Universe, Fluid, searcher, DSMC/SPH solvers) call
# host-only definitions compiled into the `atlas` static library, so link it
# when it exists (built from src/atlas/**/*.cu).
if (TARGET atlas)
    target_link_libraries(atlas_python PRIVATE atlas)
endif ()

if (DEFINED SKBUILD_PROJECT_VERSION)
    target_compile_definitions(atlas_python PRIVATE ATLAS_VERSION_STRING="${SKBUILD_PROJECT_VERSION}")
else ()
    target_compile_definitions(atlas_python PRIVATE ATLAS_VERSION_STRING="0.1.0")
endif ()

if (ATLAS_USE_NVCC AND ATLAS_DEVICE_SYSTEM STREQUAL "CUDA")
    set_source_files_properties(${ATLAS_PYTHON_SOURCES} PROPERTIES LANGUAGE CUDA)
endif ()

# When built through scikit-build-core (i.e. `pip wheel` / `python -m build`),
# install the extension at the wheel root so it imports as `import atlas`.
# Guarded by SKBUILD so an ordinary in-tree CMake build is unaffected.
if (DEFINED SKBUILD)
    install(TARGETS atlas_python LIBRARY DESTINATION . COMPONENT python)
endif ()
