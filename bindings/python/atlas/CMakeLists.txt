# ============================================================
# Atlas Python bindings (nanobind)
#
# This directory builds a backend-specific Python extension module from the
# binding sources colocated with the `atlas` Python package. It is
# configured only when ATLAS_PYTHON is enabled (guarded in the
# top-level CMakeLists.txt).
#
# The module links against atlas::core so the bindings share the
# exact same headers, backend-selection macros, and dependencies
# as the C++ library. The compiled artifact is installed inside the
# pure-Python `atlas` package.
# ============================================================

# nanobind needs the Python interpreter and the Development.Module
# component (the limited set of headers required to build extension
# modules, without a full Python library dependency).
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

# Pull in the vendored nanobind sources. Adding the submodule as a
# subdirectory exposes the nanobind_add_module() helper.
add_subdirectory("${CMAKE_SOURCE_DIR}/external/nanobind" nanobind)

set(ATLAS_PYTHON_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/module.cpp"
)

string(TOLOWER "${ATLAS_DEVICE_SYSTEM}" ATLAS_PYTHON_ENGINE)
set(ATLAS_PYTHON_MODULE "_core_${ATLAS_PYTHON_ENGINE}")
set(ATLAS_PYTHON_TBB_EXTENSION "" CACHE FILEPATH "TBB extension to include in a CUDA wheel")

# Build the extension module.
#   STABLE_ABI : build against the CPython stable ABI when available
#                so a single wheel works across Python versions.
#   NB_STATIC  : link the nanobind runtime statically into the module.
nanobind_add_module(atlas_python
        STABLE_ABI
        NB_STATIC
        NB_DOMAIN "atlas_${ATLAS_PYTHON_ENGINE}"
        ${ATLAS_PYTHON_SOURCES}
)

# The backend-specific module name is independent of the CMake target name.
set_target_properties(atlas_python PROPERTIES OUTPUT_NAME "${ATLAS_PYTHON_MODULE}")
target_compile_definitions(atlas_python PRIVATE ATLAS_PYTHON_MODULE=${ATLAS_PYTHON_MODULE})

target_link_libraries(atlas_python PRIVATE atlas::core)

# The runtime bindings (Universe, Fluid, searcher, DSMC solver) call
# host-only definitions compiled into the `atlas` static library, so link it
# when it exists (built from src/atlas/**/*.cu).
if (TARGET atlas)
    target_link_libraries(atlas_python PRIVATE atlas)
endif ()

if (DEFINED SKBUILD_PROJECT_VERSION)
    target_compile_definitions(atlas_python PRIVATE ATLAS_VERSION_STRING="${SKBUILD_PROJECT_VERSION}")
else ()
    target_compile_definitions(atlas_python PRIVATE ATLAS_VERSION_STRING="${PROJECT_VERSION}")
endif ()

if (ATLAS_USE_NVCC AND ATLAS_DEVICE_SYSTEM STREQUAL "CUDA")
    set_source_files_properties(${ATLAS_PYTHON_SOURCES} PROPERTIES LANGUAGE CUDA)
endif ()

# When built through scikit-build-core (i.e. `pip wheel` / `python -m build`),
# install the extension inside the pure-Python `atlas` package.
# Guarded by SKBUILD so an ordinary in-tree CMake build is unaffected.
if (DEFINED SKBUILD)
    install(TARGETS atlas_python
            LIBRARY DESTINATION atlas COMPONENT python
            RUNTIME DESTINATION atlas COMPONENT python)
    if (ATLAS_DEVICE_SYSTEM STREQUAL "CUDA" AND ATLAS_PYTHON_TBB_EXTENSION)
        if (NOT EXISTS "${ATLAS_PYTHON_TBB_EXTENSION}")
            message(FATAL_ERROR "ATLAS_PYTHON_TBB_EXTENSION does not exist: ${ATLAS_PYTHON_TBB_EXTENSION}")
        endif ()
        install(FILES "${ATLAS_PYTHON_TBB_EXTENSION}" DESTINATION atlas COMPONENT python)
    endif ()
endif ()
