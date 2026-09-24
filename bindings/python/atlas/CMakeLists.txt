# Backend-specific Python extension, installed inside the atlas package.

# Python and nanobind dependencies ---------------------------------------------

# Development.Module provides extension headers without requiring libpython.
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_VARIABLE nanobind_ROOT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
)
find_package(nanobind 2.13.0 EXACT CONFIG REQUIRED)

# Module identity and sources --------------------------------------------------

set(ATLAS_PYTHON_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/module.cpp"
)

string(TOLOWER "${ATLAS_DEVICE_SYSTEM}" ATLAS_PYTHON_ENGINE)
set(ATLAS_PYTHON_MODULE "_core_${ATLAS_PYTHON_ENGINE}")
set(ATLAS_PYTHON_TBB_EXTENSION "" CACHE FILEPATH "TBB extension to include in a CUDA wheel")

# Extension target ------------------------------------------------------------

# Request the stable CPython ABI where supported and a static nanobind runtime.
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

# Runtime bindings also need the compiled Core definitions when available.
if (TARGET atlas)
    target_link_libraries(atlas_python PRIVATE atlas)
endif ()

if (DEFINED SKBUILD_PROJECT_VERSION)
    target_compile_definitions(atlas_python PRIVATE ATLAS_VERSION_STRING="${SKBUILD_PROJECT_VERSION}")
else ()
    target_compile_definitions(atlas_python PRIVATE ATLAS_VERSION_STRING="${PROJECT_VERSION}")
endif ()

# Match the source language to the selected CUDA backend.
if (ATLAS_USE_NVCC AND ATLAS_DEVICE_SYSTEM STREQUAL "CUDA")
    set_source_files_properties(${ATLAS_PYTHON_SOURCES} PROPERTIES LANGUAGE CUDA)
endif ()

# Wheel installation ----------------------------------------------------------

# Only scikit-build-core installs extensions into the Python package. CUDA wheels
# can optionally include a separately built TBB extension.
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
