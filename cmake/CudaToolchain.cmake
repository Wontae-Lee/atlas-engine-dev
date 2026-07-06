# nvcc toolchain: compiler discovery, GPU/arch detection, C++20 + Thrust flags.
# nvcc compiles every Atlas TU regardless of ATLAS_DEVICE_SYSTEM; the architecture
# only matters for the CUDA variant.

find_program(ATLAS_NVIDIA_SMI_EXECUTABLE
        NAMES nvidia-smi
        HINTS /usr/bin /bin /usr/local/nvidia/bin
)

# First value of a structured nvidia-smi query, or "" if unavailable.
function(atlas_query_nvidia_smi out_var query_name)
    if (NOT ATLAS_NVIDIA_SMI_EXECUTABLE)
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif ()

    execute_process(
            COMMAND "${ATLAS_NVIDIA_SMI_EXECUTABLE}" "--query-gpu=${query_name}" "--format=csv,noheader"
            OUTPUT_VARIABLE atlas_query_output
            ERROR_VARIABLE atlas_query_error
            RESULT_VARIABLE atlas_query_result
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if (NOT atlas_query_result EQUAL 0 OR atlas_query_output STREQUAL "")
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif ()

    string(REPLACE "\n" ";" atlas_query_lines "${atlas_query_output}")
    list(GET atlas_query_lines 0 atlas_query_first_line)
    string(STRIP "${atlas_query_first_line}" atlas_query_first_line)

    set(${out_var} "${atlas_query_first_line}" PARENT_SCOPE)
endfunction()

# CUDA runtime version reported by the driver (from nvidia-smi text), or "".
function(atlas_detect_cuda_driver_runtime_version out_var)
    if (NOT ATLAS_NVIDIA_SMI_EXECUTABLE)
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif ()

    execute_process(
            COMMAND "${ATLAS_NVIDIA_SMI_EXECUTABLE}"
            OUTPUT_VARIABLE atlas_smi_output
            ERROR_VARIABLE atlas_smi_error
            RESULT_VARIABLE atlas_smi_result
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if (NOT atlas_smi_result EQUAL 0)
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif ()

    string(REGEX MATCH "CUDA Version: ([0-9]+\\.[0-9]+)" atlas_cuda_runtime_match "${atlas_smi_output}")
    set(${out_var} "${CMAKE_MATCH_1}" PARENT_SCOPE)
endfunction()

# Installed compute capability as a real arch, e.g. "8.9" -> "89-real", or "".
function(atlas_detect_cuda_architectures out_var)
    atlas_query_nvidia_smi(atlas_detected_compute_cap "compute_cap")

    if (atlas_detected_compute_cap STREQUAL "")
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif ()

    string(REPLACE "." "" atlas_detected_compute_cap "${atlas_detected_compute_cap}")
    set(${out_var} "${atlas_detected_compute_cap}-real" PARENT_SCOPE)
endfunction()

# Discover nvcc if the toolchain / CUDACXX has not already selected it.
if (NOT CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
    find_program(ATLAS_NVCC_EXECUTABLE
            NAMES nvcc
            HINTS /usr/bin /usr/local/cuda/bin
    )

    if (ATLAS_NVCC_EXECUTABLE)
        set(CMAKE_CUDA_COMPILER "${ATLAS_NVCC_EXECUTABLE}" CACHE FILEPATH "CUDA compiler" FORCE)
        message(STATUS "[ATLAS] Using nvcc: ${CMAKE_CUDA_COMPILER}")
    endif ()
endif ()

atlas_query_nvidia_smi(ATLAS_CUDA_DRIVER_VERSION "driver_version")
atlas_detect_cuda_driver_runtime_version(ATLAS_CUDA_DRIVER_RUNTIME_VERSION)
atlas_detect_cuda_architectures(ATLAS_DETECTED_CUDA_ARCHITECTURES)

if (ATLAS_CUDA_DRIVER_VERSION)
    message(STATUS "[ATLAS] NVIDIA driver version: ${ATLAS_CUDA_DRIVER_VERSION}")
endif ()

if (ATLAS_CUDA_DRIVER_RUNTIME_VERSION)
    message(STATUS "[ATLAS] NVIDIA driver CUDA runtime: ${ATLAS_CUDA_DRIVER_RUNTIME_VERSION}")
endif ()

# Pick an architecture automatically when the user gave none (or the legacy "52"
# default some toolchains inject): detected GPU arch, else 89-real.
if ((NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        OR CMAKE_CUDA_ARCHITECTURES STREQUAL ""
        OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
    if (ATLAS_DETECTED_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "${ATLAS_DETECTED_CUDA_ARCHITECTURES}" CACHE STRING "CUDA architectures" FORCE)
    else ()
        set(CMAKE_CUDA_ARCHITECTURES "89-real" CACHE STRING "CUDA architectures" FORCE)
    endif ()
endif ()

enable_language(CUDA)

if (CMAKE_CUDA_COMPILER_VERSION)
    message(STATUS "[ATLAS] CUDA toolkit version: ${CMAKE_CUDA_COMPILER_VERSION}")
endif ()

# Some CMake 3.20+ setups don't map CUDA_STANDARD 20 to an nvcc flag even when
# nvcc supports C++20; apply --std=c++20 explicitly for consistent configuration.
set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "--std=c++20")
set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "--std=c++20")
unset(CMAKE_CUDA_STANDARD CACHE)
unset(CMAKE_CUDA_STANDARD_REQUIRED CACHE)
unset(CMAKE_CUDA_STANDARD)
unset(CMAKE_CUDA_STANDARD_REQUIRED)
if (NOT CMAKE_CUDA_FLAGS MATCHES "(^| )--std=c\\+\\+20($| )")
    string(APPEND CMAKE_CUDA_FLAGS " --std=c++20")
endif ()

# Thrust-friendly flags, appended once (idempotent across reconfigure passes).
foreach (atlas_cuda_flag IN ITEMS --expt-relaxed-constexpr --extended-lambda)
    if (NOT CMAKE_CUDA_FLAGS MATCHES "(^| )${atlas_cuda_flag}($| )")
        string(APPEND CMAKE_CUDA_FLAGS " ${atlas_cuda_flag}")
    endif ()
endforeach ()

# Non-fatal warning: a toolkit newer than the driver's CUDA runtime can build
# native SASS fine but may fail PTX JIT at runtime.
if (ATLAS_DEVICE_SYSTEM STREQUAL "CUDA"
        AND CMAKE_CUDA_COMPILER_VERSION
        AND ATLAS_CUDA_DRIVER_RUNTIME_VERSION
        AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER ATLAS_CUDA_DRIVER_RUNTIME_VERSION)
    message(WARNING
            "[ATLAS] CUDA toolkit ${CMAKE_CUDA_COMPILER_VERSION} is newer than "
            "driver CUDA runtime ${ATLAS_CUDA_DRIVER_RUNTIME_VERSION}. "
            "Native SASS builds may work, but PTX JIT can fail at runtime."
    )
endif ()
