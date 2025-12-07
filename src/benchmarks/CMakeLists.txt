
# =====================================================
# Benchmarks (optional)
# =====================================================
if (ATLAS_ENABLE_BENCH)
    message(STATUS "[ATLAS] Benchmarks: ENABLED")
    set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/googlebenchmark/CMakeLists.txt")
        add_subdirectory(external/googlebenchmark)
    endif ()
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/benchmarks/CMakeLists.txt")
        add_subdirectory(src/benchmarks)
    endif ()
endif ()

