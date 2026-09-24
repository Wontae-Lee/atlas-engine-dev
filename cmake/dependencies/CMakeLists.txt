# Standalone installer for pinned third-party release archives.
# Atlas itself consumes the installed packages through find_package().

cmake_minimum_required(VERSION 3.20)
project(atlas-dependencies LANGUAGES C CXX)

include(ExternalProject)
if (POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif ()

# Installation options --------------------------------------------------------

option(ATLAS_DEPS_WITH_TBB "Build oneTBB instead of using the OS package" OFF)
option(ATLAS_DEPS_DEVELOPMENT "Install test, benchmark, and Interactive dependencies" ON)
set(ATLAS_DEPS_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/downloads" CACHE PATH "Release archive cache")

# Shared release build settings ------------------------------------------------

# Static C++20 libraries use PIC so they can also link into Python extensions.
# Hash-qualified archive names avoid collisions in a shared download directory.
function(atlas_dependency name url sha256)
    ExternalProject_Add(${name}
            URL "${url}"
            URL_HASH "SHA256=${sha256}"
            DOWNLOAD_DIR "${ATLAS_DEPS_DOWNLOAD_DIR}"
            DOWNLOAD_NAME "${name}-${sha256}.tar.gz"
            TLS_VERIFY ON
            CMAKE_ARGS
                -DCMAKE_BUILD_TYPE=Release
                -DCMAKE_CXX_STANDARD=20
                -DCMAKE_CXX_STANDARD_REQUIRED=ON
                -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                -DBUILD_SHARED_LIBS=OFF
                -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
                -DCMAKE_INSTALL_LIBDIR=lib
                -DCMAKE_PREFIX_PATH=${CMAKE_INSTALL_PREFIX}
                ${ARGN}
    )
endfunction()

# Core dependencies: Protobuf requires the matching Abseil installation.

atlas_dependency(abseil
        "https://codeload.github.com/abseil/abseil-cpp/tar.gz/refs/tags/20250512.1"
        "9b7a064305e9fd94d124ffa6cc358592eb42b5da588fb4e07d09254aa40086db"
        -DABSL_ENABLE_INSTALL=ON
        -DABSL_BUILD_TESTING=OFF
        -DABSL_PROPAGATE_CXX_STD=ON
)

atlas_dependency(protobuf
        "https://codeload.github.com/protocolbuffers/protobuf/tar.gz/refs/tags/v36.2"
        "7e7eaf580ec77f1a392ff1ab1013c13d3b04a6716f002f697f67f3c1113f0530"
        -Dprotobuf_BUILD_TESTS=OFF
        -Dprotobuf_BUILD_SHARED_LIBS=OFF
        -Dprotobuf_BUILD_PROTOC_BINARIES=ON
        -Dprotobuf_INSTALL=ON
        -Dprotobuf_LOCAL_DEPENDENCIES_ONLY=ON
        -Dprotobuf_WITH_ZLIB=OFF
        -Dprotobuf_USE_UNITY_BUILD=OFF
)
add_dependencies(protobuf abseil)

atlas_dependency(tinyobjloader
        "https://codeload.github.com/tinyobjloader/tinyobjloader/tar.gz/refs/tags/v2.0.0rc13"
        "0feb92b838f8ce4aa6eb0ccc32dff30cb64a891e0ec3bde837fca49c78d44334"
        -DTINYOBJLOADER_BUILD_TEST_LOADER=OFF
)

# Optional test, benchmark, and Interactive dependencies ------------------------

if (ATLAS_DEPS_DEVELOPMENT)
    atlas_dependency(googletest
            "https://codeload.github.com/google/googletest/tar.gz/refs/tags/v1.18.0"
            "6e3191c1455468b3fc35a417fb565c1c5071aee1b7e7f85e30cf48a98d37d8b5"
            -DBUILD_GMOCK=OFF
            -DINSTALL_GTEST=ON
    )

    atlas_dependency(benchmark
            "https://codeload.github.com/google/benchmark/tar.gz/refs/tags/v1.9.5"
            "9631341c82bac4a288bef951f8b26b41f69021794184ece969f8473977eaa340"
            -DBENCHMARK_ENABLE_TESTING=OFF
            -DBENCHMARK_ENABLE_GTEST_TESTS=OFF
    )

    atlas_dependency(json
            "https://codeload.github.com/nlohmann/json/tar.gz/refs/tags/v3.12.0"
            "4b92eb0c06d10683f7447ce9406cb97cd4b453be18d7279320f7b2f025c10187"
            -DJSON_BuildTests=OFF
            -DJSON_Install=ON
    )
endif ()

# Optional TBB installation for environments without an OS package -------------

if (ATLAS_DEPS_WITH_TBB)
    atlas_dependency(tbb
            "https://codeload.github.com/oneapi-src/oneTBB/tar.gz/refs/tags/v2021.11.0"
            "782ce0cab62df9ea125cdea253a50534862b563f1d85d4cda7ad4e77550ac363"
            -DTBB_TEST=OFF
            -DTBB_STRICT=OFF
            -DBUILD_SHARED_LIBS=ON
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    )
endif ()
