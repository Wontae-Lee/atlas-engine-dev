# External dependencies and backend definitions shared by every Atlas target.
# TBB backs the host-side parallel algorithms in every configuration. CUDAToolkit is
# needed only when nvcc compiles the sources; it also supplies the Thrust headers, which
# is why a TBB build without nvcc needs neither.

find_package(TBB REQUIRED)

if (ATLAS_USE_NVCC)
    find_package(CUDAToolkit REQUIRED)
endif ()

# Backend compile definitions applied to every Atlas target so all TUs agree on the
# backend. Exactly one of ATLAS_BACKEND_CUDA / ATLAS_BACKEND_TBB is defined:
#
#   ATLAS_BACKEND_CUDA : Thrust containers and algorithms, compiled by nvcc.
#   ATLAS_BACKEND_TBB  : std::vector and TBB. No Thrust, no CUDA toolkit.
#
# The ten headers under buffer/, memory/, parallel/, scan/ branch on these; nothing else
# in the tree does.
set(ATLAS_BACKEND_COMPILE_DEFINITIONS "")

if (ATLAS_DEVICE_SYSTEM STREQUAL "CUDA")
    message(STATUS "[ATLAS] Device system: CUDA")
    list(APPEND ATLAS_BACKEND_COMPILE_DEFINITIONS
            ATLAS_BACKEND_CUDA
            THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_TBB
            THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA
            # Thrust hides its symbols in an inline namespace whose name embeds
            # __CUDA_ARCH_LIST__. That macro exists only under nvcc, so a host-compiled TU
            # (atlas-serialization: protobuf's headers cannot go through nvcc) mangles
            # thrust::device_vector into THRUST_..._SM___CUDA_ARCH_LIST___NS while the
            # engine's nvcc TUs mangle it into THRUST_..._SM_890_NS, and the two never
            # link. Turning the ABI namespace off makes every TU agree.
            THRUST_DISABLE_ABI_NAMESPACE
            THRUST_IGNORE_ABI_NAMESPACE_ERROR
            CUB_DISABLE_NAMESPACE_MAGIC
            CUB_IGNORE_NAMESPACE_MAGIC_ERROR
    )
else ()
    message(STATUS "[ATLAS] Device system: TBB")
    list(APPEND ATLAS_BACKEND_COMPILE_DEFINITIONS
            ATLAS_BACKEND_TBB
    )
endif ()

find_package(tinyobjloader CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)

set(ATLAS_PROTO_SCHEMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/atlas/serialization/proto")
set(ATLAS_PROTO_SCHEMA "${ATLAS_PROTO_SCHEMA_DIR}/atlas_snapshot.proto")
set(ATLAS_PROTO_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(ATLAS_PROTO_SOURCES
        "${ATLAS_PROTO_OUTPUT_DIR}/atlas_snapshot.pb.cc"
        "${ATLAS_PROTO_OUTPUT_DIR}/atlas_snapshot.pb.h")

file(MAKE_DIRECTORY "${ATLAS_PROTO_OUTPUT_DIR}")

add_custom_command(
        OUTPUT ${ATLAS_PROTO_SOURCES}
        COMMAND protobuf::protoc
        ARGS
        "--proto_path=${ATLAS_PROTO_SCHEMA_DIR}"
        "--cpp_out=${ATLAS_PROTO_OUTPUT_DIR}"
        "atlas_snapshot.proto"
        DEPENDS "${ATLAS_PROTO_SCHEMA}" protobuf::protoc
        WORKING_DIRECTORY "${ATLAS_PROTO_SCHEMA_DIR}"
        VERBATIM
)

# protobuf's message_lite.h declares a static variable template of incomplete
# type (EnumTraitsImpl::Undefined), which nvcc's frontend rejects. This
# translation unit stays .cpp so the host compiler handles it; it holds no
# device code.
add_library(atlas-serialization STATIC
        ${ATLAS_PROTO_OUTPUT_DIR}/atlas_snapshot.pb.cc
        src/atlas/serialization/protobuf_snapshot.cpp
)
add_library(atlas::serialization ALIAS atlas-serialization)

target_include_directories(atlas-serialization
        PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        PRIVATE
        "${ATLAS_PROTO_OUTPUT_DIR}"
)

target_compile_definitions(atlas-serialization
        PRIVATE
        ${ATLAS_BACKEND_COMPILE_DEFINITIONS}
)

target_link_libraries(atlas-serialization
        PUBLIC
        protobuf::libprotobuf
        PRIVATE
        TBB::tbb
        tinyobjloader::tinyobjloader
)

if (ATLAS_USE_NVCC)
    target_link_libraries(atlas-serialization PRIVATE CUDA::cudart)
endif ()
