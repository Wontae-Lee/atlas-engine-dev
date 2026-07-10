# External dependencies and backend definitions shared by every Atlas target.
# TBB is the Thrust host system in every config (and the CPU device system);
# CUDAToolkit provides the Thrust headers and the runtime nvcc objects link.

find_package(TBB REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Backend compile definitions applied to every Atlas target so host- and
# device-compiled TUs agree on the Thrust systems.
set(ATLAS_BACKEND_COMPILE_DEFINITIONS
        THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_TBB
)

if (ATLAS_DEVICE_SYSTEM STREQUAL "CUDA")
    message(STATUS "[ATLAS] Device system: CUDA")
    list(APPEND ATLAS_BACKEND_COMPILE_DEFINITIONS
            ATLAS_TASKING_CUDA
            THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA
    )
else ()
    message(STATUS "[ATLAS] Device system: TBB")
    list(APPEND ATLAS_BACKEND_COMPILE_DEFINITIONS
            ATLAS_TASKING_TBB
            THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB
    )
endif ()

# Vendored tinyobj (referenced by public atlas-core headers).
add_subdirectory(external/tinyobj)

# Vendored protobuf: build both the C++ runtime and protoc from source, then
# generate the snapshot bindings and compile the serialization library.
unset(WITH_PROTOC CACHE)
set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_CONFORMANCE OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_PROTOC_BINARIES ON CACHE BOOL "" FORCE)
set(protobuf_BUILD_LIBPROTOC ON CACHE BOOL "" FORCE)
set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(protobuf_INSTALL OFF CACHE BOOL "" FORCE)
set(protobuf_USE_UNITY_BUILD OFF CACHE BOOL "" FORCE)
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)

set(ATLAS_PROTOBUF_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/protobuf")
set(ATLAS_PROTOBUF_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/protobuf")
add_subdirectory("${ATLAS_PROTOBUF_SOURCE_DIR}" "${ATLAS_PROTOBUF_BINARY_DIR}" EXCLUDE_FROM_ALL)

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
        "${CMAKE_CURRENT_SOURCE_DIR}/external/tinyobj"
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
        CUDA::cudart
)
