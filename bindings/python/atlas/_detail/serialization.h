#pragma once

#include "array.h"
#include "handles.h"

#include <atlas/serialization/protobuf_snapshot.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/unique_ptr.h>

#include <filesystem>
#include <optional>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

template <typename Buffer>
nb::object
snapshot_array(const std::optional<Buffer>& buffer) {
    return buffer ? atlas::python::numpy_copy(*buffer, buffer->size()) : nb::none();
}

}
