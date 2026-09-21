#pragma once

#include "array.h"

#include <atlas/buffer/host_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/copy.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/solver/solver.h>
#include <atlas/universe/universe.h>

#include <nanobind/nanobind.h>

#include <cstddef>
#include <limits>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

struct PySearcher final {
    explicit PySearcher(atlas::SpatialHashingSearcher searcher)
        : value(std::move(searcher)) { }

    atlas::SpatialHashingSearcher value;
    std::size_t particle_count = 0;
};

void
require_matching_grid(const atlas::SpatialHashingSearcher& searcher, const atlas::Universe& universe) {
    if (universe.cell_size() != searcher.cell_size()
        || universe.lower_corner() != searcher.lower_corner()
        || universe.grid_size() != searcher.grid_size()) {
        throw nb::value_error("universe geometry must match the searcher grid");
    }
}

template <typename T>
nb::object
searcher_array(const T* data, const std::size_t count) {
    atlas::HostBuffer<T> host(count);
    atlas::copy_device_to_host(data, atlas::raw_pointer_cast(host.data()), count);
    return atlas::python::numpy_copy(host, count);
}

}
