#pragma once

#include <atlas/core/macros.h>

#include <cstdint>

namespace atlas {

// The searcher's four device arrays, as raw pointers. Trivially copyable, so a
// kernel captures it by value instead of the searcher itself (which owns
// host-only DeviceBuffers).
//
// Cell c owns the slice sorted_index[cell_start[c] .. cell_end[c]) of the
// cell-sorted particle order; cell_start[c] < 0 marks an empty cell.
struct SpatialHashingSearcherView final {

    const std::uint32_t* cell_key {};

    const int* sorted_index {};

    const int* cell_start {};

    const int* cell_end {};
};

}
