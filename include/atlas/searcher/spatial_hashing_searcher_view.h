#pragma once

#include <atlas/core/macros.h>

#include <cstdint>

namespace atlas {

struct SpatialHashingSearcherView final {

    const std::uint32_t* cell_key {};

    const int* sorted_index {};

    const int* cell_start {};

    const int* cell_end {};
};

}