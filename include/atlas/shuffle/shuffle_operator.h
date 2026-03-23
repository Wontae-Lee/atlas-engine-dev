#pragma once

#include <atlas/core/macros.h>

#include <cstdint>

namespace atlas::system {

struct ShuffleOperator final {
    const int first_shift                 = 30;
    const int second_shift                = 27;
    const int final_shift                 = 31;
    const std::uint64_t index_offset      = 0x9e3779b97f4a7c15ull;
    const std::uint64_t first_multiplier  = 0xbf58476d1ce4e5b9ull;
    const std::uint64_t second_multiplier = 0x94d049bb133111ebull;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE std::uint64_t
    operator()(int index,
               std::uint64_t seed) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE std::uint64_t
    shuffle_key(int index,
                std::uint64_t seed) const noexcept;
};

}

namespace atlas {

using ShuffleOperator = atlas::system::ShuffleOperator;

}

#include <atlas/shuffle/shuffle_operator.hpp>