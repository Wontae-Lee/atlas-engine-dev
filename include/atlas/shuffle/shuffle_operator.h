#pragma once

#include <atlas/core/macros.h>
#include <atlas/random/seed.h>

#include <cstdint>

namespace atlas {

class ShuffleOperator final {
public:
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE std::uint64_t
    shuffle_key(int index,
                std::uint64_t seed) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE std::uint64_t
    operator()(int index,
               std::uint64_t seed) const noexcept;
};

}

#include <atlas/shuffle/shuffle_operator.hpp>