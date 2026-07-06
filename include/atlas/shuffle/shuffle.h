#pragma once

#include <atlas/core/macros.h>
#include <atlas/random/seed.h>

#include <cstdint>

namespace atlas {

class Shuffle final {
public:
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::uint64_t
    shuffle_key(const int index,
                const std::uint64_t seed) const noexcept {
        std::uint64_t value = static_cast<std::uint64_t>(index) + seed + atlas::SHUFFLE_HASH_INDEX_OFFSET;
        value               = (value ^ (value >> atlas::SHUFFLE_HASH_FIRST_SHIFT)) * atlas::SHUFFLE_HASH_FIRST_MULTIPLIER;
        value               = (value ^ (value >> atlas::SHUFFLE_HASH_SECOND_SHIFT)) * atlas::SHUFFLE_HASH_SECOND_MULTIPLIER;
        return value ^ (value >> atlas::SHUFFLE_HASH_FINAL_SHIFT);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::uint64_t
    operator()(const int index,
               const std::uint64_t seed) const noexcept {
        return shuffle_key(index, seed);
    }
};

}
