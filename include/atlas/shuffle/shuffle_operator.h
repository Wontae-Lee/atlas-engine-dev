#pragma once

#include <atlas/core/macros.h>

#include <cstdint>

namespace atlas::system {

/**
 * @brief Stateless hash-based shuffle key generator.
 *
 * @details
 * `ShuffleOperator` converts a stable element index plus an external seed into
 * a pseudo-random 64-bit sort key. The generated keys can then be used with
 * `parallel_sort_by_key()` to produce a deterministic backend-portable shuffle
 * of an associated value buffer.
 *
 * The operator is intentionally lightweight and device-callable so it can be
 * invoked inside CUDA kernels or the TBB-backed `ExecutionPolicy::device`
 * fallback used by Atlas.
 */
struct ShuffleOperator final {
    const int first_shift                 = 30;
    const int second_shift                = 27;
    const int final_shift                 = 31;
    const std::uint64_t index_offset      = 0x9e3779b97f4a7c15ull;
    const std::uint64_t first_multiplier  = 0xbf58476d1ce4e5b9ull;
    const std::uint64_t second_multiplier = 0x94d049bb133111ebull;

    /**
     * @brief Functor-call shorthand for @ref shuffle_key.
     *
     * @param index Element index in the sequence being shuffled.
     * @param seed External seed selecting a different permutation family.
     * @return Mixed 64-bit key suitable for sorting.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE std::uint64_t
    operator()(int index,
               std::uint64_t seed) const noexcept;

    /**
     * @brief Generate a pseudo-random 64-bit sort key for an element index.
     *
     * @param index Element index in the sequence being shuffled.
     * @param seed External seed selecting a different permutation family.
     * @return Mixed 64-bit key suitable for sorting.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE std::uint64_t
    shuffle_key(int index,
                std::uint64_t seed) const noexcept;
};

} // namespace atlas::system

namespace atlas {

using ShuffleOperator = atlas::system::ShuffleOperator;

} // namespace atlas

#include <atlas/shuffle/shuffle_operator.hpp>
