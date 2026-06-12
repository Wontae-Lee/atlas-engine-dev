#pragma once

/**
 * @file shuffle_operator.h
 * @brief Declares a deterministic hash-based shuffle operator for index randomization.
 *
 * @details
 * This header defines @ref atlas::ShuffleOperator, a lightweight,
 * backend-portable utility for generating pseudo-random permutations of integer
 * indices using a stateless hash function.
 *
 * ## Purpose
 * The shuffle operator is intended for:
 * - deterministic randomization of indices,
 * - building randomized iteration orders,
 * - generating reproducible pseudo-random keys for sorting or shuffling,
 * - avoiding shared RNG state in parallel or device code.
 *
 * Unlike traditional random-number generators, this operator:
 * - does not maintain mutable state,
 * - derives randomness purely from the input index and seed,
 * - is safe for use in parallel and GPU contexts.
 *
 * ## Algorithm overview
 * The implementation follows a hash-mixing pattern similar to splitmix-style
 * integer hashing:
 * - the input index is first combined with a seed and a large offset constant,
 * - several rounds of bitwise XOR-shift and multiplication are applied,
 * - the final value is a well-mixed 64-bit pseudo-random key.
 *
 * This produces:
 * - good bit dispersion,
 * - low correlation between nearby indices,
 * - deterministic and reproducible output.
 *
 * ## Determinism
 * For a fixed:
 * - input index,
 * - seed value,
 *
 * the resulting shuffle key is guaranteed to be identical across:
 * - host and device execution,
 * - repeated runs.
 *
 * ## Usage patterns
 * Typical usage includes:
 * - sorting indices by shuffle keys to obtain a randomized ordering,
 * - mapping particle indices to pseudo-random traversal sequences,
 * - generating decorrelated seeds for further sampling operations.
 *
 * ---
 */

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

} // namespace atlas

#include <atlas/shuffle/shuffle_operator.hpp>
