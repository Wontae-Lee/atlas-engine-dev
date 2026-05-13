#pragma once

/**
 * @file shuffle_operator.h
 * @brief Declares a deterministic hash-based shuffle operator for index randomization.
 *
 * @details
 * This header defines @ref atlas::system::ShuffleOperator, a lightweight,
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

namespace atlas::system {

/**
 * @brief Stateless hash-based operator for deterministic index shuffling.
 *
 * @details
 * @ref ShuffleOperator maps an integer index and a user-provided seed to a
 * 64-bit pseudo-random value using a sequence of bit-mixing operations.
 *
 * The operator is:
 * - deterministic,
 * - backend-portable (host and device),
 * - free of shared mutable state.
 *
 * ## Internal constants
 * The operator uses fixed constants for:
 * - additive offset,
 * - multiplicative mixing,
 * - XOR-shift steps,
 *
 * chosen to provide strong bit diffusion and low collision correlation.
 *
 * ## Design notes
 * - The operator does not produce a permutation by itself; instead, it produces
 *   sortable keys that can be used to derive a permutation.
 * - It is suitable for parallel contexts where thread-safe randomness is required.
 *
 * ---
 */
struct ShuffleOperator final {
    /**
     * @brief Compute a pseudo-random value from an index and seed.
     *
     * @details
     * Applies a sequence of XOR-shift and multiplication operations to produce
     * a well-mixed 64-bit value.
     *
     * This function is equivalent to @ref shuffle_key and is provided as the
     * call operator for convenience.
     *
     * @param index Input integer index.
     * @param seed User-provided seed value.
     * @return 64-bit pseudo-random value derived from @p index and @p seed.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE std::uint64_t
    operator()(int index,
               std::uint64_t seed) const noexcept;

    /**
     * @brief Generate a deterministic shuffle key for an index.
     *
     * @details
     * Produces a pseudo-random 64-bit key suitable for:
     * - sorting-based shuffling,
     * - randomized indexing,
     * - deterministic permutation construction.
     *
     * The output value is derived solely from:
     * - the input @p index,
     * - the supplied @p seed,
     * ensuring reproducibility and thread safety.
     *
     * @param index Input integer index.
     * @param seed User-provided seed value.
     * @return 64-bit shuffle key.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE std::uint64_t
    shuffle_key(int index,
                std::uint64_t seed) const noexcept;
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for @ref atlas::system::ShuffleOperator.
 */
using ShuffleOperator = atlas::system::ShuffleOperator;

} // namespace atlas

#include <atlas/shuffle/shuffle_operator.hpp>
