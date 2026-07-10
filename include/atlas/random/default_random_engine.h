#pragma once

#include <atlas/core/macros.h>

#include <cstdint>

namespace atlas {

/**
 * @brief The engine-wide pseudo-random number generator: a minimal-standard LCG.
 *
 * `x_{n+1} = 48271 * x_n mod (2^31 - 1)` — Park and Miller's minimal standard generator,
 * the same recurrence thrust's default engine uses. It is implemented here rather than
 * aliased so that **both backends draw from the identical stream**: a host build without
 * Thrust and a CUDA build must agree, or a case reproduced on the CPU would diverge from
 * the GPU run it is meant to check.
 *
 * Every member is @c ATLAS_ALL_DEVICE and the whole type is trivially copyable, so each
 * thread constructs its own engine inside a kernel, seeds it from its index, and draws
 * locally. No shared mutable state, no locking.
 *
 * @note Fast and cheap to copy, but not cryptographically strong — an acceptable
 *       trade-off for Monte-Carlo particle sampling.
 * @warning The state must never reach zero: `48271 * 0 mod m` is 0 and the engine would
 *          emit zero forever. @ref seed maps a zero seed onto @ref default_seed, so a
 *          caller passing a zero-valued hash key is safe.
 * @see atlas::uniform_real_distribution, atlas::generate_standard_normal
 */
class default_random_engine final {
public:
    /// The type @ref operator() yields.
    using result_type = std::uint32_t;

    /// Multiplier `a` of the recurrence.
    static constexpr result_type multiplier = 48271u;

    /// Modulus `m` of the recurrence; the Mersenne prime 2^31 - 1.
    static constexpr result_type modulus = 2147483647u;

    /// State used when a caller seeds with a value congruent to zero.
    static constexpr result_type default_seed = 1u;

    /**
     * @brief Construct with @ref default_seed.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    default_random_engine() noexcept
        : _state(default_seed) {
    }

    /**
     * @brief Construct and seed in one step.
     * @param s Seed; reduced modulo @ref modulus, with zero mapped to @ref default_seed.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit default_random_engine(const result_type s) noexcept {
        seed(s);
    }

    /**
     * @brief Reset the state.
     * @param s Seed; reduced modulo @ref modulus, with zero mapped to @ref default_seed.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    seed(const result_type s) noexcept {
        const result_type reduced = s % modulus;

        _state = (reduced == 0u) ? default_seed : reduced;
    }

    /**
     * @brief Advance the state and return the new value.
     *
     * The product `48271 * x` overflows 32 bits, so it is formed in 64 bits before the
     * modulo. That is the whole reason for the @c std::uint64_t temporary.
     *
     * @return The next variate, in `[min(), max()]`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE result_type
    operator()() noexcept {
        const std::uint64_t product
            = static_cast<std::uint64_t>(multiplier) * static_cast<std::uint64_t>(_state);

        _state = static_cast<result_type>(product % static_cast<std::uint64_t>(modulus));

        return _state;
    }

    /**
     * @brief Discard the next @p count variates.
     * @param count How many draws to skip.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    discard(const unsigned long long count) noexcept {
        for (unsigned long long i = 0; i < count; ++i) {
            static_cast<void>((*this)());
        }
    }

    /** @brief Smallest value @ref operator() can return. The state is never zero. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static constexpr result_type
    min() noexcept {
        return 1u;
    }

    /** @brief Largest value @ref operator() can return. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static constexpr result_type
    max() noexcept {
        return modulus - 1u;
    }

private:
    result_type _state { default_seed }; ///< Current state; always in `[1, modulus - 1]`.
};

}
