#pragma once

#include <atlas/core/macros.h>
#include <atlas/random/default_random_engine.h>

namespace atlas {

/**
 * @brief A uniform continuous distribution over the half-open interval `[min, max)`.
 *
 * Maps an @ref atlas::default_random_engine draw onto a canonical variate and rescales it.
 * The engine yields `[1, m-1]`, so `(x - 1) / (m - 1)` lands in `[0, 1)` exactly.
 * Implemented here, not aliased, for the same reason the engine is: the host and CUDA
 * backends must produce the identical stream.
 *
 * Throughout the sampling helpers this is instantiated with `min = 0, max = 1` to obtain
 * `u` in `[0, 1)`, which is then transformed (Box-Muller, inverse CDF, and so on).
 *
 * @tparam T The floating-point result type; defaults to `float`, matching the
 *           single-precision arithmetic used across the device kernels.
 * @see atlas::default_random_engine
 */
template <typename T = float>
class uniform_real_distribution final {
public:
    /// The type @ref operator() yields.
    using result_type = T;

    /**
     * @brief Construct over `[0, 1)`.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    uniform_real_distribution() noexcept
        : _min(static_cast<T>(0))
        , _max(static_cast<T>(1)) {
    }

    /**
     * @brief Construct over `[min_value, max_value)`.
     * @param min_value Inclusive lower bound.
     * @param max_value Exclusive upper bound. A value below @p min_value yields draws
     *                  outside the interval; the caller is responsible for ordering them.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    uniform_real_distribution(const T min_value, const T max_value) noexcept
        : _min(min_value)
        , _max(max_value) {
    }

    /**
     * @brief Draw one variate, advancing @p engine once.
     *
     * @param engine The generator to draw from; its state advances.
     * @return A value in `[min(), max())`.
     * @note The canonical variate is rounded to `float` before rescaling, and for the
     *       largest engine states that quotient can round up to exactly `1`. It is nudged
     *       back below one so the interval stays half-open, which callers such as
     *       @c generate_standard_normal_pair rely on when they take `log(u)`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE result_type
    operator()(default_random_engine& engine) const noexcept {
        constexpr T range = static_cast<T>(default_random_engine::max() - default_random_engine::min() + 1u);

        const T canonical = static_cast<T>(engine() - default_random_engine::min()) / range;

        const T bounded = (canonical < static_cast<T>(1)) ? canonical : nextafter_below_one();

        return _min + (_max - _min) * bounded;
    }

    /** @brief Inclusive lower bound. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE result_type
    min() const noexcept {
        return _min;
    }

    /** @brief Exclusive upper bound. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE result_type
    max() const noexcept {
        return _max;
    }

private:
    /** @brief The largest representable @p T strictly below one. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static constexpr result_type
    nextafter_below_one() noexcept {
        return static_cast<T>(1) - static_cast<T>(1) / static_cast<T>(1 << 24);
    }

    T _min; ///< Inclusive lower bound.

    T _max; ///< Exclusive upper bound.
};

}
